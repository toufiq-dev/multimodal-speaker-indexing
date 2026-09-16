"""Tests for constraint C1: word coverage must not masquerade as identity gain.

v9 and v10 ran the same identity logic on identical diarization and face
evidence. v10 scored 1.2 points lower on time-weighted speaker-name accuracy
purely because its decode emitted 58 fewer words. These tests pin the metrics
that separate the two effects.
"""
from __future__ import annotations

import pytest

from evaluation.metrics import (attributed_time_accuracy,
                                coverage_matched_accuracy,
                                speaker_name_accuracy, word_recall)
from models import DiarizationSegment, FinalSegment

# Two reference turns, 10 s each, different speakers.
REF = [DiarizationSegment(0, 10, "A"), DiarizationSegment(10, 20, "B")]


def _hyp(*rows):
    return [FinalSegment(s, e, spk, txt) for s, e, spk, txt in rows]


def test_dropped_words_sink_raw_accuracy_but_not_coverage_matched():
    """The v9/v10 failure in miniature: identical, correct labelling, but the
    second run transcribes nothing over turn B."""
    full = _hyp((0, 10, "A", "ok"), (10, 20, "B", "ok"))
    partial = _hyp((0, 10, "A", "ok"))          # same labels, fewer words

    assert speaker_name_accuracy(REF, full) == pytest.approx(1.0)
    assert speaker_name_accuracy(REF, partial) == pytest.approx(0.5)   # confound

    assert coverage_matched_accuracy(REF, full)["accuracy"] == pytest.approx(1.0)
    cm = coverage_matched_accuracy(REF, partial)
    assert cm["accuracy"] == pytest.approx(1.0)      # identity logic unchanged
    assert cm["n_turns_covered"] == 1 and cm["n_turns_dropped"] == 1
    assert cm["turn_coverage"] == pytest.approx(0.5)


def test_coverage_matched_still_charges_a_mislabelled_covered_turn():
    """It excuses silence, never a wrong name."""
    wrong = _hyp((0, 10, "A", "ok"), (10, 20, "A", "ok"))
    assert coverage_matched_accuracy(REF, wrong)["accuracy"] == pytest.approx(0.5)


def test_empty_cues_do_not_count_as_coverage():
    """A cue with no words is silence the decoder failed at, not an attribution."""
    blank = _hyp((0, 10, "A", "ok"), (10, 20, "B", "   "))
    cm = coverage_matched_accuracy(REF, blank)
    assert cm["n_turns_covered"] == 1


def test_attributed_time_ignores_partial_coverage_of_a_turn():
    """Half of turn B transcribed, correctly named: coverage-matched still
    charges the silent half (it is a covered turn), attributed-time does not."""
    half = _hyp((0, 10, "A", "ok"), (10, 15, "B", "ok"))
    assert coverage_matched_accuracy(REF, half)["accuracy"] == pytest.approx(0.75)
    at = attributed_time_accuracy(REF, half)
    assert at["accuracy"] == pytest.approx(1.0)
    assert at["hypothesised_sec"] == pytest.approx(15.0)


def test_attributed_time_charges_wrong_names_only():
    mixed = _hyp((0, 10, "A", "ok"), (10, 20, "A", "ok"))
    assert attributed_time_accuracy(REF, mixed)["accuracy"] == pytest.approx(0.5)


def test_word_recall_is_not_inflated_by_a_repetition_loop():
    """The count-ratio proxy rewards looping; aligned recall does not. This is
    why v9 (5-gram repeat 0.064) out-scored v10 (0.000) on the proxy."""
    ref = "এক দুই তিন চার পাঁচ ছয়"
    looped = "এক দুই এক দুই এক দুই এক দুই"
    r = word_recall(ref, looped)
    assert r["word_count_ratio"] > 1.0          # proxy says 133% "recall"
    assert r["word_recall"] == pytest.approx(2 / 6, abs=1e-4)


def test_word_recall_counts_only_matched_words():
    assert word_recall("a b c d", "a b c d")["word_recall"] == pytest.approx(1.0)
    assert word_recall("a b c d", "a x c d")["word_recall"] == pytest.approx(0.75)
    assert word_recall("a b c d", "")["word_recall"] == pytest.approx(0.0)


def test_ablation_summary_reports_the_noise_floor(capsys):
    """C2: with repeats the table must show the spread, and say so when it
    cannot. The illustrative rows below are the real v9/v10 pair -- same
    config, 58 words apart."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "ablate_asr", Path(__file__).resolve().parents[1] / "scripts" / "ablate_asr.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def row(name, rep, wer, words, repeat_ratio):
        return {"name": name, "rep": rep, "wer": wer, "cer": 0.3,
                "hyp_words": words, "word_recall": 0.68, "word_recall_proxy": 0.8,
                "ngram_repeat_ratio": repeat_ratio, "elapsed_sec": 90.0}

    two = [row("baseline_v9", 0, 0.3675, 924, 0.0641),
           row("baseline_v9", 1, 0.3436, 866, 0.0)]
    mod.print_summary(two)
    out = capsys.readouterr().out
    assert "NOISE FLOOR" in out and "0.0239" in out       # the WER spread
    assert "words +-58" in out

    mod.print_summary(two[:1])
    assert "--repeat 2" in capsys.readouterr().out
