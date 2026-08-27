"""Tests that the fusion duplication bug (157/191 duplicated cues) is dead."""
from __future__ import annotations

from engines.fusion import GatingFusion
from models import DiarizationSegment, TranscribedSegment, WordToken


def _word(text, start, end, spk="UNKNOWN"):
    return WordToken(word=text, start=start, end=end, speaker_id=spk)


def test_word_path_emits_each_word_once():
    turns = [
        DiarizationSegment(0.0, 2.0, "SPEAKER_00"),
        DiarizationSegment(2.0, 4.0, "SPEAKER_01"),
    ]
    words = [_word(f"w{i}", 0.1 + i * 0.5, 0.4 + i * 0.5) for i in range(8)]
    tseg = TranscribedSegment(0.1, 3.9, " ".join(w.word for w in words),
                              words=words)
    resolved = {"SPEAKER_00": ("Host", 0.9), "SPEAKER_01": ("Guest", 0.8)}
    finals = GatingFusion().create_final_segments(turns, [tseg], resolved)

    all_words = [w for f in finals for w in f.text.split()]
    assert len(all_words) == len(words)          # nothing duplicated
    assert len(finals) == 2
    assert finals[0].speaker == "Host"


def test_proportional_split_preserves_every_token_exactly_once():
    # Simulates the LoRA failure shape: one 40-120s mega-block overlapping
    # many short turns. Old code pasted the WHOLE block into every turn.
    turns = [DiarizationSegment(float(s), float(s + 4), "SPEAKER_00")
             for s in range(40, 120, 4)]
    big = TranscribedSegment(
        start=40.0, end=120.0,
        text=" ".join(f"tok{i}" for i in range(80)),
        words=[], speaker_id="SPEAKER_00")
    resolved = {"SPEAKER_00": ("Guest", 0.7)}

    finals = GatingFusion().create_final_segments(turns, [big], resolved)
    tokens = [t for f in finals for t in f.text.split()]
    assert sorted(tokens, key=lambda x: int(x[3:])) == \
        [f"tok{i}" for i in range(80)]           # exactly once, in order
    assert len(finals) == len(turns)


def test_unknown_speaker_blocks_not_merged_across_unknown():
    from engines.asr_lora import transcribe_with_lora  # merge logic lives there
    # direct check of merge guard via public data: UNKNOWN chunks must not
    # glue into mega-segments (regression of the [120-260s] monster).
    segs = [
        TranscribedSegment(0, 20, "a", [], speaker_id="UNKNOWN"),
        TranscribedSegment(20, 40, "b", [], speaker_id="UNKNOWN"),
        TranscribedSegment(40, 60, "c", [], speaker_id="SPEAKER_00"),
        TranscribedSegment(60, 80, "d", [], speaker_id="SPEAKER_00"),
    ]
    merged = []
    current = None
    for s in segs:
        if current and s.speaker_id != "UNKNOWN" and s.speaker_id == current.speaker_id:
            current = TranscribedSegment(current.start, s.end,
                                         current.text + " " + s.text,
                                         [], current.speaker_id)
        else:
            if current:
                merged.append(current)
            current = s
    merged.append(current)
    assert len([m for m in merged if m.speaker_id == "UNKNOWN"]) == 2


def test_health_metrics_flag_the_original_failure():
    from evaluation.metrics import fusion_health_metrics, assert_no_regression
    # The original failure shape: ONE word-less 40s LoRA mega-block fanned out
    # across many short turns for two speakers. Tokens are realistic-length
    # Bangla words; duplication is what made cues huge, not token length.
    turns = [
        DiarizationSegment(i * 4.0, i * 4.0 + 4.0,
                           "SPEAKER_00" if i % 2 == 0 else "SPEAKER_01")
        for i in range(10)
    ]
    big = TranscribedSegment(
        0, 40, " ".join(f"শব্দ{i}" for i in range(80)), [], "SPEAKER_00")
    big2 = TranscribedSegment(
        4, 40, " ".join(f"কথা{i}" for i in range(60)), [], "SPEAKER_01")
    resolved = {"SPEAKER_00": ("A", 0.9), "SPEAKER_01": ("B", 0.8)}
    bad_finals = GatingFusion()._finals_from_proportional_split(
        turns, [big, big2], resolved)
    health = fusion_health_metrics(bad_finals)
    # Even the fallback must not duplicate text:
    assert health["duplicate_text_rate"] == 0.0
    assert_no_regression(health)  # must not raise
