"""Metrics for joint ASR + speaker diarization + identity resolution.

Implemented here without hard runtime dependencies so unit tests run anywhere;
jiwer / pyannote.metrics are used opportunistically when installed.

References:
- WER/CER: standard edit-distance metrics.
- cpWER: concatenated minimum-permutation WER (Google USM; MeetEval toolkit).
- WDER: word diarization error rate - fraction of correctly transcribed words
  carrying the wrong speaker label.
- DER/JER: pyannote definitions (false alarm + missed + confusion / total),
  evaluated with a 0s collar.
"""
from __future__ import annotations

import itertools
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

from models import DiarizationSegment, FinalSegment


# ----------------------------------------------------------------------
# Text metrics
# ----------------------------------------------------------------------
def _edit_distance(a: Sequence, b: Sequence) -> int:
    """Classic Levenshtein distance over arbitrary token sequences."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _normalize_words(text: str) -> List[str]:
    """Bangla-aware normalization: NFC fold, strip punctuation incl. danda."""
    import re
    import unicodedata

    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[।,;:.!?<>|\[\]{}()\"'“”‘’—–\-]", " ", text)
    return [w for w in text.split() if w]


def _try_jiwer_wer(ref: str, hyp: str) -> float | None:
    try:
        import jiwer  # noqa: WPS433
        out = jiwer.wer(ref, hyp)
        return float(out)
    except Exception:
        return None


def wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate in [0, inf); uses jiwer when available."""
    jw = _try_jiwer_wer(reference, hypothesis)
    if jw is not None:
        return jw
    ref, hyp = _normalize_words(reference), _normalize_words(hypothesis)
    if not ref:
        return 0.0 if not hyp else float("inf")
    return _edit_distance(ref, hyp) / len(ref)


def cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate (better suited than WER for agglutinative Bangla)."""
    ref = list(_norm_chars(reference))
    hyp = list(_norm_chars(hypothesis))
    if not ref:
        return 0.0 if not hyp else float("inf")
    return _edit_distance(ref, hyp) / len(ref)


def _norm_chars(text: str) -> List[str]:
    import unicodedata
    text = unicodedata.normalize("NFC", text)
    drop = set(" ।,;:.!?<>|[]{}()\"'“”‘’—–-\n\t")
    return [c for c in text if c not in drop]


# ----------------------------------------------------------------------
# Joint ASR+diarization metrics
# ----------------------------------------------------------------------
def cpwer(
    reference_by_speaker: Dict[str, str],
    hypothesis_by_speaker: Dict[str, str],
) -> Tuple[float, float]:
    """Concatenated minimum-permutation WER.

    Tries every bijection between hypothesis and reference speaker labels
    (speaker counts are small, <= ~6) and returns (best_wer, best_perm_error).

    Returns:
        (min WER over permutations, number of permutations searched).
    """
    ref_spk = sorted(reference_by_speaker)
    hyp_spk = sorted(hypothesis_by_speaker)
    if not ref_spk or not hyp_spk:
        return (float("inf"), 0)

    best = float("inf")
    n_searched = 0
    # Map hypothesis speakers onto reference speakers via injections. For
    # each permutation, BOTH streams are concatenated in REFERENCE label
    # order -- concatenating the hypothesis in its own label order (the
    # original bug) makes the correct permutation score nonzero.
    shorter_is_hyp = len(hyp_spk) <= len(ref_spk)
    if shorter_is_hyp:
        perms = (
            dict(zip(p, hyp_spk))                     # ref_label -> hyp_label
            for p in itertools.permutations(ref_spk, len(hyp_spk))
        )
    else:
        perms = (
            dict(zip(ref_spk, p))                     # ref_label -> hyp_label
            for p in itertools.permutations(hyp_spk, len(ref_spk))
        )
    for mapping in perms:
        concat_ref = " ".join(reference_by_speaker[r] for r in ref_spk)
        concat_hyp = " ".join(
            hypothesis_by_speaker[mapping[r]] for r in ref_spk if r in mapping)
        best = min(best, wer(concat_ref, concat_hyp))
        n_searched += 1
    return (best, n_searched)


def wder(
    reference_words: List[Tuple[float, float, str]],   # (start, end, spk)
    hypothesis_words: List[Tuple[float, float, str]],
    tolerance: float = 0.5,
) -> float:
    """Word Diarization Error Rate: fraction of correctly-timed words that
    carry an incorrect speaker label. Directly measures who-said-what quality.
    """
    if not reference_words:
        return 0.0
    errors = 0
    matched = 0
    for rs, re_, rspk in reference_words:
        rmid = (rs + re_) / 2.0
        best_i, best_d = None, float("inf")
        for i, (hs, he, _) in enumerate(hypothesis_words):
            hmid = (hs + he) / 2.0
            d = abs(hmid - rmid)
            if d < best_d:
                best_d, best_i = d, i
        if best_i is not None and best_d <= tolerance:
            matched += 1
            if hypothesis_words[best_i][2] != rspk:
                errors += 1
    return errors / matched if matched else 0.0


# ----------------------------------------------------------------------
# Diarization metrics (pure-python DER/JER, pyannote-style accounting)
# ----------------------------------------------------------------------
def der_jer(
    reference: List[DiarizationSegment],
    hypothesis: List[DiarizationSegment],
    collar: float = 0.0,
) -> Dict[str, float]:
    """DER/JER via interval sweep at sample rate 100 Hz (collar applied by
    shrinking both sides). DER = (FA + Miss + Confusion) / ref_duration.
    JER = Jaccard-based joint error per reference speaker, averaged."""
    sr_scale = 100
    def to_cells(segs):
        cells = defaultdict(set)          # cell_index -> set(speakers)
        for s in segs:
            a = int(round((s.start + collar) * sr_scale))
            b = int(round((max(s.start + collar, s.end - collar)) * sr_scale))
            for c in range(max(0, a), max(0, b)):
                cells[c].add(s.speaker_id)
        return cells

    ref_cells, hyp_cells = to_cells(reference), to_cells(hypothesis)
    all_cells = sorted(set(ref_cells) | set(hyp_cells))

    total_ref = sum(len(v) for v in ref_cells.values())
    fa = miss = conf = 0
    for c in all_cells:
        R, H = ref_cells.get(c, set()), hyp_cells.get(c, set())
        if R and H:
            common = R & H
            if not common:
                conf += 1                 # wrong speaker entirely
            else:
                conf += len(R) - len(common) + len(H) - len(common)
            # miss/fa only when exactly one side present handled below too
        elif R:
            miss += len(R)
        elif H:
            fa += len(H)

    der = (fa + miss + conf) / total_ref if total_ref else 0.0

    # Jaccard error per reference speaker.
    ref_spk_time = defaultdict(float)
    for s in reference:
        ref_spk_time[s.speaker_id] += max(0.0, s.end - s.start)
    jer_terms = []
    for spk, dur in ref_spk_time.items():
        inter = union = 0
        for c in all_cells:
            in_r = spk in ref_cells.get(c, set())
            in_h = bool(ref_cells.get(c, set()) and
                        (ref_cells[c] & hyp_cells.get(c, set()) == {spk}))
            if in_r or in_h:
                union += 1
                if in_r and in_h:
                    inter += 1
        jer_terms.append((union - inter) / union if union else 1.0)
    jer = sum(jer_terms) / len(jer_terms) if jer_terms else 0.0

    return {"DER": round(der, 4), "JER": round(jer, 4),
            "false_alarm": round(fa / total_ref, 4) if total_ref else 0.0,
            "missed_detection": round(miss / total_ref, 4) if total_ref else 0.0,
            "confusion": round(conf / total_ref, 4) if total_ref else 0.0}


# ----------------------------------------------------------------------
# Identity-resolution metrics
# ----------------------------------------------------------------------
def speaker_name_accuracy(
    reference_turns: List[DiarizationSegment],     # speaker_id = TRUE NAME
    final_segments: List[FinalSegment],            # speaker = resolved name
) -> float:
    """Fraction of reference speaking time labelled with the correct name."""
    total = correct = 0.0
    for turn in reference_turns:
        total += turn.end - turn.start
        covered = [
            fs for fs in final_segments
            if fs.speaker == turn.speaker_id and
            min(fs.end, turn.end) > max(fs.start, turn.start)
        ]
        correct += sum(min(fs.end, turn.end) - max(fs.start, turn.start) for fs in covered)
    return correct / total if total else 0.0


def face_attribution_accuracy(
    reference_face_labels: List[Tuple[float, str]],    # (time, true_name)
    predicted_faces: List[Tuple[float, str]],
    tolerance: float = 1.0,
) -> Dict[str, float]:
    """Precision/recall/F1 of face->name attributions within a time tolerance."""
    tp = 0
    matched_pred = set()
    matched_ref = set()
    for j, (pt, pname) in enumerate(predicted_faces):
        best_i, best_d = None, float("inf")
        for i, (rt, rname) in enumerate(reference_face_labels):
            if i in matched_ref:
                continue
            d = abs(rt - pt)
            if d < best_d:
                best_d, best_i = d, i
        if best_i is not None and best_d <= tolerance and \
                reference_face_labels[best_i][1] == pname:
            tp += 1
            matched_pred.add(j)
            matched_ref.add(best_i)
    precision = tp / len(predicted_faces) if predicted_faces else 0.0
    recall = tp / len(reference_face_labels) if reference_face_labels else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4)}


# ----------------------------------------------------------------------
# Fusion output health checks (regression assertions)
# ----------------------------------------------------------------------
def fusion_health_metrics(final_segments: List[FinalSegment]) -> Dict[str, object]:
    """Cheap structural sanity metrics that catch the failure modes observed
    in the first end-to-end run:
      - duplicate_text_rate (was 82%!)
      - avg/max cue length in chars (was 2836 avg)
      - timeline coverage
      - distinct speakers
    """
    texts = [fs.text for fs in final_segments]
    duplicates = sum(1 for i, t in enumerate(texts) if t and t in texts[:i])
    durations = [(fs.end - fs.start) for fs in final_segments]
    coverage = sum(durations) if final_segments else 0.0
    span = max((fs.end for fs in final_segments), default=0.0) - \
        min((fs.start for fs in final_segments), default=0.0)
    return {
        "num_segments": len(final_segments),
        "duplicate_text_count": duplicates,
        "duplicate_text_rate": round(duplicates / len(final_segments), 4)
        if final_segments else 0.0,
        "avg_cue_chars": round(sum(len(t) for t in texts) / len(texts), 1)
        if texts else 0.0,
        "max_cue_chars": max((len(t) for t in texts), default=0),
        "coverage_ratio": round(coverage / span, 4) if span > 0 else 0.0,
        "distinct_speakers": len({fs.speaker for fs in final_segments}),
    }


def assert_no_regression(health: Dict[str, object]) -> None:
    """Smoke assertions every run must pass (would have caught the first
    failed run in seconds). Raises AssertionError with a readable message."""
    dup_rate = float(health["duplicate_text_rate"])
    assert dup_rate <= 0.05, \
        f"duplicate_text_rate={dup_rate} exceeds 5% -- fusion join regression"
    avg_cue = float(health["avg_cue_chars"])
    assert avg_cue <= 200, \
        f"avg_cue_chars={avg_cue} exceeds 200 -- unsplit mega-segments present"
    assert int(health["distinct_speakers"]) >= 2, "fewer than 2 speakers emitted"
