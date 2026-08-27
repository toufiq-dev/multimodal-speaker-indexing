"""Baseline implementations and single-modality ablations.

Baselines:
    B1  heuristic_fusion        - diarization turns + ASR text by max temporal
                                  overlap; no faces, no NLP, no anchors.
    B2  text_only               - raw ASR blocks as-is (no diarization).
    B3  audio_only_naming       - audio pipeline + generic Speaker_N labels.

Single-modality ablations (swap one component off at a time):
    A1  -faces                  : identity resolution without vision evidence
    A2  -nlp_names              : without NER/anchor names
    A3  +ground_truth_labels    : annotated labels instead of automatic naming
Each returns FinalSegment lists consumable by evaluation/metrics.py.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from models import DiarizationSegment, FaceOccurrence, TranscribedSegment, FinalSegment
from engines.fusion import GatingFusion


def heuristic_fusion(
    diarization: List[DiarizationSegment],
    transcribed: List[TranscribedSegment],
) -> List[FinalSegment]:
    """B1: classic audio-only heuristic. For each turn take the text of the
    max-overlap same-speaker transcribed block (whole block; this baseline is
    deliberately the pre-fix behaviour for comparison)."""
    finals: List[FinalSegment] = []
    spk_counter: Dict[str, int] = {}
    for d in sorted(diarization, key=lambda x: x.start):
        best, best_ov = None, 0.0
        for t in transcribed:
            if t.speaker_id != d.speaker_id:
                continue
            ov = min(t.end, d.end) - max(t.start, d.start)
            if ov > best_ov:
                best_ov, best = ov, t
        if best is None or not best.text:
            continue
        name = f"Speaker_{spk_counter.setdefault(d.speaker_id, len(spk_counter) + 1)}"
        finals.append(FinalSegment(start=d.start, end=d.end, speaker=name,
                                   text=best.text,
                                   confidence=0.5))
    return finals


def text_only(
    transcribed: List[TranscribedSegment],
) -> List[FinalSegment]:
    """B2: no diarization at all - raw ASR output as segments."""
    return [
        FinalSegment(start=t.start, end=t.end, speaker="ASR_BLOCK",
                     text=t.text, confidence=0.5)
        for t in sorted(transcribed, key=lambda x: x.start) if t.text
    ]


def run_identity_pipeline(
    diarization: List[DiarizationSegment],
    transcribed: List[TranscribedSegment],
    faces: Optional[List[FaceOccurrence]] = None,
    ordered_names: Optional[List[str]] = None,
    ground_truth_labels: Optional[Dict[str, str]] = None,
) -> List[FinalSegment]:
    """Full pipeline entry used by ablations; component toggles are expressed
    by passing empty evidence."""
    fusion = GatingFusion(ordered_names=ordered_names or [])
    resolved = fusion.resolve_identities(
        diarization, transcribed, faces or [],
        ground_truth_labels=ground_truth_labels)
    return fusion.create_final_segments(diarization, transcribed, resolved)


# ------------------------- Single-modality ablations -----------------------
def ablate_no_faces(diarization, transcribed, ordered_names=None):
    """A1: identity resolution with vision evidence removed."""
    return run_identity_pipeline(diarization, transcribed,
                                 faces=[], ordered_names=ordered_names)


def ablate_no_nlp(diarization, transcribed, faces=None):
    """A2: identity resolution with NER names and anchors removed."""
    return run_identity_pipeline(diarization, transcribed,
                                 faces=faces, ordered_names=[])


def oracle_labels(diarization, transcribed, speaker_map: Dict[str, str]):
    """A3 upper bound: ground-truth speaker-name annotations (learned ground
    truth / annotated evaluation labels). Quantifies how much of total error
    comes from naming vs from diarization+ASR themselves."""
    return run_identity_pipeline(diarization, transcribed,
                                 faces=[], ground_truth_labels=speaker_map)
