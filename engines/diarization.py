"""Speaker diarization engine using pyannote.audio."""

from __future__ import annotations

import torch
from pathlib import Path
from typing import List

from pyannote.audio import Pipeline

from config import Config
from models import DiarizationSegment


config = Config()


def _load_pipeline() -> Pipeline:
    """Load pyannote diarization pipeline with HF token."""
    try:
        pipeline = Pipeline.from_pretrained(
            config.PYANNOTE_MODEL,
            use_auth_token=config.HF_TOKEN,
        )
    except TypeError:
        pipeline = Pipeline.from_pretrained(
            config.PYANNOTE_MODEL,
            token=config.HF_TOKEN,
        )

    if config.DEVICE == "cuda":
        pipeline.to(torch.device("cuda"))

    return pipeline


def _collect_speakers(annotation) -> List[str]:
    """Extract unique speaker labels from annotation."""
    speakers = set()
    for _, _, speaker in annotation.itertracks(yield_label=True):
        speakers.add(str(speaker))
    return sorted(speakers)


def run_diarization(audio_path: str) -> List[DiarizationSegment]:
    """
    Run two-pass speaker diarization on audio file.

    Pass 1: Run without num_speakers constraint.
    Pass 2: Re-run with num_speakers=unique_speakers from pass 1 (if applicable).

    Args:
        audio_path: Path to input WAV audio file.

    Returns:
        List of DiarizationSegment objects sorted by start time.
    """
    audio = Path(audio_path).resolve()
    if not audio.exists():
        raise FileNotFoundError(f"Audio file not found: {audio}")

    pipeline = _load_pipeline()

    # Pass 1: Unconstrained
    annotation = pipeline(str(audio))
    speakers = _collect_speakers(annotation)

    if not speakers:
        torch.cuda.empty_cache()
        return []

    # Pass 2: With num_speakers hint
    try:
        annotation = pipeline(str(audio), num_speakers=len(speakers))
    except Exception:
        pass  # Fall back to pass 1 result

    segments: List[DiarizationSegment] = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append(
            DiarizationSegment(
                start=turn.start,
                end=turn.end,
                speaker_id=str(speaker),
            )
        )

    segments.sort(key=lambda s: s.start)

    torch.cuda.empty_cache()
    return segments