"""Speaker diarization engine using pyannote.audio."""

from __future__ import annotations

import os
import torch
from pathlib import Path
from typing import List, Optional

from pyannote.audio import Pipeline

from config import config
from models import DiarizationSegment
from runtime import release_gpu_memory


def _load_pipeline() -> Pipeline:
    """Load pyannote diarization pipeline with HF token (handles API version differences)."""
    hf_token = config.HF_TOKEN or os.getenv("HF_TOKEN", "")
    
    try:
        pipeline = Pipeline.from_pretrained(
            config.PYANNOTE_MODEL,
            token=hf_token if hf_token else None,
        )
    except TypeError:
        # Older pyannote versions use use_auth_token
        pipeline = Pipeline.from_pretrained(
            config.PYANNOTE_MODEL,
            use_auth_token=hf_token if hf_token else None,
        )

    if config.use_cuda():
        pipeline.to(torch.device("cuda"))

    return pipeline


def _collect_speakers(annotation) -> List[str]:
    """Extract unique speaker labels from annotation."""
    speakers = set()
    for _, _, speaker in annotation.itertracks(yield_label=True):
        speakers.add(str(speaker))
    return sorted(speakers)


def run_diarization(audio_path: str, num_speakers: Optional[int] = None) -> List[DiarizationSegment]:
    """
    Run SINGLE-PASS speaker diarization with optional speaker-count hints.

    The previous two-pass scheme (unconstrained, then re-run constrained on
    pass-1's count) doubled cost and locked in any pass-1 error. pyannote 3.x
    accepts num_speakers / min_speakers / max_speakers directly.

    Args:
        audio_path: Path to input WAV audio file.
        num_speakers: Exact speaker count if known (e.g. 5 for this show
            format); overrides config.NUM_SPEAKERS. None -> unconstrained or
            config bounds.

    Returns:
        List of DiarizationSegment objects sorted by start time.
    """
    audio = Path(audio_path).resolve()
    if not audio.exists():
        raise FileNotFoundError(f"Audio file not found: {audio}")

    pipeline = _load_pipeline()

    # Single pass with hints when available.
    kwargs = config.diarization_kwargs()
    if num_speakers is not None:
        kwargs = {"num_speakers": int(num_speakers)}

    try:
        result = pipeline(str(audio), **kwargs)
    except TypeError as e:
        # Only an unsupported-keyword error justifies dropping the speaker-count
        # hints. A blanket `except Exception` silently discarded NUM_SPEAKERS on
        # any failure — including OOM — and reported the unconstrained result as
        # if it had honoured the constraint.
        print(f"[diarization] speaker-count hints rejected by the pipeline "
              f"({e}); retrying unconstrained")
        result = pipeline(str(audio))

    # pyannote 4.x returns DiarizeOutput; extract the Annotation object.
    annotation = getattr(result, "speaker_diarization", result)

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

    if not segments:
        raise RuntimeError(
            f"Diarization produced no speech turns for {audio.name}. The audio "
            f"is silent, or the pyannote pipeline loaded without weights."
        )

    # Drop the pipeline BEFORE releasing the cache: empty_cache() only returns
    # blocks that already have no live reference, so the previous ordering
    # (cache release with `pipeline` still in scope) reclaimed nothing.
    del pipeline
    release_gpu_memory()
    return segments