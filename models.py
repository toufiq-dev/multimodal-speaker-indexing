"""Data models for the multimodal Bangla talk-show indexing system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class DiarizationSegment:
    """Represents a speaker diarization segment from pyannote.audio."""
    start: float
    end: float
    speaker_id: str


@dataclass
class WordToken:
    """Represents a single word with timestamp information from Whisper."""
    word: str
    start: float
    end: float
    speaker_id: str = "UNKNOWN"


@dataclass
class TranscribedSegment:
    """Represents a transcribed segment with word-level timestamps and speaker ID."""
    start: float
    end: float
    text: str
    words: List[WordToken]
    speaker_id: str = "UNKNOWN"


@dataclass
class FaceOccurrence:
    """Represents a detected face occurrence in a video frame."""
    frame_time: float
    box: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    track_id: int
    resolved_face_id: str = "UNKNOWN"
    face_confidence: float = 0.0
    lip_sync_score: float = 0.0
    embedding: Optional[np.ndarray] = None


@dataclass
class FinalSegment:
    """Represents the final merged segment with speaker, text, and confidence."""
    start: float
    end: float
    speaker: str
    text: str
    confidence: float = 0.0