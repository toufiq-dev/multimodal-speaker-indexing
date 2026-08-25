"""Transcription engine using faster-whisper with diarization alignment."""

from __future__ import annotations

import re
import torch
from pathlib import Path
from typing import List, Tuple, Optional

from faster_whisper import WhisperModel

from config import config
from models import DiarizationSegment, TranscribedSegment, WordToken


def _load_model() -> WhisperModel:
    """Load faster-whisper model with appropriate compute type."""
    compute_type = "float16" if config.DEVICE == "cuda" else "int8"
    model = WhisperModel(
        config.WHISPER_MODEL,
        device=config.DEVICE,
        compute_type=compute_type,
    )
    return model


def _clean_text(text: str) -> str:
    """Apply regex cleaning patterns from config."""
    cleaned = text
    for pattern in config.REGEX_FILTER_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned)
    return cleaned.strip()


def _compute_iou(word_start: float, word_end: float, seg_start: float, seg_end: float) -> float:
    """Compute Intersection over Union between word and segment."""
    intersection = max(0.0, min(word_end, seg_end) - max(word_start, seg_start))
    union = max(word_end, seg_end) - min(word_start, seg_start)
    return intersection / union if union > 0 else 0.0


def _assign_speaker_to_word(
    word_start: float,
    word_end: float,
    diarization: List[DiarizationSegment],
) -> str:
    """Assign speaker to word based on max IoU or closest midpoint."""
    if not diarization:
        return "UNKNOWN"

    best_iou = 0.0
    best_speaker = "UNKNOWN"
    word_mid = (word_start + word_end) / 2.0
    min_dist = float("inf")

    for seg in diarization:
        iou = _compute_iou(word_start, word_end, seg.start, seg.end)
        if iou > best_iou:
            best_iou = iou
            best_speaker = seg.speaker_id

        seg_mid = (seg.start + seg.end) / 2.0
        dist = abs(word_mid - seg_mid)
        if dist < min_dist:
            min_dist = dist
            if best_iou == 0.0:
                best_speaker = seg.speaker_id

    return best_speaker


def transcribe_audio(
    audio_path: str,
    model: Optional[WhisperModel] = None,
) -> Tuple[List[WordToken], str]:
    """
    Transcribe audio with word-level timestamps.

    Args:
        audio_path: Path to input WAV audio file.
        model: Optional pre-loaded WhisperModel. If None, loads internally.

    Returns:
        Tuple of (list of WordToken, full_text).
    """
    audio = Path(audio_path).resolve()
    if not audio.exists():
        raise FileNotFoundError(f"Audio file not found: {audio}")

    local_model = model or _load_model()

    try:
        segments, _ = local_model.transcribe(
            str(audio),
            word_timestamps=True,
            vad_filter=True,
            language="bn",
        )
    except Exception:
        segments, _ = local_model.transcribe(
            str(audio),
            word_timestamps=True,
            vad_filter=True,
            language=None,
        )

    words: List[WordToken] = []
    full_text_parts = []

    for segment in segments:
        full_text_parts.append(segment.text)
        for word in segment.words or []:
            words.append(
                WordToken(
                    word=word.word.strip(),
                    start=word.start,
                    end=word.end,
                )
            )

    full_text = " ".join(full_text_parts).strip()

    if model is None:
        torch.cuda.empty_cache()
    return words, full_text


def align_transcription_with_diarization(
    audio_path: str,
    diarization: List[DiarizationSegment],
) -> List[TranscribedSegment]:
    """
    Transcribe audio and align words with diarization segments.

    Args:
        audio_path: Path to input WAV audio file.
        diarization: List of DiarizationSegment from run_diarization.

    Returns:
        List of TranscribedSegment with speaker assignments.
    """
    model = _load_model()
    words, _ = transcribe_audio(audio_path, model=model)

    if not words:
        torch.cuda.empty_cache()
        return []

    for word in words:
        word.speaker_id = _assign_speaker_to_word(word.start, word.end, diarization)

    segments: List[TranscribedSegment] = []
    current_words: List[WordToken] = []
    current_speaker = words[0].speaker_id

    for word in words:
        if word.speaker_id == current_speaker:
            current_words.append(word)
        else:
            if current_words:
                text = " ".join(w.word for w in current_words)
                text = _clean_text(text)
                segments.append(
                    TranscribedSegment(
                        start=current_words[0].start,
                        end=current_words[-1].end,
                        text=text,
                        words=current_words,
                        speaker_id=current_speaker,
                    )
                )
            current_words = [word]
            current_speaker = word.speaker_id

    if current_words:
        text = " ".join(w.word for w in current_words)
        text = _clean_text(text)
        segments.append(
            TranscribedSegment(
                start=current_words[0].start,
                end=current_words[-1].end,
                text=text,
                words=current_words,
                speaker_id=current_speaker,
            )
        )

    torch.cuda.empty_cache()
    return segments