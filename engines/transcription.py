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
    device, compute_type = config.fw_device_and_compute()
    model = WhisperModel(
        config.WHISPER_MODEL,
        device=device,
        compute_type=compute_type,
    )
    return model


def _clean_text(text: str) -> str:
    """Apply regex cleaning rules from config.

    Rules are (pattern, replacement) pairs so repeated-character runs are
    COLLAPSED to one instance (r"\1") instead of deleted entirely — the old
    behaviour erased legitimate doubled Bangla graphemes.
    """
    cleaned = text
    for pattern, replacement in config.TEXT_CLEANING_RULES:
        cleaned = re.sub(pattern, replacement, cleaned)
    return cleaned.strip()


def _assign_word_to_turn(
    word_start: float,
    word_end: float,
    diarization: List[DiarizationSegment],
) -> str:
    """Assign a word to a diarization turn by MIDPOINT CONTAINMENT.

    The pipeline contract is: words carry timestamps; diarization turns own
    time intervals. A word belongs to the turn whose interval contains the
    word's midpoint. Only if NO turn contains it do we fall back to the
    nearest turn boundary. This is exact for well-formed diarization output;
    IoU-based scoring is meaningless at this scale (a 0.3s word vs a 5s turn
    yields IoU < 0.1 even for perfect alignment).
    """
    if not diarization:
        return "UNKNOWN"

    word_mid = (word_start + word_end) / 2.0

    # 1) Containment: prefer the turn that contains the midpoint. If turns
    #    overlap, pick the one with the smallest duration (most specific).
    containing = [
        seg for seg in diarization
        if seg.start <= word_mid <= seg.end
    ]
    if containing:
        return min(containing, key=lambda s: s.end - s.start).speaker_id

    # 2) Fallback: nearest boundary distance.
    def _boundary_dist(seg: DiarizationSegment) -> float:
        if word_mid < seg.start:
            return seg.start - word_mid
        if word_mid > seg.end:
            return word_mid - seg.end
        return 0.0

    return min(diarization, key=_boundary_dist).speaker_id





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
        word.speaker_id = _assign_word_to_turn(word.start, word.end, diarization)

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