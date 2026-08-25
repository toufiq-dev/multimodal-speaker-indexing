"""LoRA-adapted Whisper engine for Bengali ASR using chunked transcription."""

from __future__ import annotations

import math
import torch
import torchaudio
from pathlib import Path
from typing import List, Optional, Tuple

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

from config import config
from models import DiarizationSegment, TranscribedSegment


# Chunking constants
CHUNK_DURATION_SEC = 20.0
CHUNK_OVERLAP_SEC = 0.0
MAX_NEW_TOKENS = 256


def load_lora_whisper(
    model_name: str = "openai/whisper-large-v3",
    lora_path: Optional[str] = None,
) -> Tuple[WhisperForConditionalGeneration, WhisperProcessor]:
    """
    Load Whisper model with optional LoRA adapter.

    Args:
        model_name: Base model identifier (default: openai/whisper-large-v3).
        lora_path: Path to LoRA adapter (local or HF Hub). If None, loads base model.

    Returns:
        Tuple of (model, processor).
    """
    processor = WhisperProcessor.from_pretrained(model_name)

    if config.DEVICE == "cuda":
        model = WhisperForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_4bit=True,
        )
    else:
        model = WhisperForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map={"": config.DEVICE},
        )

    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()

    model.eval()
    return model, processor


def _compute_temporal_overlap(
    seg_start: float,
    seg_end: float,
    dia_start: float,
    dia_end: float,
) -> float:
    """Compute temporal overlap (IoU) between two segments."""
    intersection = max(0.0, min(seg_end, dia_end) - max(seg_start, dia_start))
    union = max(seg_end, dia_end) - min(seg_start, dia_start)
    return intersection / union if union > 0 else 0.0


def _assign_speaker_to_segment(
    seg_start: float,
    seg_end: float,
    diarization: List[DiarizationSegment],
) -> str:
    """Assign speaker to segment based on max temporal overlap."""
    if not diarization:
        return "UNKNOWN"

    best_overlap = 0.0
    best_speaker = "UNKNOWN"

    for dia in diarization:
        overlap = _compute_temporal_overlap(seg_start, seg_end, dia.start, dia.end)
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = dia.speaker_id

    return best_speaker


def _load_and_preprocess_audio(audio_path: str) -> Tuple[torch.Tensor, int]:
    """Load audio file and return mono waveform at target sample rate."""
    audio = Path(audio_path).resolve()
    if not audio.exists():
        raise FileNotFoundError(f"Audio file not found: {audio}")

    waveform, sample_rate = torchaudio.load(str(audio))

    if sample_rate != config.AUDIO_SR:
        resampler = torchaudio.transforms.Resample(sample_rate, config.AUDIO_SR)
        waveform = resampler(waveform)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    return waveform, config.AUDIO_SR


def _chunk_audio(
    waveform: torch.Tensor,
    sample_rate: int,
    chunk_duration: float = CHUNK_DURATION_SEC,
    overlap: float = CHUNK_OVERLAP_SEC,
) -> List[Tuple[torch.Tensor, float, float]]:
    """
    Split audio into overlapping chunks.

    Returns:
        List of (chunk_waveform, start_time, end_time)
    """
    total_samples = waveform.shape[1]
    chunk_samples = int(chunk_duration * sample_rate)
    overlap_samples = int(overlap * sample_rate)
    stride = chunk_samples - overlap_samples

    chunks = []
    for start_sample in range(0, total_samples, stride):
        end_sample = min(start_sample + chunk_samples, total_samples)
        if end_sample - start_sample < sample_rate * 0.5:
            break
        chunk = waveform[:, start_sample:end_sample]
        start_time = start_sample / sample_rate
        end_time = end_sample / sample_rate
        chunks.append((chunk, start_time, end_time))
        if end_sample >= total_samples:
            break

    return chunks


def _transcribe_chunk(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    chunk_waveform: torch.Tensor,
    language: str = "bn",
) -> str:
    """Transcribe a single audio chunk."""
    input_features = processor(
        chunk_waveform.squeeze().numpy(),
        sampling_rate=config.AUDIO_SR,
        return_tensors="pt",
    ).input_features

    if config.DEVICE == "cuda":
        input_features = input_features.to("cuda", dtype=torch.float16)
    else:
        input_features = input_features.to(config.DEVICE, dtype=torch.float32)

    forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task="transcribe")

    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids,
            max_new_tokens=MAX_NEW_TOKENS,
        )

    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    # Explicitly filter any remaining special/timestamp tokens
    import re
    text = re.sub(r"<\|[^|]+\|>", "", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def transcribe_with_lora(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    audio_path: str,
    diarization: Optional[List[DiarizationSegment]] = None,
    language: str = "bn",
) -> List[TranscribedSegment]:
    """
    Transcribe audio using LoRA-adapted Whisper via chunked approach.

    Args:
        model: Loaded Whisper model (with LoRA merged if applicable).
        processor: WhisperProcessor for feature extraction.
        audio_path: Path to input WAV audio file.
        diarization: Optional list of DiarizationSegment for speaker assignment.
        language: Language code (default: "bn" for Bengali).

    Returns:
        List of TranscribedSegment with speaker assignments and empty word list.
    """
    waveform, sample_rate = _load_and_preprocess_audio(audio_path)
    chunks = _chunk_audio(waveform, sample_rate)

    raw_segments: List[TranscribedSegment] = []

    for chunk_waveform, start_time, end_time in chunks:
        text = _transcribe_chunk(model, processor, chunk_waveform, language)
        if not text:
            continue

        speaker = _assign_speaker_to_segment(start_time, end_time, diarization or [])

        raw_segments.append(
            TranscribedSegment(
                start=start_time,
                end=end_time,
                text=text,
                words=[],
                speaker_id=speaker,
            )
        )

    if not raw_segments:
        torch.cuda.empty_cache()
        return []

    # Merge consecutive segments with same speaker
    merged: List[TranscribedSegment] = []
    current = raw_segments[0]

    for seg in raw_segments[1:]:
        if seg.speaker_id == current.speaker_id:
            current = TranscribedSegment(
                start=current.start,
                end=seg.end,
                text=current.text + " " + seg.text,
                words=[],
                speaker_id=current.speaker_id,
            )
        else:
            merged.append(current)
            current = seg

    merged.append(current)

    torch.cuda.empty_cache()
    return merged