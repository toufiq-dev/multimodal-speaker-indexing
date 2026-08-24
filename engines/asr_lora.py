"""LoRA-adapted Whisper engine for Bengali ASR."""

from __future__ import annotations

import re
import torch
import torchaudio
from pathlib import Path
from typing import List, Optional, Tuple

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

from config import Config
from models import DiarizationSegment, TranscribedSegment


config = Config()


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


def transcribe_with_lora(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    audio_path: str,
    diarization: Optional[List[DiarizationSegment]] = None,
    language: str = "bn",
) -> List[TranscribedSegment]:
    """
    Transcribe audio using LoRA-adapted Whisper (segment-level, no word timestamps).

    Args:
        model: Loaded Whisper model (with LoRA merged if applicable).
        processor: WhisperProcessor for feature extraction.
        audio_path: Path to input WAV audio file.
        diarization: Optional list of DiarizationSegment for speaker assignment.
        language: Language code (default: "bn" for Bengali).

    Returns:
        List of TranscribedSegment with speaker assignments and empty word list.
    """
    audio = Path(audio_path).resolve()
    if not audio.exists():
        raise FileNotFoundError(f"Audio file not found: {audio}")

    waveform, sample_rate = torchaudio.load(str(audio))

    if sample_rate != config.AUDIO_SR:
        resampler = torchaudio.transforms.Resample(sample_rate, config.AUDIO_SR)
        waveform = resampler(waveform)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    input_features = processor(
        waveform.squeeze().numpy(),
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
            return_timestamps=True,
        )

    decoded = processor.batch_decode(predicted_ids, skip_special_tokens=False)[0]

    timestamp_pattern = r"<\|(\d+\.\d+)\|>"
    timestamp_matches = list(re.finditer(timestamp_pattern, decoded))
    timestamps = [(float(m.group(1)), m.start()) for m in timestamp_matches]

    text_parts = re.split(timestamp_pattern, decoded)
    text_parts = [p for p in text_parts if p and not re.match(r"^\d+\.\d+$", p)]

    segments: List[TranscribedSegment] = []

    if len(timestamps) >= 2:
        for i in range(len(timestamps) - 1):
            start = timestamps[i][0]
            end = timestamps[i + 1][0]
            text = text_parts[i].strip() if i < len(text_parts) else ""

            if not text:
                continue

            speaker = _assign_speaker_to_segment(start, end, diarization or [])

            segments.append(
                TranscribedSegment(
                    start=start,
                    end=end,
                    text=text,
                    words=[],
                    speaker_id=speaker,
                )
            )

    torch.cuda.empty_cache()
    return segments