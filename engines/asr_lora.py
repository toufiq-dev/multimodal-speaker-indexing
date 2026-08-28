"""LoRA-adapted Whisper engine for Bengali ASR using chunked transcription.

This chunked path is a FALLBACK for models that cannot emit word timestamps.
The primary path is engines/transcription.py (faster-whisper with
word_timestamps=True). Fusion treats word-less segments conservatively (see
create_final_segments), so this mode no longer corrupts output even though it
cannot do precise speaker alignment.
"""
from __future__ import annotations

import re
import torch
import torchaudio
from pathlib import Path
from typing import List, Optional, Tuple

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

from config import config
from models import DiarizationSegment, TranscribedSegment
from runtime import release_gpu_memory


CHUNK_DURATION_SEC = 20.0
CHUNK_OVERLAP_SEC = 0.0
MAX_NEW_TOKENS = 256


# Correctly escaped patterns. The notebook hot-patch previously wrote
# r"<\\|[^|]+\\|>" which in a raw string means "backslash followed by
# alternation" and MANGLED special tokens instead of removing them.
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]*\|>")
_WHITESPACE_RE = re.compile(r"\s+")


def _restore_bengali_punct(text: str) -> str:
    """Lightweight Bengali punctuation restoration (regex heuristic, zero-dep).

    Mirrors engines/transcription.py, including the ENABLE_PUNCTUATION_RESTORE
    gate so the two ASR paths are scored under identical text conventions.
    """
    if not config.ENABLE_PUNCTUATION_RESTORE:
        return text
    if not text:
        return text
    if not re.search(r"[\u0980-\u09FF]", text):
        return text
    text = re.sub(r"\s*।\s*", "। ", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if "।" not in text and len(text.split()) > 18:
        words = text.split()
        mid = len(words) // 2
        text = " ".join(words[:mid]) + "। " + " ".join(words[mid:])
    return text.strip()


def _resolve_torch_device() -> str:
    if config.use_cuda():
        return "cuda:0"
    if config.DEVICE == "mps":
        return "mps"
    return "cpu"


def load_lora_whisper(
    model_name: str = "openai/whisper-large-v3",
    lora_path: Optional[str] = None,
) -> Tuple[WhisperForConditionalGeneration, WhisperProcessor]:
    """Load base Whisper in fp16, attach the LoRA adapter, then merge into fp16.

    Merging into a 4-bit base partially destroys the adapter's learned update
    (documented PEFT pitfall; the run log even warns about rounding errors).
    Whisper-medium fp16 (~3 GB) fits comfortably in any T4/M-series machine,
    so 4-bit bought nothing and cost accuracy twice over.
    """
    processor = WhisperProcessor.from_pretrained(model_name)

    device = _resolve_torch_device()
    dtype = torch.float16 if device.startswith("cuda") else torch.float32

    model = WhisperForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=dtype,
    ).to(device)

    if lora_path:
        peft_model = PeftModel.from_pretrained(model, lora_path)
        model = peft_model.merge_and_unload()

    model.eval()
    return model, processor


def _compute_temporal_overlap(seg_start, seg_end, dia_start, dia_end):
    intersection = max(0.0, min(seg_end, dia_end) - max(seg_start, dia_start))
    union = max(seg_end, dia_end) - min(seg_start, dia_start)
    return intersection / union if union > 0 else 0.0


def _assign_speaker_to_segment(seg_start, seg_end, diarization):
    """Assign by MAX OVERLAP FRACTION of the diarization turn covered.

    Falls back to UNKNOWN only when there is literally zero overlap with any
    turn (e.g. music/station-ID between speech turns).
    """
    if not diarization:
        return "UNKNOWN"
    best_overlap, best_speaker = 0.0, "UNKNOWN"
    for dia in diarization:
        overlap = _compute_temporal_overlap(seg_start, seg_end, dia.start, dia.end)
        if overlap > best_overlap:
            best_overlap, best_speaker = overlap, dia.speaker_id
    return best_speaker


def _load_and_preprocess_audio(audio_path):
    audio = Path(audio_path).resolve()
    if not audio.exists():
        raise FileNotFoundError(f"Audio file not found: {audio}")
    waveform, sample_rate = torchaudio.load(str(audio))
    if sample_rate != config.AUDIO_SR:
        waveform = torchaudio.transforms.Resample(sample_rate, config.AUDIO_SR)(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform, config.AUDIO_SR


def _chunk_audio(waveform, sample_rate, chunk_duration=CHUNK_DURATION_SEC, overlap=CHUNK_OVERLAP_SEC):
    total_samples = waveform.shape[1]
    chunk_samples = int(chunk_duration * sample_rate)
    overlap_samples = int(overlap * sample_rate)
    stride = max(1, chunk_samples - overlap_samples)
    chunks = []
    for start_sample in range(0, total_samples, stride):
        end_sample = min(start_sample + chunk_samples, total_samples)
        if end_sample - start_sample < sample_rate * 0.5:
            break
        chunks.append((waveform[:, start_sample:end_sample],
                       start_sample / sample_rate, end_sample / sample_rate))
        if end_sample >= total_samples:
            break
    return chunks


def _transcribe_chunk(model, processor, chunk_waveform, language="bn"):
    input_features = processor(
        chunk_waveform.squeeze().numpy(),
        sampling_rate=config.AUDIO_SR,
        return_tensors="pt",
    ).input_features

    model_device = next(model.parameters()).device
    dtype = torch.float16 if model_device.type == "cuda" else torch.float32
    input_features = input_features.to(model_device, dtype=dtype)

    # Robust generation config for older transformers versions.
    # NOTE: lang_to_id/task_to_id live on the tokenizer/processor vocabulary,
    # NOT on WhisperConfig — resolve via convert_tokens_to_ids.
    model.generation_config.forced_decoder_ids = None
    if not model.generation_config.suppress_tokens or len(model.generation_config.suppress_tokens) < 2:
        # transformers 4.44's Whisper head slices suppress_tokens[-2:], so the
        # sentinel must carry >=2 entries; -1 is a no-op pad token id here.
        model.generation_config.suppress_tokens = [-1, -1]

    with torch.no_grad():
        predicted_ids = model.generate(
            input_features, language=language, task="transcribe",
            max_new_tokens=MAX_NEW_TOKENS,
        )

    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    text = _SPECIAL_TOKEN_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return _restore_bengali_punct(text)


def transcribe_with_lora(model, processor, audio_path, diarization=None, language="bn"):
    waveform, sample_rate = _load_and_preprocess_audio(audio_path)
    chunks = _chunk_audio(waveform, sample_rate)
    raw_segments = []
    for chunk_waveform, start_time, end_time in chunks:
        text = _transcribe_chunk(model, processor, chunk_waveform, language)
        if not text:
            continue
        speaker = _assign_speaker_to_segment(start_time, end_time, diarization or [])
        raw_segments.append(TranscribedSegment(
            start=start_time, end=end_time, text=text, words=[], speaker_id=speaker))

    if not raw_segments:
        return []

    merged, current = [], raw_segments[0]
    for seg in raw_segments[1:]:
        if seg.speaker_id != "UNKNOWN" and seg.speaker_id == current.speaker_id:
            current = TranscribedSegment(
                start=current.start, end=seg.end,
                text=current.text + " " + seg.text,
                words=[], speaker_id=current.speaker_id)
        else:
            merged.append(current)
            current = seg
    merged.append(current)
    release_gpu_memory()
    return merged
