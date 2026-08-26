"""LoRA-adapted Whisper engine for Bengali ASR using chunked transcription."""
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


CHUNK_DURATION_SEC = 20.0
CHUNK_OVERLAP_SEC = 0.0
MAX_NEW_TOKENS = 256


def load_lora_whisper(
    model_name: str = "openai/whisper-large-v3",
    lora_path: Optional[str] = None,
) -> Tuple[WhisperForConditionalGeneration, WhisperProcessor]:
    processor = WhisperProcessor.from_pretrained(model_name)
    if config.DEVICE == "cuda":
        model = WhisperForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map={"": "cuda:0"},
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


def _compute_temporal_overlap(seg_start, seg_end, dia_start, dia_end):
    intersection = max(0.0, min(seg_end, dia_end) - max(seg_start, dia_start))
    union = max(seg_end, dia_end) - min(seg_start, dia_start)
    return intersection / union if union > 0 else 0.0


def _assign_speaker_to_segment(seg_start, seg_end, diarization):
    if not diarization: return "UNKNOWN"
    best_overlap, best_speaker = 0.0, "UNKNOWN"
    for dia in diarization:
        overlap = _compute_temporal_overlap(seg_start, seg_end, dia.start, dia.end)
        if overlap > best_overlap:
            best_overlap, best_speaker = overlap, dia.speaker_id
    return best_speaker


def _load_and_preprocess_audio(audio_path):
    audio = Path(audio_path).resolve()
    if not audio.exists(): raise FileNotFoundError(f"Audio file not found: {audio}")
    waveform, sample_rate = torchaudio.load(str(audio))
    if sample_rate != config.AUDIO_SR:
        waveform = torchaudio.transforms.Resample(sample_rate, config.AUDIO_SR)(waveform)
    if waveform.shape[0] > 1: waveform = waveform.mean(dim=0, keepdim=True)
    return waveform, config.AUDIO_SR


def _chunk_audio(waveform, sample_rate, chunk_duration=CHUNK_DURATION_SEC, overlap=CHUNK_OVERLAP_SEC):
    total_samples = waveform.shape[1]
    chunk_samples = int(chunk_duration * sample_rate)
    overlap_samples = int(overlap * sample_rate)
    stride = chunk_samples - overlap_samples
    chunks = []
    for start_sample in range(0, total_samples, stride):
        end_sample = min(start_sample + chunk_samples, total_samples)
        if end_sample - start_sample < sample_rate * 0.5: break
        chunks.append((waveform[:, start_sample:end_sample], start_sample / sample_rate, end_sample / sample_rate))
        if end_sample >= total_samples: break
    return chunks


def _transcribe_chunk(model, processor, chunk_waveform, language="bn"):
    input_features = processor(chunk_waveform.squeeze().numpy(), sampling_rate=config.AUDIO_SR, return_tensors="pt").input_features
    model_device = next(model.parameters()).device
    dtype = torch.float16 if model_device.type == "cuda" else torch.float32
    input_features = input_features.to(model_device, dtype=dtype)


    # Robust generation config for transformers 4.44.0 + 4bit quantization
    model.generation_config.forced_decoder_ids = None
    if not model.generation_config.suppress_tokens or len(model.generation_config.suppress_tokens) < 2:
        model.generation_config.suppress_tokens = [-1, -1]
    if not hasattr(model.generation_config, "lang_to_id") or not model.generation_config.lang_to_id:
        model.generation_config.lang_to_id = {f"<|{language}|>": processor.tokenizer.convert_tokens_to_ids(f"<|{language}|>")}
    if not hasattr(model.generation_config, "task_to_id") or not model.generation_config.task_to_id:
        model.generation_config.task_to_id = {"transcribe": processor.tokenizer.convert_tokens_to_ids("<|transcribe|>")}


    with torch.no_grad():
        predicted_ids = model.generate(input_features, language=language, task="transcribe", max_new_tokens=MAX_NEW_TOKENS)
        
    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    text = re.sub(r"<\\|[^|]+\\|>", "", text)
    return re.sub(r"\\s+", " ", text).strip()


def transcribe_with_lora(model, processor, audio_path, diarization=None, language="bn"):
    waveform, sample_rate = _load_and_preprocess_audio(audio_path)
    chunks = _chunk_audio(waveform, sample_rate)
    raw_segments = []
    for chunk_waveform, start_time, end_time in chunks:
        text = _transcribe_chunk(model, processor, chunk_waveform, language)
        if not text: continue
        speaker = _assign_speaker_to_segment(start_time, end_time, diarization or [])
        raw_segments.append(TranscribedSegment(start=start_time, end=end_time, text=text, words=[], speaker_id=speaker))
    if not raw_segments: return []
    merged, current = [], raw_segments[0]
    for seg in raw_segments[1:]:
        if seg.speaker_id == current.speaker_id:
            current = TranscribedSegment(start=current.start, end=seg.end, text=current.text + " " + seg.text, words=[], speaker_id=current.speaker_id)
        else:
            merged.append(current)
            current = seg
    merged.append(current)
    return merged