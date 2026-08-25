"""Engines package for multimodal Bangla talk-show indexing system."""

from engines.media import extract_audio, extract_frames
from engines.diarization import run_diarization
from engines.transcription import transcribe_audio, align_transcription_with_diarization
from engines.asr_lora import load_lora_whisper, transcribe_with_lora
from engines.nlp import extract_speaker_names_from_intro
from engines.fusion import run_fusion_pipeline, GatingFusion, GatingNetwork


# Lazy load vision to prevent OpenCV/InsightFace crashes during audio/transcription imports
def __getattr__(name):
    if name == "run_vision_pipeline":
        from engines.vision import run_vision_pipeline
        return run_vision_pipeline
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "extract_audio",
    "extract_frames",
    "run_diarization",
    "transcribe_audio",
    "align_transcription_with_diarization",
    "load_lora_whisper",
    "transcribe_with_lora",
    "run_vision_pipeline",
    "extract_speaker_names_from_intro",
    "run_fusion_pipeline",
    "GatingFusion",
    "GatingNetwork",
]