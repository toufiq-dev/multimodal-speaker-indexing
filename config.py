"""Configuration for the multimodal Bangla talk-show indexing system."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import torch


def _auto_device() -> str:
    """Auto-detect the best available device for PyTorch."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_base_dir() -> Path:
    """Resolve base directory based on execution environment."""
    # Check for Kaggle environment
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.environ.get("KAGGLE_URL_BASE"):
        return Path("/kaggle/working")
    # Check for Colab
    if os.environ.get("COLAB_RELEASE_TAG"):
        return Path("/content")
    return Path("./data")


@dataclass
class Config:
    """Application configuration with environment-based overrides and auto-detection."""

    HF_TOKEN: str = field(default_factory=lambda: os.getenv("HF_TOKEN", ""))
    DEVICE: str = field(default_factory=lambda: _auto_device())
    WHISPER_MODEL: str = field(default_factory=lambda: os.getenv("WHISPER_MODEL", "bengaliAI/tugstugi_bengaliai-asr_whisper-medium"))
    PYANNOTE_MODEL: str = "pyannote/speaker-diarization-3.1"
    USE_LORA: bool = False
    LORA_PATH: str = ""
    BANGLABERT_NER_MODEL: str = "sagorsarker/banglabert-ner"
    BANGLABERT_NER_FALLBACK: str = "sagorsarker/mbert-bengali-ner"
    FACE_SIM_THRESHOLD: float = 0.65
    VISION_FPS: int = 1
    AUDIO_SR: int = 16000
    DBSCAN_EPS: float = 0.5
    DBSCAN_MIN_SAMPLES: int = 3
    NLP_INTRO_SECONDS: int = 120
    BASE_DIR: Path = field(default_factory=_resolve_base_dir)
    DATA_INPUT_DIR: Path = field(init=False)
    DATA_REGISTRY_DIR: Path = field(init=False)
    DATA_OUTPUT_DIR: Path = field(init=False)
    REGEX_FILTER_PATTERNS: list[str] = field(default_factory=lambda: [
        r"(.)\1{2,}",           # Collapse repeated characters (3+ repetitions)
        r"\b(uh|um|ah)\b",      # Remove filler words
        r"\s+([।,;:.!?])",      # Remove spaces before punctuation
    ])

    def __post_init__(self) -> None:
        """Verify and create all required directories on initialization."""
        self.DATA_INPUT_DIR = self.BASE_DIR / "input"
        self.DATA_REGISTRY_DIR = self.BASE_DIR / "registry"
        self.DATA_OUTPUT_DIR = self.BASE_DIR / "output"

        for directory in (
            self.DATA_INPUT_DIR,
            self.DATA_REGISTRY_DIR,
            self.DATA_OUTPUT_DIR,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def get_ner_model(self) -> str:
        """Return primary NER model, fallback available via BANGLABERT_NER_FALLBACK."""
        return self.BANGLABERT_NER_MODEL


# Global config instance
config = Config()