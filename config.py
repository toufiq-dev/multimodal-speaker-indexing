"""Configuration for the multimodal Bangla talk-show indexing system."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Optional

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

    # Text-cleaning rules as (pattern, replacement) pairs.
    #
    # NOTE: The previous design stored bare patterns and always substituted "".
    # That silently DELETED legitimate repeated Bangla graphemes (e.g. "আআআ" ->
    # "") instead of collapsing them, eating characters from real transcripts.
    # Replacement strings are now explicit; runs collapse to one instance.
    TEXT_CLEANING_RULES: ClassVar[list[tuple[str, str]]] = [
        (r"(.)\1{2,}", r"\1"),                 # Collapse 3+ repeated chars -> 1
        (r"\b(uh|um|ah)\b", ""),               # English filler words
        (r"\s+([।,;:.!?])", r"\1"),            # No space before punctuation
        (r"<\|[^|]*\|>", ""),                  # Whisper special tokens
    ]

    HF_TOKEN: str = field(default_factory=lambda: os.getenv("HF_TOKEN", ""))
    DEVICE: str = field(default_factory=lambda: _auto_device())
    WHISPER_MODEL: str = field(default_factory=lambda: os.getenv(
        "WHISPER_MODEL", "bengaliAI/tugstugi_bengaliai-asr_whisper-medium"))
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
    # Vision backend: 'insightface' (RetinaFace) primary, 'yolo' (YOLOv8-face) alternative.
    # YOLO selection is grounded in the DAWN adverse-weather ablation (YOLOv8x best mAP 80.44%):
    # see thesis Ch3.5.3 — same study framed for this pipeline's face robustness.
    VISION_DETECTOR: str = field(default_factory=lambda: os.getenv("VISION_DETECTOR", "insightface"))
    YOLO_MODEL: str = field(default_factory=lambda: os.getenv(
        "YOLO_MODEL", "yolov8n-face.pt"))  # ultralytics hub: yolov8n-face, yolov8x, yolov11x etc.
    CLUSTERING: str = field(default_factory=lambda: os.getenv("CLUSTERING", "dbscan"))  # dbscan|agglomerative|hdbscan

    # --- Speaker-count hints (avoid wasteful & unstable two-pass diarization) ---
    # For a fixed-format talk show set e.g. NUM_SPEAKERS=5 via env/config.
    NUM_SPEAKERS: Optional[int] = field(
        default_factory=lambda: (
            int(os.environ["NUM_SPEAKERS"]) if os.environ.get("NUM_SPEAKERS") else None
        )
    )
    MIN_SPEAKERS: Optional[int] = None
    MAX_SPEAKERS: Optional[int] = None

    # --- Post-processing ---
    ENABLE_PUNCTUATION_RESTORE: bool = False  # requires a punctuator backend

    BASE_DIR: Path = field(default_factory=_resolve_base_dir)
    DATA_INPUT_DIR: Path = field(init=False)
    DATA_REGISTRY_DIR: Path = field(init=False)
    DATA_OUTPUT_DIR: Path = field(init=False)

    def fw_device_and_compute(self) -> tuple[str, str]:
        """Route device/compute-type for faster-whisper (CTranslate2).

        CTranslate2 ships no Metal (MPS) backend, so 'mps' must fall back to
        CPU int8; CUDA uses float16. PyTorch-only engines should keep using
        ``self.DEVICE`` directly.
        """
        if self.DEVICE == "cuda":
            return "cuda", "float16"
        return "cpu", "int8"

    def diarization_kwargs(self) -> dict:
        """Keyword args for a SINGLE pyannote pipeline call.

        Replaces the old two-pass scheme: pyannote 3.x accepts
        num_speakers OR min_speakers/max_speakers directly, which is both
        cheaper and more stable than re-running constrained on pass-1 output.
        """
        kwargs: dict = {}
        if self.NUM_SPEAKERS:
            kwargs["num_speakers"] = int(self.NUM_SPEAKERS)
        else:
            if self.MIN_SPEAKERS:
                kwargs["min_speakers"] = int(self.MIN_SPEAKERS)
            if self.MAX_SPEAKERS:
                kwargs["max_speakers"] = int(self.MAX_SPEAKERS)
        return kwargs

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
