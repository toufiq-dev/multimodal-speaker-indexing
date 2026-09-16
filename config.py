"""Configuration for the multimodal Bangla talk-show indexing system."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Optional

import torch

from runtime import (
    apply_numpy_compat,
    ensure_cudnn_on_path,
    onnx_providers,
)

# Establish the process invariants before any engine imports insightface,
# onnxruntime or ctranslate2. config is imported first by every engine, so
# this is the earliest deterministic hook available. The hard NumPy ABI
# assertion is deferred to runtime.assert_numpy_abi(), invoked by the vision
# modules — the audio/fusion/evaluation paths are ABI-independent.
apply_numpy_compat()
ensure_cudnn_on_path()


def _auto_device() -> str:
    """Auto-detect the best available device for PyTorch."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _env_float(name: str, default: float) -> float:
    """Float override from the environment; invalid values fall back."""
    raw = os.getenv(name)
    if raw is None or raw == "":
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _is_kaggle() -> bool:
    return bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
                or os.environ.get("KAGGLE_URL_BASE"))


def _resolve_base_dir() -> Path:
    """Resolve base directory based on execution environment (always absolute)."""
    if _is_kaggle():
        return Path("/kaggle/working").resolve()
    # Check for Colab
    if os.environ.get("COLAB_RELEASE_TAG"):
        return Path("/content").resolve()
    return Path("./data").resolve()


def _resolve_scratch_dir() -> Path:
    """Resolve the directory for large, disposable intermediates.

    Frames and extracted audio must NOT live under BASE_DIR on Kaggle:
    /kaggle/working is committed verbatim as notebook output, so ~3k JPEGs
    from a 53-minute show at 1 FPS (~650 MB) would be uploaded as results.
    /tmp is node-local, is not committed, and is wiped between sessions.
    """
    override = os.getenv("SCRATCH_DIR")
    if override:
        return Path(override).resolve()
    if _is_kaggle():
        return Path("/tmp/msi_scratch").resolve()
    return Path("./data/scratch").resolve()





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
    # Cosine similarity at which a face is accepted as an enrolled identity.
    #
    # CALIBRATED FROM MEASUREMENT, not guessed. Matching the on-screen face at
    # 12 timestamps of the RTV Goll Table episode against the reference photos
    # gave genuine matches of 0.45–0.65 (the speaker against his own photo),
    # while impostors scored 0.04–0.19 — a wide separation. The original 0.65
    # sat ABOVE the entire genuine distribution, so the speaker's own face was
    # rejected frame after frame and only the few strongest matches (a
    # co-panelist) survived, absorbing his turns. 0.40 sits inside the gap.
    FACE_SIM_THRESHOLD: float = field(
        default_factory=lambda: _env_float("FACE_SIM_THRESHOLD", 0.40))
    # Genuine matches lead the runner-up by 0.3–0.5; impostors by ~0.05. A
    # margin of 0.10 rejects near-ties without touching a real match.
    FACE_SIM_MARGIN: float = field(
        default_factory=lambda: _env_float("FACE_SIM_MARGIN", 0.10))
    # Keep 0.0: with correctly-identified faces, requiring a minimum presence
    # would drop legitimate short turns. Calibrate from fusion_diagnostics.json
    # if precision still needs raising.
    FACE_MIN_FRAME_FRACTION: float = 0.0
    VISION_FPS: int = 1
    # Active-speaker detection needs consecutive frames close enough in time to
    # observe articulation. At VISION_FPS=1 the interval is a full second, far
    # above phoneme rate, so mouth motion is meaningless; the tracked pass
    # therefore samples at this rate instead.
    VISION_ASD_FPS: int = 8
    # IoU above which a face in the current frame continues the previous
    # frame's track.
    ASD_TRACK_IOU: float = 0.3
    # Minimum mean mouth motion for a track to count as "speaking".
    #
    # CALIBRATED from the per-track distribution in face_tracks.json (RTV Goll
    # Table, 8 FPS): faces that are present but silent max out at 0.0233
    # (a panellist who never speaks: median 0.0150, max 0.0233), while faces
    # that are actually talking sit at 0.038-0.074 (medians 0.038-0.061).
    # 0.03 lies inside that gap, so a cutaway to a silent listener can no
    # longer be mistaken for the speaker.
    ASD_MIN_MOUTH_MOTION: float = field(
        default_factory=lambda: _env_float("ASD_MIN_MOUTH_MOTION", 0.03))
    # Fraction of a turn's tracked faces a face track must account for before it
    # may be chosen as the speaker. The camera frames the speaker for most of a
    # turn; without this, a briefly visible background face that jitters can
    # outrank them and name the turn (observed: a silent panellist from another
    # programme, and the host over a co-panelist's turns).
    ASD_MIN_TRACK_PRESENCE: float = field(
        default_factory=lambda: _env_float("ASD_MIN_TRACK_PRESENCE", 0.30))
    # Per-turn identity taken from the visible speaking face, overriding the
    # audio-derived speaker label.
    #
    # ON, now that the mouth-motion gate is calibrated (see
    # ASD_MIN_MOUTH_MOTION). It is needed to correct diarization mistakes: a
    # speaker's turn can be clustered with a co-panelist, and when their face is
    # visibly talking the override puts the turn back on the right person. With
    # the motion gate a silent listener cannot be selected, so the earlier
    # failure (a still face naming the turn) cannot recur.
    ENABLE_TURN_OVERRIDE: bool = field(
        default_factory=lambda: os.getenv("ENABLE_TURN_OVERRIDE", "1") == "1")
    # Whisper's VAD drops quiet or heavily overlapped speech before decoding,
    # which loses words in exactly the interruptions talk-shows are full of.
    # Set WHISPER_VAD_FILTER=0 to recover them (at the cost of more hallucinated
    # text in silence).
    WHISPER_VAD_FILTER: bool = field(
        default_factory=lambda: os.getenv("WHISPER_VAD_FILTER", "1") == "1")
    # Decode knobs for the ASR-recall ablation (P1). Every default below
    # reproduces faster-whisper's own default, so leaving them unset keeps the
    # v9 decode bit-for-bit; they exist to be swept and measured, not tuned by
    # eye. See scripts/ablate_asr.py and data/gt/rtv_goll_table/README.md.
    #
    # condition_on_previous_text feeds each window's text back as the next
    # window's prompt. It is the standard cause of Whisper's degenerate
    # repetition loops, and v9 cues 85-90 are exactly such a loop (the same
    # clause re-emitted five times), which the annotator had to rewrite by
    # hand. Set WHISPER_CONDITION_ON_PREV=0 to break the feedback path.
    WHISPER_CONDITION_ON_PREV: bool = field(
        default_factory=lambda: os.getenv("WHISPER_CONDITION_ON_PREV", "1") == "1")
    WHISPER_BEAM_SIZE: int = field(
        default_factory=lambda: int(os.getenv("WHISPER_BEAM_SIZE", "5")))
    # Raising the compression-ratio threshold makes the decoder tolerate more
    # repetition before discarding a window; lowering it discards sooner.
    WHISPER_COMPRESSION_RATIO_THRESHOLD: float = field(
        default_factory=lambda: float(
            os.getenv("WHISPER_COMPRESSION_RATIO_THRESHOLD", "2.4")))
    # Lowering no_speech_threshold keeps quiet/overlapped windows that would
    # otherwise be dropped as silence -- the other half of the missing-words
    # problem that WHISPER_VAD_FILTER addresses.
    WHISPER_NO_SPEECH_THRESHOLD: float = field(
        default_factory=lambda: float(os.getenv("WHISPER_NO_SPEECH_THRESHOLD", "0.6")))
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
    ENABLE_PUNCTUATION_RESTORE: bool = True  # lightweight regex heuristic (no heavy model), fixes NER on unpunctuated Whisper

    BASE_DIR: Path = field(default_factory=_resolve_base_dir)
    SCRATCH_DIR: Path = field(default_factory=_resolve_scratch_dir)
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

    def use_cuda(self) -> bool:
        """True when CUDA is both selected and actually usable."""
        return self.DEVICE == "cuda" and torch.cuda.is_available()

    def onnx_providers(self) -> list:
        """ORT execution providers with a bounded CUDA arena.

        Requesting the CUDA EP is not the same as getting it: ORT falls back
        to CPU silently. Callers that require the GPU should pair this with
        ``runtime.assert_cuda_execution_provider()``.
        """
        return onnx_providers(self.use_cuda())

    def onnx_ctx_id(self) -> int:
        """InsightFace ctx_id: 0 selects GPU 0, -1 forces CPU.

        There is no ORT Metal/MPS provider, so 'mps' maps to CPU.
        """
        return 0 if self.use_cuda() else -1

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
            self.SCRATCH_DIR,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def get_ner_model(self) -> str:
        """Return primary NER model, fallback available via BANGLABERT_NER_FALLBACK."""
        return self.BANGLABERT_NER_MODEL


# Global config instance
config = Config()
