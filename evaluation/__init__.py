"""Evaluation framework for the multimodal Bangla talk-show indexing system.

Modules:
    metrics    - WER/CER, cpWER/WDER, DER/JER, face accuracy, fusion health
    baselines  - heuristic fusion + single-modality ablation pipelines
    ablations  - systematic component-swap experiment matrix
    tracking   - experiment configs, logs, reproducibility manifests
    dataset    - multi-video manifest + ground-truth schema management
"""

from evaluation.metrics import (
    wer,
    cer,
    cpwer,
    wder,
    der_jer,
    speaker_name_accuracy,
    face_attribution_accuracy,
    fusion_health_metrics,
)
from evaluation.dataset import load_manifest, GroundTruth

__all__ = [
    "wer", "cer", "cpwer", "wder", "der_jer",
    "speaker_name_accuracy", "face_attribution_accuracy",
    "fusion_health_metrics", "load_manifest", "GroundTruth",
]
