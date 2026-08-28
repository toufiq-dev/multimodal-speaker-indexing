#!/bin/bash
# YOLO-face ablation: RetinaFace vs YOLOv8x vs YOLOv11n on cached Global TV frames
# Reuses models/bengaliAI_ct2 if available, else Systran medium
set -e
echo "YOLO-face ablation (no re-extract, re-cluster only)..."
python - << 'PY'
from pathlib import Path
import json, numpy as np
from engines.vision import run_vision_pipeline
from engines.vision_yolo import run_vision_yolo
from config import config
# Load cached faces? Here we re-run vision on cached frames (2,708 jpgs) — absolute for Kaggle /kaggle/working
from config import config as _cfg
# Frames now live under SCRATCH_DIR (/tmp on Kaggle), not the committed output
# directory. Fall back to the legacy locations for previously cached runs.
import os
candidates = [
    _cfg.SCRATCH_DIR / "frames",
    _cfg.DATA_OUTPUT_DIR / "frames",
    Path("/kaggle/working/data/output/frames"),
    Path("data/output/frames"),
]
if Path("/kaggle/working/data/input/global_tv_talkshow.mp4").exists():
    video = "/kaggle/working/data/input/global_tv_talkshow.mp4"
else:
    video = "data/input/global_tv_talkshow.mp4"

frames = []
for c in candidates:
    frames = sorted(c.glob("frame_*.jpg"))
    if frames:
        print(f"using cached frames from {c}")
        break
if not frames:
    from engines.media import extract_frames
    frames = [Path(f) for f in extract_frames(video, fps=_cfg.VISION_FPS)]
frames=[str(p) for p in frames]
# The YOLO arm now RAISES if ultralytics or the weights are missing, instead of
# silently re-running RetinaFace and reporting two identical arms.
for detector in ["insightface", "yolo"]:
    config.VISION_DETECTOR=detector
    try:
        faces = run_vision_yolo(video, frames) if detector=="yolo" else run_vision_pipeline(video, frames)
    except RuntimeError as e:
        print(f"{detector}: SKIPPED — {e}")
        continue
    print(detector, len(faces), sorted(set(f.resolved_face_id for f in faces)))
PY
echo "✓ YOLO ablation done — check evaluation/metrics face_attribution"
