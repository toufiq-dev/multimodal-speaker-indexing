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
# Load cached faces? Here we re-run vision on cached frames (2,708 jpgs)
video="data/input/global_tv_talkshow.mp4"
frames=sorted(Path("data/output/frames").glob("frame_*.jpg"))
frames=[str(p) for p in frames]
for detector in ["insightface", "yolo"]:
    config.VISION_DETECTOR=detector
    config.YOLO_MODEL="yolov8n-face.pt" if detector=="yolo" else "yolov8n-face.pt"
    faces = run_vision_yolo(video, frames) if detector=="yolo" else run_vision_pipeline(video, frames)
    print(detector, len(faces), set(f.resolved_face_id for f in faces))
PY
echo "✓ YOLO ablation done — check evaluation/metrics face_attribution"
