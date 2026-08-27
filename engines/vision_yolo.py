"""YOLO-based face detection alternative for talk-show studio robustness.

This module is the YOLO-face ablation for the thesis: it reuses the same
ArcFace embedding + registry threshold + DBSCAN/ Agglomerative pipeline
as engines/vision.py, but replaces RetinaFace (InsightFace) with a YOLO
detector. Model selection is grounded in the DAWN adverse-weather study
(YOLOv8x best mAP 80.44, Table 3.3) — same study framed for this pipeline.

DAWN has no faces, so we evaluate detector *selection* on DAWN (precision,
recall, mAP@50), and detector *robustness* on talk-show frames via CLAHE
augmentation (DAWN 2.3.2). This is a pure detector swap; identity resolution
remains in engines/fusion.py (P1 registry, P4 cluster).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering

from config import config
from models import FaceOccurrence

try:
    from ultralytics import YOLO as UltralyticsYOLO
    _YOLO_AVAILABLE = True
except Exception:
    UltralyticsYOLO = None  # type: ignore
    _YOLO_AVAILABLE = False

# ArcFace still from InsightFace for embedding (YOLO does not embed)
try:
    from insightface.app import FaceAnalysis as InsightFaceAnalysis
    _INSIGHT_AVAILABLE = True
except Exception:
    InsightFaceAnalysis = None  # type: ignore
    _INSIGHT_AVAILABLE = False


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _load_registry_embeddings_yolo(rec_model) -> Dict[str, Tuple[np.ndarray, int]]:
    """Load registry embeddings using ArcFace (same as vision.py)."""
    registry: Dict[str, Tuple[np.ndarray, int]] = {}
    reg_dir = config.DATA_REGISTRY_DIR
    if not reg_dir.exists():
        return registry
    import glob
    files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        files.extend(sorted(reg_dir.glob(ext)))
    files.sort(key=lambda p: p.name)
    for idx, img_path in enumerate(files):
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            # Use InsightFace to embed registry face (cropped by YOLO would also work,
            # but ArcFace needs aligned crop; InsightFace's get() does alignment)
            if rec_model is None:
                continue
            faces = rec_model.get(img)
            if not faces:
                continue
            largest = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
            emb = largest.embedding / (np.linalg.norm(largest.embedding) + 1e-9)
            name = img_path.stem.replace("_", " ")
            registry[name] = (emb, idx)
        except Exception:
            continue
    return registry


def _get_arcface_model():
    if not _INSIGHT_AVAILABLE:
        return None
    try:
        app = InsightFaceAnalysis(providers=['CPUExecutionProvider'], allowed_modules=['recognition'])
        app.prepare(ctx_id=-1)
        return app
    except Exception:
        return None


def run_vision_yolo(video_path: str, frame_paths: Optional[List[str]] = None) -> List[FaceOccurrence]:
    """
    YOLO-face vision pipeline. Falls back to InsightFace RetinaFace if YOLO not installed.
    """
    if not _YOLO_AVAILABLE:
        from engines.vision import run_vision_pipeline
        return run_vision_pipeline(video_path, frame_paths=frame_paths)

    # Load YOLO detector
    model_name = config.YOLO_MODEL
    # Try local cache first, else hub name
    try:
        yolo = UltralyticsYOLO(model_name)
    except Exception as e:
        # Fallback to insightface if YOLO model missing (e.g., face pt not downloaded)
        from engines.vision import run_vision_pipeline
        return run_vision_pipeline(video_path, frame_paths=frame_paths)

    # ArcFace for recognition
    rec_model = _get_arcface_model()

    # Frame list
    if frame_paths is None:
        from engines.media import extract_frames
        frame_paths = extract_frames(video_path, fps=config.VISION_FPS)

    registry = _load_registry_embeddings_yolo(rec_model)
    occurrences: List[FaceOccurrence] = []

    for frame_idx, frame_path in enumerate(frame_paths):
        frame_time = frame_idx / config.VISION_FPS
        try:
            frame = cv2.imread(frame_path)
            if frame is None:
                continue
            # YOLO inference (Ultralytics handles BGR internally, but works with cv2 BGR)
            results = yolo(frame, verbose=False)
            if not results or len(results[0].boxes) == 0:
                continue
            boxes = results[0].boxes
            for i, box in enumerate(boxes):
                # box.xyxy, conf
                xyxy = box.xyxy[0].cpu().numpy().astype(int)  # x1,y1,x2,y2
                conf = float(box.conf[0].cpu().numpy()) if hasattr(box.conf[0], 'cpu') else float(box.conf[0])
                x1, y1, x2, y2 = map(int, xyxy.tolist())
                bbox = (x1, y1, x2, y2)
                # Crop for ArcFace embedding
                crop = frame[max(0,y1):y2, max(0,x1):x2]
                embedding = None
                best_name = "UNKNOWN"
                best_sim = 0.0
                best_track = -1
                if rec_model is not None and crop.size != 0:
                    try:
                        # Resize to 112x112 for ArcFace via InsightFace's alignment (fallback: use frame)
                        faces = rec_model.get(crop)
                        if faces:
                            emb = faces[0].embedding / (np.linalg.norm(faces[0].embedding)+1e-9)
                            embedding = emb
                            for name, (reg_emb, tid) in registry.items():
                                sim = _cosine(emb, reg_emb)
                                if sim > best_sim:
                                    best_sim, best_name, best_track = sim, name, tid
                            if best_sim <= config.FACE_SIM_THRESHOLD:
                                best_name = "UNKNOWN"
                                best_track = -1
                                # keep embedding for clustering
                            else:
                                embedding = None  # known faces don't need clustering
                        else:
                            # No embedding extracted — still record occurrence for clustering if needed
                            embedding = None
                    except Exception:
                        embedding = None
                # If no embedding (e.g., rec failed), still create occurrence with bbox
                occurrences.append(FaceOccurrence(
                    frame_time=frame_time,
                    box=bbox,
                    track_id=best_track if best_track >= 0 else len(occurrences),
                    resolved_face_id=best_name,
                    face_confidence=best_sim if best_name != "UNKNOWN" else conf,
                    lip_sync_score=0.0,
                    embedding=embedding,
                ))
        except Exception:
            continue

    # Cluster UNKNOWN same as vision.py but with CLUSTERING switch
    # Collect UNKNOWN embeddings
    unknown_idx = []
    unknown_embs = []
    for idx, occ in enumerate(occurrences):
        if occ.resolved_face_id == "UNKNOWN" and occ.embedding is not None:
            unknown_idx.append(idx)
            unknown_embs.append(occ.embedding)
    if len(unknown_embs) >= config.DBSCAN_MIN_SAMPLES:
        X = np.array(unknown_embs)
        if config.CLUSTERING == "agglomerative":
            # Estimate n_clusters from diarization hint or registry remainder
            # For standalone vision pipeline, estimate as 2 if not provided
            n_clusters = 2
            # If NUM_SPEAKERS known, use it
            if config.NUM_SPEAKERS:
                n_clusters = max(1, int(config.NUM_SPEAKERS) - len(registry))
            # Cap by available embeddings
            n_clusters = min(n_clusters, len(X))
            if n_clusters >= 2:
                try:
                    agg = AgglomerativeClustering(n_clusters=n_clusters, metric='cosine', linkage='average')
                    labels = agg.fit_predict(X)
                except Exception:
                    # fallback to DBSCAN
                    db = DBSCAN(eps=config.DBSCAN_EPS, min_samples=config.DBSCAN_MIN_SAMPLES, metric='cosine')
                    labels = db.fit_predict(X)
            else:
                labels = np.zeros(len(X), dtype=int)
        elif config.CLUSTERING == "hdbscan":
            try:
                import hdbscan
                clusterer = hdbscan.HDBSCAN(min_cluster_size=config.DBSCAN_MIN_SAMPLES, metric='euclidean')
                labels = clusterer.fit_predict(X)
            except Exception:
                db = DBSCAN(eps=config.DBSCAN_EPS, min_samples=config.DBSCAN_MIN_SAMPLES, metric='cosine')
                labels = db.fit_predict(X)
        else:
            db = DBSCAN(eps=config.DBSCAN_EPS, min_samples=config.DBSCAN_MIN_SAMPLES, metric='cosine')
            labels = db.fit_predict(X)
        for idx, lab in zip(unknown_idx, labels):
            if lab >= 0:
                occurrences[idx] = FaceOccurrence(
                    frame_time=occurrences[idx].frame_time,
                    box=occurrences[idx].box,
                    track_id=int(lab),
                    resolved_face_id=f"face_cluster_{lab}",
                    face_confidence=occurrences[idx].face_confidence,
                    lip_sync_score=occurrences[idx].lip_sync_score,
                    embedding=occurrences[idx].embedding,
                )
            else:
                occurrences[idx] = FaceOccurrence(
                    frame_time=occurrences[idx].frame_time,
                    box=occurrences[idx].box,
                    track_id=-1,
                    resolved_face_id="face_cluster_noise",
                    face_confidence=occurrences[idx].face_confidence,
                    lip_sync_score=occurrences[idx].lip_sync_score,
                    embedding=occurrences[idx].embedding,
                )
    occurrences.sort(key=lambda o: o.frame_time)
    return occurrences
