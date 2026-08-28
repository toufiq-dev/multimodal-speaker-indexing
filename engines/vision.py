"""Vision engine for face detection, recognition, and lip-sync analysis using InsightFace."""

from __future__ import annotations

import os
import math
import gc
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
# NumPy 2.0 ABI patch — InsightFace 0.7.3 still references np.NaN/np.Inf removed in NumPy 2.0
if not hasattr(np, "NaN"):
    np.NaN = np.nan  # type: ignore[attr-defined]
if not hasattr(np, "Inf"):
    np.Inf = np.inf  # type: ignore[attr-defined]
if not hasattr(np, "PINF"):
    np.PINF = np.inf  # type: ignore[attr-defined]
if not hasattr(np, "NINF"):
    np.NINF = -np.inf  # type: ignore[attr-defined]

import torch
from insightface.app import FaceAnalysis
from sklearn.cluster import AgglomerativeClustering, DBSCAN

from config import config
from models import FaceOccurrence
from engines.media import extract_frames


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two normalized vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _bbox_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """Compute IoU between two bounding boxes (x1, y1, x2, y2)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def _mouth_region_diff(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int],
                       frame1: np.ndarray, frame2: np.ndarray) -> float:
    """Compute normalized absolute pixel difference in mouth region (lower half of bbox)."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    mouth_y1_1 = y1_1 + (y2_1 - y1_1) // 2
    mouth_y2_1 = y2_1
    mouth_x1_1 = x1_1
    mouth_x2_1 = x2_1

    mouth_y1_2 = y1_2 + (y2_2 - y1_2) // 2
    mouth_y2_2 = y2_2
    mouth_x1_2 = x1_2
    mouth_x2_2 = x2_2

    try:
        mouth1 = frame1[mouth_y1_1:mouth_y2_1, mouth_x1_1:mouth_x2_1]
        mouth2 = frame2[mouth_y1_2:mouth_y2_2, mouth_x1_2:mouth_x2_2]

        if mouth1.size == 0 or mouth2.size == 0:
            return 0.0

        h = min(mouth1.shape[0], mouth2.shape[0])
        w = min(mouth1.shape[1], mouth2.shape[1])
        if h == 0 or w == 0:
            return 0.0

        mouth1 = cv2.resize(mouth1, (w, h))
        mouth2 = cv2.resize(mouth2, (w, h))

        diff = cv2.absdiff(mouth1, mouth2).astype(np.float32)
        mean_diff = diff.mean() / 255.0
        return float(mean_diff)
    except Exception:
        return 0.0


def _load_registry_embeddings(app: FaceAnalysis) -> Dict[str, Tuple[np.ndarray, int]]:
    """Load face embeddings from registry images. Returns {name: (embedding, index)}."""
    registry: Dict[str, Tuple[np.ndarray, int]] = {}
    registry_dir = config.DATA_REGISTRY_DIR

    if not registry_dir.exists():
        return registry

    image_files: list = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        image_files.extend(sorted(registry_dir.glob(ext)))
    image_files.sort(key=lambda p: p.name)
    for idx, img_path in enumerate(image_files):
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            faces = app.get(img)
            if not faces:
                continue

            largest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            embedding = largest_face.embedding / np.linalg.norm(largest_face.embedding)

            name = img_path.stem.replace("_", " ")
            registry[name] = (embedding, idx)
        except Exception:
            continue

    return registry


def _process_frames_with_vision(
    frame_paths: List[str],
    registry: Dict[str, Tuple[np.ndarray, int]],
    app: FaceAnalysis,
) -> List[FaceOccurrence]:
    """Process all frames, detect faces, compute lip-sync, match to registry."""
    occurrences: List[FaceOccurrence] = []
    prev_faces: List[Tuple[Tuple[int, int, int, int], np.ndarray, np.ndarray]] = []  # (box, embedding, frame)

    for frame_idx, frame_path in enumerate(frame_paths):
        # Frames are extracted at t=0, 1/fps, 2/fps ... so frame i sits at
        # i/fps. The old '(idx+1)/fps' shifted every face occurrence a full
        # second late, corrupting audio-visual association.
        frame_time = frame_idx / config.VISION_FPS

        try:
            frame = cv2.imread(frame_path)
            if frame is None:
                continue

            faces = app.get(frame)
            if not faces:
                prev_faces = []
                continue

            for face in faces:
                box = tuple(map(int, face.bbox))
                embedding = face.embedding / np.linalg.norm(face.embedding)

                best_name = "UNKNOWN"
                best_sim = 0.0
                best_track_id = -1

                for name, (reg_emb, track_id) in registry.items():
                    sim = _cosine_similarity(embedding, reg_emb)
                    if sim > best_sim:
                        best_sim = sim
                        best_name = name
                        best_track_id = track_id

                face_confidence = best_sim
                if best_sim <= config.FACE_SIM_THRESHOLD:
                    best_name = "UNKNOWN"
                    best_track_id = -1

                lip_sync = 0.0
                if prev_faces:
                    best_iou = 0.0
                    best_prev_box = None
                    best_prev_frame = None
                    for prev_box, _, prev_frame in prev_faces:
                        iou = _bbox_iou(box, prev_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_prev_box = prev_box
                            best_prev_frame = prev_frame

                    if best_prev_box and best_iou > 0.3 and best_prev_frame is not None:
                        lip_sync = _mouth_region_diff(box, best_prev_box, frame, best_prev_frame)

                occurrences.append(FaceOccurrence(
                    frame_time=frame_time,
                    box=box,
                    track_id=best_track_id if best_track_id >= 0 else len(occurrences),
                    resolved_face_id=best_name,
                    face_confidence=face_confidence,
                    lip_sync_score=lip_sync,
                    embedding=embedding if best_name == "UNKNOWN" else None,
                ))

            prev_faces = [(tuple(map(int, f.bbox)), f.embedding / np.linalg.norm(f.embedding), frame) for f in faces]

            # Kaggle T4: periodic cache clear to prevent fragmentation over 2.7k frames
            if frame_idx % 100 == 0:
                torch.cuda.empty_cache()
                gc.collect()

        except Exception:
            continue

    torch.cuda.empty_cache()
    gc.collect()
    return occurrences


def _cluster_unknown_faces(occurrences: List[FaceOccurrence]) -> List[FaceOccurrence]:
    """Cluster UNKNOWN face embeddings: Agglomerative (with NUM_SPEAKERS hint) or DBSCAN fallback."""
    unknown_indices = []
    unknown_embeddings = []

    for idx, occ in enumerate(occurrences):
        if occ.resolved_face_id == "UNKNOWN" and occ.embedding is not None:
            unknown_indices.append(idx)
            unknown_embeddings.append(occ.embedding)

    if len(unknown_embeddings) < config.DBSCAN_MIN_SAMPLES:
        torch.cuda.empty_cache()
        return occurrences

    embeddings_array = np.array(unknown_embeddings)
    # Count already-resolved registry identities to estimate remaining speakers
    try:
        registry_names = {o.resolved_face_id for o in occurrences if o.resolved_face_id not in ("UNKNOWN", "face_cluster_noise") and not o.resolved_face_id.startswith("face_cluster_")}
        n_registry = len(registry_names)
        use_agg = (config.CLUSTERING == "agglomerative" and config.NUM_SPEAKERS is not None)
        if use_agg:
            n_clusters = max(1, int(config.NUM_SPEAKERS) - n_registry)
            n_clusters = min(n_clusters, len(unknown_embeddings))
            if n_clusters >= 2:
                clustering = AgglomerativeClustering(n_clusters=n_clusters, metric="cosine", linkage="average")
                labels = clustering.fit_predict(embeddings_array)
            else:
                labels = DBSCAN(eps=config.DBSCAN_EPS, min_samples=config.DBSCAN_MIN_SAMPLES, metric="cosine").fit(embeddings_array).labels_
        else:
            clustering = DBSCAN(eps=config.DBSCAN_EPS, min_samples=config.DBSCAN_MIN_SAMPLES, metric="cosine")
            labels = clustering.fit(embeddings_array).labels_
    except Exception:
        try:
            clustering = DBSCAN(eps=config.DBSCAN_EPS, min_samples=config.DBSCAN_MIN_SAMPLES, metric="cosine")
            labels = clustering.fit(embeddings_array).labels_
        except Exception:
            torch.cuda.empty_cache()
            return occurrences
    finally:
        torch.cuda.empty_cache()

    for idx, label in zip(unknown_indices, labels):
        if label >= 0:
            occurrences[idx] = FaceOccurrence(
                frame_time=occurrences[idx].frame_time,
                box=occurrences[idx].box,
                track_id=label,
                resolved_face_id=f"face_cluster_{label}",
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

    return occurrences


def run_vision_pipeline(video_path: str, frame_paths: Optional[List[str]] = None) -> List[FaceOccurrence]:
    """
    Full vision pipeline: extract frames, detect faces, match registry, lip-sync, cluster.

    Args:
        video_path: Path to input video file.
        frame_paths: Optional pre-extracted frame paths. If None, extracts new frames.

    Returns:
        List of FaceOccurrence sorted by frame_time.
    """
    app = FaceAnalysis(
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
        allowed_modules=['detection', 'recognition'],
    )
    # ctx_id=-1 forces CPU (onnxruntime has no MPS provider on macOS).
    app.prepare(ctx_id=0 if config.DEVICE == "cuda" else -1, det_size=(640, 640))

    registry = _load_registry_embeddings(app)

    if frame_paths is None:
        frame_paths = extract_frames(video_path, fps=config.VISION_FPS)

    occurrences = _process_frames_with_vision(frame_paths, registry, app)
    occurrences = _cluster_unknown_faces(occurrences)
    occurrences.sort(key=lambda o: o.frame_time)

    return occurrences