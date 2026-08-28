"""Vision engine for face detection, recognition, and lip-sync analysis using InsightFace."""

from __future__ import annotations

from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np

# config applies the NumPy ABI lock + compat aliases at import time (see
# runtime.apply_numpy_compat) and must therefore be imported before
# insightface, whose wheels are built against the NumPy 1.x C ABI.
from config import config
from runtime import (
    assert_cuda_execution_provider,
    assert_numpy_abi,
    release_gpu_memory,
)

# insightface/onnxruntime wheels are built against the NumPy 1.x C ABI.
assert_numpy_abi()

from insightface.app import FaceAnalysis

from models import FaceOccurrence
from engines.media import extract_frames
from engines.clustering import NOISE_LABEL, cluster_face_embeddings

#: Abort rather than hand fusion a decimated view of the video. Isolated
#: unreadable frames are normal; a systemic failure (wrong provider, corrupt
#: JPEGs, OOM) is not, and used to surface only as "Detected 0 face occurrences".
MAX_FRAME_FAILURE_RATE = 0.10


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


def _clamp_box(box: Tuple[int, int, int, int],
               frame: np.ndarray) -> Tuple[int, int, int, int]:
    """Clip a detector bbox to the frame.

    InsightFace emits boxes that run off the edge, and negative coordinates do
    NOT raise under NumPy slicing — ``frame[-10:50, -20:30]`` silently yields
    an empty (0, 0) array, so an out-of-frame face reads as "no mouth motion"
    rather than as an error.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(x1), w))
    y1 = max(0, min(int(y1), h))
    x2 = max(x1, min(int(x2), w))
    y2 = max(y1, min(int(y2), h))
    return x1, y1, x2, y2


def _mouth_region_diff(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int],
                       frame1: np.ndarray, frame2: np.ndarray) -> float:
    """Normalized absolute pixel difference in the mouth region (lower bbox half).

    Diagnostic only: at VISION_FPS=1 the two frames are a full second apart,
    far above phoneme rate, so this is a coarse motion proxy and NOT a
    lip-sync signal. engines/fusion.py deliberately does not consume it.
    """
    x1_1, y1_1, x2_1, y2_1 = _clamp_box(box1, frame1)
    x1_2, y1_2, x2_2, y2_2 = _clamp_box(box2, frame2)

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
    """Load face embeddings from registry images. Returns {name: (embedding, index)}.

    Failures are reported rather than swallowed: with a three-photo registry,
    one silently skipped image removes a third of the P1 identity evidence and
    the pipeline degrades all the way to generic Speaker_N labels with no
    indication of why.
    """
    registry: Dict[str, Tuple[np.ndarray, int]] = {}
    registry_dir = config.DATA_REGISTRY_DIR

    if not registry_dir.exists():
        print(f"[vision] registry directory {registry_dir} does not exist — "
              f"P1 registry identification disabled")
        return registry

    image_files: list = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        image_files.extend(sorted(registry_dir.glob(ext)))
    image_files.sort(key=lambda p: p.name)
    skipped: List[str] = []
    for idx, img_path in enumerate(image_files):
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                skipped.append(f"{img_path.name} (unreadable)")
                continue

            faces = app.get(img)
            if not faces:
                skipped.append(f"{img_path.name} (no face detected)")
                continue

            largest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            embedding = largest_face.embedding / np.linalg.norm(largest_face.embedding)

            name = img_path.stem.replace("_", " ")
            registry[name] = (embedding, idx)
        except Exception as e:
            skipped.append(f"{img_path.name} ({e.__class__.__name__}: {e})")

    if skipped:
        print(f"[vision] registry: skipped {len(skipped)}/{len(image_files)} "
              f"image(s): {'; '.join(skipped)}")
    if image_files and not registry:
        raise RuntimeError(
            f"{len(image_files)} registry image(s) in {registry_dir} but not one "
            f"usable embedding — every face would resolve to Speaker_N. "
            f"Check the photos are frontal and that ArcFace loaded."
        )
    print(f"[vision] registry loaded: {sorted(registry)}")
    return registry


def _process_frames_with_vision(
    frame_paths: List[str],
    registry: Dict[str, Tuple[np.ndarray, int]],
    app: FaceAnalysis,
) -> List[FaceOccurrence]:
    """Process all frames, detect faces, compute lip-sync, match to registry."""
    occurrences: List[FaceOccurrence] = []
    prev_faces: List[Tuple[Tuple[int, int, int, int], np.ndarray, np.ndarray]] = []  # (box, embedding, frame)
    failures: List[str] = []

    for frame_idx, frame_path in enumerate(frame_paths):
        # Frames are extracted at t=0, 1/fps, 2/fps ... so frame i sits at
        # i/fps. The old '(idx+1)/fps' shifted every face occurrence a full
        # second late, corrupting audio-visual association.
        frame_time = frame_idx / config.VISION_FPS

        try:
            frame = cv2.imread(frame_path)
            if frame is None:
                failures.append(f"{frame_idx}: unreadable")
                continue

            faces = app.get(frame)
            if not faces:
                prev_faces = []
                continue

            for face in faces:
                box = _clamp_box(tuple(map(int, face.bbox)), frame)
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

            prev_faces = [(_clamp_box(tuple(map(int, f.bbox)), frame),
                           f.embedding / np.linalg.norm(f.embedding), frame)
                          for f in faces]

            # Kaggle T4: periodic reclaim to limit fragmentation over ~2.7k frames.
            if frame_idx % 500 == 0:
                release_gpu_memory()

        except Exception as e:
            failures.append(f"{frame_idx}: {e.__class__.__name__}: {e}")
            continue

    if failures:
        rate = len(failures) / max(len(frame_paths), 1)
        print(f"[vision] {len(failures)}/{len(frame_paths)} frames failed "
              f"({rate:.1%}); first: {failures[:3]}")
        if rate > MAX_FRAME_FAILURE_RATE:
            raise RuntimeError(
                f"{rate:.1%} of frames failed (limit {MAX_FRAME_FAILURE_RATE:.0%}). "
                f"Refusing to emit partial vision evidence into fusion — the "
                f"identity cascade cannot distinguish 'absent' from 'not looked at'. "
                f"First failures: {failures[:5]}"
            )

    release_gpu_memory()
    return occurrences


def _cluster_unknown_faces(occurrences: List[FaceOccurrence]) -> List[FaceOccurrence]:
    """Label UNKNOWN faces with identity clusters (P4 evidence for fusion).

    Delegates to engines.clustering, which bounds the fit set so the dense
    n x n distance matrix cannot exhaust host RAM on a full-length show, and
    which is shared verbatim with the YOLO ablation so the two detector arms
    differ only in detection.
    """
    unknown_indices: List[int] = []
    unknown_embeddings: List[np.ndarray] = []

    for idx, occ in enumerate(occurrences):
        if occ.resolved_face_id == "UNKNOWN" and occ.embedding is not None:
            unknown_indices.append(idx)
            unknown_embeddings.append(occ.embedding)

    if len(unknown_embeddings) < config.DBSCAN_MIN_SAMPLES:
        return occurrences

    # Agglomerative needs a cluster count: the speakers not already pinned to a
    # registry identity. Requires the NUM_SPEAKERS hint; without it, DBSCAN.
    registry_names = {
        o.resolved_face_id for o in occurrences
        if o.resolved_face_id != "UNKNOWN"
        and not o.resolved_face_id.startswith("face_cluster_")
    }
    algorithm = config.CLUSTERING
    n_clusters: Optional[int] = None
    if algorithm == "agglomerative":
        if config.NUM_SPEAKERS:
            n_clusters = max(1, int(config.NUM_SPEAKERS) - len(registry_names))
        else:
            print("[vision] CLUSTERING=agglomerative needs NUM_SPEAKERS; "
                  "falling back to DBSCAN")
            algorithm = "dbscan"

    try:
        labels = cluster_face_embeddings(
            np.asarray(unknown_embeddings),
            algorithm=algorithm,
            n_clusters=n_clusters,
            eps=config.DBSCAN_EPS,
            min_samples=config.DBSCAN_MIN_SAMPLES,
        )
    except MemoryError as e:
        raise RuntimeError(
            f"Face clustering ran out of memory on {len(unknown_embeddings)} "
            f"embeddings. Lower engines.clustering.MAX_FIT_SAMPLES or reduce "
            f"VISION_FPS."
        ) from e

    n_clustered = len({int(l) for l in labels if l >= 0})
    n_noise = int(sum(1 for l in labels if l < 0))
    print(f"[vision] clustered {len(unknown_embeddings)} unknown faces into "
          f"{n_clustered} identities ({n_noise} noise) via {algorithm}")

    for idx, label in zip(unknown_indices, labels):
        label = int(label)
        is_noise = label == NOISE_LABEL
        occurrences[idx] = FaceOccurrence(
            frame_time=occurrences[idx].frame_time,
            box=occurrences[idx].box,
            track_id=NOISE_LABEL if is_noise else label,
            resolved_face_id="face_cluster_noise" if is_noise else f"face_cluster_{label}",
            face_confidence=occurrences[idx].face_confidence,
            lip_sync_score=occurrences[idx].lip_sync_score,
            embedding=occurrences[idx].embedding,
        )

    release_gpu_memory()
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
    # Requesting the CUDA EP is not the same as getting it: ORT falls back to
    # CPU without raising, which turns a GPU run into a silent ~20x slowdown.
    if config.use_cuda():
        assert_cuda_execution_provider()

    app = FaceAnalysis(
        providers=config.onnx_providers(),
        allowed_modules=['detection', 'recognition'],
    )
    # ctx_id=-1 forces CPU (onnxruntime has no MPS provider on macOS).
    app.prepare(ctx_id=config.onnx_ctx_id(), det_size=(640, 640))

    registry = _load_registry_embeddings(app)

    if frame_paths is None:
        frame_paths = extract_frames(video_path, fps=config.VISION_FPS)

    try:
        occurrences = _process_frames_with_vision(frame_paths, registry, app)
        occurrences = _cluster_unknown_faces(occurrences)
    finally:
        # ONNX Runtime device memory is invisible to torch's allocator; only
        # dropping the session returns it.
        del app
        release_gpu_memory()

    occurrences.sort(key=lambda o: o.frame_time)
    return occurrences