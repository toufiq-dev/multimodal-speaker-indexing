"""Memory-bounded clustering of unknown face embeddings.

Shared by ``engines/vision.py`` (RetinaFace) and ``engines/vision_yolo.py``
(YOLO ablation) so the two detector arms differ *only* in detection — which
is the whole point of the ablation. Previously each module carried its own
near-copy of this logic and they had already drifted (the YOLO copy defaulted
to ``n_clusters=2`` and used euclidean HDBSCAN; the RetinaFace copy did not).

Why the sample cap exists
-------------------------
``AgglomerativeClustering`` materialises a dense ``n x n`` float64 distance
matrix, and ``DBSCAN`` with a non-precomputed metric performs O(n^2) distance
work. At 1 FPS over a 53-minute show with ~4 faces per frame that is ~12k
embeddings (1.15 GB); a busy panel at ~10 faces/frame reaches ~30k (7.2 GB)
on top of SciPy's linkage copy. Kaggle offers ~13-16 GB of *host* RAM shared
with everything else, so this — not VRAM — is the real OOM in the pipeline.

We therefore fit on a bounded random subsample and assign the remainder by
nearest cluster centroid. For ArcFace embeddings this is a standard and
well-behaved approximation: identities form tight, near-spherical cosine
clusters, so a few thousand samples recover the same centroids as the full
set. The subsample uses a fixed seed because ablation cells must be
bit-reproducible across runs.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN

#: Fitting ceiling. 6000^2 x 8 B ~= 288 MB for the dense distance matrix,
#: which leaves ample headroom next to the frame cache and ORT arena.
MAX_FIT_SAMPLES = 6000

#: Deterministic subsample: ablation cells must be reproducible.
SUBSAMPLE_SEED = 0

NOISE_LABEL = -1


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-9)


def _fit_labels(
    x: np.ndarray,
    algorithm: str,
    n_clusters: Optional[int],
    eps: float,
    min_samples: int,
) -> np.ndarray:
    """Cluster the (already bounded) fitting set; -1 marks noise."""
    if algorithm == "agglomerative" and n_clusters and n_clusters >= 2:
        n_clusters = min(int(n_clusters), len(x))
        if n_clusters >= 2:
            return AgglomerativeClustering(
                n_clusters=n_clusters, metric="cosine", linkage="average",
            ).fit_predict(x)

    if algorithm == "hdbscan":
        try:
            import hdbscan  # type: ignore
            return hdbscan.HDBSCAN(
                min_cluster_size=max(2, min_samples), metric="euclidean",
            ).fit_predict(x)
        except Exception:
            pass  # optional extra; fall through to DBSCAN

    return DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit_predict(x)


def _centroids(x: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-cluster L2-normalised centroids, plus their label ids."""
    ids = np.array(sorted({int(l) for l in labels if l >= 0}), dtype=int)
    if ids.size == 0:
        return np.empty((0, x.shape[1]), dtype=x.dtype), ids
    cents = np.stack([x[labels == i].mean(axis=0) for i in ids])
    return _l2_normalize(cents), ids


def cluster_face_embeddings(
    embeddings: np.ndarray,
    *,
    algorithm: str = "dbscan",
    n_clusters: Optional[int] = None,
    eps: float = 0.5,
    min_samples: int = 3,
    max_fit_samples: int = MAX_FIT_SAMPLES,
    seed: int = SUBSAMPLE_SEED,
) -> np.ndarray:
    """Assign a cluster label to every embedding under a fixed memory budget.

    Args:
        embeddings: ``(n, d)`` face embeddings; need not be normalised.
        algorithm: ``dbscan`` | ``agglomerative`` | ``hdbscan``. Unknown
            values and unavailable extras fall back to DBSCAN.
        n_clusters: Required by ``agglomerative``; ignored otherwise.
        eps: DBSCAN cosine-distance radius. Also used as the assignment
            radius for out-of-sample points under density algorithms, so a
            point far from every centroid stays noise rather than being
            forced into the nearest identity.
        min_samples: DBSCAN/HDBSCAN core-point threshold.
        max_fit_samples: Cap on the number of points passed to the clusterer.
        seed: Subsample RNG seed.

    Returns:
        ``(n,)`` int array of labels, ``-1`` for noise.
    """
    x = np.asarray(embeddings, dtype=np.float32)
    if x.ndim != 2 or len(x) == 0:
        return np.full(len(x), NOISE_LABEL, dtype=int)

    x = _l2_normalize(x)
    n = len(x)

    if n <= max_fit_samples:
        return _fit_labels(x, algorithm, n_clusters, eps, min_samples).astype(int)

    rng = np.random.default_rng(seed)
    fit_idx = rng.choice(n, size=max_fit_samples, replace=False)
    fit_labels = _fit_labels(x[fit_idx], algorithm, n_clusters, eps, min_samples)

    cents, ids = _centroids(x[fit_idx], fit_labels)
    if ids.size == 0:
        return np.full(n, NOISE_LABEL, dtype=int)

    labels = np.full(n, NOISE_LABEL, dtype=int)
    labels[fit_idx] = fit_labels

    # Out-of-sample assignment by cosine distance to the fitted centroids.
    # Chunked so the (n_out x k) similarity matrix stays small.
    out_mask = np.ones(n, dtype=bool)
    out_mask[fit_idx] = False
    out_idx = np.flatnonzero(out_mask)
    # Density algorithms keep their noise semantics; a partitioning algorithm
    # has no notion of noise, so every point gets its nearest centroid.
    radius = None if algorithm == "agglomerative" else eps

    for start in range(0, len(out_idx), 4096):
        block = out_idx[start:start + 4096]
        sims = x[block] @ cents.T
        best = sims.argmax(axis=1)
        assigned = ids[best]
        if radius is not None:
            too_far = (1.0 - sims[np.arange(len(block)), best]) > radius
            assigned = np.where(too_far, NOISE_LABEL, assigned)
        labels[block] = assigned

    return labels
