"""Active-speaker selection: choose the face that is *speaking*, not the one that matches best.

Why this module exists
----------------------
The original identity rule asked "which enrolled face is visible during this
turn, and which of those matches the registry best?". In multi-camera broadcast
several faces are visible at once, and a still, well-framed **listener** can
match the registry more reliably than the speaker — who may be turned, small, or
badly lit. On the RTV Goll Table episode the speaker's own face cleared the
similarity threshold once in 616 detections while a silent co-panelist cleared
it 55 times, so the silent panellist won the vote and named every turn.

Similarity answers *who is present*. Speaker indexing needs *who is speaking*.
The observable difference is motion: the speaking face is the one whose mouth is
moving. These helpers implement that choice and the matching track-level
identity aggregation.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def group_by_track(faces: List) -> Dict[int, List]:
    """Group face occurrences by their temporal track id (negative ids dropped)."""
    out: Dict[int, List] = {}
    for f in faces:
        tid = getattr(f, "face_track_id", -1)
        if tid is None or tid < 0:
            continue
        out.setdefault(tid, []).append(f)
    return out


def track_mean_motion(faces: List) -> float:
    """Mean mouth motion of the faces belonging to one track."""
    motions = [float(getattr(f, "mouth_motion", 0.0) or 0.0) for f in faces]
    return sum(motions) / len(motions) if motions else 0.0


def speaking_track(faces: List, min_motion: float = 0.0,
                   min_presence: float = 0.0) -> Optional[int]:
    """Track id whose mouth moves most among the prominently visible faces.

    ``min_presence`` is the fraction of the turn's tracked faces a track must
    account for before it is even considered. Without it, a two-frame
    background face that happens to jitter outranks the person the camera is
    actually on: the v6 run named turns after "Zahed Ur Rahman" (a panellist
    of a different programme, present in the frame but silent) and after the
    host during Shahriar's turns for exactly that reason. The camera frames the
    speaker for most of a turn, so motion should only choose *among* the
    faces that actually occupy it.

    ``min_motion`` then gates the winner: with 0.0 the most-moving eligible
    track is always accepted; raise it to require real articulation.
    """
    tracks = group_by_track(faces)
    if not tracks:
        return None
    total = sum(len(group) for group in tracks.values())
    if total <= 0:
        return None

    eligible = {tid: group for tid, group in tracks.items()
                if len(group) / total >= min_presence}
    if not eligible:
        return None

    best_tid, best_motion = None, None
    for tid, group in eligible.items():
        motion = track_mean_motion(group)
        if best_motion is None or motion > best_motion:
            best_tid, best_motion = tid, motion
    if best_tid is None or (best_motion or 0.0) < min_motion:
        return None
    return best_tid


def track_identity(faces: List) -> Optional[tuple]:
    """Dominant registry identity within a track's faces, or None.

    Faces of one track share an identity because track-level aggregation
    assigned it; this returns the majority non-cluster match together with the
    mean similarity of its frames.
    """
    votes: Dict[str, List[float]] = {}
    for f in faces:
        fid = getattr(f, "resolved_face_id", "UNKNOWN")
        if not fid or fid == "UNKNOWN" or fid.startswith("face_cluster_"):
            continue
        votes.setdefault(fid, []).append(float(f.face_confidence))
    if not votes:
        return None
    name = max(votes, key=lambda n: len(votes[n]))
    sims = votes[name]
    return name, round(sum(sims) / len(sims), 3)


def track_mean_embedding(faces: List) -> Optional[np.ndarray]:
    """L2-normalised mean embedding of a track — stabler than any single frame.

    A partially-turned face may sit below the similarity threshold frame by
    frame, yet its track average is close to the enrolled embedding. This is
    what lets the speaking face be named at all.
    """
    embs = [f.embedding for f in faces
            if getattr(f, "embedding", None) is not None]
    if not embs:
        return None
    stack = np.stack([e / (np.linalg.norm(e) + 1e-9) for e in embs])
    mean = stack.mean(axis=0)
    norm = np.linalg.norm(mean)
    return mean / norm if norm > 0 else None
