"""Tests for active-speaker selection (mouth motion) and track-level identity.

These cover the failure that motivated them: a still, well-framed listener
matching the registry better than the speaker, and therefore naming the
speaker's turns.
"""
from __future__ import annotations

import numpy as np
import pytest

from engines.active_speaker import (
    group_by_track,
    speaking_track,
    track_identity,
    track_mean_embedding,
)
from models import FaceOccurrence


def _face(t, track, motion, name="UNKNOWN", conf=0.0, emb=None):
    return FaceOccurrence(frame_time=t, box=(0, 0, 10, 10), track_id=0,
                          resolved_face_id=name, face_confidence=conf,
                          face_track_id=track, mouth_motion=motion,
                          embedding=emb)


# ── grouping / selection ───────────────────────────────────────────────

def test_group_by_track_drops_untracked_faces():
    groups = group_by_track([_face(0, 1, 0.1), _face(0.1, 1, 0.2),
                             _face(0.2, -1, 0.9)])
    assert set(groups) == {1}
    assert len(groups[1]) == 2


def test_speaking_track_picks_the_moving_face():
    faces = ([_face(i * 0.1, 7, 0.01) for i in range(5)]
             + [_face(i * 0.1, 9, 0.30) for i in range(5)])
    assert speaking_track(faces) == 9


def test_motion_decides_even_when_the_listener_matches_better():
    # The listener has a confident registry identity but a still mouth; the
    # speaker has no identity yet but is moving. Motion must win.
    faces = ([_face(i * 0.1, 7, 0.0, name="Shahriar", conf=0.90)
              for i in range(10)]
             + [_face(i * 0.1, 9, 0.20, name="UNKNOWN") for i in range(3)])
    assert speaking_track(faces) == 9


def test_speaking_track_respects_min_motion():
    faces = [_face(0, 1, 0.001)]
    assert speaking_track(faces, min_motion=0.05) is None
    assert speaking_track(faces, min_motion=0.0) == 1


def test_speaking_track_is_none_without_tracks():
    assert speaking_track([_face(0, -1, 5.0)]) is None


# ── track identity ─────────────────────────────────────────────────────

def test_track_identity_is_the_majority_with_mean_confidence():
    faces = [_face(0, 1, 0.1, "Tushar", 0.70),
             _face(0.1, 1, 0.1, "Tushar", 0.80),
             _face(0.2, 1, 0.1, "UNKNOWN")]
    assert track_identity(faces) == ("Tushar", 0.75)


def test_track_identity_ignores_clusters_and_unknowns():
    faces = [_face(0, 1, 0.1, "face_cluster_1", 0.5),
             _face(0.1, 1, 0.1, "UNKNOWN")]
    assert track_identity(faces) is None


def test_track_mean_embedding_is_l2_normalised():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    mean = track_mean_embedding([_face(0, 1, 0.1, emb=a),
                                 _face(0.1, 1, 0.1, emb=b)])
    assert mean is not None
    assert abs(float(np.linalg.norm(mean)) - 1.0) < 1e-5


def test_track_mean_embedding_is_none_without_embeddings():
    assert track_mean_embedding([_face(0, 1, 0.1)]) is None


# ── track-level identity aggregation (vision) ──────────────────────────

def test_track_aggregation_lifts_a_weak_per_frame_match(monkeypatch):
    """Frames that individually miss the threshold can be named as a track.

    Reproduces why the speaking face contributed nothing before: each frame
    scores 0.6 against the reference (threshold 0.65), but the track mean is
    1.0. This is the mechanism that lets a badly-turned speaker be identified.
    """
    # engines.vision imports insightface/onnxruntime behind the NumPy ABI
    # guard; off the target environment the heavy native stack may be absent.
    monkeypatch.setenv("ALLOW_NUMPY_ABI_DRIFT", "1")
    try:
        from engines.vision import _resolve_track_identities
    except Exception as exc:            # pragma: no cover - environment dependent
        pytest.skip(f"engines.vision unavailable here: {exc}")

    registry = {"Tushar": (np.array([1.0, 0.0], dtype=np.float32), 0)}
    f1 = np.array([0.6, 0.8], dtype=np.float32)
    f2 = np.array([0.6, -0.8], dtype=np.float32)
    occurrences = [
        FaceOccurrence(frame_time=0.0, box=(0, 0, 1, 1), track_id=0,
                       resolved_face_id="UNKNOWN", face_confidence=0.6,
                       embedding=f1, face_track_id=1),
        FaceOccurrence(frame_time=0.125, box=(0, 0, 1, 1), track_id=0,
                       resolved_face_id="UNKNOWN", face_confidence=0.6,
                       embedding=f2, face_track_id=1),
    ]

    out = _resolve_track_identities(occurrences, registry)

    assert [o.resolved_face_id for o in out] == ["Tushar", "Tushar"]
    assert all(o.face_confidence > 0.9 for o in out)
