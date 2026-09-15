"""Tests for deterministic identity resolution (no untrained gate)."""
from __future__ import annotations

from engines.fusion import GatingFusion
from models import DiarizationSegment, TranscribedSegment, WordToken, FaceOccurrence


DIA = [
    DiarizationSegment(0.0, 30.0, "SPEAKER_00"),   # host
    DiarizationSegment(30.0, 60.0, "SPEAKER_01"),
    DiarizationSegment(60.0, 90.0, "SPEAKER_02"),
]


def _seg(start, end, spk, text):
    return TranscribedSegment(start=start, end=end, text=text, words=[], speaker_id=spk)


def test_host_anchor_identifies_self_introduction():
    trans = [
        _seg(0, 20, "SPEAKER_00",
             "আপনাদের সাথে আছে আমি রফতান আঞ্জুমান নিকোল"),
        _seg(30, 50, "SPEAKER_01", "আমি মনে করি এটা ঠিক নয়"),
    ]
    resolved = GatingFusion().resolve_identities(DIA, trans, [])
    assert resolved["SPEAKER_00"][0] == "রফতান আঞ্জুমান নিকোল"
    assert resolved["SPEAKER_00"][1] >= 0.6


def test_cooccurrence_beats_positional_matching():
    # NER order: ["নিকোল", "নাভিদ"]. SPEAKER_01 mentions "নাভিদ" constantly;
    # positional matching would wrongly give SPEAKER_00's slot ordering.
    ordered_names = ["নিকোল", "নাভিদ"]
    trans = [
        _seg(0, 20, "SPEAKER_00", "স্বাগতম সবাইকে"),
        _seg(30, 55, "SPEAKER_01", "নাভিদ বলছেন নাভিদের মতো নাভিদ"),
        _seg(62, 80, "SPEAKER_02", "রাজনীতির আলোচনা চলছে"),
    ]
    resolved = GatingFusion(ordered_names=ordered_names).resolve_identities(
        DIA, trans, [])
    assert resolved["SPEAKER_01"][0] == "নাভিদ"


def test_registry_face_wins_over_text():
    face = FaceOccurrence(frame_time=40.0, box=(0, 0, 10, 10), track_id=0,
                          resolved_face_id="Nahid", face_confidence=0.82)
    trans = [_seg(35, 55, "SPEAKER_01", "কিছু কথা")]
    resolved = GatingFusion(ordered_names=["Wrong"]).resolve_identities(
        DIA, trans, [face])
    assert resolved["SPEAKER_01"] == ("Nahid", 0.82)


def test_ground_truth_labels_override_everything():
    trans = [_seg(5, 25, "SPEAKER_00", "আমি কেউ নই")]
    resolved = GatingFusion().resolve_identities(
        DIA, trans, [], ground_truth_labels={"SPEAKER_02": "সাংবাদিক"})
    assert resolved["SPEAKER_02"] == ("সাংবাদিক", 1.0)


def test_unresolved_speakers_get_generic_labels():
    trans = [_seg(5, 25, "SPEAKER_00", "সাধারণ কথা")]
    resolved = GatingFusion().resolve_identities(DIA, trans, [])
    names = [v[0] for v in resolved.values()]
    assert len(set(names)) == 3                      # unique labels
    assert any(n.startswith("Speaker_") for n in names)


# ── Face voting: margin + presence (RC-1 regression) ───────────────────
#
# The old rule took the single highest-similarity face across ALL of a
# speaker's frames. These tests pin the replacement: each face votes, a name
# needs a clear top-1 lead AND enough presence, and the confidence is the
# mean similarity rather than a copied constant.

def _face(t, name, conf, runner=0.0):
    return FaceOccurrence(frame_time=t, box=(0, 0, 10, 10), track_id=0,
                          resolved_face_id=name, face_confidence=conf,
                          runner_up_confidence=runner)


def test_ambiguous_face_is_not_named():
    # top-1 leads the runner-up by 0.02, below FACE_SIM_MARGIN (0.05):
    # two enrolled people look alike, so no name may be asserted.
    f = _face(40.0, "Nahid", 0.80, runner=0.78)
    trans = [_seg(35, 55, "SPEAKER_01", "কিছু কথা")]
    resolved = GatingFusion().resolve_identities(DIA, trans, [f])
    assert resolved["SPEAKER_01"][0] != "Nahid"


def test_low_presence_identity_is_rejected():
    # Only 1 of 4 faces in the turn is the registry match (25% < 40%):
    # a fleeting appearance must not name the whole speaker.
    faces = [_face(31.0, "Nahid", 0.85)]
    faces += [_face(35.0 + i, "UNKNOWN", 0.0) for i in range(3)]
    trans = [_seg(35, 55, "SPEAKER_01", "কিছু কথা")]
    resolved = GatingFusion().resolve_identities(DIA, trans, faces)
    assert resolved["SPEAKER_01"][0] != "Nahid"


def test_dominant_identity_names_the_speaker_with_mean_confidence():
    # 3 of 4 faces are Nahid -> presence 0.75 >= 0.4; confidence is the mean
    # of 0.80, 0.82, 0.84, not a copied constant.
    faces = [_face(32.0 + i, "Nahid", 0.80 + i * 0.02) for i in range(3)]
    faces.append(_face(50.0, "UNKNOWN", 0.0))
    trans = [_seg(35, 55, "SPEAKER_01", "কিছু কথা")]
    resolved = GatingFusion().resolve_identities(DIA, trans, faces)
    assert resolved["SPEAKER_01"][0] == "Nahid"
    assert resolved["SPEAKER_01"][1] == 0.82
