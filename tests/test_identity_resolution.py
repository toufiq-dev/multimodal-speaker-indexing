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
