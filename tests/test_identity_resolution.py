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
    # An anchor is trusted only when corroborated by a registry identity or an
    # NER name: two runs showed unvalidated 'আমি <clause>' captures becoming
    # speaker labels. Here the NER list supplies the corroboration.
    trans = [
        _seg(0, 20, "SPEAKER_00",
             "আপনাদের সাথে আছে আমি রফতান আঞ্জুমান নিকোল"),
        _seg(30, 50, "SPEAKER_01", "আমি মনে করি এটা ঠিক নয়"),
    ]
    resolved = GatingFusion(
        ordered_names=["রফতান আঞ্জুমান নিকোল"]).resolve_identities(DIA, trans, [])
    assert resolved["SPEAKER_00"][0] == "রফতান আঞ্জুমান নিকোল"
    assert resolved["SPEAKER_00"][1] >= 0.6


def test_uncorroborated_anchor_is_not_used():
    """Even a real-looking name is refused without corroboration."""
    trans = [_seg(0, 20, "SPEAKER_00", "আমি রফতান আঞ্জুমান নিকোল")]
    resolved = GatingFusion().resolve_identities(DIA, trans, [])
    assert resolved["SPEAKER_00"][0].startswith("Speaker_")


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


def test_ambiguous_face_is_not_named(monkeypatch):
    # top-1 leads the runner-up by 0.02, below the margin: two enrolled people
    # look alike, so no name may be asserted. The gates ship disabled (they
    # must be calibrated from fusion_diagnostics.json), hence enabling here.
    from config import config
    monkeypatch.setattr(config, "FACE_SIM_MARGIN", 0.05)

    f = _face(40.0, "Nahid", 0.80, runner=0.78)
    trans = [_seg(35, 55, "SPEAKER_01", "কিছু কথা")]
    resolved = GatingFusion().resolve_identities(DIA, trans, [f])
    assert resolved["SPEAKER_01"][0] != "Nahid"


def test_low_presence_identity_is_rejected(monkeypatch):
    # Only 1 of 4 faces in the turn is the registry match (25% < 40%):
    # a fleeting appearance must not name the whole speaker.
    from config import config
    monkeypatch.setattr(config, "FACE_MIN_FRAME_FRACTION", 0.4)

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


def test_uncorroborated_anchor_is_not_a_speaker_label():
    """An unvalidated 'আমি <clause>' must never become a speaker name.

    Regression: the v2 RTV run labelled speakers 'গিয়েছি তদ্বির করতে',
    'বলে দিচ্ছি' and 'যখন' because the greedy anchor capture was emitted
    without any check that it was a person's name.
    """
    trans = [
        _seg(35, 55, "SPEAKER_01", "আমি গিয়েছি তদ্বির করতে"),
        _seg(62, 80, "SPEAKER_02", "আমি বলে দিচ্ছি"),
    ]
    resolved = GatingFusion().resolve_identities(DIA, trans, [])
    names = [v[0] for v in resolved.values()]
    for bad in ("গিয়েছি তদ্বির করতে", "বলে দিচ্ছি", "যখন", "তদ্বির", "আর"):
        assert bad not in names, f"{bad!r} leaked into the speaker labels"
    assert any(n.startswith("Speaker_") for n in names)


def test_registry_diagnostics_reports_gaps_and_presence():
    """The calibration data must expose margin and presence per speaker."""
    faces = [_face(32.0 + i, "Nahid", 0.80 + i * 0.01, runner=0.70)
             for i in range(2)]
    faces.append(_face(50.0, "UNKNOWN", 0.0))

    fusion = GatingFusion()
    speaker_faces = fusion._aggregate_faces_per_speaker(DIA, faces)
    diag = fusion.registry_face_diagnostics(speaker_faces)

    d = diag["SPEAKER_01"]
    assert d["n_faces"] == 3
    assert d["votes"] == {"Nahid": 2}
    assert d["winner"] == "Nahid"
    assert d["winner_presence"] == round(2 / 3, 3)
    assert d["best_margin"] == round(0.81 - 0.70, 3)


# ── Per-turn identity from the speaking face track (Fixes 1 & 3) ───────
#
# The decisive failure: a still, well-framed listener matched the registry
# confidently across many frames, so the pooled vote named the speaker's turns
# after the non-speaker. Identity is now decided per TURN, from the track whose
# mouth is moving.

def _trackface(t, track, motion, name="UNKNOWN", conf=0.0):
    return FaceOccurrence(frame_time=t, box=(0, 0, 10, 10), track_id=0,
                          resolved_face_id=name, face_confidence=conf,
                          face_track_id=track, mouth_motion=motion)


def _one_turn():
    return [DiarizationSegment(0.0, 30.0, "SPEAKER_00")]


def test_turn_uses_the_speaking_track_not_the_best_matching_listener():
    dia = _one_turn()
    faces = []
    # A listener: 20 still frames, confidently matched to Shahriar.
    for i in range(20):
        faces.append(_trackface(i * 0.5, track=1, motion=0.0,
                                name="Shahriar", conf=0.78))
    # The speaker: fewer frames, weaker similarity, but the mouth is moving.
    for i in range(10):
        faces.append(_trackface(i * 0.5, track=2, motion=0.30,
                                name="Tushar", conf=0.60))

    fusion = GatingFusion()
    resolved = fusion.resolve_identities(dia, [], faces)
    finals = fusion.create_final_segments(
        dia, [_seg(0, 30, "SPEAKER_00", "কিছু কথা")], resolved)

    assert finals[0].speaker == "Tushar"
    assert finals[0].confidence == 0.6


def test_without_track_information_the_speaker_label_is_used():
    # Faces produced by callers that do not track must keep old behaviour.
    dia = _one_turn()
    faces = [_face(1.0, "Shahriar", 0.80)]
    fusion = GatingFusion()
    resolved = fusion.resolve_identities(dia, [], faces)

    assert fusion.turn_identities == {}
    finals = fusion.create_final_segments(
        dia, [_seg(0, 30, "SPEAKER_00", "কিছু কথা")], resolved)
    assert finals[0].speaker == "Shahriar"


def test_speaking_track_without_a_registry_identity_falls_back():
    # The moving face has no name (e.g. never matched): the turn keeps the
    # per-speaker fallback rather than inventing a label.
    dia = _one_turn()
    faces = [_trackface(1.0, track=2, motion=0.30, name="UNKNOWN"),
             _face(1.5, "Shahriar", 0.80)]
    fusion = GatingFusion()
    resolved = fusion.resolve_identities(dia, [], faces)
    finals = fusion.create_final_segments(
        dia, [_seg(0, 30, "SPEAKER_00", "কিছু কথা")], resolved)
    assert finals[0].speaker == "Shahriar"
