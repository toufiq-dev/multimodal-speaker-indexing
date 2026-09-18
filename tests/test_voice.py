"""Tests for the voice-reference registry (engines/voice.py).

Everything here runs without pyannote, without a GPU and without network: the
embedding model is the only part of the feature that needs any of those, and it
is kept behind :class:`VoiceEmbedder`. What is tested is the part that decides
identity -- vector reduction, windowing, gating, the fusion policy, and the
diagnostics a threshold is chosen from -- because that is the part that can be
wrong silently.

The constraint these tests exist to protect is C1-adjacent: adding a second
biometric must not be able to change an existing run's output unless it is
explicitly switched on, and when it is on, ``fallback`` must be strictly
additive.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from config import config
from engines.fusion import GatingFusion
from engines.voice import (ClusterVoice, VoiceMatch, Voiceprint,
                           accepted_matches, cosine, enrolled_cross_matrix,
                           load_voiceprints, l2_normalize, match_clusters,
                           name_from_stem, registry_audio_files,
                           robust_centroid, save_voiceprints, subsample,
                           threshold_sweep, voice_diagnostics, window_spans)
from models import DiarizationSegment, FaceOccurrence, TranscribedSegment

DIA = [
    DiarizationSegment(0.0, 30.0, "SPEAKER_00"),    # no face evidence
    DiarizationSegment(30.0, 60.0, "SPEAKER_01"),   # face evidence
    DiarizationSegment(60.0, 90.0, "SPEAKER_02"),
]


def _unit(*values) -> np.ndarray:
    return l2_normalize(np.array(values, dtype=np.float32))


def _cluster(speaker, vector, **kw) -> ClusterVoice:
    return ClusterVoice(speaker_id=speaker, embedding=_unit(*vector), **kw)


def _vp(name, vector, **kw) -> Voiceprint:
    return Voiceprint(name=name, embedding=_unit(*vector), **kw)


# ---------------------------------------------------------------------------
# Vector primitives
# ---------------------------------------------------------------------------

def test_cosine_is_scale_invariant_and_bounded():
    a, b = _unit(1, 0, 0), _unit(2, 0, 0)
    assert cosine(a, b) == pytest.approx(1.0)
    assert cosine(_unit(1, 0, 0), _unit(0, 1, 0)) == pytest.approx(0.0)
    assert cosine(_unit(1, 0, 0), _unit(-1, 0, 0)) == pytest.approx(-1.0)
    # A zero vector must not produce NaN: it is the shape a failed embedding
    # takes, and NaN would silently poison every downstream comparison.
    assert np.isfinite(cosine(np.zeros(3, dtype=np.float32), _unit(1, 0, 0)))


def test_robust_centroid_discards_an_outlier():
    """The enrolment guarantee: one other voice in a reference clip must not
    drag the voiceprint."""
    target = _unit(1.0, 0.0, 0.0)
    embeddings = np.stack([target] * 6 + [_unit(0.0, 1.0, 0.0)])
    centroid, coherence, kept = robust_centroid(embeddings)

    assert cosine(centroid, target) > 0.99
    assert kept.sum() == 6 and not kept[-1]
    assert coherence > 0.99


def test_robust_centroid_keeps_a_single_embedding():
    """A one-window voiceprint is reported, not silently dropped: the caller
    sees n_kept=1 and can decide for itself."""
    centroid, coherence, kept = robust_centroid(np.stack([_unit(1, 0, 0)]))
    assert kept.sum() == 1 and coherence == pytest.approx(1.0)


def test_robust_centroid_rejects_an_empty_set():
    with pytest.raises(ValueError):
        robust_centroid(np.zeros((0, 3), dtype=np.float32))


def test_robust_centroid_never_trims_below_min_keep():
    embeddings = np.stack([_unit(*v) for v in
                           [(1, 0), (0.9, 0.1), (0.8, 0.2), (0, 1)]])
    _, _, kept = robust_centroid(embeddings, trim=0.9, min_keep=3)
    assert kept.sum() >= 3


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def test_window_spans_never_straddle_a_region_boundary():
    """A window spanning two turns would embed two speakers into one cluster
    voiceprint, which is the exact contamination the feature exists to avoid."""
    spans = window_spans(20.0, window=3.0, hop=1.5, regions=[(10.0, 16.0)])
    assert spans, "expected windows inside the region"
    for start, end in spans:
        assert start >= 10.0 - 1e-6 and end <= 16.0 + 1e-6


def test_window_spans_returns_a_short_region_whole():
    spans = window_spans(20.0, window=3.0, hop=1.5, regions=[(10.0, 12.0)])
    assert spans == [(10.0, 12.0)]


def test_window_spans_skips_regions_below_the_floor():
    assert window_spans(20.0, 3.0, 1.5, regions=[(10.0, 10.2)]) == []


def test_window_spans_covers_the_tail():
    """A hop that does not divide the region must not lose the last speech."""
    spans = window_spans(20.0, window=3.0, hop=2.0, regions=[(0.0, 8.0)])
    assert spans[-1][1] == pytest.approx(8.0)


def test_subsample_is_evenly_spaced_and_deterministic():
    items = list(range(100))
    picked = subsample(items, 5)
    assert picked == subsample(items, 5)
    assert len(picked) == 5
    assert picked[0] == 0 and picked[-1] == 99
    assert subsample(items[:3], 5) == items[:3]


# ---------------------------------------------------------------------------
# Registry scan / naming
# ---------------------------------------------------------------------------

def test_registry_audio_files_and_name_matching(tmp_path):
    (tmp_path / "Dr_Abdul_Noor_Tushar.mp3").write_bytes(b"\x00")
    (tmp_path / "MA_Aziz.wav").write_bytes(b"\x00")
    (tmp_path / "ignored.txt").write_text("x")
    (tmp_path / "photo.jpg").write_bytes(b"\x00")

    files = registry_audio_files(tmp_path)
    assert [p.name for p in files] == ["Dr_Abdul_Noor_Tushar.mp3", "MA_Aziz.wav"]
    # The naming rule must be identical to the face registry's, or the two
    # biometrics would key the same person under different strings.
    assert name_from_stem("Dr_Abdul_Noor_Tushar") == "Dr Abdul Noor Tushar"


def test_registry_audio_files_on_a_missing_directory(tmp_path):
    assert registry_audio_files(tmp_path / "nope") == []


# ---------------------------------------------------------------------------
# Gating
# ---------------------------------------------------------------------------

ENROLLED = {
    "Tushar": _vp("Tushar", [1.0, 0.0, 0.0]),
    "Shahriar": _vp("Shahriar", [0.0, 1.0, 0.0]),
    "Absent": _vp("Absent", [0.0, 0.0, 1.0]),
}


def test_match_accepts_a_clear_winner():
    clusters = {"SPEAKER_00": _cluster("SPEAKER_00", [0.95, 0.05, 0.0])}
    match = match_clusters(clusters, ENROLLED, threshold=0.45, margin=0.05)["SPEAKER_00"]
    assert match.accepted and match.name == "Tushar"
    assert match.margin > 0.05


def test_match_rejects_a_cluster_equidistant_from_two_people():
    """Two panellists on one studio microphone score alike; the gate must
    abstain rather than pick a name."""
    clusters = {"SPEAKER_00": _cluster("SPEAKER_00", [0.7, 0.7, 0.0])}
    match = match_clusters(clusters, ENROLLED, threshold=0.45, margin=0.10)["SPEAKER_00"]
    assert not match.accepted and match.reason == "ambiguous_margin"


def test_match_rejects_below_threshold():
    clusters = {"SPEAKER_00": _cluster("SPEAKER_00", [0.1, -0.9, 0.1])}
    match = match_clusters(clusters, ENROLLED, threshold=0.60, margin=0.05)["SPEAKER_00"]
    assert not match.accepted and match.reason == "below_threshold"


def test_match_without_a_cluster_voiceprint_abstains():
    clusters = {"SPEAKER_00": ClusterVoice(speaker_id="SPEAKER_00")}
    match = match_clusters(clusters, ENROLLED)["SPEAKER_00"]
    assert not match.accepted and match.reason == "no_voiceprint"


def test_match_with_an_empty_registry_abstains():
    clusters = {"SPEAKER_00": _cluster("SPEAKER_00", [1.0, 0.0, 0.0])}
    match = match_clusters(clusters, {})["SPEAKER_00"]
    assert not match.accepted and match.reason == "no_enrolled_voiceprints"


def test_low_coherence_gate_is_optional_but_effective():
    clusters = {"SPEAKER_00": _cluster("SPEAKER_00", [1.0, 0.0, 0.0],
                                       coherence=0.2)}
    assert match_clusters(clusters, ENROLLED, threshold=0.45, margin=0.05,
                          min_coherence=0.0)["SPEAKER_00"].accepted
    gated = match_clusters(clusters, ENROLLED, threshold=0.45, margin=0.05,
                           min_coherence=0.5)["SPEAKER_00"]
    assert not gated.accepted and gated.reason == "low_coherence"


def test_accepted_matches_shape_matches_the_fusion_contract():
    clusters = {
        "SPEAKER_00": _cluster("SPEAKER_00", [0.95, 0.05, 0.0]),
        "SPEAKER_01": _cluster("SPEAKER_01", [0.1, -0.9, 0.1]),  # below floor
    }
    matches = match_clusters(clusters, ENROLLED, threshold=0.45, margin=0.05)
    mapped = accepted_matches(matches)
    assert mapped["SPEAKER_00"][0] == "Tushar"
    assert mapped["SPEAKER_00"][1] == pytest.approx(
        cosine(clusters["SPEAKER_00"].embedding, ENROLLED["Tushar"].embedding),
        abs=1e-3)
    assert "SPEAKER_01" not in mapped
    assert isinstance(matches["SPEAKER_00"], VoiceMatch)


def test_enrolled_cross_matrix_excludes_self_and_is_symmetric():
    cross = enrolled_cross_matrix(ENROLLED)
    assert "Tushar" not in cross["Tushar"]
    assert cross["Tushar"]["Shahriar"] == pytest.approx(cross["Shahriar"]["Tushar"])


# ---------------------------------------------------------------------------
# Fusion policy
# ---------------------------------------------------------------------------

def _face(name, conf=0.75, runner=0.10):
    return FaceOccurrence(
        frame_time=40.0, box=(0, 0, 10, 10), track_id=0,
        resolved_face_id=name, face_confidence=conf,
        runner_up_face_id="Other", runner_up_confidence=runner)


def _trans():
    return [TranscribedSegment(start=5.0, end=25.0, text="কিছু কথা",
                               words=[], speaker_id="SPEAKER_00")]


def test_policy_off_ignores_injected_voice_matches():
    """The default must be provably inert: same output with and without
    voice evidence present."""
    baseline = GatingFusion().resolve_identities(DIA, _trans(), [])
    with_voice = GatingFusion(
        voice_matches={"SPEAKER_00": ("Tushar", 0.95)},
        voice_policy="off",
    ).resolve_identities(DIA, _trans(), [])
    assert baseline == with_voice


def test_fallback_names_a_speaker_the_face_pass_left_unnamed():
    fusion = GatingFusion(voice_matches={"SPEAKER_00": ("Tushar", 0.88)},
                          voice_policy="fallback")
    resolved = fusion.resolve_identities(DIA, _trans(), [])
    assert resolved["SPEAKER_00"] == ("Tushar", 0.88)
    assert fusion.last_voice_decisions["SPEAKER_00"]["action"] == "filled"


def test_fallback_cannot_overwrite_a_face_match():
    """The additive guarantee: the face pass is authoritative under fallback."""
    fusion = GatingFusion(voice_matches={"SPEAKER_01": ("Tushar", 0.99)},
                          voice_policy="fallback")
    resolved = fusion.resolve_identities(DIA, _trans(), [_face("Shahriar")])
    assert resolved["SPEAKER_01"][0] == "Shahriar"
    assert fusion.last_voice_decisions["SPEAKER_01"]["action"] == "kept_face"


def test_override_replaces_a_face_match_when_materially_stronger():
    fusion = GatingFusion(voice_matches={"SPEAKER_01": ("Tushar", 0.92)},
                          voice_policy="override")
    resolved = fusion.resolve_identities(DIA, _trans(), [_face("Shahriar", conf=0.75)])
    assert resolved["SPEAKER_01"] == ("Tushar", 0.92)
    assert fusion.last_voice_decisions["SPEAKER_01"]["action"] == "overrode_face"


def test_override_keeps_the_face_match_when_the_gain_is_small():
    """Two comparable sources must not fight over one cluster."""
    fusion = GatingFusion(voice_matches={"SPEAKER_01": ("Tushar", 0.80)},
                          voice_policy="override")
    resolved = fusion.resolve_identities(DIA, _trans(), [_face("Shahriar", conf=0.78)])
    assert resolved["SPEAKER_01"][0] == "Shahriar"
    assert fusion.last_voice_decisions["SPEAKER_01"]["action"] == "kept_face"


def test_voice_agreement_with_face_is_recorded_as_agreement():
    fusion = GatingFusion(voice_matches={"SPEAKER_01": ("Shahriar", 0.91)},
                          voice_policy="override")
    resolved = fusion.resolve_identities(DIA, _trans(), [_face("Shahriar")])
    assert resolved["SPEAKER_01"][0] == "Shahriar"
    assert fusion.last_voice_decisions["SPEAKER_01"]["action"] == "agreed_with_face"


def test_voice_never_reuses_a_name_already_taken():
    """One person cannot be two diarization clusters."""
    matches = {"SPEAKER_00": ("Tushar", 0.90), "SPEAKER_02": ("Tushar", 0.90)}
    fusion = GatingFusion(voice_matches=matches, voice_policy="fallback")
    resolved = fusion.resolve_identities(DIA, _trans(), [])
    names = [resolved[s][0] for s in ("SPEAKER_00", "SPEAKER_02")]
    assert names.count("Tushar") == 1
    assert fusion.last_voice_decisions["SPEAKER_02"]["action"] == "skipped_name_taken"


def test_voice_evidence_reaches_the_final_segments():
    from engines.fusion import run_fusion_pipeline
    segments = run_fusion_pipeline(
        DIA, _trans(), [], ordered_names=[],
        voice_matches={"SPEAKER_00": ("Tushar", 0.9)}, voice_policy="fallback")
    assert [s.speaker for s in segments] == ["Tushar"]


# ---------------------------------------------------------------------------
# Diagnostics + calibration
# ---------------------------------------------------------------------------

def test_voice_diagnostics_reports_every_vector_and_the_impostor_matrix():
    clusters = {
        "SPEAKER_00": _cluster("SPEAKER_00", [0.95, 0.05, 0.0], n_windows=12,
                               n_kept=9, coherence=0.9),
        "SPEAKER_01": ClusterVoice(speaker_id="SPEAKER_01"),
    }
    matches = match_clusters(clusters, ENROLLED, threshold=0.45, margin=0.05)
    diag = voice_diagnostics(clusters, ENROLLED, matches, embedder=None,
                             policy="fallback")

    assert diag["summary"]["n_accepted"] == 1
    assert diag["summary"]["accepted"] == {"SPEAKER_00": "Tushar"}
    assert diag["summary"]["clusters_without_voiceprint"] == ["SPEAKER_01"]
    # The chosen name must come with the full vector it was chosen from, or the
    # threshold cannot be re-judged after the run.
    node = diag["clusters"]["SPEAKER_00"]
    assert node["name"] == "Tushar" and len(node["scores"]) == len(ENROLLED)
    assert "enrolled_cross" in diag and "Absent" in diag["enrolled"]
    json.dumps(diag)  # must be JSON-serialisable: it is written to disk


def test_threshold_sweep_scores_against_truth():
    clusters = {
        "SPEAKER_00": _cluster("SPEAKER_00", [0.95, 0.05, 0.0]),   # Tushar
        "SPEAKER_01": _cluster("SPEAKER_01", [0.05, 0.95, 0.0]),   # Shahriar
        "SPEAKER_02": _cluster("SPEAKER_02", [0.7, 0.7, 0.0]),     # ambiguous
    }
    truth = {"SPEAKER_00": "Tushar", "SPEAKER_01": "Shahriar"}
    rows = threshold_sweep(clusters, ENROLLED, truth=truth,
                           thresholds=[0.45], margins=[0.05, 0.60])

    loose = next(r for r in rows if r["margin"] == 0.05)
    strict = next(r for r in rows if r["margin"] == 0.60)
    assert (loose["correct"], loose["wrong"]) == (2, 0)
    assert loose["accuracy"] == pytest.approx(1.0)
    # A margin wide enough to reject the genuinely ambiguous cluster leaves
    # both real matches intact and reports nothing wrongly named.
    assert strict["wrong"] == 0 and strict["correct"] == 2


def test_threshold_sweep_reports_a_wrong_name_as_wrong():
    """A sweep that names a cluster wrongly must say so: that is the row a
    reader uses to reject a threshold."""
    clusters = {"SPEAKER_00": _cluster("SPEAKER_00", [0.0, 0.0, 1.0])}
    rows = threshold_sweep(clusters, ENROLLED, truth={"SPEAKER_00": "Tushar"},
                           thresholds=[0.45], margins=[0.05])
    assert rows[0]["wrong"] == 1 and rows[0]["accuracy"] == 0.0
    assert rows[0]["wrong_detail"] == ["SPEAKER_00:Absent!=Tushar"]


def test_voiceprints_round_trip_through_disk(tmp_path):
    clusters = {"SPEAKER_00": _cluster("SPEAKER_00", [0.95, 0.05, 0.0],
                                       n_windows=10, n_kept=7, coherence=0.88)}
    path = tmp_path / "voiceprints.npz"
    save_voiceprints(path, clusters, ENROLLED)

    loaded_clusters, loaded_enrolled = load_voiceprints(path)
    assert set(loaded_enrolled) == set(ENROLLED)
    assert loaded_clusters["SPEAKER_00"].n_kept == 7
    assert cosine(loaded_clusters["SPEAKER_00"].embedding,
                  clusters["SPEAKER_00"].embedding) > 0.999
    # Re-scoring from disk must reproduce the live decision exactly.
    before = match_clusters(clusters, ENROLLED, threshold=0.45, margin=0.05)
    after = match_clusters(loaded_clusters, loaded_enrolled,
                           threshold=0.45, margin=0.05)
    assert after["SPEAKER_00"].name == before["SPEAKER_00"].name


def test_load_voiceprints_missing_file_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_voiceprints(tmp_path / "absent.npz")


# ---------------------------------------------------------------------------
# The pipeline entry point degrades safely
# ---------------------------------------------------------------------------

def test_build_voice_evidence_is_inert_when_policy_is_off(monkeypatch):
    from engines.voice import build_voice_evidence
    monkeypatch.setattr(config, "VOICE_POLICY", "off")
    evidence = build_voice_evidence(DIA, "unused.wav")
    assert evidence.matches == {} and not evidence.available
    assert evidence.note == "VOICE_POLICY=off"


def test_build_voice_evidence_without_references_does_not_load_a_model(
        monkeypatch, tmp_path):
    """A face-only registry is a valid configuration, not an error: the model
    must never be loaded (or downloaded) just to discover there is no audio."""
    from engines.voice import build_voice_evidence

    class ExplodingEmbedder:
        model_id = "never"

        def load(self):
            raise AssertionError("the embedding model must not be loaded here")

    monkeypatch.setattr(config, "VOICE_POLICY", "fallback")
    monkeypatch.setattr(config, "DATA_REGISTRY_DIR", tmp_path)
    evidence = build_voice_evidence(DIA, "unused.wav",
                                    embedder=ExplodingEmbedder())
    assert evidence.matches == {}
    assert evidence.note == "no audio references in the registry"


def test_build_voice_evidence_survives_a_failing_embedder(monkeypatch, tmp_path):
    """A gated model or a GPU OOM must cost the run its voice evidence, never
    the run itself."""
    from engines.voice import build_voice_evidence

    class FailingEmbedder:
        model_id = "boom"

        def load(self):
            raise RuntimeError("401 gated")

    (tmp_path / "Someone.mp3").write_bytes(b"\x00")
    monkeypatch.setattr(config, "VOICE_POLICY", "fallback")
    monkeypatch.setattr(config, "DATA_REGISTRY_DIR", tmp_path)
    evidence = build_voice_evidence(DIA, "unused.wav",
                                    embedder=FailingEmbedder())
    assert evidence.matches == {}
    assert "401 gated" in evidence.note


# ---------------------------------------------------------------------------
# Truth derivation for the calibration tool (scripts/voice_verify.py)
# ---------------------------------------------------------------------------

def _write_run(tmp_path, turns, entries):
    from scripts.voice_verify import derive_cluster_names

    dia = tmp_path / "diarization.json"
    txt = tmp_path / "transcript.json"
    dia.write_text(json.dumps(turns), encoding="utf-8")
    txt.write_text(json.dumps(entries), encoding="utf-8")
    return derive_cluster_names(dia, txt)


def test_truth_derivation_picks_the_dominant_annotated_name(tmp_path):
    truth, stats = _write_run(
        tmp_path,
        [{"start": 0.0, "end": 10.0, "speaker_id": "SPEAKER_00"}],
        [{"start": 0.0, "end": 7.0, "speaker": "A", "overlap": False},
         {"start": 7.0, "end": 10.0, "speaker": "B", "overlap": False}],
    )
    assert truth == {"SPEAKER_00": "A"}
    assert stats["SPEAKER_00"]["purity"] == pytest.approx(0.7)
    assert stats["SPEAKER_00"]["n_annotated_names"] == 2


def test_truth_purity_cannot_exceed_one_when_annotations_overlap(tmp_path):
    """The annotation marks simultaneous speech with separate overlapping
    entries, so a duration-denominated purity would read as more than perfect
    for exactly the turns this feature targets."""
    truth, stats = _write_run(
        tmp_path,
        [{"start": 0.0, "end": 10.0, "speaker_id": "SPEAKER_00"}],
        [{"start": 0.0, "end": 10.0, "speaker": "A", "overlap": False},
         {"start": 0.0, "end": 10.0, "speaker": "B", "overlap": True}],
    )
    assert truth["SPEAKER_00"] == "A"
    assert stats["SPEAKER_00"]["purity"] <= 1.0
    assert stats["SPEAKER_00"]["purity"] == pytest.approx(0.5)
    assert stats["SPEAKER_00"]["annotated_overlap_sec"] == pytest.approx(10.0)


def test_truth_derivation_ignores_a_cluster_with_no_annotation(tmp_path):
    truth, stats = _write_run(
        tmp_path,
        [{"start": 0.0, "end": 10.0, "speaker_id": "SPEAKER_00"},
         {"start": 100.0, "end": 110.0, "speaker_id": "SPEAKER_09"}],
        [{"start": 0.0, "end": 10.0, "speaker": "A", "overlap": False}],
    )
    assert truth == {"SPEAKER_00": "A"}
    assert "SPEAKER_09" not in stats
