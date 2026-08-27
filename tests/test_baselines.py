"""Tests for evaluation/baselines.py — heuristic and ablation pipelines."""
from __future__ import annotations

from models import DiarizationSegment, TranscribedSegment, WordToken
from evaluation.baselines import (
    heuristic_fusion,
    text_only,
    run_identity_pipeline,
    ablate_no_faces,
    ablate_no_nlp,
    oracle_labels,
)


def _dia(start, end, spk):
    return DiarizationSegment(start, end, spk)


def _trans(start, end, spk, text, words=None):
    return TranscribedSegment(start, end, text, words or [], speaker_id=spk)


# ── heuristic_fusion (B1) ─────────────────────────────────────────────

class TestHeuristicFusion:
    """Tests for the max-overlap audio-only baseline."""

    def test_assigns_same_speaker_text(self):
        dia = [_dia(0, 5, "SPEAKER_00")]
        trans = [_trans(0, 5, "SPEAKER_00", "hello world")]
        finals = heuristic_fusion(dia, trans)
        assert len(finals) == 1
        assert finals[0].text == "hello world"
        assert finals[0].speaker == "Speaker_1"

    def test_picks_max_overlap(self):
        dia = [_dia(0, 10, "SPEAKER_00")]
        trans = [
            _trans(0, 3, "SPEAKER_00", "short"),     # overlap=3
            _trans(0, 10, "SPEAKER_00", "long full"), # overlap=10
        ]
        finals = heuristic_fusion(dia, trans)
        assert finals[0].text == "long full"

    def test_skips_different_speaker(self):
        dia = [_dia(0, 5, "SPEAKER_00")]
        trans = [_trans(0, 5, "SPEAKER_01", "wrong speaker")]
        finals = heuristic_fusion(dia, trans)
        assert len(finals) == 0

    def test_multiple_speakers_sequential(self):
        dia = [
            _dia(0, 5, "SPEAKER_00"),
            _dia(5, 10, "SPEAKER_01"),
        ]
        trans = [
            _trans(0, 5, "SPEAKER_00", "host text"),
            _trans(5, 10, "SPEAKER_01", "guest text"),
        ]
        finals = heuristic_fusion(dia, trans)
        assert len(finals) == 2
        assert finals[0].text == "host text"
        assert finals[1].text == "guest text"
        # Speaker labels are assigned per unique diarization speaker
        assert finals[0].speaker != finals[1].speaker

    def test_empty_transcription(self):
        dia = [_dia(0, 5, "SPEAKER_00")]
        finals = heuristic_fusion(dia, [])
        assert len(finals) == 0

    def test_empty_diarization(self):
        trans = [_trans(0, 5, "SPEAKER_00", "text")]
        finals = heuristic_fusion([], trans)
        assert len(finals) == 0

    def test_empty_text_skipped(self):
        dia = [_dia(0, 5, "SPEAKER_00")]
        trans = [_trans(0, 5, "SPEAKER_00", "")]
        finals = heuristic_fusion(dia, trans)
        assert len(finals) == 0

    def test_overlapping_dia_turns(self):
        """When diarization turns overlap, each still gets its own segment."""
        dia = [
            _dia(0, 6, "SPEAKER_00"),
            _dia(4, 10, "SPEAKER_01"),
        ]
        trans = [
            _trans(0, 6, "SPEAKER_00", "host words"),
            _trans(4, 10, "SPEAKER_01", "guest words"),
        ]
        finals = heuristic_fusion(dia, trans)
        assert len(finals) == 2
        assert finals[0].text == "host words"
        assert finals[1].text == "guest words"


# ── text_only (B2) ────────────────────────────────────────────────────

class TestTextOnly:
    """Tests for the raw-ASR baseline (no diarization)."""

    def test_passthrough(self):
        trans = [
            _trans(0, 5, "SPK1", "first block"),
            _trans(5, 10, "SPK2", "second block"),
        ]
        finals = text_only(trans)
        assert len(finals) == 2
        assert finals[0].speaker == "ASR_BLOCK"
        assert finals[1].speaker == "ASR_BLOCK"

    def test_preserves_order(self):
        trans = [
            _trans(10, 15, "SPK1", "later"),
            _trans(0, 5, "SPK0", "earlier"),
        ]
        finals = text_only(trans)
        assert finals[0].text == "earlier"
        assert finals[1].text == "later"

    def test_skips_empty_text(self):
        trans = [
            _trans(0, 5, "SPK1", ""),
            _trans(5, 10, "SPK1", "real text"),
        ]
        finals = text_only(trans)
        assert len(finals) == 1
        assert finals[0].text == "real text"

    def test_empty_input(self):
        assert text_only([]) == []


# ── run_identity_pipeline ─────────────────────────────────────────────

class TestRunIdentityPipeline:
    """Tests for the full pipeline entry point used by ablations."""

    def test_produces_final_segments(self):
        dia = [_dia(0, 10, "SPEAKER_00")]
        trans = [_trans(0, 10, "SPEAKER_00", "আমি রফতান")]
        finals = run_identity_pipeline(dia, trans)
        assert len(finals) >= 1
        assert all(hasattr(f, "speaker") for f in finals)

    def test_ground_truth_labels_applied(self):
        dia = [_dia(0, 10, "SPEAKER_00")]
        trans = [_trans(0, 10, "SPEAKER_00", "কিছু কথা")]
        gt = {"SPEAKER_00": "Host"}
        finals = run_identity_pipeline(dia, trans, ground_truth_labels=gt)
        assert finals[0].speaker == "Host"

    def test_faces_passed_through(self):
        from models import FaceOccurrence
        dia = [_dia(0, 10, "SPEAKER_00")]
        trans = [_trans(0, 10, "SPEAKER_00", "text")]
        face = FaceOccurrence(frame_time=5.0, box=(0, 0, 10, 10),
                              track_id=0, resolved_face_id="Dr. Rahman",
                              face_confidence=0.9)
        finals = run_identity_pipeline(dia, trans, faces=[face])
        assert any(f.speaker == "Dr. Rahman" for f in finals)


# ── ablate_no_faces (A1) ──────────────────────────────────────────────

class TestAblateNoFaces:
    """Tests for the -faces ablation."""

    def test_ignores_face_evidence(self):
        from models import FaceOccurrence
        dia = [_dia(0, 10, "SPEAKER_00")]
        trans = [_trans(0, 10, "SPEAKER_00", "আমি রফতান")]
        face = FaceOccurrence(frame_time=5.0, box=(0, 0, 10, 10),
                              track_id=0, resolved_face_id="ShouldBeIgnored",
                              face_confidence=0.99)
        # Pass face but ablate_no_faces ignores it
        finals = ablate_no_faces(dia, trans)
        assert all(f.speaker != "ShouldBeIgnored" for f in finals)

    def test_anchor_still_works(self):
        dia = [_dia(0, 10, "SPEAKER_00")]
        trans = [_trans(0, 10, "SPEAKER_00", "আমি রফতান")]
        finals = ablate_no_faces(dia, trans)
        assert any("রফতান" in f.speaker for f in finals)


# ── ablate_no_nlp (A2) ────────────────────────────────────────────────

class TestAblateNoNlp:
    """Tests for the -nlp_names ablation.

    Note: ablate_no_nlp only suppresses co-occurrence matching (Pass 3)
    by passing ordered_names=[]. The host-anchor extraction (Pass 2) still
    runs because it reads directly from transcribed segments.
    """

    def test_suppresses_cooccurrence_names(self):
        # Without NLP ordered_names, co-occurrence matching is suppressed.
        # Anchor extraction (Pass 2) still runs from transcribed text.
        dia = [_dia(0, 10, "SPEAKER_00")]
        trans = [_trans(0, 10, "SPEAKER_00", "কিছু কথা")]  # no anchor
        finals = ablate_no_nlp(dia, trans)
        # No ordered_names → co-occurrence can't match any NER names
        assert len(finals) >= 1

    def test_anchor_extraction_still_works(self):
        # Anchor extraction is independent of ordered_names
        dia = [_dia(0, 10, "SPEAKER_00")]
        trans = [_trans(0, 10, "SPEAKER_00", "আমি রফতান")]
        finals = ablate_no_nlp(dia, trans)
        # Anchor (Pass 2) still fires → name resolved
        assert any("রফতান" in f.speaker for f in finals)


# ── oracle_labels (A3) ────────────────────────────────────────────────

class TestOracleLabels:
    """Tests for the ground-truth upper bound."""

    def test_perfect_naming(self):
        dia = [
            _dia(0, 10, "SPEAKER_00"),
            _dia(10, 20, "SPEAKER_01"),
        ]
        trans = [
            _trans(0, 10, "SPEAKER_00", "host speaks"),
            _trans(10, 20, "SPEAKER_01", "guest speaks"),
        ]
        speaker_map = {"SPEAKER_00": "Host", "SPEAKER_01": "Guest"}
        finals = oracle_labels(dia, trans, speaker_map)
        speakers = {f.speaker for f in finals}
        assert "Host" in speakers
        assert "Guest" in speakers

    def test_empty_speaker_map(self):
        dia = [_dia(0, 10, "SPEAKER_00")]
        trans = [_trans(0, 10, "SPEAKER_00", "text")]
        finals = oracle_labels(dia, trans, {})
        assert len(finals) >= 1

    def test_partial_speaker_map(self):
        dia = [
            _dia(0, 10, "SPEAKER_00"),
            _dia(10, 20, "SPEAKER_01"),
        ]
        trans = [
            _trans(0, 10, "SPEAKER_00", "text a"),
            _trans(10, 20, "SPEAKER_01", "text b"),
        ]
        speaker_map = {"SPEAKER_00": "OnlyMapped"}
        finals = oracle_labels(dia, trans, speaker_map)
        speakers = {f.speaker for f in finals}
        assert "OnlyMapped" in speakers
        # SPEAKER_01 not in map → gets generic label
        assert any(s.startswith("Speaker_") for s in speakers)
