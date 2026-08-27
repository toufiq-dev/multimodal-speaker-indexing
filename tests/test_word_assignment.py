"""Tests for the midpoint-containment word-assignment rule."""
from __future__ import annotations

from engines.transcription import _assign_word_to_turn
from models import DiarizationSegment


DIA = [
    DiarizationSegment(0.0, 5.0, "SPEAKER_00"),
    DiarizationSegment(5.2, 9.0, "SPEAKER_01"),
    DiarizationSegment(10.0, 14.0, "SPEAKER_00"),
]


def test_word_inside_turn_gets_that_speaker():
    assert _assign_word_to_turn(1.0, 1.3, DIA) == "SPEAKER_00"
    assert _assign_word_to_turn(6.0, 6.4, DIA) == "SPEAKER_01"


def test_word_midpoint_rule_not_iou():
    # A 0.3s word perfectly inside a 4s turn: IoU would be ~0.07 (tiny),
    # containment is exact. The old IoU code effectively chose arbitrarily.
    assert _assign_word_to_turn(7.0, 7.3, DIA) == "SPEAKER_01"


def test_word_in_gap_goes_to_nearest_turn():
    # 9.5 sits in the 9.0-10.0 gap; nearest boundary is SPEAKER_01's end (0.5)
    # vs SPEAKER_00's start at 10.0 (0.5) -> tie, min() picks first -> SPEAKER_01
    spk = _assign_word_to_turn(9.35, 9.65, DIA)
    assert spk in {"SPEAKER_00", "SPEAKER_01"}
    # Unambiguous side: 9.05-9.15 clearly nearest to SPEAKER_01 end.
    assert _assign_word_to_turn(9.05, 9.15, DIA) == "SPEAKER_01"


def test_empty_diarization_returns_unknown():
    assert _assign_word_to_turn(1.0, 2.0, []) == "UNKNOWN"


def test_overlapping_turns_pick_most_specific():
    overlap = [
        DiarizationSegment(0.0, 10.0, "SPEAKER_00"),
        DiarizationSegment(4.0, 6.0, "SPEAKER_02"),
    ]
    assert _assign_word_to_turn(4.8, 5.0, overlap) == "SPEAKER_02"
