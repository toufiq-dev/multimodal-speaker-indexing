"""Tests for evaluation metrics and dataset management."""
from __future__ import annotations

import json

import pytest

from models import DiarizationSegment, FinalSegment
from evaluation.metrics import (
    wer, cer, cpwer, der_jer, speaker_name_accuracy,
    fusion_health_metrics, assert_no_regression,
)
from evaluation.dataset import (
    load_manifest, parse_rttm, validate_registry, VideoEntry,
)


def test_wer_perfect_and_imperfect():
    assert wer("আমি ভাত খাই", "আমি ভাত খাই") == 0.0
    assert wer("আমি ভাত খাই", "আমি ভাত খায়") > 0.0


def test_cer_counts_characters():
    assert cer("abc", "abc") == 0.0
    assert cer("abcd", "abxd") == pytest.approx(0.25)


def test_cpwer_finds_min_permutation():
    ref = {"A": "one two three", "B": "four five"}
    hyp_swapped = {"X": "four five", "Y": "one two three"}
    best, perms = cpwer(ref, hyp_swapped)
    assert best == pytest.approx(0.0)     # swap resolves it
    assert perms >= 2


def test_der_jer_perfect_and_degenerate():
    ref = [DiarizationSegment(0, 10, "SPK1"), DiarizationSegment(10, 20, "SPK2")]
    assert der_jer(ref, ref)["DER"] == pytest.approx(0.0)
    hyp_all_wrong = [DiarizationSegment(0, 20, "OTHER")]
    assert der_jer(ref, hyp_all_wrong)["DER"] > 0.3


def test_speaker_name_accuracy():
    ref = [DiarizationSegment(0, 10, "Host"), DiarizationSegment(10, 20, "Guest")]
    finals = [
        FinalSegment(0, 10, "Host", "x"),
        FinalSegment(10, 20, "WRONG", "y"),
    ]
    assert speaker_name_accuracy(ref, finals) == pytest.approx(0.5)


def test_health_assertions_catch_original_failure():
    # Reproduce the failed run: 32 duplicated mega-cues.
    segs = [FinalSegment(i * 8.0, i * 8.0 + 8.0, "Speaker_1", "MEGA" * 800)
            for i in range(32)]
    health = fusion_health_metrics(segs)
    with pytest.raises(AssertionError):
        assert_no_regression(health)


def test_manifest_roundtrip(tmp_path):
    manifest = {"videos": [{
        "id": "v1", "source": "yt:x", "media_path": "videos/v1.mp4",
        "num_speakers": 5,
        "ground_truth": {"rttm": "gt/v1.rttm"},
    }]}
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(manifest))
    entries = load_manifest(p)
    assert entries[0].id == "v1"
    assert entries[0].resolve(tmp_path)["media"] == tmp_path / "videos/v1.mp4"


def test_manifest_rejects_missing_keys(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"videos": [{"id": "x"}]}))
    with pytest.raises(ValueError):
        load_manifest(p)


def test_rttm_parsing(tmp_path):
    rttm = tmp_path / "ref.rttm"
    rttm.write_text(
        "SPEAKER v1 1 0.0 5.0 <NA> <NA> Host <NA>\n"
        "SPEAKER v1 1 5.0 7.5 <NA> <NA> Guest <NA>\n")
    turns = parse_rttm(rttm)
    assert [(t.speaker_id, t.start, t.end) for t in turns] == \
        [("Host", 0.0, 5.0), ("Guest", 5.0, 12.5)]


def test_registry_validation_detects_missing_photos(tmp_path):
    reg = tmp_path / "registry"
    reg.mkdir()
    (reg / "host.jpg").write_bytes(b"x")
    missing = validate_registry(reg, ["host", "guest 1"])
    assert missing == ["guest 1"]
    assert validate_registry(tmp_path / "nope", ["a"]) == ["a"]
