"""Regression tests for the Kaggle-readiness audit.

Every test here corresponds to a defect that shipped because nothing covered
it. They are deliberately dependency-light so they run on any machine, not
only on the CUDA target.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import runtime
from engines.clustering import MAX_FIT_SAMPLES, NOISE_LABEL, cluster_face_embeddings


# --------------------------------------------------------------------------
# runtime: ABI lock and provider policy
# --------------------------------------------------------------------------

def test_numpy_compat_restores_removed_aliases():
    runtime.apply_numpy_compat()
    assert np.isnan(np.NaN)
    assert np.isinf(np.Inf) and np.Inf > 0
    assert np.isinf(np.NINF) and np.NINF < 0


def test_numpy_abi_assert_can_be_bypassed_for_dev(monkeypatch):
    monkeypatch.setenv("ALLOW_NUMPY_ABI_DRIFT", "1")
    runtime.assert_numpy_abi()  # must not raise regardless of local NumPy


def test_numpy_abi_assert_rejects_wrong_major(monkeypatch):
    monkeypatch.delenv("ALLOW_NUMPY_ABI_DRIFT", raising=False)
    monkeypatch.setattr(np, "__version__", "2.1.0")
    with pytest.raises(RuntimeError, match="ABI lock broken"):
        runtime.assert_numpy_abi()


def test_onnx_providers_bounds_the_cuda_arena():
    """An unbounded ORT arena competes with torch for the same device."""
    providers = runtime.onnx_providers(use_cuda=True, gpu_mem_limit_gb=2)
    assert providers[0][0] == "CUDAExecutionProvider"
    opts = providers[0][1]
    assert opts["gpu_mem_limit"] == 2 * 1024 ** 3
    assert opts["arena_extend_strategy"] == "kSameAsRequested"
    assert providers[-1] == "CPUExecutionProvider", "CPU must remain the fallback"


def test_onnx_providers_cpu_only_when_cuda_off():
    assert runtime.onnx_providers(use_cuda=False) == ["CPUExecutionProvider"]


def test_assert_cuda_ep_raises_when_provider_missing(monkeypatch):
    monkeypatch.setattr(runtime, "available_onnx_providers",
                        lambda: ["CPUExecutionProvider"])
    with pytest.raises(RuntimeError, match="chromadb"):
        runtime.assert_cuda_execution_provider()


def test_with_model_frees_before_returning():
    """The model must be unreachable by the time the cache is released.

    A `with ... as model:` form cannot satisfy this: the `as` target stays
    bound in the caller's frame after the block, so the release would run
    against a live reference and reclaim nothing.
    """
    import weakref

    class Model:
        pass

    captured = {}

    def use(m):
        captured["ref"] = weakref.ref(m)
        assert captured["ref"]() is not None
        return "result"

    released = {}
    real_release = runtime.release_gpu_memory

    def spy():
        released["alive_at_release"] = captured["ref"]() is not None
        real_release()

    runtime.release_gpu_memory = spy
    try:
        assert runtime.with_model(Model, use, "dummy") == "result"
    finally:
        runtime.release_gpu_memory = real_release

    assert released["alive_at_release"] is False, (
        "model was still referenced when the cache was released — VRAM would "
        "not have been returned")
    assert captured["ref"]() is None


# --------------------------------------------------------------------------
# clustering: bounded memory, shared by both vision backends
# --------------------------------------------------------------------------

def _synthetic_identities(n_per: int, k: int, dim: int = 64, seed: int = 7):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(k, dim))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    pts = np.concatenate([c + 0.02 * rng.normal(size=(n_per, dim)) for c in centers])
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


def test_clustering_recovers_identities():
    labels = cluster_face_embeddings(_synthetic_identities(40, 3),
                                     algorithm="dbscan", eps=0.3, min_samples=3)
    assert len({int(l) for l in labels if l >= 0}) == 3


def test_agglomerative_honours_requested_cluster_count():
    labels = cluster_face_embeddings(_synthetic_identities(30, 4),
                                     algorithm="agglomerative", n_clusters=4)
    assert len(set(labels.tolist())) == 4
    assert NOISE_LABEL not in labels, "partitioning must not emit noise"


def test_clustering_bounds_the_distance_matrix(monkeypatch):
    """The n^2 matrix, not VRAM, is the real OOM on a full-length show.

    Fit must be capped and out-of-sample points assigned by centroid, while
    still labelling every input.
    """
    seen = {}
    import engines.clustering as clustering

    real_fit = clustering._fit_labels

    def spy(x, *a, **kw):
        seen["n_fit"] = len(x)
        return real_fit(x, *a, **kw)

    monkeypatch.setattr(clustering, "_fit_labels", spy)

    n = 900
    labels = clustering.cluster_face_embeddings(
        _synthetic_identities(n // 3, 3), algorithm="dbscan",
        eps=0.3, min_samples=3, max_fit_samples=200)

    assert seen["n_fit"] == 200, "clusterer was handed the full set"
    assert len(labels) == n, "out-of-sample points must still be labelled"
    assert len({int(l) for l in labels if l >= 0}) == 3


def test_clustering_is_deterministic():
    """Ablation cells must be reproducible across runs."""
    x = _synthetic_identities(300, 3)
    a = cluster_face_embeddings(x, algorithm="dbscan", eps=0.3,
                                min_samples=3, max_fit_samples=100)
    b = cluster_face_embeddings(x, algorithm="dbscan", eps=0.3,
                                min_samples=3, max_fit_samples=100)
    assert np.array_equal(a, b)


def test_clustering_handles_degenerate_input():
    assert len(cluster_face_embeddings(np.zeros((0, 64)))) == 0
    labels = cluster_face_embeddings(np.ones((2, 64)), min_samples=5)
    assert all(l == NOISE_LABEL for l in labels)


def test_max_fit_samples_keeps_matrix_under_a_gigabyte():
    assert (MAX_FIT_SAMPLES ** 2) * 8 < 1_000_000_000


# --------------------------------------------------------------------------
# evaluation: the manifest writer and the ablation matrix
# --------------------------------------------------------------------------

def test_experiment_run_save_round_trips(tmp_path):
    """Previously raised NameError: `asdict` was used but never imported."""
    from evaluation.tracking import ExperimentRun

    run = ExperimentRun(name="audit", config={"WHISPER_MODEL": "x"})
    run.finish({"DER": 0.21})
    path = run.save(tmp_path)

    payload = json.loads(path.read_text())
    assert payload["name"] == "audit"
    assert payload["metrics"]["DER"] == 0.21
    assert payload["finished_at"] and payload["run_id"]
    assert "numpy" in payload["packages"], "ABI-relevant versions must be recorded"


def test_text_only_baseline_cell_is_reachable():
    """The B2 cell was shadowed by the B1 branch and never executed."""
    from evaluation.ablations import default_matrix, run_cell
    from models import DiarizationSegment, TranscribedSegment, WordToken

    cell = next(c for c in default_matrix() if c.name == "text_only_baseline")

    words = [WordToken("ka", 0.0, 1.0), WordToken("kha", 1.0, 2.0)]
    transcribed = [TranscribedSegment(0.0, 2.0, "ka kha", words, "SPEAKER_00")]
    diarization = [DiarizationSegment(0.0, 2.0, "SPEAKER_00")]

    out = run_cell(
        cell, diarization=diarization, transcribed_word=transcribed,
        transcribed_chunk=None, faces=[], ordered_names=[],
        reference_turns_named=[DiarizationSegment(0.0, 2.0, "Host")],
        reference_texts_by_speaker={"Host": "ka kha"},
    )
    speakers = {fs.speaker for fs in out["final_segments"]}
    assert speakers == {"ASR_BLOCK"}, (
        f"text_only baseline must bypass diarization; got {speakers}")


# --------------------------------------------------------------------------
# config: Kaggle-safe I/O scaffolding
# --------------------------------------------------------------------------

def test_scratch_is_separate_from_committed_output():
    """/kaggle/working is committed as notebook output; frames must not go there."""
    from config import config
    assert config.SCRATCH_DIR != config.DATA_OUTPUT_DIR
    assert config.DATA_OUTPUT_DIR not in config.SCRATCH_DIR.parents
    assert config.SCRATCH_DIR.exists()


def test_kaggle_env_routes_scratch_to_tmp(monkeypatch):
    import config as config_module

    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    monkeypatch.delenv("SCRATCH_DIR", raising=False)
    assert config_module._resolve_base_dir() == Path("/kaggle/working")
    # .resolve() maps /tmp -> /private/tmp on macOS; compare resolved forms.
    assert config_module._resolve_scratch_dir() == Path("/tmp/msi_scratch").resolve()


def test_punctuation_restore_is_actually_gated(monkeypatch):
    """The flag was dead config while the rewrite ran unconditionally."""
    from config import config
    from engines.transcription import _restore_bengali_punct

    long_bn = " ".join(["শব্দ"] * 25)

    monkeypatch.setattr(config, "ENABLE_PUNCTUATION_RESTORE", True)
    assert "।" in _restore_bengali_punct(long_bn)

    monkeypatch.setattr(config, "ENABLE_PUNCTUATION_RESTORE", False)
    assert _restore_bengali_punct(long_bn) == long_bn


# --------------------------------------------------------------------------
# nlp: silent NER truncation
# --------------------------------------------------------------------------

def test_ner_windows_cover_the_whole_intro():
    """TokenClassificationPipeline truncates at 512 subwords with no warning."""
    from engines.nlp import _NER_WINDOW_CHARS, _split_windows

    text = ". ".join(f"বাক্য {i} এখানে শেষ" for i in range(400))
    windows = _split_windows(text)

    assert len(windows) > 1, "a 120s intro must be split, not truncated"
    assert all(len(w) <= _NER_WINDOW_CHARS + 1 for _, w in windows)
    # Offsets must rebase exactly onto the source string.
    assert all(text[base:base + len(w)] == w for base, w in windows)
    assert "".join(w for _, w in windows) == text, "no characters may be dropped"


def test_ner_short_intro_is_a_single_window():
    from engines.nlp import _split_windows
    assert _split_windows("আমি রফিকুল ইসলাম।") == [(0, "আমি রফিকুল ইসলাম।")]


# --------------------------------------------------------------------------
# ASR checkpoint selection
#
# config.WHISPER_MODEL names a *Transformers* repository, but faster-whisper
# can only load a CTranslate2 directory. The bootstrap must therefore convert
# and re-point WHISPER_MODEL; previously it did neither, so a clean setup
# failed in stage 3 with an opaque loader error.
# --------------------------------------------------------------------------

def test_conversion_script_does_not_hardcode_a_local_venv():
    script = (Path(__file__).resolve().parents[1]
              / "scripts" / "convert_bengali_ct2.sh").read_text("utf-8")
    # Comments are allowed to name the old path; executable lines are not.
    code = "\n".join(l for l in script.splitlines()
                     if not l.lstrip().startswith("#"))
    assert "~/.venv" not in code, (
        "a hardcoded interpreter path does not exist on Kaggle or Colab and "
        "aborts the script under `set -e`")
    assert "ctranslate2.converters.transformers" in code


def test_bootstrap_selects_the_converted_ct2_directory(tmp_path, monkeypatch):
    import kaggle_setup

    model_dir = tmp_path / "bengaliAI_ct2"
    model_dir.mkdir()
    (model_dir / "model.bin").write_bytes(b"stub")
    monkeypatch.setenv("CT2_MODEL_DIR", str(model_dir))
    monkeypatch.delenv("WHISPER_MODEL", raising=False)

    # An existing conversion must be reused, never rebuilt.
    def _no_subprocess(*a, **k):
        raise AssertionError("re-converted an existing CT2 directory")
    monkeypatch.setattr(kaggle_setup.subprocess, "run", _no_subprocess)

    class _Probe:
        def __init__(self, *a, **k):
            self.model = type("m", (), {"is_multilingual": True})()
    monkeypatch.setattr(sys.modules["faster_whisper"], "WhisperModel", _Probe)

    kaggle_setup.convert_asr_model()
    assert kaggle_setup.os.environ["WHISPER_MODEL"] == str(model_dir)


def test_bootstrap_rejects_an_english_only_conversion(tmp_path, monkeypatch):
    import kaggle_setup

    model_dir = tmp_path / "en_only_ct2"
    model_dir.mkdir()
    (model_dir / "model.bin").write_bytes(b"stub")
    monkeypatch.setenv("CT2_MODEL_DIR", str(model_dir))

    class _Probe:
        def __init__(self, *a, **k):
            self.model = type("m", (), {"is_multilingual": False})()
    monkeypatch.setattr(sys.modules["faster_whisper"], "WhisperModel", _Probe)

    # Silently decoding Bangla through an English decoder is worse than a crash.
    with pytest.raises(RuntimeError, match="English-only"):
        kaggle_setup.convert_asr_model()
