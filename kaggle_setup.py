"""Kaggle bootstrap for the multimodal speaker-indexing pipeline.

Run this FIRST, as a single cell, before importing anything from the repo.

Ordering rationale
------------------
The previous version installed dependencies from a requirements list embedded
in this file and *then* cloned the repository. That had three consequences:

1. Two copies of the dependency set existed and had already drifted.
2. ``open("requirements.txt", "w")`` is relative, and ``setup_repository()``
   chdir's into the clone — so re-running setup in the same session
   overwrote the repository's own requirements.txt with the stale copy.
3. Nothing verified that the CUDA execution provider survived installation.

The order is now: clone -> install from the repo's pinned files -> verify.
The repository is the single source of truth for dependency versions.
"""

import os
import shutil
import subprocess
import sys
import warnings

# Rust download accelerator; must be set before huggingface_hub is imported.
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
warnings.filterwarnings("ignore")

REPO_URL = "https://github.com/toufiq-dev/multimodal-speaker-indexing.git"
REPO_DIR = "/kaggle/working/multimodal-speaker-indexing"
WORKING = "/kaggle/working"

TORCH_PINS = ["torch==2.5.1+cu121", "torchaudio==2.5.1+cu121", "torchvision==0.20.1+cu121"]
TORCH_INDEX = "https://download.pytorch.org/whl/cu121"
NUMPY_PIN = "numpy==1.26.4"
ORT_GPU_PIN = "onnxruntime-gpu==1.19.2"

# The Bengali ASR checkpoint is published as a Transformers model; faster-whisper
# needs a CTranslate2 directory. The conversion output is a ~800 MB *derived*
# artifact, so it lives under the scratch tree rather than /kaggle/working,
# which is committed verbatim as notebook output.
CT2_SRC = "bengaliAI/tugstugi_bengaliai-asr_whisper-medium"
CT2_DIR = "/tmp/msi_scratch/models/bengaliAI_ct2"


def _pip(*args, check=True):
    """Run pip, surfacing the tail of stderr on failure."""
    result = subprocess.run([sys.executable, "-m", "pip", *args],
                            capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ pip {' '.join(args[:3])}... failed:\n{result.stderr[-3000:]}")
        if check:
            raise RuntimeError(f"pip {args[0]} failed")
    return result.returncode == 0


# ==========================================================================
# 1. CLONE REPOSITORY (first: it owns the dependency pins)
# ==========================================================================

def clone_repository():
    if os.path.exists(REPO_DIR):
        shutil.rmtree(REPO_DIR)
        print("Removed existing repository")

    print("Cloning repository...")
    result = subprocess.run(["git", "clone", REPO_URL],
                            capture_output=True, text=True, cwd=WORKING)
    if result.returncode != 0:
        raise RuntimeError(f"git clone failed: {result.stderr}")

    os.chdir(REPO_DIR)
    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)

    # Kaggle re-run safety: drop cached modules that froze an old BASE_DIR.
    import importlib
    for mod in list(sys.modules):
        if mod == "config" or mod == "runtime" or mod.startswith(("config.", "models", "engines", "evaluation")):
            del sys.modules[mod]
    importlib.invalidate_caches()
    print(f"✅ Repository cloned to {REPO_DIR}")


# ==========================================================================
# 2. INSTALL DEPENDENCIES
# ==========================================================================

def install_pytorch():
    """CUDA 12.1 build, from the PyTorch index."""
    print("Installing PyTorch (CUDA 12.1)...")
    _pip("install", "--index-url", TORCH_INDEX, *TORCH_PINS)
    print("✅ PyTorch installed")


def install_numpy():
    """NumPy first: locks the C ABI for insightface / onnxruntime wheels."""
    print(f"Installing {NUMPY_PIN} (must precede binary wheels)...")
    _pip("install", NUMPY_PIN)
    print("✅ NumPy installed")


def install_insightface():
    """insightface 0.7.3 is source-only and needs build deps up front.

    Its setup.py imports Cython and NumPy headers, which pip's build isolation
    does not provide, so a clean image fails to build the wheel.
    """
    print("Installing insightface (seeded build deps, no isolation)...")
    _pip("install", "cython", NUMPY_PIN)
    ok = _pip("install", "insightface==0.7.3", "--no-build-isolation", check=False)
    if not ok:
        print("⚠️ --no-build-isolation failed; retrying with default isolation")
        _pip("install", "insightface==0.7.3")
    print("✅ insightface installed")


def install_requirements():
    """Install the repository's pinned set under the NumPy constraint."""
    req = os.path.join(REPO_DIR, "requirements.txt")
    con = os.path.join(REPO_DIR, "constraints.txt")
    print(f"Installing {req} with constraints {con}...")
    _pip("install", "-r", req, "-c", con)
    print("✅ Requirements installed")


def fix_onnxruntime_conflict():
    """Restore the CUDA execution provider after chromadb clobbers it.

    chromadb hard-depends on the CPU ``onnxruntime`` wheel, which installs into
    the SAME ``onnxruntime`` import path as ``onnxruntime-gpu``. Whichever lands
    last wins, and when the CPU build wins, InsightFace's
    ``providers=['CUDAExecutionProvider', ...]`` silently degrades to CPU —
    turning a GPU run into a ~20x slowdown with no error anywhere in the log.
    """
    print("Resolving onnxruntime / onnxruntime-gpu conflict...")
    _pip("uninstall", "-y", "onnxruntime", check=False)
    _pip("install", "--force-reinstall", "--no-deps", ORT_GPU_PIN)

    import onnxruntime as ort
    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" not in providers:
        raise RuntimeError(
            f"CUDA execution provider still missing after repair; ORT sees {providers}. "
            f"Vision would run on CPU."
        )
    print(f"✅ ORT providers: {providers}")


def configure_cudnn_path():
    """Expose the torch wheel's cuDNN 9 to CTranslate2's dynamic loader."""
    sys.path.insert(0, REPO_DIR)
    from runtime import ensure_cudnn_on_path
    lib = ensure_cudnn_on_path()
    print(f"✅ cuDNN path: {lib}" if lib else "⚠️ wheel-local cuDNN not found")


# ==========================================================================
# 3. HUGGING FACE AUTH
# ==========================================================================

def setup_hf_token():
    """Load the HF token from Kaggle Secrets; gated models need it."""
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or ""
    try:
        from kaggle_secrets import UserSecretsClient
        secrets = UserSecretsClient()
        for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            try:
                tok = secrets.get_secret(key)
                if tok and len(tok) > 20:
                    hf_token = tok
                    print(f"✅ HF token loaded from Kaggle Secrets ({key})")
                    break
            except Exception:
                continue
    except Exception as e:
        print(f"⚠️ Could not read Kaggle Secrets: {e}")

    if hf_token and len(hf_token) > 20:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        subprocess.run(["huggingface-cli", "login", "--token", hf_token,
                        "--add-to-git-credential"], capture_output=True, text=True)
        print(f"✅ HF token ready (length {len(hf_token)})")
    else:
        # Not fatal here: the diarization stage raises with a clear message,
        # and the non-gated stages remain testable.
        print("⚠️ HF_TOKEN missing — pyannote/speaker-diarization-3.1 will 401. "
              "Add it under Kaggle Secrets (🔒) and restart the kernel.")
    return hf_token


# ==========================================================================
# 4. DIRECTORIES & DATASET
# ==========================================================================

def create_directories():
    """Outputs under /kaggle/working; heavy intermediates under /tmp.

    /kaggle/working is committed verbatim as notebook output, so frames
    (~3k JPEGs for a 53-minute show at 1 FPS) and extracted audio must not
    live there. config.SCRATCH_DIR resolves to /tmp/msi_scratch on Kaggle.
    """
    for d in ("/kaggle/working/input", "/kaggle/working/registry",
              "/kaggle/working/output", "/kaggle/working/data/inputs",
              "/kaggle/working/data/registry", "/kaggle/working/data/output",
              "/tmp/msi_scratch"):
        os.makedirs(d, exist_ok=True)
    print("✅ Data + scratch directories created")


def download_dataset():
    """Fetch the Jamuna TV Rajniti talk show."""
    video_url = "https://youtu.be/qcMkD62HErQ"
    output_path = "/kaggle/working/data/inputs/full_show.mp4"

    if os.path.exists(output_path):
        print(f"✅ Video already present: {output_path}")
        return output_path

    print("Downloading video...")
    result = subprocess.run(
        ["yt-dlp", "-f", "bestvideo[height<=720]+bestaudio/best[height<=720]",
         "--merge-output-format", "mp4", "-o", output_path, video_url],
        capture_output=True, text=True)
    if result.returncode != 0 or not os.path.exists(output_path):
        print(f"❌ Download failed: {result.stderr[-1500:]}")
        return None

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ Video downloaded: {output_path} ({size_mb:.1f} MB)")
    return output_path


def ensure_ffmpeg():
    if not shutil.which("ffmpeg"):
        print("⚠️ ffmpeg missing — installing via apt")
        subprocess.run(["apt", "update", "-qq"], capture_output=True)
        subprocess.run(["apt", "install", "-y", "ffmpeg"], capture_output=True)
    print(f"✅ ffmpeg: {shutil.which('ffmpeg')} | ffprobe: {shutil.which('ffprobe')}")


# ==========================================================================
# 5. ASR MODEL CONVERSION
# ==========================================================================

def convert_asr_model():
    """Convert the Bengali Whisper checkpoint to CTranslate2 and select it.

    faster-whisper cannot consume a Transformers checkpoint: it requires a CT2
    directory (``model.bin`` plus a CT2 ``config.json`` and ``vocabulary.json``).
    ``config.WHISPER_MODEL`` names the *source* repository, so without this step
    the transcription stage fails at model load — the single most likely reason
    for an otherwise clean setup to die in stage 3.

    Set ``CT2_MODEL_DIR`` to reuse a conversion from a Kaggle dataset instead.
    """
    out = os.environ.get("CT2_MODEL_DIR", CT2_DIR)
    weights = os.path.join(out, "model.bin")

    if os.path.exists(weights):
        print(f"✅ CT2 model already present: {out}")
    else:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        print(f"Converting {CT2_SRC} → {out} (int8); this takes a few minutes...")
        # Prefer the console script, falling back to the module form. The old
        # hardcoded ~/.venv/bin/ct2-transformers-converter exists on neither
        # Kaggle nor Colab.
        exe = shutil.which("ct2-transformers-converter")
        cmd = [exe] if exe else [sys.executable, "-m",
                                 "ctranslate2.converters.transformers"]
        result = subprocess.run(
            cmd + ["--model", CT2_SRC, "--output_dir", out,
                   "--quantization", "int8", "--force",
                   "--copy_files", "tokenizer.json", "preprocessor_config.json"],
            capture_output=True, text=True)
        if result.returncode != 0 or not os.path.exists(weights):
            raise RuntimeError(
                "CTranslate2 conversion failed — transcription cannot run.\n"
                + result.stderr[-2500:])
        print(f"✅ Converted: {out}")

    # An English-only conversion does NOT make faster-whisper reject
    # language="bn"; it silently decodes Bangla through an English decoder.
    from faster_whisper import WhisperModel
    probe = WhisperModel(out, device="cpu", compute_type="int8")
    multilingual = probe.model.is_multilingual
    del probe
    if not multilingual:
        raise RuntimeError(
            f"{out} converted as English-only; language='bn' would be silently "
            f"downgraded to 'en'. Re-convert with the multilingual tokenizer.")

    # Select it for this process. config reads WHISPER_MODEL from the
    # environment at import time, so also patch an already-imported instance.
    os.environ["WHISPER_MODEL"] = out
    mod = sys.modules.get("config")
    if mod is not None:
        mod.config.WHISPER_MODEL = out
    print(f"✅ WHISPER_MODEL={out} (multilingual)")


# ==========================================================================
# 6. PREFLIGHT — turn every silent degradation into a loud failure
# ==========================================================================

def preflight():
    """Assert the invariants that otherwise fail silently mid-run."""
    print("\n--- Preflight ---")
    import numpy as np
    import torch
    import onnxruntime as ort

    from runtime import NUMPY_ABI_LOCK, describe_devices

    assert np.__version__.startswith(NUMPY_ABI_LOCK), (
        f"NumPy ABI lock broken: {np.__version__} (need {NUMPY_ABI_LOCK}.x). "
        f"insightface/onnxruntime are compiled against NumPy 1.x.")
    assert torch.cuda.is_available(), (
        "No CUDA device. Settings → Accelerator → GPU T4 x2.")
    assert "CUDAExecutionProvider" in ort.get_available_providers(), (
        f"ORT is CPU-only ({ort.get_available_providers()}); a CPU onnxruntime "
        f"wheel is shadowing onnxruntime-gpu.")
    assert os.environ.get("HF_TOKEN"), "HF_TOKEN unset — gated pyannote will 401."

    whisper_model = os.environ.get("WHISPER_MODEL", "")
    assert whisper_model and os.path.exists(os.path.join(whisper_model, "model.bin")), (
        f"WHISPER_MODEL={whisper_model!r} is not a CTranslate2 directory — "
        f"faster-whisper cannot load a Transformers checkpoint. "
        f"Run convert_asr_model().")

    print(describe_devices())

    try:
        import ctranslate2
        print(f"✅ ctranslate2 {ctranslate2.__version__}")
    except Exception as e:
        print(f"⚠️ ctranslate2 unavailable: {e}")

    print("✅ Preflight passed")


def verify_imports():
    """Import every engine, INCLUDING vision.

    vision was previously excluded from this check — which meant the one
    module whose import can actually fail (OpenCV / InsightFace / ORT / NumPy
    ABI) was the one module never smoke-tested.
    """
    print("\n--- Verifying Imports ---")
    modules = ["runtime", "config", "models", "engines.media", "engines.diarization",
               "engines.transcription", "engines.asr_lora", "engines.nlp",
               "engines.fusion", "engines.clustering", "engines.vision",
               "engines.rag", "evaluation.metrics", "evaluation.tracking"]
    failed = []
    for module in modules:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except Exception as e:
            print(f"  ❌ {module}: {e.__class__.__name__}: {e}")
            failed.append(module)
    if failed:
        raise RuntimeError(f"Imports failed: {failed}")
    print("✅ All imports successful")


def verify_registry():
    """A silently empty registry disables P1 identity resolution entirely."""
    from config import config
    photos = [p for p in config.DATA_REGISTRY_DIR.iterdir()
              if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}] \
        if config.DATA_REGISTRY_DIR.exists() else []
    if not photos:
        print(f"⚠️ No registry photos in {config.DATA_REGISTRY_DIR} — every speaker "
              f"will fall through the cascade to a generic Speaker_N label.")
    else:
        print(f"✅ Registry photos: {[p.stem for p in photos]}")


# ==========================================================================
# MAIN
# ==========================================================================

def main():
    clone_repository()
    install_pytorch()
    install_numpy()
    install_insightface()
    install_requirements()
    fix_onnxruntime_conflict()
    configure_cudnn_path()
    ensure_ffmpeg()
    setup_hf_token()
    create_directories()
    download_dataset()
    convert_asr_model()
    preflight()
    verify_imports()
    verify_registry()

    print("""
=============================================================================
✅ SETUP COMPLETE

    from main import run_pipeline
    segments = run_pipeline(
        video_path="/kaggle/working/data/inputs/full_show.mp4",
        registry_dir="/kaggle/working/data/registry",
        output_dir="/kaggle/working/output",
    )

ASR model    : $WHISPER_MODEL (CTranslate2 int8, already selected)
Outputs      : /kaggle/working/output/{result.json,subtitles.srt,rag_index/}
Intermediates: /tmp/msi_scratch/{audio,frames,models}   (not committed as output)
=============================================================================
""")


if __name__ == "__main__" or os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
    main()
