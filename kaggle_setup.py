# Kaggle Setup Script for Multimodal Speaker Indexing
# Run this cell FIRST before any other cells

import os
# Enable hf-transfer Rust accelerator for 10× faster HF downloads (must be before huggingface import)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import sys
import subprocess
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# 1. INSTALL PYTORCH (CUDA 12.1) - Must use explicit index
# ============================================================================

def install_pytorch():
    """Install PyTorch with CUDA 12.1 from PyTorch index."""
    print("Installing PyTorch with CUDA 12.1...")
    result = subprocess.run([
        sys.executable, "-m", "pip", "install",
        "--index-url", "https://download.pytorch.org/whl/cu121",
        "torch==2.5.1+cu121",
        "torchaudio==2.5.1+cu121",
        "torchvision==0.20.1+cu121"
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ PyTorch install failed:\n{result.stderr[-3000:]}")
        raise RuntimeError("PyTorch installation failed")
    print("✅ PyTorch installed")

# ============================================================================
# 2. INSTALL NUMPY FIRST - Locks ABI for binary wheels
# ============================================================================

def install_numpy():
    """Install numpy==1.26.4 first to lock ABI for binary wheels (insightface, onnxruntime, etc.)"""
    print("Installing numpy==1.26.4 (must be first)...")
    result = subprocess.run([
        sys.executable, "-m", "pip", "install", "numpy==1.26.4"
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ NumPy install failed:\n{result.stderr[-3000:]}")
        raise RuntimeError("NumPy installation failed")
    print("✅ NumPy installed")

# ============================================================================
# 3. INSTALL REMAINING REQUIREMENTS
# ============================================================================

def install_requirements():
    """Install all requirements with versions compatible with Kaggle environment."""
    # Enable hf-transfer for 10× faster gated model downloads (must be before huggingface_hub import)
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    requirements = """
# Audio Processing
# CRITICAL: PyPI package is 'pyannote-audio' (hyphen), NOT 'pyannote.audio' (dot)
pyannote-audio==3.3.2
faster-whisper==1.1.1
librosa==0.10.2.post1

# Transformers & NLP
transformers==4.44.0
huggingface-hub==0.25.2
tokenizers==0.19.1
accelerate==0.33.0
sentencepiece==0.2.0
hf-transfer==0.1.9

# Vision (InsightFace requires numpy<2.0 — numpy already installed first)
insightface==0.7.3
opencv-python-headless==4.10.0.84
scikit-learn==1.5.1
scikit-image==0.24.0

# Face Analysis Dependencies
# onnxruntime-gpu only available on Linux; use onnxruntime on other platforms
onnxruntime-gpu==1.19.2

# Clustering & Metrics
scipy==1.13.1

# Data Processing
pandas==2.2.2
tqdm==4.66.4
pyyaml==6.0.1

# Utilities
ffmpeg-python==0.2.0
requests==2.32.3
python-dotenv==1.0.1

# LoRA / PEFT (QLoRA 4-bit quantization needs bitsandbytes)
peft==0.12.0
bitsandbytes==0.43.0

# Protobuf compatibility (required by Google Cloud libs, transformers)
protobuf==5.29.3

# FSSpec compatibility
fsspec==2025.3.0

# Rich for progress bars
rich==13.9.4

# Jupyter/Notebook (for Kaggle/Colab execution)
ipykernel==6.29.5
jupyter-client==8.6.2

# Download utilities - pinned to existing PyPI version (avoid 2026 future 404)
yt-dlp==2024.12.23

# Evaluation
jiwer==3.0.4
pyannote.metrics==3.2.1

# RAG (Chapter 6.3.1 extension only)
sentence-transformers==3.3.1
chromadb==0.5.15
"""
    
    # Write requirements to file
    with open("requirements.txt", "w") as f:
        f.write(requirements.strip())
    
    # Install with pip
    print("Installing remaining requirements...")
    result = subprocess.run([
        sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Install failed:\n{result.stderr[-3000:]}")
        raise RuntimeError("Requirements installation failed")
    else:
        print("✅ Requirements installed successfully")

# Run installation in correct order
install_pytorch()
install_numpy()
install_requirements()

# ============================================================================
# 4. VERIFY NUMPY VERSION - CRITICAL FOR BINARY WHEEL COMPATIBILITY
# ============================================================================

import numpy as np
print(f"\nCurrent numpy version: {np.__version__}")

if not np.__version__.startswith("1.26"):
    print("⚠️ Wrong numpy version! Expected 1.26.x")
    print("Reinstalling numpy==1.26.4 with --force-reinstall...")
    result = subprocess.run([
        sys.executable, "-m", "pip", "install", "--force-reinstall", "numpy==1.26.4"
    ], capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ NumPy reinstalled successfully")
        print("🔴 RESTART KERNEL NOW (Runtime → Restart session) then re-run")
    else:
        print(f"❌ Failed: {result.stderr}")
else:
    print("✅ NumPy version OK (1.26.x)")
    print("✅ Binary wheel ABI compatibility confirmed")

# ============================================================================
# 5. FIX NUMPY COMPATIBILITY (Runtime Patch)
# ============================================================================

import numpy as np
if not hasattr(np, "NaN"):
    np.NaN = np.nan
if not hasattr(np, "Inf"):
    np.Inf = np.inf
if not hasattr(np, "PINF"):
    np.PINF = np.inf
if not hasattr(np, "NINF"):
    np.NINF = -np.inf

print("✅ NumPy compatibility patch applied")

# ============================================================================
# 6. SETUP HUGGING FACE TOKEN
# ============================================================================

def setup_hf_token():
    """Load HF token from Kaggle Secrets with fail-fast for gated models."""
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or ""
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        # Try HF_TOKEN first, then HUGGING_FACE_HUB_TOKEN fallback
        for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            try:
                tok = user_secrets.get_secret(key)
                if tok and len(tok) > 20:
                    hf_token = tok
                    os.environ["HF_TOKEN"] = hf_token
                    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
                    print(f"✅ Hugging Face token loaded from Kaggle Secrets ({key})")
                    break
            except Exception:
                continue
    except Exception as e:
        print(f"⚠️ Could not load HF_TOKEN from secrets: {e}")
    if not hf_token or len(hf_token) < 20:
        print("⚠️ HF_TOKEN missing or too short — gated pyannote/speaker-diarization-3.1 will fail with 401")
        print("   Add HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) in Kaggle Secrets (left sidebar 🔒) and restart kernel")
        # Do not raise hard — allow smoke tests, but pipeline will fail fast with clear message
        if not hf_token:
            hf_token = ""
    else:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        print(f"✅ HF token ready (length {len(hf_token)})")
    return hf_token

setup_hf_token()

# Login to Hugging Face CLI (required for pyannote models)
if os.environ.get("HF_TOKEN"):
    result = subprocess.run([
        "huggingface-cli", "login", "--token", os.environ["HF_TOKEN"], "--add-to-git-credential"
    ], capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ Hugging Face CLI login successful")
    else:
        print(f"⚠️ HF CLI login warning: {result.stderr.strip()}")

# ============================================================================
# 7. VERIFY GPU AVAILABILITY
# ============================================================================

import torch
print("\n--- Hardware Verification ---")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"PyTorch CUDA version: {torch.version.cuda}")
else:
    print("⚠️ No GPU detected. Enable GPU in Kaggle session settings (Settings → Accelerator → GPU T4 x2)")

# ============================================================================
# 8. CLONE REPOSITORY & SETUP PATHS
# ============================================================================

def setup_repository():
    """Clone the repository and setup Python paths with Kaggle cache bust."""
    repo_url = "https://github.com/toufiq-dev/multimodal-speaker-indexing.git"
    repo_dir = "/kaggle/working/multimodal-speaker-indexing"
    
    # Remove existing clone if any
    if os.path.exists(repo_dir):
        import shutil
        shutil.rmtree(repo_dir)
        print("Removed existing repository")
    
    # Clone fresh
    print("Cloning repository...")
    result = subprocess.run(["git", "clone", repo_url], capture_output=True, text=True, cwd="/kaggle/working")
    if result.returncode != 0:
        print(f"Error cloning: {result.stderr}")
        return False
    
    # Change to repo directory
    os.chdir(repo_dir)
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    # Kaggle re-run safety: clear cached imports that froze old BASE_DIR
    import importlib
    for mod in list(sys.modules.keys()):
        if mod == "config" or mod.startswith(("config.", "models", "engines")):
            del sys.modules[mod]
    importlib.invalidate_caches()
    # Verify ffmpeg binary exists (not just python wrapper)
    import shutil
    if not shutil.which("ffmpeg"):
        print("⚠️ ffmpeg binary not found — installing via apt (required for media extraction)")
        subprocess.run(["apt", "update", "-qq"], capture_output=True)
        subprocess.run(["apt", "install", "-y", "ffmpeg"], capture_output=True)
    else:
        print(f"✅ ffmpeg found: {shutil.which('ffmpeg')}")
    if not shutil.which("ffprobe"):
        print("⚠️ ffprobe missing (part of ffmpeg)")
    
    print(f"✅ Repository cloned to {repo_dir} (cache busted)")
    return True

setup_repository()

# ============================================================================
# 9. CREATE DATA DIRECTORIES
# ============================================================================

def create_directories():
    """Create necessary data directories."""
    dirs = [
        "/kaggle/working/input",
        "/kaggle/working/registry", 
        "/kaggle/working/output",
        "/kaggle/working/data/inputs",
        "/kaggle/working/data/registry",
        "/kaggle/working/data/output",
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    
    print("✅ Data directories created")

create_directories()

# ============================================================================
# 10. DOWNLOAD DATASET (Jamuna TV Rajniti Talk Show)
# ============================================================================

def download_dataset():
    """Download the talk show video using yt-dlp."""
    video_url = "https://youtu.be/qcMkD62HErQ"
    output_path = "/kaggle/working/data/inputs/full_show.mp4"
    
    if os.path.exists(output_path):
        print(f"✅ Video already exists: {output_path}")
        return output_path
    
    print("Downloading video...")
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[height<=720]+bestaudio/best[height<=720]",
        "--merge-output-format", "mp4",
        "-o", output_path,
        video_url
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error downloading: {result.stderr}")
        return None
    
    # Verify download
    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✅ Video downloaded: {output_path} ({size_mb:.1f} MB)")
        
        # Get duration
        import subprocess
        try:
            dur_cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", 
                      "-of", "csv=p=0", output_path]
            dur_result = subprocess.run(dur_cmd, capture_output=True, text=True, check=True)
            duration = float(dur_result.stdout.strip())
            print(f"   Duration: {duration:.1f}s ({duration/60:.1f} min)")
        except:
            pass
        return output_path
    else:
        print("❌ Download failed - file not found")
        return None

video_path = download_dataset()

# ============================================================================
# 11. VERIFY IMPORTS WORK
# ============================================================================

def verify_imports():
    """Verify all engine modules can be imported without errors."""
    print("\n--- Verifying Imports ---")
    
    modules_to_test = [
        ("config", "config"),
        ("models", "models"),
        ("engines.media", "engines.media"),
        ("engines.diarization", "engines.diarization"),
        ("engines.transcription", "engines.transcription"),
        ("engines.asr_lora", "engines.asr_lora"),
        ("engines.nlp", "engines.nlp"),
        ("engines.fusion", "engines.fusion"),
        # REMOVED: ("engines", "engines") - triggers lazy vision load
    ]
    
    for name, module in modules_to_test:
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            return False
    
    print("✅ All imports successful")
    return True

verify_imports()

# ============================================================================
# 12. PRINT USAGE INSTRUCTIONS
# ============================================================================

print("""
=============================================================================
✅ SETUP COMPLETE - Ready to run pipeline!
=============================================================================

Next steps - Run these cells in order:

# Cell 1: Run audio-only pipeline (diarization + transcription)
from engines.media import extract_audio
from engines.diarization import run_diarization
from engines.transcription import align_transcription_with_diarization

audio_path = extract_audio("data/inputs/full_show.mp4")
diarization = run_diarization(audio_path)
transcribed = align_transcription_with_diarization(audio_path, diarization)

# Cell 2: Run vision pipeline (face detection + recognition)
from engines.vision import run_vision_pipeline
from engines.media import extract_frames

frame_paths = extract_frames("data/inputs/full_show.mp4", fps=1)
faces = run_vision_pipeline("data/inputs/full_show.mp4", frame_paths=frame_paths)

# Cell 3: Extract speaker names from intro
from engines.nlp import extract_speaker_names_from_intro

ordered_names = extract_speaker_names_from_intro(transcribed)

# Cell 4: Run fusion to get final speaker-indexed segments
from engines.fusion import run_fusion_pipeline

final_segments = run_fusion_pipeline(diarization, transcribed, faces, ordered_names)

# Cell 5: View results
for seg in final_segments[:10]:
    print(f"[{seg.start:.1f}s - {seg.end:.1f}s] {seg.speaker}: {seg.text[:80]}...")

# Or run full pipeline:
from main import run_pipeline
final_segments = run_pipeline("data/inputs/full_show.mp4")

=============================================================================
""")

print("🎉 Setup complete! You can now run the pipeline cells above.")