# Kaggle Setup Script for Multimodal Speaker Indexing
# Run this cell FIRST before any other cells

import os
import sys
import subprocess
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# 1. INSTALL DEPENDENCIES WITH COMPATIBLE VERSIONS
# ============================================================================

def install_requirements():
    """Install all requirements with versions compatible with Kaggle environment."""
    requirements = """
# Core ML Framework (CUDA 12.1)
torch==2.3.1+cu121
torchaudio==2.3.1+cu121
torchvision==0.18.1+cu121
--index-url https://download.pytorch.org/whl/cu121

# Audio Processing
pyannote.audio==3.3.2
faster-whisper==1.1.1
librosa==0.10.2.post1

# Transformers & NLP
transformers==4.44.0
huggingface-hub==0.25.2
tokenizers==0.19.1
accelerate==0.33.0
sentencepiece==0.2.0

# Vision
insightface==0.7.3
opencv-python-headless==4.10.0.84
scikit-learn==1.5.1
scikit-image==0.24.0

# Face Analysis Dependencies
onnxruntime-gpu==1.19.2

# Clustering & Metrics
scipy==1.13.1
numpy==1.26.4

# Data Processing
pandas==2.2.2
tqdm==4.66.4
pyyaml==6.0.1

# Utilities
ffmpeg-python==0.2.0
requests==2.32.3
python-dotenv==1.0.1

# LoRA / PEFT
peft==0.12.0

# Protobuf compatibility (critical)
protobuf==5.29.3

# FSSpec compatibility
fsspec==2025.3.0

# Rich for progress bars
rich==13.9.4

# Jupyter/Notebook
ipykernel==6.29.5
jupyter-client==8.6.2

# Download utilities
yt-dlp==2024.8.6

# Evaluation
jiwer==3.0.4
"""
    
    # Write requirements to file
    with open("requirements.txt", "w") as f:
        f.write(requirements.strip())
    
    # Install with pip
    print("Installing requirements...")
    result = subprocess.run([
        sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Warning: Some packages may have failed to install:")
        print(result.stderr[-2000:])
    else:
        print("✅ Requirements installed successfully")

# Run installation
install_requirements()

# ============================================================================
# 2. FIX NUMPY COMPATIBILITY (Runtime Patch)
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
# 3. SETUP HUGGING FACE TOKEN
# ============================================================================

def setup_hf_token():
    """Load HF token from Kaggle Secrets."""
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        hf_token = user_secrets.get_secret("HF_TOKEN")
        os.environ["HF_TOKEN"] = hf_token
        print("✅ Hugging Face token loaded from Kaggle Secrets")
    except Exception as e:
        print(f"⚠️ Could not load HF_TOKEN from secrets: {e}")
        print("   Make sure to add HF_TOKEN in Kaggle Secrets (left sidebar 🔒)")

setup_hf_token()

# ============================================================================
# 4. VERIFY GPU AVAILABILITY
# ============================================================================

import torch
print("\n--- Hardware Verification ---")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    print(f"PyTorch CUDA version: {torch.version.cuda}")
else:
    print("⚠️ No GPU detected. Enable GPU in Kaggle session settings (Settings → Accelerator → GPU)")

# ============================================================================
# 5. CLONE REPOSITORY & SETUP PATHS
# ============================================================================

def setup_repository():
    """Clone the repository and setup Python paths."""
    repo_url = "https://github.com/toufiq-dev/multimodal-speaker-indexing.git"
    repo_dir = "/kaggle/working/multimodal-speaker-indexing"
    
    # Remove existing clone if any
    if os.path.exists(repo_dir):
        import shutil
        shutil.rmtree(repo_dir)
        print("Removed existing repository")
    
    # Clone fresh
    print("Cloning repository...")
    result = subprocess.run(["git", "clone", repo_url], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error cloning: {result.stderr}")
        return False
    
    # Change to repo directory
    os.chdir(repo_dir)
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    
    print(f"✅ Repository cloned to {repo_dir}")
    return True

setup_repository()

# ============================================================================
# 6. CREATE DATA DIRECTORIES
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
# 7. DOWNLOAD DATASET (Jamuna TV Rajniti Talk Show)
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
# 8. VERIFY IMPORTS WORK
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
        ("engines", "engines"),  # This tests lazy loading of vision
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
# 9. PRINT USAGE INSTRUCTIONS
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