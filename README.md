# Multimodal Speaker Indexing

A language-agnostic framework for **speaker identity resolution** in multi-speaker videos using audio-visual fusion and semantic context. Originally developed for Bengali talk-show indexing (DIU MSc thesis), the architecture generalizes to any language/domain.

## Overview

| Component | Technology | Purpose |
|-----------|------------|---------|
| **ASR** | faster-whisper / Whisper-LoRA | Word/segment-level transcription |
| **Diarization** | pyannote.audio 3.1 | Speaker turn segmentation |
| **Face Recognition** | InsightFace (RetinaFace + ArcFace) | Face detection, embedding, tracking |
| **Lip-Sync** | Mouth region pixel diff | Audio-visual synchronization score |
| **NER** | BanglaBERT / mBERT-Bengali | Speaker name extraction from intro |
| **Fusion** | MLP Gating Network + Heuristics | Multimodal identity resolution |

## Pipeline

```
Video
├── Audio → 16kHz WAV → Diarization (pyannote) → Speaker segments
├── Audio → Transcription (Whisper/LoRA) → Words + timestamps
├── Video → Frames @ 1 FPS → Face detection (InsightFace) → Embeddings + lip-sync
└── Transcript (first 120s) → NER → Speaker names

Fusion:
  For each diarization segment:
    1. Aggregate overlapping faces → best face per speaker
    2. Extract features: [audio_conf, face_conf, lip_sync]
    3. Gating network → P(match)
    4. Identity assignment: registered name → NLP name → cluster → generic
  Output: FinalSegment(start, end, speaker, text, confidence)
```

## Installation

```bash
# Clone
git clone https://github.com/toufiq-dev/multimodal-speaker-indexing.git
cd multimodal-speaker-indexing

# Install dependencies (requires Python 3.10+, CUDA 11.8+ for GPU)
pip install -r requirements.txt

# For GPU: ensure bitsandbytes compiles correctly
pip install bitsandbytes==0.43.0
```

### Key Dependencies

- `faster-whisper==1.0.3`
- `pyannote.audio==3.1.1`
- `torch>=2.0`, `torchaudio>=2.0`
- `transformers==4.40.0`, `peft>=0.10.0`, `bitsandbytes>=0.43.0`
- `insightface==0.7.3`, `onnxruntime-gpu`
- `numpy<2.0` (critical for InsightFace compatibility)
- `scikit-learn`, `opencv-python-headless`, `torchaudio`

## Configuration

Set environment variables:

```bash
export HF_TOKEN="your_huggingface_token"        # Required for pyannote/gated models
export WHISPER_MODEL="large-v3"                 # Or medium, small
```

Or modify `config.py` directly:

```python
from config import Config
cfg = Config()
cfg.FACE_SIM_THRESHOLD = 0.65
cfg.DBSCAN_EPS = 0.5
cfg.NLP_INTRO_SECONDS = 120
```

## Usage

### Prepare Face Registry

Place reference photos in `data/registry/` (one per known person):

```
data/registry/
├── dr_rahman.jpg
├── host_karim.jpg
└── guest_shila.png
```

Filenames become speaker names (underscores → spaces).

### Run Pipeline

```bash
# Using faster-whisper (default)
python main.py -i video.mp4 -r data/registry -o output

# Using LoRA-adapted Whisper (better for Bengali)
python main.py -i video.mp4 -r data/registry -o output --use_lora --lora_path your-org/bengali-talkshow-adapter
```

### Outputs

```
output/
├── result.json      # List of FinalSegment objects
├── subtitles.srt    # SRT format with speaker labels
├── audio/           # Extracted 16kHz WAV
└── frames/          # Extracted frames @ 1 FPS
```

**result.json** format:
```json
[
  {
    "start": 12.3,
    "end": 18.7,
    "speaker": "Dr. Rahman",
    "text": "আমি আজকের আলোচনায় অংশ নিতে পারি।",
    "confidence": 0.92
  }
]
```

## LoRA Adapter Training

Train a Bengali Talk-Show adapter for Whisper-large-v3:

```bash
# On Kaggle (30h/week free T4×2)
# See notebooks/03_train_whisper_lora.ipynb
```

Config (QLoRA 4-bit):
- Rank: 32, Alpha: 64
- Target modules: `q_proj`, `v_proj`
- LR: 1e-3, 3-5 epochs
- Dataset: Bengali-Loop (158h) + in-domain talk-show clips

## Project Structure

```
multimodal-speaker-indexing/
├── config.py              # Central configuration
├── models.py              # Frozen dataclasses
├── main.py                # CLI entry point
├── requirements.txt
├── engines/
│   ├── media.py           # Audio/frame extraction (ffmpeg)
│   ├── diarization.py     # pyannote 3.1 two-pass
│   ├── transcription.py   # faster-whisper + alignment
│   ├── asr_lora.py        # Whisper-LoRA chunked inference
│   ├── vision.py          # InsightFace + DBSCAN clustering
│   ├── nlp.py             # BanglaBERT NER name extraction
│   └── fusion.py          # GatingNetwork + identity resolution
├── data/
│   ├── input/             # Videos (gitignored)
│   ├── registry/          # Reference face photos
│   └── output/            # Pipeline outputs
└── notebooks/             # Kaggle training/benchmark notebooks
```

## Benchmarks (Bengali-Loop)

| Task | Model | Metric |
|------|-------|--------|
| ASR | Whisper-large-v3 (zero-shot) | 34% WER |
| ASR | Whisper-LoRA (this work) | **~25% WER** (target) |
| Diarization | pyannote 3.1 | 40% DER |
| Diarization + Fusion | This work | **<30% DER** (target) |

## Thesis Context

Developed for **DIU MSc Data Science 18-credit capstone** (2024-2025).
Addresses the research gap: **semantic identity mapping** vs. blind clustering in low-resource Bengali multimedia.

Key citations:
- Bengali-Loop benchmark: `arXiv:2602.14291`
- Audio-visual diarization: `AVA-AVD (arXiv:2111.14448)`
- Semantic speaker naming: `Zhang et al. 2025 (arXiv:2509.15082)`

## License

MIT License - see LICENSE file.