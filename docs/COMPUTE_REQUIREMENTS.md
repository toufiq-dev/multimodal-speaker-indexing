# Compute Requirements

## Development machine (used for this revision: Apple Silicon Mac)

| Component | Requirement | Notes |
|---|---|---|
| Python | 3.10–3.12 | 3.14 works for the unit-test venv (pure-logic tests stub heavy deps) |
| Unit tests | CPU only, < 1 s | `.venv/bin/pip install pytest` — ML stack NOT required (conftest stubs torch/transformers/insightface) |
| faster-whisper ASR | CPU int8 via CTranslate2; **no Metal/MPS backend exists** | `config.fw_device_and_compute()` routes `mps` → `cpu, int8` automatically. ~2× realtime for medium on M-series |
| pyannote 3.1 diarization | CPU works; pin to CPU locally | slow but fine for ≤ 5-min development clips |
| InsightFace / onnxruntime | CPU (`ctx_id=-1`); onnxruntime has no MPS provider | handled in `engines/vision.py` |
| Disk | ~6 GB models (whisper-medium fp16 ≈ 3 GB + pyannote + insightface) + video |

## GPU runs (full-length episodes)

| Environment | ASR | Diarization | Faces | Verdict |
|---|---|---|---|---|
| Colab free T4 | CUDA fp16 | fast | fast | good mid-tier |
| Kaggle T4×2/P100 | CUDA fp16 | fast | fast | keep for full episodes & training (30 h/wk quota) |
| RunPod/Vast RTX 3090 (~$0.2–0.4/h) | full | fast | fast | optional |

**Memory**: Whisper-medium fp16 ≈ 3 GB — fits any T4 (16 GB). 4-bit
quantization is unnecessary and measurably harmful at merge time (PEFT
rounding-error warning observed in the failed run's log).

## Recommended workflow split

1. Develop/validate locally on a 3–5 min slice (CPU, minutes per iteration).
2. Full episodes on Kaggle/Colab with pinned repo commit + cached artifacts.
3. Convert merged LoRA → CTranslate2 **once offline**, ship as dataset/HF repo;
   never convert inside an ephemeral notebook session.

## Reproducibility

`evaluation/tracking.py` records config hash, git state, package versions,
stage timings and metrics with every run — the missing piece that made the
first run unrepeatable.
