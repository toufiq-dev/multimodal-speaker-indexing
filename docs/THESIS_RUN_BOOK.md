# Thesis Run Book — Recording Verified End-to-End Results

This run book is the *operating procedure* for turning one Bengali talk-show
video into a **verified, recorded** thesis result. It exists because the
pipeline was run successfully only once on Kaggle; every later attempt died on
silent-degradation defects that are now fixed but were never exercised through
a single repeatable entry point.

The strategy has three layers:

1. **A golden environment** — exactly the pinned dependency universe the code
   was audited against (NumPy 1.26.x, torch cu121, InsightFace 0.7.3,
   onnxruntime-gpu 1.19.2). Nothing else is trusted for recorded results.
2. **A preflight** — `kaggle_setup.preflight()` / `runtime` assertions turn
   every silent degradation (CPU fallback, English-only ASR, NumPy ABI drift,
   missing gated token) into a loud failure *before* a run.
3. **A recorded run** — stage outputs are cached under scratch, the final
   `result.json`/`.srt` are written to an episode output directory, and a
   reproducibility manifest + evaluation metrics are written next to them.

---

## 0. The single most important warning

**Do not record thesis results from a local `.venv` built before the ABI-lock
fix.** As of this writing the repository's `requirements.txt`/`constraints.txt`
were last touched **2026-08-28**; the local `.venv` at the repo root was built
**2026-08-26** and contains:

| Package | Installed (local `.venv`) | Required by repo |
|---|---|---|
| numpy | **2.5.2** | **1.26.4** |
| torch | 2.13.0 | 2.5.1+cu121 |
| insightface | **1.0.1** | **0.7.3** |
| onnxruntime | **1.29.0** (CPU) | onnxruntime-gpu 1.19.2 |
| transformers | 5.16.1 | 4.44.0 |

The 98 unit tests pass in that venv only because the tests mock the model
imports; a full run in it would fail the `runtime.assert_numpy_abi()` check in
`engines/vision.py` (or worse, silently run the vision stage on CPU with a
modern wheel). **Rebuild the environment with the pinned files before any
recorded run** (Section 2 below).

---

## 1. Decide where you will run

Two supported targets, chosen deliberately:

### 1a. Recommended: Kaggle T4 (for long episodes ≥ 15 min)

- You already have a working Kaggle setup (`kaggle_setup.py`) and the repo's
  GPU-execution audit (thesis §4.9) was written *for* this target.
- Cost per ~45-min episode: roughly diarization + transcription on GPU. T4
  free-tier is 30 h/week.

### 1b. Local M-series Mac (for short clips ≤ 5 min, and for iterating)

- You already have `data/input/global_tv_talkshow.mp4` (280 MB, 45 min) and
  `data/input/atn_news.mp4` (10 MB, 3 min).
- A short clip (5–8 min, `ffmpeg -t`) is the right way to validate a change
  before committing GPU hours.

**Golden rule:** run the *exact same* `run_episode.py` (Section 4) on both. The
only difference is the environment, never the procedure.

---

## 2. Golden environment (do this once per machine)

### 2a. Local (macOS, short clips only)

The repo's `requirements.txt` installs the CUDA `torch`/`onnxruntime-gpu`
builds, which do not exist for macOS. **Local CPU-only runs therefore need a
separate, documented venv**, not the repo `.venv`. Create it fresh:

```bash
cd ~/Developer/multimodal-speaker-indexing
/usr/bin/python3 -m venv .venv-cpu          # NOT the repo .venv
.venv-cpu/bin/pip install -U pip

# Install the CPU-safe subset (no CUDA torch, no onnxruntime-gpu, no chromadb).
# vision uses ONNX CPU via insightface's own dependency; use the CPU torch index.
.venv-cpu/bin/pip install \
  numpy==1.26.4 \
  torch==2.5.1 torchaudio==2.5.1 torchvision==0.20.1 \
  pyannote-audio==3.3.2 \
  faster-whisper==1.1.1 ctranslate2==4.5.0 \
  transformers==4.44.0 huggingface-hub==0.25.2 tokenizers==0.19.1 \
  accelerate==0.33.0 sentencepiece==0.2.0 \
  insightface==0.7.3 opencv-python-headless==4.10.0.84 \
  scikit-learn==1.5.1 scikit-image==0.24.0 \
  scipy==1.13.1 pandas==2.2.2 tqdm pyyaml \
  ffmpeg-python requests python-dotenv \
  peft==0.12.0 protobuf==5.29.3 fsspec rich
```

Then verify:

```bash
.venv-cpu/bin/python - <<'PY'
import numpy, torch, onnxruntime
from runtime import NUMPY_ABI_LOCK
assert numpy.__version__.startswith(NUMPY_ABI_LOCK), numpy.__version__
assert not torch.cuda.is_available()          # CPU run is fine
print("numpy", numpy.__version__, "| torch", torch.__version__)
print("ort providers", onnxruntime.get_available_providers())
PY
```

Notes:
- `insightface==0.7.3` is source-only; on macOS it needs a compiler. If the
  build fails, this is a signal to use Kaggle rather than fight a local wheel.
- `runtime.assert_numpy_abi()` is satisfied (numpy is 1.26.4), so the vision
  module will import. `onnx_providers()` returns CPU on this machine.
- Do **not** run `requirements.txt` verbatim locally (it would try to install
  the CUDA-local torch wheels and fail).

### 2b. Kaggle

Follow `kaggle_setup.py` exactly: clone → pinned install → ORT repair →
cuDNN path → HF token → preflight → imports. The script already performs the
onnxruntime-gpu repair and verifies the CUDA EP is present.

**HF gated model:** `pyannote/speaker-diarization-3.1` requires accepting its
gated license on huggingface.co with the account owning `HF_TOKEN`, and
`HF_TOKEN` must be set as a Kaggle Secret before the kernel starts.

---

## 3. Preflight checklist (run before every recorded run)

`kaggle_setup.preflight()` asserts these on Kaggle. Replicate the checks
locally before a short run:

- [ ] NumPy is `1.26.x` (`runtime.assert_numpy_abi()`).
- [ ] `WHISPER_MODEL` points at a CTranslate2 directory containing `model.bin`
      and the model is multilingual (`faster_whisper` load probe).
- [ ] On GPU: `CUDAExecutionProvider` is present in `ort.get_available_providers()`.
- [ ] `HF_TOKEN` is set and the pyannote gated license is accepted.
- [ ] The face registry directory contains a **frontal photo per expected
      speaker**, and no filename has a trailing space before the extension.
- [ ] `ffmpeg`/`ffprobe` are on `PATH`.
- [ ] Input video exists and is decodable (`ffprobe` duration > 0).

---

## 4. The runner: `scripts/run_episode.py`

A single session-scoped script performs the *whole* recorded run for one
episode:

1. **Resolve** the input video, registry, and output dir.
2. **Stage** (only if missing): `audio.wav`, diarization RTTM/JSON, frames,
   and the ASR word JSON under the scratch dir. Re-running reuses stages, so a
   fusion bug costs seconds, not 30 minutes.
3. **Run** `main.run_pipeline(..., build_rag=True)` with the staged audio.
4. **Verify + record** the outputs into the episode directory:
   - `result.json` / `subtitles.srt` (the pipeline outputs),
   - `run_<id>.json` reproducibility manifest (`evaluation.tracking`),
   - `health.json` — `fusion_health_metrics()` + `assert_no_regression()`,
   - the stage timings from the pipeline log.
5. **Print a one-line verdict**: PASS + cpWER/DER/health, or FAIL + which stage.

When ground truth (RTTM + speaker map) exists for the episode, the runner also
computes DER/JER/cpWER/speaker-name accuracy and writes `metrics.json`.

---

## 5. Operating rhythm that prevents suffering

1. **Validate on a 3–5 min clip first.** Crop the long video:
   ```bash
   ffmpeg -y -ss 00:01:00 -t 300 -i data/input/global_tv_talkshow.mp4 \
     -c copy data/input/global_clip_5min.mp4
   ```
   A clip exercises every stage (diarization, ASR, vision, NER, fusion, RAG)
   in ~10–20 min on CPU or ~2–4 min on a T4.
2. **Only then run the full episode.** The runner's stage cache means the full
   run reuses the validated clip stages where they overlap.
3. **Record every episode** with the runner into `data/output/<episode_id>/`;
   the manifest + health.json + metrics.json are the thesis evidence.
4. **Treat the Kaggle notebook as a thin shell** that calls
   `run_episode.py`. Never hot-patch inside the notebook (that was the
   original "notebook-as-codebase" failure).

---

## 6. What a "clean" result must contain (the thesis evidence)

For the thesis you need, per episode:

| Artifact | Contents | Verifies |
|---|---|---|
| `result.json` | FinalSegment list (start/end/speaker/text/confidence) | pipeline ran end-to-end |
| `subtitles.srt` | speaker-labeled subtitles | human-readable check |
| `health.json` | duplicate rate ≤5%, avg cue ≤200 chars, ≥2 speakers | no fusion regression |
| `run_*.json` | config hash, git commit, package versions, stage timings | reproducibility |
| `metrics.json` | DER/JER/cpWER/WDER/speaker-name accuracy (if GT exists) | the actual thesis numbers |
| `registry/*.jpg` | the face references used | P1 identity evidence |
| `result_bengali.json` | run with the Bengali-specialized CT2 model | the *named-speaker* result |

The pipeline's own `fusion_health_metrics()` + `assert_no_regression()` are the
gate: if health fails, the run is not recorded as valid.

---

## 7. Current machine state (verified 2026-09-03)

- `data/input/global_tv_talkshow.mp4` (280 MB, 45 min) and
  `data/input/atn_news.mp4` (10 MB, 3 min) are present.
- `data/registry/` contains 5 photos: `Abu_Hena_Razzaki.jpg`,
  `Dr_Abdul_Noor_Tushar.jpg`, `MA_Aziz.jpg`, `Matiur_Rahman_Chowdhury.jpg`,
  `Zahed_Ur_Rahman.jpg`. *(Fixed: `Matiur_Rahman_Chowdhury .jpg` had a space
  before the extension, which would have produced a `"Matiur Rahman
  Chowdhury "` speaker label with a trailing space.)*
- `data/output/` contains earlier results (`result.json`, `result_bengali.json`,
  `subtitles*.srt`, `diarization.json`, `frames/`) — **these predate the
  ABI-lock fix and must be re-derived before being used as thesis evidence.**
- Local `yt-dlp` and `ffmpeg` are installed.
- The repo `.venv` is **stale** (Section 0). Build `.venv-cpu` or use Kaggle.
