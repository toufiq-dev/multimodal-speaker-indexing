# Kaggle Run Book — verified end-to-end

**Target video:** https://youtu.be/t2l5UlEBpz0 (RTV "Goll Table" talkshow, 6m49s).

This is the single authoritative guide. Run the cells in order. Every cell below
was checked against the repository: the functions exist, the flags are real, and
the failure modes are documented in §6.

---

## 0. Preconditions (configuration, not bugs)

Set these **before** running anything. They cannot be worked around in code.

| Requirement | Where |
|---|---|
| Accelerator = **GPU T4 x2** | notebook right-hand **Settings** panel |
| **Internet = ON** | notebook right-hand **Settings** panel |
| `HF_TOKEN` Secret **created and attached** | Add-ons → Secrets (below) |
| Gated pyannote licence **accepted** by that account | huggingface.co/pyannote/speaker-diarization-3.1 |

**HF_TOKEN setup:**
1. kaggle.com → your account → **Settings → Secrets → Add a new secret**.
2. Name it exactly `HF_TOKEN`, value = your `hf_...` token.
3. In the notebook: **Add-ons → Secrets → attach** it to this notebook.
4. Sign in to Hugging Face as that token's account and accept the gated licence
   for `pyannote/speaker-diarization-3.1` (otherwise diarization 403s later).
5. If you added it after the session started: **Runtime → Restart session**.

---

## 1. Setup — ONE cell, safe to re-run any number of times

```python
import os, sys, subprocess, importlib

REPO = "/kaggle/working/multimodal-speaker-indexing"
if not os.path.isdir(REPO):
    subprocess.run(["git", "clone",
        "https://github.com/toufiq-dev/multimodal-speaker-indexing.git", REPO],
        check=True)
os.chdir(REPO); sys.path.insert(0, REPO)
subprocess.run(["git", "pull", "--ff-only"], check=True)

# CRITICAL: Python caches imported modules in sys.modules. A `git pull` updates
# the FILE on disk, but the kernel keeps serving the OLD module object — which is
# exactly how "module 'kaggle_setup' has no attribute 'bootstrap'" happens. Drop
# the cached copies so the freshly pulled code is what actually runs.
for _m in list(sys.modules):
    if _m in ("kaggle_setup", "config", "runtime", "models") \
            or _m.startswith(("engines", "evaluation")):
        del sys.modules[_m]
importlib.invalidate_caches()

# Import without triggering the full auto-setup, then drive it explicitly and
# idempotently.
_saved = os.environ.pop("KAGGLE_KERNEL_RUN_TYPE", None)
import kaggle_setup as ks
if _saved:
    os.environ["KAGGLE_KERNEL_RUN_TYPE"] = _saved

print("bootstrap available:", hasattr(ks, "bootstrap"))
ks.bootstrap()
```

**How to read the result.** `bootstrap()` does exactly one stage and stops with a
`NEXT ACTION` banner:

- **"Pinning NumPy … NEXT ACTION: restart"** → restart, re-run this same cell.
- **"Installing dependencies … NEXT ACTION: restart"** → restart, re-run.
- **"HF_TOKEN is not set"** → do §0 steps, restart, re-run.
- **"✅ bootstrap complete"** → environment ready; continue to §2.

It never redoes finished work, so repeating the cell is always safe.

---

## 2. Diagnostic — does ASR work on Kaggle?

This is the decisive test for the repetition-loop failure seen on macOS.

```python
import subprocess, os
url = "https://youtu.be/t2l5UlEBpz0"
out = "/kaggle/working/data/inputs/rtv.mp4"
os.makedirs("/kaggle/working/data/inputs", exist_ok=True)
if not os.path.exists(out):
    subprocess.run(["yt-dlp", "-f", "best[height<=720]", "-o", out, url], check=True)

subprocess.run(["ffmpeg", "-y", "-ss", "5", "-t", "30", "-i", out,
                "-vn", "-ar", "16000", "-ac", "1", "/kaggle/working/t30.wav"],
               check=True)

from faster_whisper import WhisperModel
import config as cfg
model = WhisperModel(cfg.config.WHISPER_MODEL, device="cuda", compute_type="float16")
segs, info = model.transcribe("/kaggle/working/t30.wav", language="bn",
                              condition_on_previous_text=False, temperature=0.0)
print("lang:", info.language, "prob:", round(info.language_probability, 3))
for s in list(segs)[:5]:
    print("TEXT:", s.text[:150])
```

- **Real Bengali text** → the macOS failure was platform-local; proceed to §3.
- **Same character repeated** → a genuine model/content issue; stop and report.

---

## 3. Face registry for this episode

Upload photos as a **private Kaggle Dataset** (files do not persist in
`/kaggle/working`): Datasets → New Dataset → name it `msi-registry` → upload →
Private. Then **Add Input** it in the notebook and copy:

```python
import os, shutil
src, dst = "/kaggle/input/msi-registry", "/kaggle/working/registry/rtv_goll_table"
os.makedirs(dst, exist_ok=True)
copied = [f for f in os.listdir(src)
          if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
for f in copied:
    shutil.copy(os.path.join(src, f), os.path.join(dst, f))
print("registry:", copied)
assert copied, "no images found — check the dataset mount path"
```

Skipping this is allowed: the pipeline falls back to face clusters + NER, just
with fewer real names.

---

## 4. Run the full pipeline

```python
import os
os.chdir("/kaggle/working/multimodal-speaker-indexing")
!python scripts/run_episode.py \
    /kaggle/working/data/inputs/rtv.mp4 \
    --registry /kaggle/working/registry/rtv_goll_table \
    --id rtv_goll_table_ep \
    --output-dir /kaggle/working/output/rtv_goll_table_ep
```

Only a run whose fusion-health checks pass is recorded. Expect `✔ recorded:`
plus `dup_rate`, `avg_cue`, `speakers`. Add `--no-rag` to skip the retrieval
stage while testing.

For a cheap first pass, make a clip and run that instead:

```python
!ffmpeg -y -ss 60 -t 300 -i /kaggle/working/data/inputs/rtv.mp4 \
    -c copy /kaggle/working/data/inputs/rtv_clip5m.mp4
```

---

## 5. Inspect the result and export a transcript

```python
import json
from pathlib import Path
out = Path("/kaggle/working/output/rtv_goll_table_ep")
print((out / "health.json").read_text())
res = json.loads((out / "result.json").read_text())
for s in res[:8]:
    print(f"[{s['start']:7.1f}-{s['end']:7.1f}] {s['speaker']}: {s['text'][:70]}")
print("segments:", len(res), "| speakers:", sorted({s['speaker'] for s in res}))
```

```python
!python scripts/export_transcript.py \
    /kaggle/working/output/rtv_goll_table_ep/result.json \
    --out /kaggle/working/output/rtv_goll_table_ep/transcript \
    --title "RTV Goll Table transcript"
```

Download everything from the notebook's **Output** tab (or commit the notebook —
see §7 for intermittent power).

---

## 6. Troubleshooting (every error seen so far, and its cause)

| Error | Cause | Fix |
|---|---|---|
| `AttributeError: module 'kaggle_setup' has no attribute 'bootstrap'` | kernel served a **cached module** from before `git pull` | the §1 cell purges `sys.modules`; if it persists, Runtime → Restart, re-run §1 |
| `RecursionError` inside `numpy/_core` → `faster_whisper/feature_extractor.py` | kernel imported NumPy 2.x before pip pinned 1.26.x | restart, re-run §1 (bootstrap pins first, then stops) |
| `AssertionError: HF_TOKEN unset` | token not created **or not attached**, or set after session start | §0 steps, then restart and re-run §1 |
| `401/403` from pyannote at the diarization stage | gated licence not accepted by the token's account | accept it at huggingface.co/pyannote/speaker-diarization-3.1 |
| `ORT is CPU-only` | CPU `onnxruntime` wheel shadowed `onnxruntime-gpu` | bootstrap runs `fix_onnxruntime_conflict()`; if it persists, restart and re-run §1 |
| `run_episode.py` prints `✗ episode … failed` and a `health.json` | genuine pipeline failure | send the message + `health.json` |
| Session ends mid-run / power cut | session limit or outage | see §7 |

---

## 7. Intermittent power / session limits

- **Run on laptop battery** and **tether to your phone**, so a mains cut does not
  drop the browser session.
- For long jobs prefer **Save Version** (committed run): it executes on Kaggle's
  servers, so you can close the laptop; outputs are saved with the notebook
  version. A committed run starts fresh, so §1 re-runs — which is why
  `bootstrap()` is idempotent and stops for a restart only when it must.
- The runner writes per-video scratch, so re-running after a cut reuses the
  extracted media rather than starting from zero.

---

## 8. Active-speaker detection (the tracked high-FPS pass)

Identity is attributed to the face whose **mouth is moving**, not to whichever
visible face matches the registry best. In multi-camera footage a still,
well-framed listener can match better than the speaker — on the RTV episode the
speaker's face cleared the similarity threshold once in 616 detections while a
silent co-panelist cleared it 55 times — so the old pooled vote named the
speaker's turns after a non-speaker.

How it works now:

1. Frames are extracted at `VISION_ASD_FPS` (default **8**) instead of 1 FPS,
   because mouth motion is meaningless across one-second gaps.
2. `engines/vision.py` follows each face across frames (IoU) and measures the
   mouth region against **that track's own** previous frame.
3. Each track's mean embedding is matched to the registry, so a face that misses
   the threshold frame-by-frame can still be named as a track.
4. `engines/fusion.py` resolves identity **per turn**, from the track with the
   most mouth motion (`engines/active_speaker.py`), rather than pooling every
   face visible across all of a speaker's turns.

Consequences to expect:

- **More frames**: ~3.3k for a 6m49s clip (vs ~410 at 1 FPS), so vision takes a
  little longer and the scratch cache key changes to `frames_8fps`.
- **Frames are cached per rate**, so a rate change re-extracts rather than
  reusing mismatched frames.
- Override the rate with `--vision-fps N` on `scripts/run_episode.py`.

Settings that matter for calibration: `ASD_MIN_MOUTH_MOTION` (0.0 accepts the
most-moving track unconditionally) and `ASD_TRACK_IOU` (0.3).
