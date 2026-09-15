# Kaggle Run Book — Full End-to-End Test (RTV "Goll Table")

**Target video:** https://youtu.be/t2l5UlEBpz0 — "আব্দুন নূর তুষারের সঙ্গে কেন
তর্কে জড়ালেন ব্যারিস্টার শাহরিয়ার?" (RTV, Goll Table talkshow).

**What you will run, in order:**
1. Kaggle settings + HF token + face-image dataset upload.
2. `kaggle_setup` (clones the repo, pinned install, preflight).
3. Skip the old Jamuna video; download the RTV video with `yt-dlp`.
4. Build the face registry for this episode from the uploaded dataset.
5. Run the **5-minute clip** first (validation), then the **full episode**.
6. Read the recorded outputs + health check.

---

## 0. Why this sequence (verified against the repo)

- `kaggle_setup.py` ends with `if __name__ == "__main__" or os.environ.get("KAGGLE_KERNEL_RUN_TYPE"): main()`, so **on Kaggle it auto-runs** the full setup (clone → pinned install → ORT repair → cuDNN → HF token → dirs → **downloads the OLD Jamuna video** → CT2 convert → preflight → imports). We **do not** want its built-in `download_dataset()` (it fetches a 53-min Jamuna show we don't need). So we call the setup functions directly and **skip** the auto-download, or we let it download and ignore it.
- `scripts/run_episode.py` (already in the repo) drives the real pipeline with per-video scratch and only records runs whose fusion-health checks pass. We call it via `python scripts/run_episode.py ...`.
- `engines/vision.py` reads the registry from `config.DATA_REGISTRY_DIR` — the runner sets that from `--registry`. So we pass `--registry` pointing at a directory with our photos.

---

## 1. Kaggle session settings (do these in the UI first)

1. Create a **new Notebook** (or use your existing one).
2. In the right-hand **Settings** panel:
   - **Accelerator:** `GPU T4 x2` (or `GPU P100`).
   - **Internet:** ON (needed to clone, `pip install`, `yt-dlp`).
   - **Language:** Python.
3. **Add your Hugging Face token as a Secret** (left sidebar 🔒 → "Add-New Secret"):
   - Name: `HF_TOKEN`
   - Value: your `hf_...` token (the account that accepted the gated
     `pyannote/speaker-diarization-3.1` license).
4. **Upload a Dataset containing the face registry photos** (see §2).

---

## 2. Do I need to upload the registry images? How?

**Yes — you need the face photos available in the Kaggle session** if you want
registry-based (P1) identity resolution. Kaggle `/kaggle/working` is wiped each
session, so you cannot persist files there. The supported way is a **Kaggle
Dataset** mounted read-only at `/kaggle/input/<dataset-name>/`.

### 2a. Create the Kaggle Dataset (UI)
1. Go to **kaggle.com → Datasets → New Dataset**.
2. Give it a name, e.g. `msi-registry` (the slug becomes the mount path).
3. Upload the photos. **Recommended layout** (so you can add per-episode
   registries later):
   ```
   msi-registry/
   ├── README.md                  (provenance/consent, see §2c)
   ├── Abu_Hena_Razzaki.jpg
   ├── Dr_Abdul_Noor_Tushar.jpg   (this RTV episode's known panelist)
   ├── MA_Aziz.jpg
   ├── Matiur_Rahman_Chowdhury.jpg
   └── Zahed_Ur_Rahman.jpg
   ```
   Keep filenames `First_Last.jpg` (no spaces, underscores → spaces become the
   speaker label).
4. **Make it Private** (recommended) so real faces are not public.
5. Create. In the notebook, click **"Add Input"** (top-right) → search your
   dataset → it mounts at `/kaggle/input/msi-registry`.

### 2b. What the notebook will do with it
Cell 6 copies only the `.jpg` files you need for THIS episode into a working
registry dir (e.g. `/kaggle/working/registry/rtv_goll_table/`), then the runner
is pointed at that dir with `--registry`.

### 2c. Provenance/consent (do this now, it protects you)
In the dataset's `README.md`, record for each image: source URL, license/terms,
and whether you have permission. If any photo is a third-party image you do not
hold rights to, **do not** make the dataset public, and consider replacing it
with a photo you own or a synthetic placeholder. The thesis should state that
registry photos are private reference data.

---

## 3. The notebook cells (run in order)

> Paste each block into its own cell. Run top-to-bottom. Do not skip cells.
> If a cell fails, **copy the full error text** and send it to me — do not
> edit code blindly.

---

### CELL 0 — Environment header (run first)

```python
# ============================================================
# 0. ENV HEADER — run first
# ============================================================
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import sys, subprocess, warnings, shutil
warnings.filterwarnings("ignore")

print("Python:", sys.version.split()[0])
print("HF_HUB_ENABLE_HF_TRANSFER =", os.environ.get("HF_HUB_ENABLE_HF_TRANSFER"))
```

---

### CELL 1 — Run the hardened setup (clone + pinned install + preflight)

```python
# ============================================================
# 1. kaggle_setup — clones the repo, installs the PINNED stack,
#    repairs onnxruntime-gpu, converts the Bengali CT2 ASR model,
#    and runs the preflight that fails loudly on any silent issue.
#    On Kaggle, importing it AUTO-RUNS main().
# ============================================================
import kaggle_setup

# kaggle_setup.main() already ran on import (KAGGLE_KERNEL_RUN_TYPE is set).
# It downloaded the OLD Jamuna video into /kaggle/working/data/inputs/.
# We do NOT need it — we download our own target video next.
print("kaggle_setup finished. cwd =", os.getcwd())
```

If this cell takes a long time (it pip-installs torch/insightface/etc. and
converts the CT2 model), that is normal on the first run of a fresh kernel.

---

### CELL 2 — Verify the environment (should all print ✅)

```python
# ============================================================
# 2. VERIFY the golden environment
# ============================================================
import numpy, torch, onnxruntime
from runtime import NUMPY_ABI_LOCK

print("numpy:", numpy.__version__, "| need", NUMPY_ABI_LOCK + ".x")
assert numpy.__version__.startswith(NUMPY_ABI_LOCK), "numpy ABI drift"
print("cuda available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "no GPU — enable T4 in Settings"
print("gpu:", torch.cuda.get_device_name(0))
print("ort providers:", onnxruntime.get_available_providers())
assert "CUDAExecutionProvider" in onnxruntime.get_available_providers(), "ORT CPU-only"
print("HF_TOKEN set:", bool(os.environ.get("HF_TOKEN")))
print("✅ environment OK")
```

---

### CELL 3 — Download the RTV target video with yt-dlp

```python
# ============================================================
# 3. DOWNLOAD the RTV "Goll Table" video (our target)
# ============================================================
video_url = "https://youtu.be/t2l5UlEBpz0"
out_dir = "/kaggle/working/data/inputs"
os.makedirs(out_dir, exist_ok=True)
video_path = os.path.join(out_dir, "rtv_goll_table.mp4")

if os.path.exists(video_path):
    print("Video already present:", video_path)
else:
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[height<=720]+bestaudio/best[height<=720]",
        "--merge-output-format", "mp4",
        "-o", video_path,
        video_url,
    ]
    print("Downloading (this can take a few minutes)...")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(video_path):
        print("STDERR:", r.stderr[-2000:])
        raise RuntimeError("yt-dlp download failed")
print("✅ video:", video_path, round(os.path.getsize(video_path)/1e6, 1), "MB")

# duration
dur = subprocess.run(
    ["ffprobe","-v","quiet","-show_entries","format=duration","-of","csv=p=0", video_path],
    capture_output=True, text=True)
print("duration(s):", float(dur.stdout.strip()))
```

---

### CELL 4 — Prepare the face registry for THIS episode

```python
# ============================================================
# 4. FACE REGISTRY — copy the photos we need from the uploaded
#    Kaggle Dataset (/kaggle/input/msi-registry) into a working
#    registry dir for this episode.
# ============================================================
# The dataset you added via "Add Input" mounts here. Adjust the slug
# if you named it differently.
dataset_dir = "/kaggle/input/msi-registry"
registry_dir = "/kaggle/working/registry/rtv_goll_table"
os.makedirs(registry_dir, exist_ok=True)

if os.path.isdir(dataset_dir):
    copied = []
    for f in sorted(os.listdir(dataset_dir)):
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            shutil.copy(os.path.join(dataset_dir, f), os.path.join(registry_dir, f))
            copied.append(f)
    print("Copied from dataset:", copied)
else:
    print("⚠️ Dataset not found at", dataset_dir, "— put photos directly in", registry_dir)

# Show what the registry now contains
print("Registry photos:")
for f in sorted(os.listdir(registry_dir)):
    print("  -", f)
assert any(f.lower().endswith((".jpg",".jpeg",".png",".webp")) for f in os.listdir(registry_dir)), \
    "no registry photos — P1 face identification disabled"
```

If you do **not** have a Kaggle Dataset, you can instead upload photos via the
notebook's **"Add Input" → "New Dataset"** flow, or (for a quick test) skip the
registry entirely and rely on face clusters + NER.

---

### CELL 5 — Make a 5-minute validation clip

```python
# ============================================================
# 5. 5-MINUTE CLIP — validate the full pipeline cheaply first
# ============================================================
clip_path = "/kaggle/working/data/inputs/rtv_goll_table_clip5m.mp4"
if not os.path.exists(clip_path):
    # 5 minutes starting at 60s (adjust if the intro is elsewhere)
    r = subprocess.run([
        "ffmpeg", "-y", "-ss", "00:01:00", "-t", "300",
        "-i", video_path, "-c", "copy", clip_path,
    ], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-1500:])
        raise RuntimeError("ffmpeg clip failed")
print("✅ clip:", clip_path, round(os.path.getsize(clip_path)/1e6,1), "MB")
```

---

### CELL 6 — Run the pipeline on the 5-minute clip (first validation)

```python
# ============================================================
# 6. RUN the runner on the 5-minute clip.
#    Only records a result if fusion-health checks pass.
# ============================================================
import sys, os
sys.path.insert(0, "/kaggle/working/multimodal-speaker-indexing")
os.chdir("/kaggle/working/multimodal-speaker-indexing")

!python scripts/run_episode.py \
    /kaggle/working/data/inputs/rtv_goll_table_clip5m.mp4 \
    --registry /kaggle/working/registry/rtv_goll_table \
    --id rtv_goll_table_5m \
    --output-dir /kaggle/working/output/rtv_goll_table_5m
```

**Stop here and check the output.** You should see a `✔ recorded:` line with
`dup_rate=0.0`, `avg_cue=...`, `speakers=...`. If it fails, **send me the full
error** before running the full episode.

---

### CELL 7 — Inspect the clip's result + health (verify it looks sane)

```python
# ============================================================
# 7. INSPECT the clip result + health
# ============================================================
import json
from pathlib import Path

out_dir = Path("/kaggle/working/output/rtv_goll_table_5m")
print("--- health.json ---")
print((out_dir / "health.json").read_text() if (out_dir/"health.json").exists() else "NO health.json")
print("\n--- first 8 result segments ---")
res = json.loads((out_dir/"result.json").read_text())
for s in res[:8]:
    print(f"[{s['start']:7.1f}-{s['end']:7.1f}] {s['speaker']:<24} conf={s['confidence']:.2f}  {s['text'][:60]}")
print("\nTotal segments:", len(res))
print("Speakers:", sorted({s['speaker'] for s in res}))
```

If the speakers are all `face_cluster_N`/`Speaker_N` and no real names appear,
that is **expected** on a 5-min clip if the intro (host anchor) falls outside
the clip window, or if the Bengali ASR isn't producing proper script. Do not
panic — the full episode is the real test.

---

### CELL 8 — Run the FULL episode

```python
# ============================================================
# 8. RUN the FULL RTV episode (the thesis-grade run)
#    Expect this to take a while on a T4: diarization + ASR on
#    a ~30-60 min video can be 20-60+ minutes.
# ============================================================
import sys, os
sys.path.insert(0, "/kaggle/working/multimodal-speaker-indexing")
os.chdir("/kaggle/working/multimodal-speaker-indexing")

!python scripts/run_episode.py \
    /kaggle/working/data/inputs/rtv_goll_table.mp4 \
    --registry /kaggle/working/registry/rtv_goll_table \
    --id rtv_goll_table_ep \
    --output-dir /kaggle/working/output/rtv_goll_table_ep
```

**Success criteria (thesis-grade):**
- Proper Bengali script in the transcript (not Devanagari/romanized).
- The host self-intro anchor (P2) resolves the host by name.
- The registry (P1) resolves a known panelist (Tushar) if he appears on-camera.
- NER (P3) resolves ≥1 additional name.
- `health.json` shows `dup_rate ≤ 0.05`, `avg_cue ≤ 200`, `≥ 2 speakers`.

---

### CELL 9 — Inspect the full result + health + manifest

```python
# ============================================================
# 9. INSPECT the full run's result, health, and reproducibility
# ============================================================
import json, glob
from pathlib import Path

out_dir = Path("/kaggle/working/output/rtv_goll_table_ep")
for name in ("health.json", "metrics.json"):
    p = out_dir / name
    print(f"--- {name} ---")
    print(p.read_text() if p.exists() else "(absent)")

print("\n--- result.json summary ---")
res = json.loads((out_dir/"result.json").read_text())
print("Total segments:", len(res))
from collections import Counter
print("Speaker time (s):", dict(Counter(round(s['end']-s['start'],1) for s in res)))
# per-speaker segment counts + rough duration
spk_dur = {}
for s in res:
    spk_dur[s['speaker']] = spk_dur.get(s['speaker'], 0) + (s['end'] - s['start'])
for spk, d in sorted(spk_dur.items(), key=lambda kv: -kv[1]):
    print(f"  {spk:<28} {d:8.1f}s")

print("\n--- reproducibility manifest ---")
mf = sorted(glob.glob(str(out_dir / "run_*.json")))
print(mf[-1] if mf else "no manifest")
if mf:
    m = json.loads(Path(mf[-1]).read_text())
    print("commit:", m.get("git", {}).get("commit"))
    print("numpy :", m.get("packages", {}).get("numpy"))
    print("torch :", m.get("packages", {}).get("torch"))
```

---

### CELL 10 — (Optional) Download the results to your machine

Kaggle auto-commits notebook output. To download locally, use the **Output**
tab in the notebook (top-right) → "Download", or from the Kaggle UI on the
notebook page. The files you want are under:
`/kaggle/working/output/rtv_goll_table_ep/` (`result.json`, `subtitles.srt`,
`health.json`, `run_*.json`, `rag_index/`).

---

## 4. What to send me if something fails

When a cell errors, paste me:
1. **Which cell number** failed.
2. The **full traceback/error text** (scroll to the bottom; include the last
   `----> ` line and the exception type + message).
3. Any ✅/❌ lines printed just before the failure.
4. The `health.json` content if the runner printed a `✗ episode ... failed`
   line.

Do **not** edit code blindly — many failures are environment/version issues the
preflight is designed to catch, and the fix is usually in the setup, not the
pipeline.

---

## 5. Recovery: `RecursionError` in `convert_asr_model()` (NumPy mismatch)

**Symptom:** `import kaggle_setup` crashes with
`RecursionError: maximum recursion depth exceeded` inside
`numpy/_core/...` → `faster_whisper/feature_extractor.py`.

**Cause:** the Kaggle base image ships **NumPy 2.x**, and the running kernel
imported it *before* `pip install numpy==1.26.4` downgraded the file on disk.
Pip changes disk, but Python keeps the already-imported module in `sys.modules`,
so `faster_whisper`'s Mel-filterbank math runs on the wrong NumPy and recurses
inside NumPy's dtype `repr`. The repo pins NumPy 1.26.x for the
`insightface`/`onnxruntime` ABI.

**Fix — restart the kernel, then resume without reinstalling:**

1. Kaggle menu: **Run → Restart session** (or the ⟳ button).
2. Run this cell (packages are already on disk, so this is cheap):

```python
import os, sys
repo = "/kaggle/working/multimodal-speaker-indexing"
os.chdir(repo); sys.path.insert(0, repo)

# numpy must be 1.26.x in the *restarted* process
import numpy; print("numpy in memory:", numpy.__version__)
assert numpy.__version__.startswith("1.26"), \
    "run: !pip install numpy==1.26.4  then restart again"

# Import without triggering the full ~20-min auto-setup, then finish it.
_saved = os.environ.pop("KAGGLE_KERNEL_RUN_TYPE", None)
import kaggle_setup as ks
if _saved:
    os.environ["KAGGLE_KERNEL_RUN_TYPE"] = _saved

ks.finish_setup()   # convert CT2 -> preflight -> verify imports -> verify registry
```

3. If the assert says NumPy is still 2.x, run `!pip install numpy==1.26.4` and
   **restart again**, then re-run the cell.

`convert_asr_model()` now asserts the ABI itself, so any recurrence raises a
clear "restart the kernel" message instead of a RecursionError.
