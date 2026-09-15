#!/usr/bin/env python3
"""One-command Kaggle runner: fix the environment, then run the pipeline.

Notebook usage (a single shell line — immune to copy/paste indentation):

    !cd /kaggle/working/multimodal-speaker-indexing && git pull -q && \
        python scripts/kaggle_run.py --id rtv_goll_table_ep_v2

What it does, in order:
  1. reports NumPy / CUDA / ONNX Runtime and the CT2 model state,
  2. restores HF_TOKEN and WHISPER_MODEL into the environment (a kernel
     restart clears os.environ, which made subprocess runs 401),
  3. re-converts the CT2 model if ``/tmp`` was wiped by a session change,
  4. re-creates the face registry from an attached Kaggle dataset if empty,
  5. finds the episode video under ``/kaggle/input``,
  6. runs ``scripts/run_episode.py`` with the given episode id.

It is safe to re-run: every step is skipped when already satisfied.
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)
sys.path.insert(0, str(REPO))

# Importing kaggle_setup on Kaggle triggers its full main(); we only want the
# helper functions here, so hide the Kaggle marker during the import.
_saved = os.environ.pop("KAGGLE_KERNEL_RUN_TYPE", None)
import kaggle_setup as ks  # noqa: E402
if _saved:
    os.environ["KAGGLE_KERNEL_RUN_TYPE"] = _saved


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def find_video(explicit: str | None = None) -> str | None:
    if explicit:
        return explicit if os.path.exists(explicit) else None
    mp4s = glob.glob("/kaggle/input/**/*.mp4", recursive=True)
    if not mp4s:
        return None
    return next((c for c in mp4s if "rtv" in c.lower()), mp4s[0])


def ensure_registry(dst: str) -> list:
    """Populate the registry from an attached dataset when it is empty."""
    os.makedirs(dst, exist_ok=True)
    have = [f for f in os.listdir(dst) if f.lower().endswith(IMAGE_EXTS)]
    if have:
        return have
    for p in glob.glob("/kaggle/input/**/*", recursive=True):
        if p.lower().endswith(IMAGE_EXTS):
            shutil.copy(p, os.path.join(dst, os.path.basename(p)))
    return [f for f in os.listdir(dst) if f.lower().endswith(IMAGE_EXTS)]


def report() -> None:
    print("=== ENVIRONMENT ===")
    import numpy
    print("numpy        :", numpy.__version__)
    try:
        import torch
        print("cuda         :", torch.cuda.is_available())
    except Exception as exc:
        print("cuda         : ERROR", exc)
    try:
        import onnxruntime as ort
        print("ort          :", ort.get_available_providers())
    except Exception as exc:
        print("ort          : ERROR", exc)


def export_cudnn_path() -> str | None:
    """Put torch's bundled cuDNN 9 on LD_LIBRARY_PATH for child processes.

    CTranslate2 ``dlopen``s cuDNN, and glibc fixes the library search path at
    process start — so this must be in the environment *before* the pipeline
    subprocess launches, not set inside it.
    """
    try:
        from runtime import cudnn_library_dir
    except Exception as exc:
        print("cudnn path   : ERROR", exc)
        return None
    lib = cudnn_library_dir()
    if not lib:
        print("cudnn path   : not found in the torch wheel")
        return None
    current = os.environ.get("LD_LIBRARY_PATH", "")
    if lib not in current.split(os.pathsep):
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
            p for p in (lib, current) if p)
    print("cudnn path   :", lib)
    return lib


def summarize(out_dir: str) -> None:
    """Print the health block and the per-speaker table for an episode."""
    import json

    result_path = os.path.join(out_dir, "result.json")
    if not os.path.exists(result_path):
        print("no result.json at", result_path)
        return
    with open(result_path, encoding="utf-8") as fh:
        res = json.load(fh)

    health_path = os.path.join(out_dir, "health.json")
    if os.path.exists(health_path):
        with open(health_path, encoding="utf-8") as fh:
            print(fh.read())

    durations: dict = {}
    for s in res:
        durations[s["speaker"]] = durations.get(s["speaker"], 0.0) + (s["end"] - s["start"])

    print(f"\nsegments: {len(res)}")
    print(f"{'speaker':<36}{'segs':>5}{'secs':>8}{'mean_conf':>11}")
    for spk, dur in sorted(durations.items(), key=lambda kv: -kv[1]):
        segs = [s for s in res if s["speaker"] == spk]
        mean_conf = sum(s["confidence"] for s in segs) / len(segs)
        print(f"{spk:<36}{len(segs):>5}{dur:>8.1f}{mean_conf:>11.3f}")

    print("\nfirst 12 segments:")
    for s in res[:12]:
        print(f"[{s['start']:6.1f}-{s['end']:6.1f}] "
              f"{s['speaker'][:30]:<30} {s['text'][:60]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--id", default="episode")
    ap.add_argument("--registry", default="/kaggle/working/registry/rtv_goll_table")
    ap.add_argument("--video", default=None)
    ap.add_argument("--no-rag", action="store_true")
    ap.add_argument("--report-only", action="store_true",
                    help="Skip the run; just summarise an existing output dir.")
    args = ap.parse_args()

    out_dir = f"/kaggle/working/output/{args.id}"
    if args.report_only:
        summarize(out_dir)
        return 0

    report()
    export_cudnn_path()

    ct2_ok = os.path.exists(os.path.join(ks.CT2_DIR, "model.bin"))
    print("ct2 model    :", ks.CT2_DIR, "->", ct2_ok)

    env = ks.prepare_env()
    print("HF_TOKEN     :", env["HF_TOKEN"])
    print("WHISPER_MODEL:", env["WHISPER_MODEL"])
    if not env["HF_TOKEN"]:
        print("\nHF_TOKEN is missing. Add it under Add-ons -> Secrets, attach it "
              "to this notebook, restart the session, and re-run.")
        return 2

    if not ct2_ok:
        print("\nCT2 model missing (a new session clears /tmp) -> converting...")
        ks.convert_asr_model()
        print("WHISPER_MODEL:", ks.prepare_env()["WHISPER_MODEL"])

    registry = ensure_registry(args.registry)
    print("registry     :", registry or "EMPTY (names will fall back to clusters)")
    assert registry, (
        "no face photos found. Attach the msi-registry dataset (Add Input), "
        "or pass --registry with a folder that contains them."
    )

    video = find_video(args.video)
    print("video        :", video)
    if not video:
        print("No .mp4 under /kaggle/input — attach the video dataset.")
        return 2

    cmd = [sys.executable, "scripts/run_episode.py", video,
           "--registry", args.registry,
           "--id", args.id,
           "--output-dir", f"/kaggle/working/output/{args.id}"]
    if args.no_rag:
        cmd.append("--no-rag")

    print("\n=== RUN ===")
    print(" ".join(cmd))
    rc = subprocess.call(cmd)
    print("\n=== PIPELINE EXIT:", rc, "===")
    if rc == 0:
        print(f"outputs: {out_dir}/")
        print("\n=== RESULT ===")
        summarize(out_dir)
    else:
        print("Run failed. Re-run with the same command after fixing the error above,")
        print("or inspect an existing run with:  --report-only")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
