#!/usr/bin/env python3
"""P1: measure ASR recall against the ground truth, one decode config at a time.

Isolates the ASR stage. Diarization, vision and fusion are NOT run, so each
config costs one Whisper pass over the episode instead of a full pipeline, and
the resulting WER/CER move can be attributed to the decode change alone.

Measured against data/gt/rtv_goll_table/transcript.json, whose text is a
LOWER BOUND on true error (it was produced by correcting v9's own output).
The comparison between configs is still sound: the same bias applies to all.

Kaggle:
    !cd /kaggle/working/multimodal-speaker-indexing && git pull -q && \
        python scripts/ablate_asr.py --video /kaggle/input/.../rtv_goll_table.mp4

macOS is refused: Apple-Silicon CPU decode produces a degenerate repetition
loop that looks like valid Bangla (docs/PLATFORM_ASR_FINDING.md).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)
sys.path.insert(0, str(REPO))

from config import config                                   # noqa: E402
from evaluation.metrics import (_edit_distance, _normalize_words,  # noqa: E402
                                _norm_chars, ngram_repetition)

GT = REPO / "data" / "gt" / "rtv_goll_table" / "transcript.json"
OUT_DIR = REPO / "data" / "gt" / "rtv_goll_table" / "asr_ablation"

# Named configs. 'baseline' reproduces the v9 decode exactly; every other row
# changes ONE thing relative to it, except the explicitly combined rows.
CONFIGS = [
    {"name": "baseline_v9",      "quant": "int8",    "vad": True,  "cond": True},
    {"name": "novad",            "quant": "int8",    "vad": False, "cond": True},
    {"name": "nocond",           "quant": "int8",    "vad": True,  "cond": False},
    {"name": "novad_nocond",     "quant": "int8",    "vad": False, "cond": False},
    {"name": "fp16",             "quant": "float16", "vad": True,  "cond": True},
    {"name": "fp16_novad_nocond","quant": "float16", "vad": False, "cond": False},
]


def _wer(ref: str, hyp: str) -> float:
    r, h = _normalize_words(ref), _normalize_words(hyp)
    return _edit_distance(r, h) / len(r) if r else float("inf")


def _cer(ref: str, hyp: str) -> float:
    r, h = _norm_chars(ref), _norm_chars(hyp)
    return _edit_distance(r, h) / len(r) if r else float("inf")


def repetition_stats(text: str) -> dict:
    """Degenerate-loop share; see evaluation.metrics.ngram_repetition."""
    return {"ngram_repeat_ratio": ngram_repetition(text)["ngram_repeat_ratio"]}


def ensure_ct2(quant: str, scratch: Path) -> str:
    """Convert the checkpoint at the requested quantization (cached)."""
    out = scratch / f"bengaliAI_ct2_{quant}"
    if (out / "model.bin").exists():
        print(f"[ct2] reusing {out}")
        return str(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[ct2] converting at {quant} -> {out}")
    subprocess.run(
        [sys.executable, "-m", "ctranslate2.converters.transformers",
         "--model", os.getenv("WHISPER_MODEL_SRC",
                              "bengaliAI/tugstugi_bengaliai-asr_whisper-medium"),
         "--output_dir", str(out), "--quantization", quant,
         "--copy_files", "tokenizer.json", "preprocessor_config.json", "--force"],
        check=True)
    return str(out)


def print_summary(rows: list) -> None:
    """Render the comparison table. Kept separate so it is unit-testable:
    a formatting crash here would land after every Whisper pass has run."""
    ok = [r for r in rows if "error" not in r]
    base = next((r for r in ok if r["name"] == "baseline_v9"), None)
    hdr = (f"{'config':<20}{'WER':>8}{'CER':>8}{'words':>7}{'recall':>8}"
           f"{'rep':>7}{'sec':>7}")
    print("\n" + hdr + ("    dWER" if base else ""))
    print("-" * (len(hdr) + (8 if base else 0)))
    for r in ok:
        line = (f"{r['name']:<20}{r['wer']:>8.4f}{r['cer']:>8.4f}"
                f"{r['hyp_words']:>7}{r['word_recall_proxy']:>8.3f}"
                f"{r['ngram_repeat_ratio']:>7.3f}{r['elapsed_sec']:>7.1f}")
        if base:
            line += f"  {r['wer'] - base['wer']:+.4f}"
        print(line)
    for r in rows:
        if "error" in r:
            print(f"{r['name']:<20} FAILED: {r['error'][:80]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--only", nargs="*", default=None,
                    help="run only these config names")
    ap.add_argument("--scratch", default="/tmp/msi_scratch/models")
    ap.add_argument("--force-cpu", action="store_true",
                    help="override the macOS/CPU refusal (results are invalid)")
    a = ap.parse_args()

    import torch
    if not torch.cuda.is_available() and not a.force_cpu:
        print("REFUSING: no CUDA. Apple-Silicon/CPU decode yields a degenerate\n"
              "repetition loop that still reports language=bn, so the numbers\n"
              "would look plausible and be wrong. See docs/PLATFORM_ASR_FINDING.md.\n"
              "Run this on Kaggle (GPU T4), or pass --force-cpu to override.",
              file=sys.stderr)
        return 2
    print(f"platform={platform.platform()} cuda={torch.cuda.is_available()}")

    from engines.media import extract_audio
    from engines.transcription import transcribe_audio, _load_model

    ref_text = " ".join(t["text"] for t in
                        sorted(json.loads(GT.read_text(encoding="utf-8")),
                               key=lambda x: x["start"]))
    ref_words = len(_normalize_words(ref_text))
    print(f"[gt] reference words = {ref_words}")

    audio = extract_audio(a.video)
    print(f"[audio] {audio}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, by_quant = [], {}
    todo = [c for c in CONFIGS if not a.only or c["name"] in a.only]

    for cfg in todo:
        print(f"\n=== {cfg['name']} ===")
        model_dir = by_quant.get(cfg["quant"]) or ensure_ct2(
            cfg["quant"], Path(a.scratch))
        by_quant[cfg["quant"]] = model_dir

        config.WHISPER_MODEL = model_dir
        config.WHISPER_VAD_FILTER = cfg["vad"]
        config.WHISPER_CONDITION_ON_PREV = cfg["cond"]
        try:
            model = _load_model()
            t0 = time.time()
            words, text = transcribe_audio(audio, model=model)
            elapsed = time.time() - t0
        except Exception as exc:
            print(f"[{cfg['name']}] FAILED: {exc}")
            rows.append({**cfg, "error": str(exc)})
            continue
        finally:
            try:
                del model
                import torch as _t; _t.cuda.empty_cache()
            except Exception:
                pass

        hyp_words = len(_normalize_words(text))
        row = {**cfg,
               "wer": round(_wer(ref_text, text), 4),
               "cer": round(_cer(ref_text, text), 4),
               "hyp_words": hyp_words,
               "gt_words": ref_words,
               "word_recall_proxy": round(hyp_words / ref_words, 4),
               "n_word_timestamps": len(words),
               "elapsed_sec": round(elapsed, 1),
               **repetition_stats(text)}
        rows.append(row)
        (OUT_DIR / f"transcript_{cfg['name']}.txt").write_text(text, encoding="utf-8")
        print(json.dumps(row, indent=2))

    (OUT_DIR / "results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print_summary(rows)
    print(f"\nwrote {OUT_DIR/'results.json'}")
    print("Then run the winning config through the full pipeline and score it:\n"
          "  python scripts/kaggle_run.py --id rtv_goll_table_ep_v10\n"
          "  python -m evaluation.score_run --hyp <out>/subtitles.srt --tag v10 \\\n"
          "      --compare data/gt/rtv_goll_table/metrics_v9.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
