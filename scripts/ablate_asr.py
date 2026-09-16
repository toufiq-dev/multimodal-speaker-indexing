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
                                _norm_chars, ngram_repetition, word_recall)

GT = REPO / "data" / "gt" / "rtv_goll_table" / "transcript.json"
OUT_DIR = REPO / "data" / "gt" / "rtv_goll_table" / "asr_ablation"

# Named configs. 'baseline' reproduces the v9 decode exactly; every other row
# changes ONE thing relative to it, except the explicitly combined rows.
#
# 'temp' False decodes greedily at temperature 0 with no sampling fallback.
# That row exists for C2, not for WER: v9 and v10 ran this same code and this
# same config and produced different word sets, and the temperature fallback is
# the only stochastic step in the decode. If --repeat shows the baseline's
# spread collapse to zero under 'nofallback', the irreproducibility is
# explained rather than merely observed.
CONFIGS = [
    {"name": "baseline_v9",       "quant": "int8",    "vad": True,  "cond": True,  "temp": True},
    {"name": "novad",             "quant": "int8",    "vad": False, "cond": True,  "temp": True},
    {"name": "nocond",            "quant": "int8",    "vad": True,  "cond": False, "temp": True},
    {"name": "nofallback",        "quant": "int8",    "vad": True,  "cond": True,  "temp": False},
    {"name": "novad_nocond",      "quant": "int8",    "vad": False, "cond": False, "temp": True},
    {"name": "fp16",              "quant": "float16", "vad": True,  "cond": True,  "temp": True},
    {"name": "fp16_novad_nocond", "quant": "float16", "vad": False, "cond": False, "temp": True},
    {"name": "fp16_novad_nocond_nofallback",
     "quant": "float16", "vad": False, "cond": False, "temp": False},
]

# Keys that define a decode condition; everything else in a row is a result.
COND_KEYS = ("quant", "vad", "cond", "temp", "src")


def _wer(ref: str, hyp: str) -> float:
    r, h = _normalize_words(ref), _normalize_words(hyp)
    return _edit_distance(r, h) / len(r) if r else float("inf")


def _cer(ref: str, hyp: str) -> float:
    r, h = _norm_chars(ref), _norm_chars(hyp)
    return _edit_distance(r, h) / len(r) if r else float("inf")


def repetition_stats(text: str) -> dict:
    """Degenerate-loop share; see evaluation.metrics.ngram_repetition."""
    return {"ngram_repeat_ratio": ngram_repetition(text)["ngram_repeat_ratio"]}


def ensure_ct2(quant: str, scratch: Path, src_model: str | None = None) -> str:
    """Convert the checkpoint at the requested quantization (cached)."""
    src_model = src_model or os.getenv(
        "WHISPER_MODEL_SRC", "bengaliAI/tugstugi_bengaliai-asr_whisper-medium")
    slug = src_model.rstrip("/").split("/")[-1]
    out = scratch / f"{slug}_ct2_{quant}"
    if (out / "model.bin").exists():
        print(f"[ct2] reusing {out}")
        return str(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[ct2] converting {src_model} at {quant} -> {out}")
    subprocess.run(
        [sys.executable, "-m", "ctranslate2.converters.transformers",
         "--model", src_model,
         "--output_dir", str(out), "--quantization", quant,
         "--copy_files", "tokenizer.json", "preprocessor_config.json", "--force"],
        check=True)
    return str(out)


def _spread(vals: list) -> float:
    """max - min. With --repeat 2 this IS the noise floor: per C2, a delta
    smaller than this is not a result."""
    return (max(vals) - min(vals)) if vals else 0.0


def print_summary(rows: list) -> None:
    """Render the comparison table, one line per config with the across-repeat
    spread beside each mean. Kept separate so it is unit-testable: a formatting
    crash here would land after every Whisper pass has run."""
    ok = [r for r in rows if "error" not in r]
    groups: dict = {}
    for r in ok:
        groups.setdefault(r["name"], []).append(r)

    def mean(name, key):
        vals = [g[key] for g in groups[name] if key in g]
        return sum(vals) / len(vals) if vals else float("nan")

    base = "baseline_v9" if "baseline_v9" in groups else None
    hdr = (f"{'config':<22}{'n':>3}{'WER':>8}{'+-':>7}{'recall':>8}{'+-':>7}"
           f"{'words':>7}{'rep':>7}{'+-':>7}{'sec':>7}")
    print("\n" + hdr + ("    dWER" if base else ""))
    print("-" * (len(hdr) + (8 if base else 0)))
    for name, g in groups.items():
        rec_key = "word_recall" if "word_recall" in g[0] else "word_recall_proxy"
        line = (f"{name:<22}{len(g):>3}{mean(name, 'wer'):>8.4f}"
                f"{_spread([x['wer'] for x in g]):>7.4f}"
                f"{mean(name, rec_key):>8.3f}"
                f"{_spread([x[rec_key] for x in g]):>7.3f}"
                f"{mean(name, 'hyp_words'):>7.0f}"
                f"{mean(name, 'ngram_repeat_ratio'):>7.3f}"
                f"{_spread([x['ngram_repeat_ratio'] for x in g]):>7.3f}"
                f"{mean(name, 'elapsed_sec'):>7.1f}")
        if base:
            line += f"  {mean(name, 'wer') - mean(base, 'wer'):+.4f}"
        print(line)
    for r in rows:
        if "error" in r:
            print(f"{r['name']:<22} FAILED: {r['error'][:80]}")

    if base and len(groups[base]) > 1:
        g = groups[base]
        print(f"\nNOISE FLOOR from {len(g)} repeats of {base} (C2): "
              f"WER +-{_spread([x['wer'] for x in g]):.4f}  "
              f"words +-{_spread([x['hyp_words'] for x in g]):.0f}  "
              f"repeat +-{_spread([x['ngram_repeat_ratio'] for x in g]):.4f}\n"
              "Any row differing from baseline by less than that is noise.")
    elif base:
        print("\nNo noise floor measured: re-run with --repeat 2 (C2). A single "
              "run of a config cannot establish an effect.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--only", nargs="*", default=None,
                    help="run only these config names")
    ap.add_argument("--repeat", type=int, default=1,
                    help="decodes per config. C2: use 2. The spread across "
                         "repeats is the noise floor every later delta is "
                         "judged against; one decode cannot establish an effect")
    ap.add_argument("--extra-model", default=None,
                    help="HF id of an additional checkpoint to sweep (e.g. a "
                         "larger Bengali Whisper). Verify the id exists before "
                         "using it; nothing is hard-coded here on trust")
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
    rows, cache = [], {}
    todo = [c for c in CONFIGS if not a.only or c["name"] in a.only]
    if a.extra_model:
        todo += [{"name": f"extra_{a.extra_model.split('/')[-1]}", "quant": "float16",
                  "vad": True, "cond": True, "temp": True, "src": a.extra_model}]
    if a.repeat < 2:
        print("[c2] --repeat 1: no noise floor will be measured. Deltas from "
              "this table are not yet results.")

    for rep in range(a.repeat):
        for cfg in todo:
            print(f"\n=== {cfg['name']} (rep {rep + 1}/{a.repeat}) ===")
            key = (cfg.get("src"), cfg["quant"])
            model_dir = cache.get(key) or ensure_ct2(
                cfg["quant"], Path(a.scratch), cfg.get("src"))
            cache[key] = model_dir

            config.WHISPER_MODEL = model_dir
            config.WHISPER_VAD_FILTER = cfg["vad"]
            config.WHISPER_CONDITION_ON_PREV = cfg["cond"]
            config.WHISPER_TEMPERATURE_FALLBACK = cfg.get("temp", True)
            model = None
            try:
                model = _load_model()
                t0 = time.time()
                words, text = transcribe_audio(audio, model=model)
                elapsed = time.time() - t0
            except Exception as exc:
                print(f"[{cfg['name']}] FAILED: {exc}")
                rows.append({**cfg, "rep": rep, "error": str(exc)})
                continue
            finally:
                try:
                    del model
                    import torch as _t; _t.cuda.empty_cache()
                except Exception:
                    pass

            # Aligned recall, not the count ratio: a repetition loop raises
            # hyp_words without recovering a single reference word.
            rec = word_recall(ref_text, text)
            row = {**cfg, "rep": rep,
                   "wer": round(_wer(ref_text, text), 4),
                   "cer": round(_cer(ref_text, text), 4),
                   "hyp_words": rec["hyp_words"],
                   "gt_words": ref_words,
                   "word_recall": rec["word_recall"],
                   "matched_words": rec["matched_words"],
                   "word_recall_proxy": rec["word_count_ratio"],
                   "n_word_timestamps": len(words),
                   "elapsed_sec": round(elapsed, 1),
                   **repetition_stats(text)}
            rows.append(row)
            suffix = f"_rep{rep}" if a.repeat > 1 else ""
            (OUT_DIR / f"transcript_{cfg['name']}{suffix}.txt").write_text(
                text, encoding="utf-8")
            print(json.dumps(row, indent=2))
            # Written after every decode: a crash in a later config must not
            # cost the passes already paid for.
            (OUT_DIR / "results.json").write_text(
                json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print_summary(rows)
    print(f"\nwrote {OUT_DIR/'results.json'}")
    print("Then run the winning config through the full pipeline TWICE (C2) and "
          "score each with a fresh run id:\n"
          "  python scripts/kaggle_run.py --id rtv_goll_table_ep_v11a\n"
          "  python -m evaluation.score_run --hyp <out>/subtitles.srt --tag v11a \\\n"
          "      --compare data/gt/rtv_goll_table/metrics_v9.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
