"""Score ANY run's subtitles.srt against the validated ground truth.

`score_v9.py` aligns hypothesis to reference by cue index, which is only valid
for v9 (the annotation was edited from v9, so the boundaries coincide). Any
later run produces its own boundaries, so this scorer is purely time-based and
makes no assumption that hypothesis cues correspond to reference turns.

Primary metric for tracking interventions:
    time-weighted speaker-name accuracy over the reliable UEM,
    reported separately for overlapping and non-overlapping reference time.

Run:  python -m evaluation.score_run --hyp /path/to/subtitles.srt --tag v10
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from evaluation.metrics import (_edit_distance, _normalize_words, _norm_chars,
                                cpwer, der_jer, ngram_repetition,
                                speaker_name_accuracy)
from evaluation.normalize_gt import CANONICAL, GT_DIR, V9_SRT, _parse_srt_tolerant, \
    _split_speaker, _canonicalise
from models import DiarizationSegment, FinalSegment

EPISODE_SEC = 409.0


def _wer(ref: str, hyp: str) -> float:
    r, h = _normalize_words(ref), _normalize_words(hyp)
    return _edit_distance(r, h) / len(r) if r else (0.0 if not h else float("inf"))


def _cer(ref: str, hyp: str) -> float:
    r, h = _norm_chars(ref), _norm_chars(hyp)
    return _edit_distance(r, h) / len(r) if r else (0.0 if not h else float("inf"))


def _union(intervals) -> List[List[float]]:
    out: List[List[float]] = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def _clip(segs, regions):
    kept = []
    for sg in segs:
        for rs, re_ in regions:
            a, b = max(sg.start, rs), min(sg.end, re_)
            if b - a > 1e-9:
                kept.append(DiarizationSegment(a, b, sg.speaker_id))
    return kept


def load_srt(path: Path) -> List[FinalSegment]:
    out: List[FinalSegment] = []
    for rec in _parse_srt_tolerant(path):
        if rec["start"] is None or not rec["lines"]:
            continue
        for line in rec["lines"]:
            try:
                name, text = _split_speaker(line)
            except ValueError:
                continue
            try:
                name = _canonicalise(name)
            except ValueError:
                pass                       # keep face_cluster_N / Speaker_N as-is
            out.append(FinalSegment(rec["start"], rec["end"], name, text))
    return out


def score(hyp_segs: List[FinalSegment], tag: str) -> Dict[str, object]:
    gt = json.loads((GT_DIR / "transcript.json").read_text(encoding="utf-8"))
    reliable = [t for t in gt if t["timing_quality"] == "reliable"]
    uem = _union([(t["start"], t["end"]) for t in reliable])
    uem_sec = sum(e - s for s, e in uem)

    def turns(rows):
        return [DiarizationSegment(t["start"], t["end"], t["speaker"]) for t in rows]

    rep: Dict[str, object] = {
        "tag": tag,
        "n_hyp_segments": len(hyp_segs),
        "scoring_region_sec": round(uem_sec, 2),
        "scoring_region_pct": round(100 * uem_sec / EPISODE_SEC, 1),
        "_primary_metric": "speaker_name_accuracy.reliable_uem",
    }

    # Time-weighted speaker-name accuracy (union-based; cannot exceed 1.0).
    overlapping = [t for t in reliable if t["overlap"]]
    clean = [t for t in reliable if not t["overlap"]]
    rep["speaker_name_accuracy"] = {
        "reliable_uem": round(speaker_name_accuracy(turns(reliable), hyp_segs), 4),
        "overlapping_turns": round(
            speaker_name_accuracy(turns(overlapping), hyp_segs), 4) if overlapping else None,
        "non_overlapping_turns": round(
            speaker_name_accuracy(turns(clean), hyp_segs), 4) if clean else None,
        "n_overlapping": len(overlapping), "n_non_overlapping": len(clean),
        "per_speaker": {
            name: round(speaker_name_accuracy(
                turns([t for t in reliable if t["speaker"] == name]), hyp_segs), 4)
            for name in CANONICAL.values()},
        "all_turns_incl_unreliable": round(
            speaker_name_accuracy(turns(gt), hyp_segs), 4),
    }

    # DER / JER, UEM-restricted.
    hyp_turns = [DiarizationSegment(s.start, s.end, s.speaker) for s in hyp_segs]
    rep["der_jer"] = {
        "_warning": ("reference boundaries are v9's own, so this reflects "
                     "speaker confusion far more than boundary quality"),
        "uem_collar_0.25": der_jer(turns(reliable), _clip(hyp_turns, uem), 0.25),
        "uem_collar_0.0": der_jer(turns(reliable), _clip(hyp_turns, uem), 0.0),
    }

    # ASR.
    ref_all = " ".join(t["text"] for t in sorted(gt, key=lambda x: x["start"]))
    hyp_all = " ".join(s.text for s in sorted(hyp_segs, key=lambda x: x.start))
    ref_by = defaultdict(list); hyp_by = defaultdict(list)
    for t in sorted(gt, key=lambda x: x["start"]):
        ref_by[t["speaker"]].append(t["text"])
    for s in sorted(hyp_segs, key=lambda x: x.start):
        hyp_by[s.speaker].append(s.text)
    cp, nperm = cpwer({k: " ".join(v) for k, v in ref_by.items()},
                      {k: " ".join(v) for k, v in hyp_by.items()})
    rep["asr"] = {
        "_warning": "LOWER BOUND: the annotation was edited from v9's own text",
        "corpus_wer": round(_wer(ref_all, hyp_all), 4),
        "corpus_cer": round(_cer(ref_all, hyp_all), 4),
        "cpwer": round(cp, 4), "cpwer_permutations": nperm,
        "gt_words": len(_normalize_words(ref_all)),
        "hyp_words": len(_normalize_words(hyp_all)),
        "word_recall_proxy": round(
            len(_normalize_words(hyp_all)) / len(_normalize_words(ref_all)), 4),
    }
    # Decoder looping, the failure that forced hand-rewriting of 360-387 s.
    rep["repetition"] = {
        "_definition": "share of duplicated 5-grams; reference text scores 0.0044",
        "hypothesis": ngram_repetition(hyp_all),
        "reference": ngram_repetition(ref_all),
    }
    rep["label_inventory"] = dict(Counter(s.speaker for s in hyp_segs))
    rep["spurious_labels"] = sorted(
        {s.speaker for s in hyp_segs} - set(CANONICAL.values()))
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyp", type=Path, default=V9_SRT,
                    help="subtitles.srt to grade (default: the v9 baseline)")
    ap.add_argument("--tag", default=None, help="label for this run")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--compare", type=Path, default=None,
                    help="an earlier score_run JSON to diff against")
    a = ap.parse_args()
    tag = a.tag or a.hyp.stem
    rep = score(load_srt(a.hyp), tag)

    if a.compare and a.compare.exists():
        base = json.loads(a.compare.read_text())
        rep["delta_vs"] = base.get("tag")
        rep["delta"] = {
            "speaker_name_accuracy.reliable_uem": round(
                rep["speaker_name_accuracy"]["reliable_uem"]
                - base["speaker_name_accuracy"]["reliable_uem"], 4),
            "speaker_name_accuracy.overlapping_turns": round(
                (rep["speaker_name_accuracy"]["overlapping_turns"] or 0)
                - (base["speaker_name_accuracy"]["overlapping_turns"] or 0), 4),
            "corpus_wer": round(rep["asr"]["corpus_wer"]
                                - base["asr"]["corpus_wer"], 4),
            "DER": round(rep["der_jer"]["uem_collar_0.25"]["DER"]
                         - base["der_jer"]["uem_collar_0.25"]["DER"], 4),
        }

    out = a.out or (GT_DIR / f"metrics_{tag}.json")
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
