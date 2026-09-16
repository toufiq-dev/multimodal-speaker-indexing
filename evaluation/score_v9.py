"""Score the v9 system output against the validated ground truth.

Reports speaker-name accuracy, WER/CER and DER/JER for the untouched v9
subtitles, with every number scoped to the subset of the annotation whose
timestamps support it. Nothing here tunes anything; it only measures.

Run:  python -m evaluation.score_v9
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

from evaluation.metrics import (_edit_distance, _normalize_words, _norm_chars,
                                cpwer, der_jer, speaker_name_accuracy)
from evaluation.normalize_gt import (CANONICAL, DISPLAY_TO_KEY, GT_DIR, V9_SRT,
                                     _parse_srt_tolerant, _split_speaker,
                                     _canonicalise)
from models import DiarizationSegment, FinalSegment

OUT = GT_DIR / "v9_baseline_metrics.json"


def _wer(ref: str, hyp: str) -> float:
    r, h = _normalize_words(ref), _normalize_words(hyp)
    return _edit_distance(r, h) / len(r) if r else (0.0 if not h else float("inf"))


def _cer(ref: str, hyp: str) -> float:
    r, h = _norm_chars(ref), _norm_chars(hyp)
    return _edit_distance(r, h) / len(r) if r else (0.0 if not h else float("inf"))


def _union(intervals):
    out = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def _clip(segs, regions):
    """Restrict segments to a scoring region (UEM), splitting where needed."""
    kept = []
    for sg in segs:
        for rs, re_ in regions:
            a, b = max(sg.start, rs), min(sg.end, re_)
            if b - a > 1e-9:
                kept.append(DiarizationSegment(start=a, end=b,
                                               speaker_id=sg.speaker_id))
    return kept


def load_v9() -> Dict[int, FinalSegment]:
    out: Dict[int, FinalSegment] = {}
    for rec in _parse_srt_tolerant(V9_SRT):
        if rec["index"] is None:
            continue
        name, text = _split_speaker(rec["lines"][0])
        try:
            name = _canonicalise(name)
        except ValueError:
            pass                      # keep non-canonical labels (face_cluster_N)
        out[rec["index"]] = FinalSegment(start=rec["start"], end=rec["end"],
                                         speaker=name, text=text)
    return out


def main() -> None:
    gt = json.loads((GT_DIR / "transcript.json").read_text(encoding="utf-8"))
    v9 = load_v9()
    report: Dict[str, object] = {
        "episode": "rtv_goll_table",
        "hypothesis": "v9_system.srt (untouched)",
        "n_gt_turns": len(gt), "n_v9_cues": len(v9),
    }

    # ---- A. Cue-level speaker label accuracy (exact, shared boundaries) ----
    aligned = [t for t in gt if t["src_v9_index"] is not None]
    conf = Counter()
    correct = words_ok = words_tot = 0
    per_spk = defaultdict(lambda: {"n": 0, "correct": 0})
    for t in aligned:
        pred = v9[t["src_v9_index"]].speaker
        ref = t["speaker"]
        conf[(ref, pred)] += 1
        nw = len(_normalize_words(t["text"]))
        words_tot += nw
        per_spk[ref]["n"] += 1
        if pred == ref:
            correct += 1
            words_ok += nw
            per_spk[ref]["correct"] += 1
    report["cue_level_speaker_accuracy"] = {
        "_definition": ("fraction of the 89 GT turns that share a v9 cue whose "
                        "v9 label equals the true speaker; boundaries identical "
                        "on both sides, so this isolates identity assignment"),
        "n_aligned_turns": len(aligned),
        "accuracy": round(correct / len(aligned), 4),
        "word_weighted_accuracy": round(words_ok / words_tot, 4),
        "per_speaker_recall": {k: round(v["correct"] / v["n"], 4)
                               for k, v in sorted(per_spk.items())},
        "per_speaker_n": {k: v["n"] for k, v in sorted(per_spk.items())},
        "confusion_ref_to_pred": {f"{r} -> {p}": c
                                  for (r, p), c in conf.most_common()},
    }

    # ---- B. Accuracy split by overlap, the first-class target ----
    # Reported under three definitions of "overlapping" so the conclusion can
    # be seen not to depend on how the degenerate long cue is treated.
    pools = {
        "as_annotated": gt,
        "excl_implausible_span": [t for t in gt
                                  if t["timing_quality"] != "implausible_span"],
        "reliable_timing_only": [t for t in gt
                                 if t["timing_quality"] == "reliable"],
    }
    acc_ov: Dict[str, object] = {"_definition": (
        "a turn is 'overlapping' if its time span intersects any other "
        "annotated turn; v9 label correctness is then split on that flag")}
    for pname, pool in pools.items():
        flags = {id(a): any(min(a["end"], b["end"]) - max(a["start"], b["start"]) > 1e-9
                            for b in pool if b is not a) for a in aligned}
        entry = {}
        for flag, key in ((True, "overlapping"), (False, "non_overlapping")):
            sub = [t for t in aligned if flags[id(t)] is flag]
            if sub:
                ok = sum(1 for t in sub
                         if v9[t["src_v9_index"]].speaker == t["speaker"])
                entry[key] = {"n": len(sub), "accuracy": round(ok / len(sub), 4)}
        acc_ov[pname] = entry
    report["accuracy_by_overlap"] = acc_ov

    # ---- C. Time-weighted speaker-name accuracy ----
    hyp_segs = list(v9.values())
    def turns(rows):
        return [DiarizationSegment(start=t["start"], end=t["end"],
                                   speaker_id=t["speaker"]) for t in rows]
    reliable = [t for t in gt if t["timing_quality"] == "reliable"]
    report["time_weighted_speaker_name_accuracy"] = {
        "all_turns": round(speaker_name_accuracy(turns(gt), hyp_segs), 4),
        "reliable_timing_only": round(
            speaker_name_accuracy(turns(reliable), hyp_segs), 4),
        "_note": "reliable_timing_only is the defensible figure",
    }

    # ---- D. DER / JER (label assignment on fixed boundaries) ----
    hyp_turns = [DiarizationSegment(start=s.start, end=s.end, speaker_id=s.speaker)
                 for s in hyp_segs]
    uem = _union([(t["start"], t["end"]) for t in reliable])
    uem_sec = round(sum(e - s for s, e in uem), 2)
    ref_uem, hyp_uem = turns(reliable), _clip(hyp_turns, uem)
    report["der_jer"] = {
        "_warning": ("GT timestamps ARE v9's own timestamps, so miss/false-alarm "
                     "are structurally near-zero: this measures speaker "
                     "CONFUSION on fixed boundaries, NOT diarization boundary "
                     "quality. A true DER needs independently annotated "
                     "boundaries, which this annotation does not provide."),
        "scoring_region_sec": uem_sec,
        "scoring_region_pct_of_episode": round(100 * uem_sec / 409.0, 1),
        "uem_restricted_collar_0.25": der_jer(ref_uem, hyp_uem, 0.25),
        "uem_restricted_collar_0.0": der_jer(ref_uem, hyp_uem, 0.0),
        "_unrestricted_note": ("without the UEM the hypothesis is scored over "
                               "353.6s against a 261.5s reference, inflating "
                               "false alarm to ~0.39; reported for transparency"),
        "no_uem_collar_0.25": der_jer(turns(reliable), hyp_turns, 0.25),
    }

    # ---- E. WER / CER (lower bound: GT was edited from v9 text) ----
    gt_sorted = sorted(gt, key=lambda t: t["start"])
    ref_all = " ".join(t["text"] for t in gt_sorted)
    hyp_all = " ".join(v9[i].text for i in sorted(v9))
    per_cue = [(_wer(t["text"], v9[t["src_v9_index"]].text),
                len(_normalize_words(t["text"]))) for t in aligned]
    tot_w = sum(w for _, w in per_cue)
    ref_by_spk = defaultdict(list); hyp_by_spk = defaultdict(list)
    for t in gt_sorted:
        ref_by_spk[t["speaker"]].append(t["text"])
    for i in sorted(v9):
        hyp_by_spk[v9[i].speaker].append(v9[i].text)
    cp, nperm = cpwer({k: " ".join(v) for k, v in ref_by_spk.items()},
                      {k: " ".join(v) for k, v in hyp_by_spk.items()})
    report["asr"] = {
        "_warning": ("the annotation was produced by CORRECTING v9's own text, "
                     "so these are LOWER BOUNDS on true error"),
        "corpus_wer": round(_wer(ref_all, hyp_all), 4),
        "corpus_cer": round(_cer(ref_all, hyp_all), 4),
        "aligned_cue_wer_wordweighted": round(
            sum(w * n for w, n in per_cue) / tot_w, 4) if tot_w else None,
        "cpwer": round(cp, 4), "cpwer_permutations": nperm,
        "gt_words": len(_normalize_words(ref_all)),
        "v9_words": len(_normalize_words(hyp_all)),
        "words_only_in_untimed_gt_insertions": sum(
            len(_normalize_words(t["text"])) for t in gt
            if t["timing_quality"] == "inferred"),
    }

    # ---- F. Label inventory actually emitted by v9 ----
    report["v9_label_inventory"] = dict(Counter(s.speaker for s in hyp_segs))

    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
