#!/usr/bin/env python3
"""Re-score a finished run's voiceprints and choose the voice gates from data.

Why this exists
---------------
``engines/voice.py`` deliberately refuses to guess its thresholds. A run with
``VOICE_POLICY != off`` writes three artefacts:

    <run>/voiceprints.npz          the raw cluster and enrolment embeddings
    <run>/voice_diagnostics.json   what the gates did at the time
    <run>/diarization.json         the turns the clusters came from

This script reads the first and third and re-decides every cluster at any
threshold, so the gates are chosen from a measured genuine/impostor separation
on THIS episode with THIS registry -- not from a default, and not by re-running
the pipeline. It needs no GPU, no audio and no network: the expensive embedding
pass already happened, and the Kaggle session that produced the audio can be
long gone.

How the truth is derived (and why it is not circular)
----------------------------------------------------
Scoring the voice matcher needs to know which real person each diarization
cluster is. Neither file states that directly: the run labels clusters
``SPEAKER_xx``, and the annotation labels the same audio with real names. The
mapping is recovered by temporal overlap -- the run supplies the segmentation,
the ANNOTATION supplies the identity. Nothing on the hypothesis side is used to
decide what the right answer is.

The derived ``purity`` is itself evidence: a cluster whose time is spread
across several annotated speakers is a diarization error, and no labelling
method can fix it.

What to look for
----------------
1. ``max off-diagonal`` in the cross-matrix -- the impostor ceiling for this
   registry. No threshold at or below it can ever separate that pair of
   enrolled people; the honest conclusion is that voice cannot tell them apart,
   not that the threshold needs lowering.
2. ``coherence`` per cluster. A low value means the cluster contains more than
   one voice, so its centroid is nobody's voiceprint.
3. The sweep. ``wrong`` counts clusters the gate would name INCORRECTLY. A row
   that buys one more ``correct`` at the cost of one ``wrong`` is not obviously
   a gain: in a time-weighted metric the two turns are rarely the same length.

Usage (single line, from the repo root):

    python scripts/voice_verify.py --run-dir data/output/rtv_goll_table_ep_v16

    python scripts/voice_verify.py --run-dir /kaggle/working/output/rtv_goll_table_ep_v16 \
        --gt-dir data/gt/rtv_goll_table --save voice_sweep.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import config  # noqa: E402
from engines.voice import (enrolled_cross_matrix, load_voiceprints,  # noqa: E402
                           match_clusters, threshold_sweep)


# ---------------------------------------------------------------------------
# Ground truth for the voice matcher
# ---------------------------------------------------------------------------

def derive_cluster_names(
    diarization: Path,
    transcript: Path,
) -> Tuple[Dict[str, str], Dict[str, Dict]]:
    """Cluster -> canonical name by overlapping the run against the annotation.

    Returns ``(truth, stats)``. ``stats[spk]`` carries the purity diagnostics:
    how much of the cluster's time the winning name explains, how many distinct
    annotated speakers appear inside it, and how much of it is annotated
    overlapping speech -- which is the condition this whole feature exists to
    address.
    """
    turns = json.loads(Path(diarization).read_text(encoding="utf-8"))
    entries = json.loads(Path(transcript).read_text(encoding="utf-8"))

    per_name: Dict[str, Dict[str, float]] = {}
    total_sec: Dict[str, float] = {}
    overlap_sec: Dict[str, float] = {}
    n_turns: Dict[str, int] = {}

    for turn in turns:
        spk = str(turn["speaker_id"])
        start, end = float(turn["start"]), float(turn["end"])
        if end <= start:
            continue
        total_sec[spk] = total_sec.get(spk, 0.0) + (end - start)
        n_turns[spk] = n_turns.get(spk, 0) + 1
        for entry in entries:
            shared = (min(end, float(entry["end"]))
                      - max(start, float(entry["start"])))
            if shared <= 0:
                continue
            name = entry.get("speaker") or "UNKNOWN"
            bucket = per_name.setdefault(spk, {})
            bucket[name] = bucket.get(name, 0.0) + shared
            if entry.get("overlap"):
                overlap_sec[spk] = overlap_sec.get(spk, 0.0) + shared

    truth: Dict[str, str] = {}
    stats: Dict[str, Dict] = {}
    for spk, scores in per_name.items():
        name, seconds = max(scores.items(), key=lambda kv: kv[1])
        attributed = sum(scores.values())
        cluster_sec = total_sec.get(spk, 0.0)
        truth[spk] = name
        stats[spk] = {
            "name": name,
            "n_turns": n_turns.get(spk, 0),
            "cluster_sec": round(cluster_sec, 1),
            "dominant_sec": round(seconds, 1),
            # Share of the ANNOTATED speech inside this cluster belonging to
            # the winning name. Measured against the annotated total, not the
            # cluster duration: the annotation marks simultaneous speech with
            # separate overlapping entries, so dividing by the cluster
            # duration can exceed 1.0 and would read as more than perfect.
            "purity": round(seconds / attributed, 3) if attributed else 0.0,
            "n_annotated_names": len(scores),
            "annotated_overlap_sec": round(overlap_sec.get(spk, 0.0), 1),
        }
    return truth, stats


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _table(rows: List[List[str]], header: List[str]) -> str:
    widths = [
        max([len(header[i])] + [len(r[i]) for r in rows]) for i in range(len(header))
    ]
    out = ["  ".join(h.ljust(widths[i]) for i, h in enumerate(header)).rstrip()]
    out.append("  ".join("-" * w for w in widths))
    for row in rows:
        out.append("  ".join(c.ljust(widths[i])
                             for i, c in enumerate(row)).rstrip())
    return "\n".join(out)


def report(
    clusters,
    enrolled,
    truth: Optional[Dict[str, str]],
    stats: Dict[str, Dict],
    run_dir: Path,
    save: Optional[Path],
) -> int:
    print(f"run        : {run_dir}")
    print(f"clusters   : {len(clusters)} with voiceprints")
    print(f"enrolled   : {len(enrolled)}")

    # --- enrolment -----------------------------------------------------
    print("\n=== ENROLMENT (voice references) ===")
    rows = []
    for name, vp in sorted(enrolled.items()):
        verdict = "OK" if vp.n_kept >= 3 and vp.coherence >= 0.5 else "WEAK"
        rows.append([name, str(vp.n_windows), str(vp.n_kept),
                     f"{vp.duration_sec:.0f}", f"{vp.coherence:.3f}", verdict])
    print(_table(rows, ["name", "windows", "kept", "sec", "coherence", ""]))
    weak = [r[0] for r in rows if r[5] == "WEAK"]
    if weak:
        print(f"\n  WEAK enrolment (few windows or low coherence): {', '.join(weak)}")
        print("  A reference clip of a broadcast often contains other speakers;")
        print("  low coherence means the trimmed centroid is still mixed.")

    # --- impostor ceiling ----------------------------------------------
    names = sorted(enrolled)
    cross = enrolled_cross_matrix(enrolled)
    print("\n=== ENROLLED CROSS-SIMILARITY (the impostor ceiling) ===")
    # Printed as a full square (diagonal 1.0) so the columns line up.
    rows = [[a] + [f"{1.0 if a == b else cross[a][b]:.3f}" for b in names]
            for a in names]
    print(_table(rows, ["name"] + [n[:14] for n in names]))
    off = [(cross[a][b], a, b) for i, a in enumerate(names)
           for b in names[i + 1:]]
    if off:
        worst, a, b = max(off)
        print(f"\n  max off-diagonal = {worst:.3f}  ({a} vs {b})")
        print(f"  -> no VOICE_SIM_THRESHOLD at or below {worst:.3f} can separate "
              f"that pair.")

    # --- per-cluster at the configured gates ---------------------------
    current = match_clusters(clusters, enrolled)
    print(f"\n=== CLUSTERS (current gates: threshold="
          f"{config.VOICE_SIM_THRESHOLD}, margin={config.VOICE_SIM_MARGIN}) ===")
    rows = []
    for spk in sorted(clusters):
        match, cluster = current[spk], clusters[spk]
        purity = (f"{stats[spk]['purity']:.2f}" if spk in stats else "-")
        rows.append([
            spk, str(cluster.n_turns_used), f"{cluster.used_sec:.0f}",
            f"{cluster.coherence:.3f}", purity,
            (match.name or "-")[:26], f"{match.similarity:.3f}",
            (match.runner_up or "-")[:26], f"{match.runner_up_similarity:.3f}",
            f"{match.margin:.3f}",
            "ACCEPT" if match.accepted else match.reason.upper(),
        ])
    print(_table(rows, ["cluster", "turns", "used_s", "coh", "purity",
                        "best", "sim", "runner-up", "sim", "margin", "decision"]))
    if stats:
        print("\n  purity = share of the cluster's annotated time belonging to "
              "the winning name;")
        print("  a low value is a DIARIZATION error and no gate here can fix it.")
        impure = [s for s, v in stats.items() if v["purity"] < 0.8]
        if impure:
            print(f"  impure clusters (<0.80): "
                  + ", ".join(f"{s}={stats[s]['purity']:.2f}" for s in sorted(impure)))

    # --- sweep ---------------------------------------------------------
    sweep = threshold_sweep(clusters, enrolled, truth=truth)
    print("\n=== THRESHOLD SWEEP ===")
    if truth:
        print(f"truth derived from the annotation for {len(truth)} cluster(s): "
              + ", ".join(f"{k}={v}" for k, v in sorted(truth.items())))
        frontier: Dict[int, Dict] = {}
        for row in sweep:
            best = frontier.get(row["wrong"])
            if best is None or (row["correct"], row["margin"]) > (
                    best["correct"], best["margin"]):
                frontier[row["wrong"]] = row
        rows = [[f"{r['threshold']:.2f}", f"{r['margin']:.2f}",
                 str(r["n_accepted"]), str(r["correct"]), str(r["wrong"]),
                 str(r["missed"]), ("; ".join(r["wrong_detail"]) or "-")[:44]]
                for _, r in sorted(frontier.items())]
        print(_table(rows, ["thresh", "margin", "named", "correct", "wrong",
                            "missed", "misnamed"]))
        print("\n  (Pareto frontier only: for each number of misnamed clusters, "
              "the")
        print("   gate that names the most correctly. 'wrong' > 0 is a "
              "regression risk.)")
        zero = frontier.get(0)
        if zero:
            print(f"\n  RECOMMENDED (no misnaming, most correct): "
                  f"VOICE_SIM_THRESHOLD={zero['threshold']:.2f} "
                  f"VOICE_SIM_MARGIN={zero['margin']:.2f} "
                  f"-> {zero['correct']}/{len(truth)} named correctly, "
                  f"{zero['missed']} left unnamed")
        else:
            print("\n  Every gate that names anything misnames something. "
                  "Voice is")
            print("  not safely usable on this registry/episode as enrolled.")
    else:
        print("  no --gt-dir/--truth supplied, so nothing can be scored.")

    if save:
        Path(save).write_text(json.dumps({
            "run_dir": str(run_dir),
            "truth": truth,
            "cluster_stats": stats,
            "sweep": sweep,
            "current_gates": {
                "threshold": config.VOICE_SIM_THRESHOLD,
                "margin": config.VOICE_SIM_MARGIN,
                "accepted": {s: m.name for s, m in current.items() if m.accepted},
            },
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nsweep -> {save}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True,
                    help="run output dir containing voiceprints.npz")
    ap.add_argument("--npz", default=None,
                    help="explicit voiceprints.npz (default: <run-dir>/voiceprints.npz)")
    ap.add_argument("--diarization", default=None,
                    help="turn list (default: <run-dir>/diarization.json)")
    ap.add_argument("--gt-dir", default=None,
                    help="annotation dir with transcript.json")
    ap.add_argument("--truth", nargs="+", default=None, metavar="SPEAKER=Name",
                    help="explicit cluster=name pairs, overriding --gt-dir")
    ap.add_argument("--save", default=None, help="write the full sweep to JSON")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    npz = Path(args.npz).resolve() if args.npz else run_dir / "voiceprints.npz"
    if not npz.exists():
        print(f"no voiceprints at {npz}", file=sys.stderr)
        print("This file is written only when VOICE_POLICY is not 'off'. Re-run "
              "the episode with VOICE_POLICY=fallback (or override).",
              file=sys.stderr)
        return 2

    clusters, enrolled = load_voiceprints(npz)

    truth: Optional[Dict[str, str]] = None
    stats: Dict[str, Dict] = {}
    if args.truth:
        truth = {}
        for item in args.truth:
            if "=" not in item:
                print(f"--truth expects SPEAKER=Name, got {item!r}", file=sys.stderr)
                return 2
            key, value = item.split("=", 1)
            truth[key.strip()] = value.strip()
    elif args.gt_dir:
        gt_dir = Path(args.gt_dir).resolve()
        transcript = gt_dir / "transcript.json"
        diarization = (Path(args.diarization).resolve() if args.diarization
                       else run_dir / "diarization.json")
        if not transcript.exists():
            print(f"no transcript.json in {gt_dir}; continuing unscored",
                  file=sys.stderr)
        elif not diarization.exists():
            print(f"no diarization.json in {run_dir} (runs before the voice "
                  f"feature did not persist it); pass --truth SPEAKER=Name "
                  f"explicitly, or re-run the episode.", file=sys.stderr)
        else:
            truth, stats = derive_cluster_names(diarization, transcript)
            print(f"truth derived from {diarization.name} x {transcript.name}\n")
    return report(clusters, enrolled, truth, stats, run_dir, args.save)


if __name__ == "__main__":
    raise SystemExit(main())
