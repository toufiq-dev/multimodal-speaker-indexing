"""Normalise and validate the hand-edited talk-show ground-truth SRT.

The draft annotation (``groundtruth_raw.srt``) was produced by editing the
system's own v9 subtitles, so it inherits v9's segmentation *and* v9's
segmentation faults, and hand-typed insertions carry no timestamps at all.

This module turns that draft into the ``evaluation/dataset.py`` schema
(RTTM + transcript.json + speaker_map.json) and, critically, attaches a
per-turn ``timing_quality`` tier so that time-based metrics can be
restricted to the spans whose timestamps are actually trustworthy.

Repairs applied (all deterministic, all logged to reliability.json):
  R1  a cue whose index line was lost -- recovered by matching its
      timestamps against the v9 SRT it was edited from (this is the cue
      that a naive SRT parser reports as the speaker "00").
  R2  a cue containing two speaker lines -- split into two turns.
  R3  hand-typed cues with no timestamps -- interpolated between their
      timed neighbours and marked ``inferred`` (excluded from DER/JER).

Timing tiers:
  reliable          v9 timestamp, plausible duration and speaking rate
  implausible_rate  v9 timestamp, but chars/sec far above human speech
  implausible_span  v9 timestamp, but duration absurd for the text length
  inferred          no original timestamp at all (text-accurate only)

Run:  python -m evaluation.normalize_gt
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

GT_DIR = Path(__file__).resolve().parent.parent / "data" / "gt" / "rtv_goll_table"
RAW_SRT = GT_DIR / "groundtruth_raw.srt"
V9_SRT = GT_DIR / "v9_system.srt"
EPISODE_ID = "rtv_goll_table"
EPISODE_DURATION = 409.0

# Canonical labels. Keys are RTTM-safe; values are the display names used in
# transcript.json and emitted by fusion (they match data/registry/*.jpg stems).
CANONICAL: Dict[str, str] = {
    "Dr_Abdul_Noor_Tushar": "Dr Abdul Noor Tushar",
    "Barrister_A_S_M_Shahriar_Kabir": "Barrister A S M Shahriar Kabir",
    "Dr_Md_Tawohidul_Haque": "Dr Md Tawohidul Haque",
}
DISPLAY_TO_KEY = {v: k for k, v in CANONICAL.items()}

# Reliability thresholds. Bengali conversational speech runs roughly
# 10-18 characters/second; 35 is a generous ceiling that only flags
# timestamps that are physically impossible, not merely fast.
MAX_CHARS_PER_SEC = 35.0
MIN_CHARS_PER_SEC_LONG = 5.0      # below this over a long span => bad boundary
LONG_SPAN_SEC = 20.0

TS_RE = re.compile(
    r"^(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})$")


def _t(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def _fmt(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


@dataclass
class Turn:
    start: Optional[float]
    end: Optional[float]
    speaker: str                  # display name
    text: str
    src_index: Optional[int]      # v9 cue index this came from
    timing_quality: str = "reliable"
    repairs: List[str] = field(default_factory=list)
    overlap: bool = False

    @property
    def duration(self) -> float:
        if self.start is None or self.end is None:
            return 0.0
        return max(0.0, self.end - self.start)


def _parse_srt_tolerant(path: Path) -> List[dict]:
    """Parse an SRT that may have lost index lines or timestamp lines."""
    records: List[dict] = []
    for block in path.read_text(encoding="utf-8").strip().split("\n\n"):
        lines = [ln for ln in block.split("\n") if ln.strip()]
        if not lines:
            continue
        idx: Optional[int] = None
        start = end = None
        body = lines
        if lines[0].strip().isdigit() and len(lines) > 1 and TS_RE.match(lines[1].strip()):
            idx = int(lines[0].strip())
            m = TS_RE.match(lines[1].strip())
            start, end = _t(*m.groups()[:4]), _t(*m.groups()[4:])
            body = lines[2:]
        elif TS_RE.match(lines[0].strip()):
            m = TS_RE.match(lines[0].strip())
            start, end = _t(*m.groups()[:4]), _t(*m.groups()[4:])
            body = lines[1:]
        records.append({"index": idx, "start": start, "end": end, "lines": body})
    return records


def _split_speaker(line: str) -> tuple[str, str]:
    name, sep, rest = line.partition(":")
    if not sep:
        raise ValueError(f"Annotation line has no 'Speaker: text' form: {line!r}")
    return name.strip(), rest.strip()


def _canonicalise(name: str) -> str:
    """Map a raw annotation label onto exactly one canonical display name."""
    key = unicodedata.normalize("NFC", name).strip()
    if key in DISPLAY_TO_KEY:
        return key
    squashed = re.sub(r"[^a-z]", "", key.lower())
    for display in DISPLAY_TO_KEY:
        if re.sub(r"[^a-z]", "", display.lower()) == squashed:
            return display
    raise ValueError(f"Unknown speaker label {name!r}; expected one of "
                     f"{sorted(DISPLAY_TO_KEY)}")


def build_turns() -> tuple[List[Turn], dict]:
    raw = _parse_srt_tolerant(RAW_SRT)
    v9 = {r["index"]: r for r in _parse_srt_tolerant(V9_SRT) if r["index"]}
    v9_by_time = {(round(r["start"], 3), round(r["end"], 3)): i
                  for i, r in v9.items()}

    log: Dict[str, list] = {"R1_index_recovered": [], "R2_cue_split": [],
                            "R3_timing_inferred": [], "label_fixes": []}
    turns: List[Turn] = []

    for rec in raw:
        idx, start, end = rec["index"], rec["start"], rec["end"]
        repairs: List[str] = []

        # R1: timestamps but no index -> recover index from the v9 SRT.
        if idx is None and start is not None:
            hit = v9_by_time.get((round(start, 3), round(end, 3)))
            if hit is not None:
                idx = hit
                repairs.append("R1_index_recovered")
                log["R1_index_recovered"].append(
                    {"recovered_index": hit, "start": start, "end": end,
                     "note": "cue read as speaker '00' by a naive SRT parser"})

        # R2: more than one speaker line in a single cue -> split.
        if len(rec["lines"]) > 1:
            repairs.append("R2_cue_split")
            log["R2_cue_split"].append(
                {"src_index": idx, "start": start, "n_speakers": len(rec["lines"]),
                 "speakers": [_split_speaker(l)[0] for l in rec["lines"]]})

        for line in rec["lines"]:
            rawname, text = _split_speaker(line)
            name = _canonicalise(rawname)
            if name != rawname:
                log["label_fixes"].append({"from": rawname, "to": name})
            turns.append(Turn(start=start, end=end, speaker=name, text=text,
                              src_index=idx, repairs=list(repairs)))

    _infer_missing_times(turns, log)
    _grade_timing(turns)
    _mark_overlaps(turns)
    return turns, log


def _infer_missing_times(turns: List[Turn], log: dict) -> None:
    """R3: give untimed hand-typed turns a bounded, clearly-marked placement."""
    n = len(turns)
    i = 0
    while i < n:
        if turns[i].start is not None:
            i += 1
            continue
        j = i
        while j < n and turns[j].start is None:
            j += 1
        lo = max((t.end for t in turns[:i] if t.end is not None), default=0.0)
        hi = min((t.start for t in turns[j:] if t.start is not None),
                 default=EPISODE_DURATION)
        count = j - i
        if hi > lo:
            step = (hi - lo) / count
            for k in range(count):
                turns[i + k].start = lo + k * step
                turns[i + k].end = lo + (k + 1) * step
        else:
            # Neighbours overlap; anchor at lo with a nominal width.
            for k in range(count):
                turns[i + k].start = lo + k * 0.001
                turns[i + k].end = lo + (k + 1) * 0.001
        for k in range(count):
            turns[i + k].repairs.append("R3_timing_inferred")
            log["R3_timing_inferred"].append(
                {"speaker": turns[i + k].speaker,
                 "text": turns[i + k].text[:60],
                 "placed_in": [round(lo, 3), round(hi, 3)],
                 "bounded": hi > lo})
        i = j


def _grade_timing(turns: List[Turn]) -> None:
    for t in turns:
        if "R3_timing_inferred" in t.repairs:
            t.timing_quality = "inferred"
            continue
        chars = len(t.text)
        dur = t.duration
        rate = chars / dur if dur > 0 else float("inf")
        if dur >= LONG_SPAN_SEC and rate < MIN_CHARS_PER_SEC_LONG:
            t.timing_quality = "implausible_span"
        elif rate > MAX_CHARS_PER_SEC:
            t.timing_quality = "implausible_rate"
        else:
            t.timing_quality = "reliable"


def _mark_overlaps(turns: List[Turn]) -> None:
    for a in range(len(turns)):
        for b in range(a + 1, len(turns)):
            ta, tb = turns[a], turns[b]
            if min(ta.end, tb.end) - max(ta.start, tb.start) > 1e-9:
                ta.overlap = tb.overlap = True


def write_outputs(turns: List[Turn], log: dict) -> dict:
    GT_DIR.mkdir(parents=True, exist_ok=True)
    ordered = sorted(turns, key=lambda t: (t.start, t.end))

    # transcript.json -- full annotation, every turn, with reliability flags.
    transcript = [{"start": round(t.start, 3), "end": round(t.end, 3),
                   "speaker": t.speaker, "text": t.text,
                   "src_v9_index": t.src_index,
                   "timing_quality": t.timing_quality,
                   "overlap": t.overlap,
                   "repairs": t.repairs} for t in ordered]
    (GT_DIR / "transcript.json").write_text(
        json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8")

    # RTTMs: everything, and the DER/JER-scorable subset.
    def rttm(rows: List[Turn]) -> str:
        return "".join(
            f"SPEAKER {EPISODE_ID} 1 {t.start:.3f} {t.duration:.3f} "
            f"<NA> <NA> {DISPLAY_TO_KEY[t.speaker]} <NA> <NA>\n" for t in rows)

    (GT_DIR / "reference.rttm").write_text(rttm(ordered), encoding="utf-8")
    scorable = [t for t in ordered if t.timing_quality == "reliable"]
    (GT_DIR / "reference_reliable.rttm").write_text(rttm(scorable), encoding="utf-8")

    (GT_DIR / "speaker_map.json").write_text(json.dumps({
        "_comment": ("Canonical identities for this episode. The diarization "
                     "cluster -> name mapping (SPEAKER_00 -> ...) is a property "
                     "of a given run, not of the annotation; derive it from that "
                     "run's result.json. Zahed Ur Rahman is NOT in this episode."),
        "canonical": CANONICAL,
        "registry_photo": {k: f"data/registry/{k}.jpg" for k in CANONICAL},
        "absent_from_episode": ["Zahed Ur Rahman"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    by_q: Dict[str, int] = {}
    time_by_q: Dict[str, float] = {}
    for t in ordered:
        by_q[t.timing_quality] = by_q.get(t.timing_quality, 0) + 1
        time_by_q[t.timing_quality] = round(
            time_by_q.get(t.timing_quality, 0.0) + t.duration, 2)

    report = {
        "episode": EPISODE_ID,
        "duration_sec": EPISODE_DURATION,
        "n_turns": len(ordered),
        "turns_by_speaker": {name: sum(1 for t in ordered if t.speaker == name)
                             for name in CANONICAL.values()},
        "speaking_time_by_speaker": {
            name: round(sum(t.duration for t in ordered if t.speaker == name), 2)
            for name in CANONICAL.values()},
        "turns_by_timing_quality": by_q,
        "seconds_by_timing_quality": time_by_q,
        "n_overlapping_turns": sum(1 for t in ordered if t.overlap),
        "scorable_for_der": len(scorable),
        "excluded_from_der": [
            {"start": round(t.start, 3), "end": round(t.end, 3),
             "speaker": t.speaker, "reason": t.timing_quality,
             "src_v9_index": t.src_index, "text": t.text[:70]}
            for t in ordered if t.timing_quality != "reliable"],
        "repairs": log,
        "caveats": [
            "Timestamps are the v9 system's own segmentation, not re-annotated. "
            "DER/JER computed against this reference therefore measure speaker "
            "LABEL assignment on fixed boundaries, not boundary quality.",
            "Transcript text was produced by correcting v9's output, so WER "
            "against it is a LOWER BOUND on true WER (annotator anchoring).",
            "Turns marked 'inferred' have accurate text but no real timestamps.",
        ],
    }
    (GT_DIR / "reliability.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    turns, log = build_turns()
    report = write_outputs(turns, log)
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("excluded_from_der", "repairs")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
