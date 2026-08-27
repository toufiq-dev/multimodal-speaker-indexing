"""Dataset management: multi-video manifest + ground-truth schema.

Manifest format (dataset/manifest.json):

{
  "videos": [
    {
      "id": "jamuna_rajniti_ep12",
      "source": "https://youtu.be/...",
      "media_path": "videos/jamuna_ep12.mp4",
      "duration_sec": 3192.2,
      "language": "bn",
      "num_speakers": 5,
      "registry_dir": "registries/jamuna_ep12/",   // face reference photos
      "ground_truth": {
        "transcript_json": "gt/jamuna_ep12/transcript.json",
        "rttm":           "gt/jamuna_ep12/reference.rttm",
        "speaker_map":    "gt/jamuna_ep12/speaker_map.json"
      },
      "split": "test",
      "notes": "1 host + 4 guests; fixed weekly format"
    }
  ]
}

Ground-truth schemas:
- transcript.json:  [{"start": s, "end": e, "speaker": "<true name>",
                      "text": "..."}, ...]   (word-level optional)
- reference.rttm:   standard RTTM SPEAKER records
- speaker_map.json: {"SPEAKER_00": "<true name>", ...} diarization->name map
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from models import DiarizationSegment


MANIFEST_VERSION = 1
_REQUIRED_VIDEO_KEYS = {"id", "source", "media_path"}


@dataclass
class VideoEntry:
    id: str
    source: str
    media_path: str
    duration_sec: Optional[float] = None
    language: str = "bn"
    num_speakers: Optional[int] = None
    registry_dir: Optional[str] = None
    split: str = "test"
    notes: str = ""
    gt_transcript_json: Optional[str] = None
    gt_rttm: Optional[str] = None
    gt_speaker_map: Optional[str] = None

    def resolve(self, manifest_dir: Path) -> Dict[str, Optional[Path]]:
        """Resolve relative paths against the manifest location."""
        return {
            "media": manifest_dir / self.media_path if self.media_path else None,
            "registry": manifest_dir / self.registry_dir if self.registry_dir else None,
            "transcript": manifest_dir / self.gt_transcript_json if self.gt_transcript_json else None,
            "rttm": manifest_dir / self.gt_rttm if self.gt_rttm else None,
            "speaker_map": manifest_dir / self.gt_speaker_map if self.gt_speaker_map else None,
        }


@dataclass
class GroundTruth:
    """Container for a video's annotated evaluation data."""
    video_id: str
    turns: List[DiarizationSegment] = field(default_factory=list)  # spk = true name
    texts: List[Dict] = field(default_factory=list)                # start/end/speaker/text
    word_speakers: List[tuple] = field(default_factory=list)       # (s,e,name)
    face_labels: List[tuple] = field(default_factory=list)         # (time,name)


def load_manifest(manifest_path: str | Path) -> List[VideoEntry]:
    """Load and validate a dataset manifest. Raises ValueError with details."""
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data.get("videos"), list) or not data["videos"]:
        raise ValueError("Manifest must contain a non-empty 'videos' list")
    entries: List[VideoEntry] = []
    ids = set()
    for i, v in enumerate(data["videos"]):
        missing = _REQUIRED_VIDEO_KEYS - set(v)
        if missing:
            raise ValueError(f"videos[{i}] missing keys: {sorted(missing)}")
        if v["id"] in ids:
            raise ValueError(f"Duplicate video id: {v['id']}")
        ids.add(v["id"])
        gt = v.get("ground_truth", {})
        entries.append(VideoEntry(
            id=v["id"], source=v["source"], media_path=v["media_path"],
            duration_sec=v.get("duration_sec"), language=v.get("language", "bn"),
            num_speakers=v.get("num_speakers"),
            registry_dir=v.get("registry_dir"), split=v.get("split", "test"),
            notes=v.get("notes", ""),
            gt_transcript_json=gt.get("transcript_json"),
            gt_rttm=gt.get("rttm"),
            gt_speaker_map=gt.get("speaker_map"),
        ))
    return entries


def parse_rttm(rttm_path: str | Path) -> List[DiarizationSegment]:
    """Parse standard RTTM into DiarizationSegments (label kept verbatim)."""
    segs: List[DiarizationSegment] = []
    for line in Path(rttm_path).read_text().splitlines():
        parts = line.split()
        if len(parts) < 8 or parts[0] != "SPEAKER":
            continue
        start, dur, label = float(parts[3]), float(parts[4]), parts[7]
        segs.append(DiarizationSegment(start=start, end=start + dur,
                                       speaker_id=label))
    segs.sort(key=lambda s: s.start)
    return segs


def load_ground_truth(entry: VideoEntry, manifest_dir: Path) -> GroundTruth:
    """Load all available ground truth for one manifest entry."""
    paths = entry.resolve(manifest_dir)
    gt = GroundTruth(video_id=entry.id)

    if paths["rttm"] and paths["rttm"].exists():
        gt.turns = parse_rttm(paths["rttm"])

    if paths["transcript"] and paths["transcript"].exists():
        gt.texts = json.loads(paths["transcript"].read_text(encoding="utf-8"))
        # If no RTTM, derive turns from the annotated transcript.
        if not gt.turns:
            gt.turns = [
                DiarizationSegment(start=t["start"], end=t["end"],
                                   speaker_id=t.get("speaker", "UNKNOWN"))
                for t in gt.texts
            ]
        for t in gt.texts:
            for w in t.get("words", []) or []:
                gt.word_speakers.append((w["start"], w["end"],
                                         t.get("speaker", "UNKNOWN")))
    return gt


def validate_registry(registry_dir: Path, expected_names: List[str]) -> List[str]:
    """Check that every expected speaker has at least one registry photo.
    Returns list of missing names (empty == OK). This directly addresses the
    Kaggle failure where the registry directory existed but stayed empty."""
    missing: List[str] = []
    if not registry_dir.exists():
        return sorted(expected_names)
    have = {p.stem.lower() for p in registry_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}}
    for name in expected_names:
        key = name.lower()
        if key not in have and key.replace(" ", "_") not in have \
                and key.replace(" ", "-") not in have:
            missing.append(name)
    return missing
