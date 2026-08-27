"""Experiment tracking: configs, logs, reproducibility manifests.

Every pipeline run records:
  - full config snapshot + hash
  - git commit / dirty state of the repo
  - package versions (torch, transformers, peft, faster-whisper, pyannote)
  - stage artifact paths and timings
  - final metrics

This exists because the first end-to-end run was executed with notebook
hot-patches that diverged from the repo, leaving nothing reproducible.
"""
from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


def _git_state() -> Dict[str, Optional[str]]:
    def _run(*args: str) -> Optional[str]:
        try:
            return subprocess.run(
                ["git", *args], capture_output=True, text=True,
                timeout=10).stdout.strip() or None
        except Exception:
            return None
    commit = _run("rev-parse", "HEAD")
    dirty = _run("status", "--porcelain")
    return {"commit": commit, "dirty": bool(dirty)}


def _package_versions() -> Dict[str, str]:
    versions = {"python": sys.version.split()[0]}
    for mod in ("torch", "transformers", "peft", "faster_whisper",
                "pyannote.audio", "insightface"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            versions[mod] = "not-installed"
    return versions


@dataclass
class ExperimentRun:
    name: str
    config: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)

    started_at: str = field(default_factory=lambda:
                            datetime.datetime.utcnow().isoformat() + "Z")
    finished_at: Optional[str] = None
    git: Dict[str, Optional[str]] = field(default_factory=_git_state)
    packages: Dict[str, str] = field(default_factory=_package_versions)
    run_id: str = ""

    def __post_init__(self):
        if not self.run_id:
            payload = json.dumps(self.config, sort_keys=True) + self.started_at
            self.run_id = hashlib.sha1(payload.encode()).hexdigest()[:12]

    def log_stage(self, stage: str, artifact_path: str, seconds: float) -> None:
        """Record a cached stage artifact (diarization RTTM, ASR JSON...)."""
        self.artifacts[stage] = f"{artifact_path} ({seconds:.1f}s)"

    def finish(self, metrics: Dict[str, Any]) -> None:
        self.metrics = _jsonable(metrics)
        self.finished_at = datetime.datetime.utcnow().isoformat() + "Z"

    def save(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"run_{self.run_id}.json"
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2))
        return path


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float):
        return obj if obj != float("inf") else "inf"
    if hasattr(obj, "__dataclass_fields__"):
        return dataclasses.asdict(obj)
    return obj


def config_snapshot(cfg) -> Dict[str, Any]:
    """Snapshot of a Config instance's scalar fields (paths stringified)."""
    out = {}
    for f in dataclasses.fields(cfg):
        v = getattr(cfg, f.name)
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[f.name] = str(v) if isinstance(v, Path) else v
    return out
