"""Voice-reference identity: enrolment, per-cluster voiceprints, matching.

Why this exists
---------------
Identity in this pipeline is decided by a cascade whose strongest acoustic-free
evidence is the face registry: "who is on screen". That answers the wrong
question for the failure this project is actually about. The measured split on
the RTV episode is ~95% speaker-name accuracy on non-overlapping turns against
~77% on overlapping ones, and in an overlap the camera shows *both* speakers --
so a face match, however confident, cannot say which of them is talking. A
voiceprint can: it is derived from the speech itself, not from who is visible.

This module is deliberately the mirror image of the face registry:

    face registry                     voice registry
    ----------------                  -------------------
    data/registry/<Name>.jpg          data/registry/<Name>.mp3
    ArcFace embedding (insightface)   speaker-verification embedding (pyannote)
    per-speaker majority vote         per-cluster duration-weighted centroid
    FACE_SIM_THRESHOLD/_MARGIN        VOICE_SIM_THRESHOLD/_MARGIN

Three properties are non-negotiable, and they are why the code looks the way it
does:

1. **Off by default.** ``config.VOICE_POLICY`` defaults to ``"off"`` and
   ``GatingFusion`` then behaves exactly as before. A new evidence source with
   uncalibrated gates must not be able to change an existing run's output.

2. **A mean is not a voiceprint.** A reference clip of a broadcast contains the
   interviewer, the panel, music and silence; a real episode cluster contains
   whichever speakers diarization confused. Averaging raw windows drags the
   centroid toward every voice in the file. Windows are therefore embedded
   independently and reduced with an *outlier-trimmed* centroid, and each
   cluster reports its own coherence so an impure cluster is visible rather
   than silently trusted.

3. **Measure before trusting.** Nothing here picks a threshold. Every cluster's
   full similarity vector against every enrolled speaker is written to
   ``voice_diagnostics.json``, alongside the enrolled-speaker cross-matrix,
   which is the impostor distribution for this exact registry. The gates are
   then set from that separation -- ``scripts/voice_verify.py`` re-scores a
   finished run's saved voiceprints offline, with no GPU and no audio.

No model is imported at module scope: ``engines.fusion`` and the evaluation
paths must stay importable (and testable) without pyannote or torch.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from config import config
from models import DiarizationSegment


# ---------------------------------------------------------------------------
# Vector + audio primitives (no model required — these are unit-testable)
# ---------------------------------------------------------------------------

def l2_normalize(x: np.ndarray) -> np.ndarray:
    """Unit-norm a vector, leaving a zero vector as zeros rather than NaN."""
    v = np.asarray(x, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 1e-12 else v


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two embeddings, clamped against float drift."""
    return float(np.clip(float(np.dot(l2_normalize(a), l2_normalize(b))), -1.0, 1.0))


def robust_centroid(
    embeddings: Sequence[np.ndarray] | np.ndarray,
    trim: float = 0.25,
    min_keep: int = 3,
    iters: int = 2,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Trimmed-mean centroid of an embedding set.

    Returns ``(centroid, coherence, kept_mask)`` where coherence is the mean
    cosine of the retained embeddings to the centroid. This is the single
    reason a reference clip of a live broadcast can be used as enrolment: the
    lowest-similarity quarter of its windows is discarded each pass, so the
    windows that contain the *other* voices in the clip stop pulling the
    centroid away from the target speaker.

    With fewer than three embeddings no trimming happens — there is nothing to
    trim with, and a 1- or 2-window voiceprint is reported as such (low
    ``n_kept``) rather than being silently rejected.
    """
    E = np.atleast_2d(np.asarray(embeddings, dtype=np.float32))
    if E.size == 0:
        raise ValueError("cannot build a centroid from zero embeddings")
    E = np.stack([l2_normalize(row) for row in E])

    if E.shape[0] == 1:
        return E[0], 1.0, np.ones(1, dtype=bool)

    keep = np.ones(E.shape[0], dtype=bool)
    for _ in range(max(0, int(iters))):
        if int(keep.sum()) <= max(min_keep, 1):
            break
        centroid = l2_normalize(E[keep].mean(axis=0))
        sims = E @ centroid
        cut = float(np.quantile(sims[keep], trim))
        proposed = keep & (sims >= cut)
        if int(proposed.sum()) < max(min_keep, 1) or proposed.sum() == keep.sum():
            break
        keep = proposed

    centroid = l2_normalize(E[keep].mean(axis=0))
    coherence = float(np.mean(E[keep] @ centroid)) if keep.any() else 0.0
    return centroid, coherence, keep


def decode_audio(path: str | Path, sr: Optional[int] = None) -> np.ndarray:
    """Decode any ffmpeg-readable file to mono float32 in [-1, 1].

    ffmpeg is used rather than librosa/soundfile so that mp3, m4a and the
    episode's own WAV all take one code path. The pipeline already requires
    ffmpeg for media extraction, and going through it removes the class of bug
    where a reference clip decodes to silence because a codec backend was not
    installed. Returns the float32 signal; use :func:`decode_pcm16` when the
    caller only needs slices of a long file.
    """
    return decode_pcm16(path, sr).astype(np.float32) / 32768.0


def decode_pcm16(path: str | Path, sr: Optional[int] = None) -> np.ndarray:
    """Decode to mono int16. Halves the memory of a long episode's audio."""
    rate = int(sr or config.AUDIO_SR)
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"audio not found: {src}")
    cmd = [
        "ffmpeg", "-v", "error", "-nostdin", "-i", str(src),
        "-vn", "-ac", "1", "-ar", str(rate), "-f", "s16le", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", "replace").strip()[:400]
        raise RuntimeError(f"ffmpeg failed to decode {src.name}: {detail}")
    if not proc.stdout:
        raise RuntimeError(f"ffmpeg decoded zero samples from {src.name}")
    return np.frombuffer(proc.stdout, dtype="<i2").copy()


def _rms(window: np.ndarray) -> float:
    if window.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(window, dtype=np.float64))))


def window_spans(
    duration: float,
    window: float,
    hop: float,
    regions: Optional[Sequence[Tuple[float, float]]] = None,
    min_region: float = 0.4,
) -> List[Tuple[float, float]]:
    """Window spans in seconds, optionally confined to speech regions.

    When ``regions`` is given, windows never straddle a region boundary: the
    cluster voiceprint must be built only from that speaker's own turns, so a
    window spanning a turn boundary would embed two speakers and pollute the
    centroid. A region shorter than one window is returned whole (a 1.2 s turn
    is still evidence) unless it is below ``min_region``.
    """
    spans: List[Tuple[float, float]] = []
    for start, end in (regions if regions is not None else [(0.0, duration)]):
        span = float(end) - float(start)
        if span < min_region:
            continue
        if span <= window + 1e-6:
            spans.append((float(start), float(end)))
            continue
        t = float(start)
        last_end = float(start)
        while t + window <= float(end) + 1e-6:
            spans.append((t, t + window))
            last_end = t + window
            t += hop
        # A tail shorter than `hop` would otherwise be dropped; anchor the final
        # window to the region end instead of losing the last of the speech.
        if float(end) - last_end > 0.25 and float(end) - window > float(start) + 1e-6:
            spans.append((float(end) - window, float(end)))
    return spans


def subsample(items: List, cap: int) -> List:
    """Deterministically keep at most ``cap`` items, evenly spaced.

    Even spacing rather than the first N: taking the first 40 windows of a
    12-minute clip would enrol one topic and one microphone position.
    """
    if cap <= 0 or len(items) <= cap:
        return items
    idx = np.linspace(0, len(items) - 1, cap).round().astype(int)
    return [items[i] for i in sorted(set(int(i) for i in idx))]


def _write_wav(path: Path, wav: np.ndarray, sr: int) -> None:
    """Write mono float32 as 16-bit PCM using the standard library."""
    clipped = np.clip(np.asarray(wav, dtype=np.float32), -1.0, 1.0)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(int(sr))
        fh.writeframes((clipped * 32767.0).astype("<i2").tobytes())


def registry_audio_files(registry_dir: Optional[Path] = None) -> List[Path]:
    """Sorted audio files in the registry directory (the voice references)."""
    root = Path(registry_dir or config.DATA_REGISTRY_DIR)
    if not root.is_dir():
        return []
    exts = tuple(config.VOICE_AUDIO_EXTS)
    return sorted(
        (p for p in root.iterdir() if p.is_file() and p.suffix.lower() in exts),
        key=lambda p: p.name.lower(),
    )


def name_from_stem(stem: str) -> str:
    """Registry filename -> display name, identical to the face convention.

    ``Dr_Abdul_Noor_Tushar.mp3`` and ``Dr_Abdul_Noor_Tushar.jpg`` must resolve
    to the same identity string, or the two registries could never be compared.
    """
    return stem.replace("_", " ").strip()


def _probe_sample_rate(model) -> Optional[int]:
    """Sample rate the loaded model expects, if it advertises one."""
    for attr in ("audio", "specifications"):
        obj = getattr(model, attr, None)
        rate = getattr(obj, "sample_rate", None)
        if isinstance(rate, int) and rate > 0:
            return rate
    return None


# ---------------------------------------------------------------------------
# Enrolment
# ---------------------------------------------------------------------------

@dataclass
class Voiceprint:
    """An enrolled speaker's voiceprint and the evidence behind it."""

    name: str
    embedding: np.ndarray
    n_windows: int = 0
    n_kept: int = 0
    duration_sec: float = 0.0
    coherence: float = 0.0
    sources: List[str] = field(default_factory=list)

    def report(self) -> Dict:
        return {
            "n_windows": self.n_windows,
            "n_kept": self.n_kept,
            "duration_sec": round(self.duration_sec, 1),
            "coherence": round(self.coherence, 4),
            "sources": list(self.sources),
        }


@dataclass
class ClusterVoice:
    """A diarization cluster's voiceprint and how much speech backs it."""

    speaker_id: str
    embedding: Optional[np.ndarray] = None
    n_turns: int = 0
    n_turns_used: int = 0
    speech_sec: float = 0.0
    used_sec: float = 0.0
    n_windows: int = 0
    n_kept: int = 0
    n_silent: int = 0
    coherence: float = 0.0
    skipped_short_sec: float = 0.0

    def report(self) -> Dict:
        return {
            "n_turns": self.n_turns,
            "n_turns_used": self.n_turns_used,
            "speech_sec": round(self.speech_sec, 1),
            "used_sec": round(self.used_sec, 1),
            "skipped_short_sec": round(self.skipped_short_sec, 1),
            "n_windows": self.n_windows,
            "n_kept": self.n_kept,
            "n_silent": self.n_silent,
            "coherence": round(self.coherence, 4),
            "has_voiceprint": self.embedding is not None,
        }


# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------

class VoiceEmbedder:
    """Thin wrapper over a pyannote speaker-verification embedding model.

    Loading is lazy and the model is chosen from a candidate list: the gated
    ``pyannote/wespeaker-voxceleb-resnet34-LM`` is preferred because the
    diarization pipeline already downloads it (so its licence is accepted for
    any account able to run diarization at all), with ``pyannote/embedding`` as
    the fallback. If neither loads, callers get a RuntimeError with the real
    Hugging Face error attached, and the caller is expected to continue
    face-only rather than fail the episode.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        hf_token: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self._requested_model = model_id or config.VOICE_MODEL or None
        self._hf_token = (
            hf_token if hf_token is not None
            else (config.HF_TOKEN or os.getenv("HF_TOKEN", ""))
        )
        self._device = device
        self._inference = None
        self._model = None
        self.model_id: Optional[str] = None
        self.sample_rate: int = int(config.AUDIO_SR)

    @property
    def loaded(self) -> bool:
        return self._inference is not None

    def load(self) -> "VoiceEmbedder":
        if self._inference is not None:
            return self

        import torch  # lazy: keeps this module importable without torch
        from pyannote.audio import Inference, Model

        candidates = ([self._requested_model] if self._requested_model
                      else list(config.VOICE_MODEL_CANDIDATES))
        errors: List[str] = []
        for model_id in candidates:
            if not model_id:
                continue
            try:
                model = self._from_pretrained(Model, model_id)
            except Exception as exc:  # gated 401, no network, bad id, ...
                errors.append(f"{model_id}: {exc.__class__.__name__}: {exc}")
                continue

            inference = Inference(model, window="whole")
            device = self._device or config.DEVICE
            try:
                if device == "cuda" and torch.cuda.is_available():
                    inference.to(torch.device("cuda"))
                elif device == "mps" and torch.backends.mps.is_available():
                    inference.to(torch.device("mps"))
            except Exception as exc:
                # A device move failing must not lose a loaded model: CPU
                # embedding of a few hundred windows is slow but correct.
                print(f"[voice] could not move model to {device} "
                      f"({exc.__class__.__name__}: {exc}); using CPU")

            self._model, self._inference, self.model_id = model, inference, model_id
            # Prefer the rate the model itself advertises, then the inference
            # wrapper's own expectation, before falling back to 16 kHz.
            rate = (getattr(inference, "sample_rate", None)
                    or _probe_sample_rate(model)
                    or config.AUDIO_SR)
            self.sample_rate = int(rate)
            print(f"[voice] embedding model: {model_id} "
                  f"(device={device}, sr={self.sample_rate})")
            return self

        raise RuntimeError(
            "no voice embedding model could be loaded. Tried: "
            + "; ".join(errors)
            + ". Set HF_TOKEN and accept the model licence, or leave "
              "VOICE_POLICY=off to run face-only.")

    def _from_pretrained(self, Model, model_id: str):
        """Handle the token= / use_auth_token= API split across pyannote versions."""
        attempts = []
        if self._hf_token:
            attempts.append({"token": self._hf_token})
            attempts.append({"use_auth_token": self._hf_token})
        attempts.append({})
        last_type_error: Optional[TypeError] = None
        for kwargs in attempts:
            try:
                return Model.from_pretrained(model_id, **kwargs)
            except TypeError as exc:
                last_type_error = exc
                continue
        if last_type_error is not None:
            raise last_type_error
        raise RuntimeError(f"could not load {model_id}")

    def embed_window(self, wav: np.ndarray) -> np.ndarray:
        """L2-normalised embedding of one already-sliced mono window."""
        self.load()
        import torch

        signal = np.asarray(wav, dtype=np.float32).reshape(-1)
        if signal.size == 0:
            raise ValueError("cannot embed an empty window")
        tensor = torch.from_numpy(signal).unsqueeze(0)  # (channel, time)
        try:
            out = self._inference(
                {"waveform": tensor, "sample_rate": self.sample_rate})
        except Exception:
            # Documented pyannote path is a file on disk; a version whose
            # in-memory dict handling differs must not lose the whole feature.
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
                tmp = Path(fh.name)
            try:
                _write_wav(tmp, signal, self.sample_rate)
                out = self._inference(str(tmp))
            finally:
                tmp.unlink(missing_ok=True)
        return l2_normalize(np.asarray(out, dtype=np.float32).reshape(-1))

    def embed_windows(self, windows: Sequence[np.ndarray]) -> np.ndarray:
        """Embed many windows, returning an (n, dim) matrix of normalised rows.

        A single window that fails (a decoder edge case, a device OOM on one
        shape) is skipped rather than aborting the run: one bad window among
        forty must not cost the episode its voiceprint.
        """
        rows: List[np.ndarray] = []
        for i, wav in enumerate(windows):
            try:
                rows.append(self.embed_window(wav))
            except Exception as exc:
                print(f"[voice] window {i} failed to embed "
                      f"({exc.__class__.__name__}: {exc}); skipped")
        if not rows:
            raise RuntimeError("no window could be embedded")
        return np.stack(rows)


# ---------------------------------------------------------------------------
# Enrolment from the registry
# ---------------------------------------------------------------------------

def _cache_signature(files: Sequence[Path], model_id: str) -> str:
    """Invalidate the cache when a reference file or the model changes.

    mtime is included, not just size: re-downloading a corrected clip for the
    same person at the same bitrate is exactly the case a size-only key would
    silently keep serving the old voiceprint for.
    """
    parts = [f"model={model_id}", f"v=1"]
    for p in files:
        try:
            st = p.stat()
            parts.append(f"{p.name}:{st.st_size}:{int(st.st_mtime)}")
        except OSError:
            parts.append(f"{p.name}:missing")
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()


def _save_cache(path: Path, prints: Dict[str, Voiceprint], signature: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = sorted(prints)
    matrix = (np.stack([prints[n].embedding for n in names])
              if names else np.zeros((0, 0), dtype=np.float32))
    meta = {n: prints[n].report() for n in names}
    np.savez(
        path,
        names=np.array(json.dumps(names)),
        matrix=matrix.astype(np.float32),
        meta=np.array(json.dumps(meta)),
        signature=np.array(signature),
    )


def _load_cache(path: Path, signature: str) -> Optional[Dict[str, Voiceprint]]:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if str(data["signature"]) != signature:
                return None
            names = json.loads(str(data["names"]))
            matrix = np.asarray(data["matrix"], dtype=np.float32)
            meta = json.loads(str(data["meta"]))
    except Exception:
        return None  # a corrupt cache is a cache miss, never a failure

    prints: Dict[str, Voiceprint] = {}
    for i, name in enumerate(names):
        info = meta.get(name, {})
        prints[name] = Voiceprint(
            name=name,
            embedding=l2_normalize(matrix[i]),
            n_windows=int(info.get("n_windows", 0)),
            n_kept=int(info.get("n_kept", 0)),
            duration_sec=float(info.get("duration_sec", 0.0)),
            coherence=float(info.get("coherence", 0.0)),
            sources=list(info.get("sources", [])),
        )
    return prints


def enroll_voice_registry(
    registry_dir: Optional[Path] = None,
    embedder: Optional[VoiceEmbedder] = None,
    cache_path: Optional[Path] = None,
    force: bool = False,
) -> Dict[str, Voiceprint]:
    """Build one voiceprint per enrolled name from the registry audio files.

    Several clips for the same person are pooled and reduced in a single
    trimmed centroid, so ``Name.mp3`` plus ``Name_interview.wav`` strengthen one
    identity instead of competing. Returns ``{}`` when no audio references
    exist, which is the normal case for the face-only registry and must not be
    an error.
    """
    files = registry_audio_files(registry_dir)
    if not files:
        print("[voice] no audio references in the registry "
              f"({Path(registry_dir or config.DATA_REGISTRY_DIR)}) — "
              "voice evidence unavailable")
        return {}

    embedder = embedder or VoiceEmbedder()
    embedder.load()
    cache_path = Path(cache_path) if cache_path else (
        Path(registry_dir or config.DATA_REGISTRY_DIR) / "voiceprints.npz")
    signature = _cache_signature(files, embedder.model_id or "?")

    if not force:
        cached = _load_cache(cache_path, signature)
        if cached:
            print(f"[voice] enrolled {len(cached)} identities from cache "
                  f"({cache_path.name})")
            return cached

    by_name: Dict[str, List[Path]] = {}
    for p in files:
        by_name.setdefault(name_from_stem(p.stem), []).append(p)

    prints: Dict[str, Voiceprint] = {}
    for name, paths in sorted(by_name.items()):
        pooled: List[np.ndarray] = []
        duration = 0.0
        for path in paths:
            try:
                wav = decode_audio(path, embedder.sample_rate)
            except Exception as exc:
                print(f"[voice] {path.name} could not be decoded "
                      f"({exc.__class__.__name__}: {exc}); skipped")
                continue
            duration += len(wav) / float(embedder.sample_rate)
            spans = window_spans(
                len(wav) / float(embedder.sample_rate),
                config.VOICE_ENROLL_WINDOW_SEC,
                config.VOICE_ENROLL_HOP_SEC,
            )
            spans = subsample(spans, config.VOICE_MAX_ENROLL_WINDOWS)
            windows = [
                wav[int(s * embedder.sample_rate):int(e * embedder.sample_rate)]
                for s, e in spans
            ]
            windows = [w for w in windows if _rms(w) >= config.VOICE_MIN_RMS]
            if not windows:
                print(f"[voice] {path.name} yielded no speech windows "
                      f"(all below RMS {config.VOICE_MIN_RMS}); skipped")
                continue
            try:
                pooled.append(embedder.embed_windows(windows))
            except Exception as exc:
                print(f"[voice] {path.name} failed to embed "
                      f"({exc.__class__.__name__}: {exc}); skipped")

        if not pooled:
            print(f"[voice] no usable audio for {name}; not enrolled")
            continue

        matrix = np.vstack(pooled)
        centroid, coherence, kept = robust_centroid(matrix)
        prints[name] = Voiceprint(
            name=name,
            embedding=centroid,
            n_windows=int(matrix.shape[0]),
            n_kept=int(kept.sum()),
            duration_sec=duration,
            coherence=coherence,
            sources=[p.name for p in paths],
        )
        print(f"[voice] enrolled {name}: {matrix.shape[0]} windows "
              f"({int(kept.sum())} kept), {duration:.0f}s, "
              f"coherence={coherence:.3f}")

    if prints:
        _save_cache(cache_path, prints, signature)
        print(f"[voice] voiceprint cache -> {cache_path}")
    return prints


# ---------------------------------------------------------------------------
# Per-cluster voiceprints from the episode itself
# ---------------------------------------------------------------------------

def cluster_voiceprints(
    diarization: Sequence[DiarizationSegment],
    audio_path: str | Path,
    embedder: Optional[VoiceEmbedder] = None,
    min_turn_sec: Optional[float] = None,
    max_windows: Optional[int] = None,
) -> Dict[str, ClusterVoice]:
    """Build one voiceprint per diarization speaker from their own turns.

    Only the speaker's own turn interiors are windowed, so an overlapped turn
    contributes the mixture rather than a clean voice -- which is reported, not
    hidden: ``n_silent`` and ``coherence`` expose exactly that, and the
    duration accounting (``used_sec`` vs ``skipped_short_sec``) shows how much
    of a cluster was too short to embed at all.
    """
    min_turn_sec = (config.VOICE_MIN_TURN_SEC if min_turn_sec is None
                    else float(min_turn_sec))
    max_windows = (config.VOICE_MAX_CLUSTER_WINDOWS if max_windows is None
                   else int(max_windows))

    embedder = embedder or VoiceEmbedder()
    embedder.load()
    sr = embedder.sample_rate
    pcm = decode_pcm16(audio_path, sr)
    total_sec = len(pcm) / float(sr)

    by_speaker: Dict[str, List[DiarizationSegment]] = {}
    for seg in sorted(diarization, key=lambda s: s.start):
        by_speaker.setdefault(seg.speaker_id, []).append(seg)

    out: Dict[str, ClusterVoice] = {}
    for speaker_id, turns in by_speaker.items():
        cluster = ClusterVoice(speaker_id=speaker_id, n_turns=len(turns))
        regions: List[Tuple[float, float]] = []
        for seg in turns:
            dur = float(seg.end) - float(seg.start)
            cluster.speech_sec += max(0.0, dur)
            if dur < min_turn_sec:
                cluster.skipped_short_sec += max(0.0, dur)
                continue
            regions.append((max(0.0, float(seg.start)),
                            min(total_sec, float(seg.end))))
        cluster.n_turns_used = len(regions)
        cluster.used_sec = sum(e - s for s, e in regions)

        if not regions:
            out[speaker_id] = cluster
            continue

        spans = window_spans(total_sec, config.VOICE_ENROLL_WINDOW_SEC,
                             config.VOICE_ENROLL_HOP_SEC, regions=regions)
        spans = subsample(spans, max_windows)
        windows: List[np.ndarray] = []
        for s, e in spans:
            w = pcm[int(s * sr):int(e * sr)]
            if w.size == 0 or _rms(w.astype(np.float32) / 32768.0) < config.VOICE_MIN_RMS:
                cluster.n_silent += 1
                continue
            windows.append(w.astype(np.float32) / 32768.0)

        if not windows:
            out[speaker_id] = cluster
            continue

        try:
            matrix = embedder.embed_windows(windows)
        except Exception as exc:
            print(f"[voice] cluster {speaker_id} failed to embed "
                  f"({exc.__class__.__name__}: {exc})")
            out[speaker_id] = cluster
            continue

        centroid, coherence, kept = robust_centroid(matrix)
        cluster.embedding = centroid
        cluster.n_windows = int(matrix.shape[0])
        cluster.n_kept = int(kept.sum())
        cluster.coherence = coherence
        out[speaker_id] = cluster

    for speaker_id, c in out.items():
        state = "ok" if c.embedding is not None else "no voiceprint"
        print(f"[voice] cluster {speaker_id}: {c.n_turns} turns, "
              f"{c.used_sec:.0f}s used, {c.n_windows} windows "
              f"({c.n_kept} kept) -> {state}"
              + (f", coherence={c.coherence:.3f}" if c.embedding is not None else ""))
    return out


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

@dataclass
class VoiceMatch:
    """The outcome of matching one cluster against the enrolled voiceprints."""

    speaker_id: str
    name: Optional[str] = None
    similarity: float = 0.0
    runner_up: Optional[str] = None
    runner_up_similarity: float = 0.0
    margin: float = 0.0
    accepted: bool = False
    reason: str = "no_voiceprint"
    scores: Dict[str, float] = field(default_factory=dict)

    def report(self) -> Dict:
        return {
            "name": self.name,
            "similarity": round(self.similarity, 4),
            "runner_up": self.runner_up,
            "runner_up_similarity": round(self.runner_up_similarity, 4),
            "margin": round(self.margin, 4),
            "accepted": self.accepted,
            "reason": self.reason,
            "scores": {k: round(v, 4) for k, v in
                       sorted(self.scores.items(), key=lambda kv: -kv[1])},
        }


def match_clusters(
    clusters: Dict[str, ClusterVoice],
    enrolled: Dict[str, Voiceprint],
    threshold: Optional[float] = None,
    margin: Optional[float] = None,
    min_coherence: Optional[float] = None,
) -> Dict[str, VoiceMatch]:
    """Rank every cluster against every enrolled speaker and gate the result.

    Mirrors the face gate exactly -- an absolute floor plus a top-1/top-2
    margin -- because the failure mode is identical: a cluster that is almost
    equally close to two enrolled people must abstain rather than pick one.
    The margin matters more here than for faces, since two panellists on the
    same microphone in the same studio share channel and room colouration.
    """
    threshold = (config.VOICE_SIM_THRESHOLD if threshold is None
                 else float(threshold))
    margin = config.VOICE_SIM_MARGIN if margin is None else float(margin)
    min_coherence = (config.VOICE_MIN_COHERENCE if min_coherence is None
                     else float(min_coherence))

    out: Dict[str, VoiceMatch] = {}
    for speaker_id, cluster in clusters.items():
        match = VoiceMatch(speaker_id=speaker_id)
        if cluster.embedding is None or not enrolled:
            match.reason = ("no_enrolled_voiceprints" if not enrolled
                            else "no_voiceprint")
            out[speaker_id] = match
            continue

        scores = {name: cosine(cluster.embedding, vp.embedding)
                  for name, vp in enrolled.items()}
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        match.scores = scores
        match.name = ranked[0][0]
        match.similarity = ranked[0][1]
        if len(ranked) > 1:
            match.runner_up, match.runner_up_similarity = ranked[1]
        match.margin = match.similarity - match.runner_up_similarity

        if min_coherence > 0.0 and cluster.coherence < min_coherence:
            match.accepted = False
            match.reason = "low_coherence"
        elif match.similarity < threshold:
            match.accepted = False
            match.reason = "below_threshold"
        elif match.margin < margin:
            match.accepted = False
            match.reason = "ambiguous_margin"
        else:
            match.accepted = True
            match.reason = "ok"
        out[speaker_id] = match
    return out


def accepted_matches(matches: Dict[str, VoiceMatch]) -> Dict[str, Tuple[str, float]]:
    """Accepted matches as the ``speaker_id -> (name, confidence)`` fusion expects.

    The reported confidence is the cosine similarity itself, on the same
    "higher is better, ~1.0 is certain" scale as the face pass, so the cascade
    can compare the two sources without a conversion table.
    """
    return {spk: (m.name, round(m.similarity, 3))
            for spk, m in matches.items()
            if m.accepted and m.name}


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def enrolled_cross_matrix(enrolled: Dict[str, Voiceprint]) -> Dict[str, Dict[str, float]]:
    """Similarity between every pair of enrolled speakers.

    This is the impostor distribution for *this registry*, and it is the single
    most useful number for choosing VOICE_SIM_THRESHOLD: if two enrolled people
    already score 0.7 against each other, no threshold above 0.7 can ever
    separate them, and the honest conclusion is that this registry cannot
    distinguish those two by voice.
    """
    names = sorted(enrolled)
    return {
        a: {b: round(cosine(enrolled[a].embedding, enrolled[b].embedding), 4)
            for b in names if b != a}
        for a in names
    }


def voice_diagnostics(
    clusters: Dict[str, ClusterVoice],
    enrolled: Dict[str, Voiceprint],
    matches: Dict[str, VoiceMatch],
    embedder: Optional[VoiceEmbedder] = None,
    policy: Optional[str] = None,
) -> Dict:
    """Everything needed to judge -- and re-threshold -- the voice evidence."""
    policy = policy or config.VOICE_POLICY
    accepted = [m for m in matches.values() if m.accepted]
    missing = [spk for spk, c in clusters.items() if c.embedding is None]
    reasons: Dict[str, int] = {}
    for m in matches.values():
        if not m.accepted:
            reasons[m.reason] = reasons.get(m.reason, 0) + 1

    return {
        "_what": ("Voice-reference evidence. Read the summary before turning "
                  "VOICE_POLICY on: `enrolled_cross` is the impostor "
                  "distribution for this registry and `clusters` holds every "
                  "cluster's full similarity vector, so the gates are set from "
                  "measured separation rather than guessed."),
        "policy": policy,
        "model": (embedder.model_id if embedder else None),
        "thresholds": {
            "VOICE_SIM_THRESHOLD": config.VOICE_SIM_THRESHOLD,
            "VOICE_SIM_MARGIN": config.VOICE_SIM_MARGIN,
            "VOICE_OVERRIDE_MARGIN": config.VOICE_OVERRIDE_MARGIN,
            "VOICE_MIN_COHERENCE": config.VOICE_MIN_COHERENCE,
            "VOICE_MIN_TURN_SEC": config.VOICE_MIN_TURN_SEC,
        },
        "summary": {
            "n_enrolled": len(enrolled),
            "enrolled_names": sorted(enrolled),
            "n_clusters": len(clusters),
            "n_clusters_with_voiceprint": len(clusters) - len(missing),
            "clusters_without_voiceprint": sorted(missing),
            "n_accepted": len(accepted),
            "accepted": {m.speaker_id: m.name for m in accepted},
            "rejected_reasons": reasons,
        },
        "enrolled": {n: vp.report() for n, vp in sorted(enrolled.items())},
        "enrolled_cross": enrolled_cross_matrix(enrolled),
        "clusters": {spk: {**c.report(), **matches[spk].report()}
                     for spk, c in sorted(clusters.items())},
    }


def threshold_sweep(
    clusters: Dict[str, ClusterVoice],
    enrolled: Dict[str, Voiceprint],
    truth: Optional[Dict[str, str]] = None,
    thresholds: Optional[Sequence[float]] = None,
    margins: Optional[Sequence[float]] = None,
) -> List[Dict]:
    """Re-score saved voiceprints across a threshold grid, offline.

    With ``truth`` (cluster -> canonical name) each row reports how many
    clusters the gate would name correctly, name wrongly, or leave alone. That
    is the whole point: the voice gates are chosen from the resulting curve on
    this episode, on this registry, rather than from a literature default.

    A row where ``wrong > 0`` is a regression risk and must be read next to the
    accuracy it buys -- naming one more cluster correctly while misnaming
    another is not obviously a gain in a time-weighted metric.
    """
    if thresholds is None:
        thresholds = [round(0.25 + 0.05 * i, 2) for i in range(12)]
    if margins is None:
        margins = [round(0.00 + 0.02 * i, 2) for i in range(8)]

    rows: List[Dict] = []
    for t in thresholds:
        for m in margins:
            matches = match_clusters(clusters, enrolled, threshold=t, margin=m)
            row: Dict = {
                "threshold": round(float(t), 3),
                "margin": round(float(m), 3),
                "n_accepted": sum(1 for x in matches.values() if x.accepted),
            }
            if truth:
                correct, wrong, missed, absent = [], [], [], []
                for spk, match in matches.items():
                    expected = truth.get(spk)
                    if not match.accepted:
                        if expected:
                            missed.append(spk)
                        continue
                    if expected is None:
                        absent.append((spk, match.name))
                    elif match.name == expected:
                        correct.append(spk)
                    else:
                        wrong.append((spk, match.name, expected))
                row.update({
                    "correct": len(correct),
                    "wrong": len(wrong),
                    "missed": len(missed),
                    "unlabelled_accepts": len(absent),
                    "accuracy": (round(len(correct) / len(truth), 3)
                                 if truth else None),
                    "wrong_detail": [f"{s}:{got}!={want}" for s, got, want in wrong],
                    "missed_detail": missed,
                })
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Persistence for offline re-scoring
# ---------------------------------------------------------------------------

def save_voiceprints(
    path: Path,
    clusters: Dict[str, ClusterVoice],
    enrolled: Dict[str, Voiceprint],
) -> None:
    """Persist raw embeddings so gates can be re-swept without the GPU.

    Storing the vectors (not just the decisions) is what makes calibration
    cheap: voice_verify.py can explore the full threshold grid in milliseconds
    on a laptop, days after the Kaggle session that produced the audio is gone.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cluster_ids = [s for s in sorted(clusters) if clusters[s].embedding is not None]
    enrolled_names = sorted(enrolled)
    np.savez(
        path,
        cluster_ids=np.array(json.dumps(cluster_ids)),
        cluster_matrix=np.stack([clusters[s].embedding for s in cluster_ids])
        if cluster_ids else np.zeros((0, 0), dtype=np.float32),
        cluster_meta=np.array(json.dumps({s: clusters[s].report() for s in cluster_ids})),
        enrolled_names=np.array(json.dumps(enrolled_names)),
        enrolled_matrix=np.stack([enrolled[n].embedding for n in enrolled_names])
        if enrolled_names else np.zeros((0, 0), dtype=np.float32),
        enrolled_meta=np.array(json.dumps(
            {n: enrolled[n].report() for n in enrolled_names})),
    )


def load_voiceprints(path: Path) -> Tuple[Dict[str, ClusterVoice], Dict[str, Voiceprint]]:
    """Inverse of :func:`save_voiceprints`, for offline calibration."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no saved voiceprints at {path}")
    with np.load(path, allow_pickle=False) as data:
        cluster_ids = json.loads(str(data["cluster_ids"]))
        cluster_matrix = np.asarray(data["cluster_matrix"], dtype=np.float32)
        cluster_meta = json.loads(str(data["cluster_meta"]))
        enrolled_names = json.loads(str(data["enrolled_names"]))
        enrolled_matrix = np.asarray(data["enrolled_matrix"], dtype=np.float32)
        enrolled_meta = json.loads(str(data["enrolled_meta"]))

    clusters: Dict[str, ClusterVoice] = {}
    for i, sid in enumerate(cluster_ids):
        info = cluster_meta.get(sid, {})
        clusters[sid] = ClusterVoice(
            speaker_id=sid,
            embedding=l2_normalize(cluster_matrix[i]),
            n_turns=int(info.get("n_turns", 0)),
            n_turns_used=int(info.get("n_turns_used", 0)),
            speech_sec=float(info.get("speech_sec", 0.0)),
            used_sec=float(info.get("used_sec", 0.0)),
            n_windows=int(info.get("n_windows", 0)),
            n_kept=int(info.get("n_kept", 0)),
            n_silent=int(info.get("n_silent", 0)),
            coherence=float(info.get("coherence", 0.0)),
        )
    enrolled: Dict[str, Voiceprint] = {}
    for i, name in enumerate(enrolled_names):
        info = enrolled_meta.get(name, {})
        enrolled[name] = Voiceprint(
            name=name,
            embedding=l2_normalize(enrolled_matrix[i]),
            n_windows=int(info.get("n_windows", 0)),
            n_kept=int(info.get("n_kept", 0)),
            duration_sec=float(info.get("duration_sec", 0.0)),
            coherence=float(info.get("coherence", 0.0)),
            sources=list(info.get("sources", [])),
        )
    return clusters, enrolled


# ---------------------------------------------------------------------------
# One-call entry point used by the pipeline
# ---------------------------------------------------------------------------

@dataclass
class VoiceEvidence:
    """Everything one voice stage produced, including the raw vectors.

    Carrying ``clusters``/``enrolled`` back to the caller is deliberate: the
    embedding pass is the expensive part of this feature, and the only way to
    re-threshold a finished run offline (scripts/voice_verify.py) is to have
    kept the vectors that pass produced.
    """

    matches: Dict[str, Tuple[str, float]] = field(default_factory=dict)
    diagnostics: Dict = field(default_factory=dict)
    clusters: Optional[Dict[str, ClusterVoice]] = None
    enrolled: Optional[Dict[str, Voiceprint]] = None
    embedder: Optional[VoiceEmbedder] = None
    policy: str = "off"
    note: str = ""

    @property
    def available(self) -> bool:
        return bool(self.matches)

    def persist(self, out_dir: Path) -> None:
        """Write voice_diagnostics.json and voiceprints.npz beside the result.

        Both are plain artefacts of the run: the JSON is what a human reads to
        judge the evidence, the npz is what a later calibration reads. Neither
        affects result.json, so a failure here is reported and swallowed.
        """
        out_dir = Path(out_dir)
        if self.diagnostics:
            (out_dir / "voice_diagnostics.json").write_text(
                json.dumps(self.diagnostics, ensure_ascii=False, indent=2),
                encoding="utf-8")
            print(f"    voice_diagnostics.json <- policy={self.policy}")
        if self.clusters is not None and self.enrolled:
            save_voiceprints(out_dir / "voiceprints.npz", self.clusters, self.enrolled)
            print("    voiceprints.npz <- offline threshold re-scoring "
                  f"({len(self.clusters)} clusters x {len(self.enrolled)} enrolled)")


def build_voice_evidence(
    diarization: Sequence[DiarizationSegment],
    audio_path: str | Path,
    embedder: Optional[VoiceEmbedder] = None,
) -> "VoiceEvidence":
    """Enrol, embed every cluster, match, and package everything for the caller.

    Returns a :class:`VoiceEvidence`. ``matches`` is what fusion consumes; the
    raw ``clusters``/``enrolled`` are returned too so the caller can persist
    them for offline re-thresholding without paying for a second pass over the
    audio (the embedding pass is the expensive part of this feature).

    Any failure -- no reference audio, a gated model, an unreadable clip --
    degrades to "no voice evidence" with a printed reason, because a missing
    supporting evidence source must never fail an otherwise valid episode.
    """
    policy = config.VOICE_POLICY
    evidence = VoiceEvidence(policy=policy)
    if policy == "off":
        evidence.note = "VOICE_POLICY=off"
        return evidence

    try:
        embedder = embedder or VoiceEmbedder()
        enrolled = enroll_voice_registry(embedder=embedder)
        if not enrolled:
            evidence.note = "no audio references in the registry"
            return evidence
        clusters = cluster_voiceprints(diarization, audio_path, embedder=embedder)
        matches = match_clusters(clusters, enrolled)
        evidence.matches = accepted_matches(matches)
        evidence.diagnostics = voice_diagnostics(
            clusters, enrolled, matches, embedder=embedder, policy=policy)
        evidence.clusters = clusters
        evidence.enrolled = enrolled
        evidence.embedder = embedder
        return evidence
    except Exception as exc:
        print(f"[voice] voice evidence unavailable "
              f"({exc.__class__.__name__}: {exc}); continuing face-only")
        evidence.note = f"{exc.__class__.__name__}: {exc}"
        return evidence
