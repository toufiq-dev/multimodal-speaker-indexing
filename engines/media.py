"""Media extraction utilities for audio and video frames."""

from __future__ import annotations

import subprocess
import shutil
from pathlib import Path
from typing import List

from config import config


def _ensure_dir(path: Path) -> None:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def _run_ffmpeg(cmd: List[str], description: str) -> None:
    """Run ffmpeg command with error handling."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"{description} failed (exit code {e.returncode}): {e.stderr.strip()}"
        ) from e
    except FileNotFoundError as e:
        raise RuntimeError(
            f"ffmpeg not found. Please install ffmpeg: {e}"
        ) from e


def extract_audio(video_path: str) -> str:
    """
    Extract audio from video as 16kHz mono 16-bit WAV using ffmpeg.

    Args:
        video_path: Path to input video file.

    Returns:
        Absolute path to extracted WAV file.

    Raises:
        RuntimeError: If ffmpeg fails or video file doesn't exist.
    """
    video = Path(video_path).resolve()
    if not video.exists():
        raise FileNotFoundError(f"Video file not found: {video}")

    # Scratch, not output: on Kaggle DATA_OUTPUT_DIR is committed as notebook
    # output, and a 53-minute WAV at 16 kHz is ~100 MB of dead weight there.
    output_dir = config.SCRATCH_DIR / "audio"
    _ensure_dir(output_dir)

    output_path = output_dir / f"{video.stem}.wav"

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(config.AUDIO_SR),
        "-ac", "1",
        "-f", "wav",
        str(output_path),
    ]

    try:
        _run_ffmpeg(cmd, "Audio extraction")
    except Exception:
        if output_path.exists():
            output_path.unlink(missing_ok=True)
        raise

    return str(output_path.resolve())


def extract_frames(video_path: str, fps: int = 1) -> List[str]:
    """
    Extract frames from video at specified FPS using ffmpeg.

    Args:
        video_path: Path to input video file.
        fps: Frames per second to extract (default: 1).

    Returns:
        Sorted list of absolute paths to extracted JPEG frames.

    Raises:
        RuntimeError: If ffmpeg fails or video file doesn't exist.
    """
    video = Path(video_path).resolve()
    if not video.exists():
        raise FileNotFoundError(f"Video file not found: {video}")

    # Scratch, not output: ~3k JPEGs for a 53-minute show at 1 FPS.
    output_dir = config.SCRATCH_DIR / "frames"
    _ensure_dir(output_dir)

    # Stale frames from a previous, longer video would otherwise be picked up
    # by the glob below and silently attributed to this run's timeline.
    for stale in output_dir.glob("frame_*.jpg"):
        stale.unlink(missing_ok=True)

    output_pattern = output_dir / "frame_%06d.jpg"

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video),
        "-vf", f"fps={fps}",
        "-q:v", "2",
        str(output_pattern),
    ]

    try:
        _run_ffmpeg(cmd, "Frame extraction")
    except Exception:
        for f in output_dir.glob("frame_*.jpg"):
            f.unlink(missing_ok=True)
        raise

    frames = sorted(output_dir.glob("frame_*.jpg"))
    if not frames:
        raise RuntimeError(
            f"ffmpeg reported success but produced no frames in {output_dir}. "
            f"Check that {video.name} contains a video stream."
        )
    return [str(f.resolve()) for f in frames]