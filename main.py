"""Main pipeline for multimodal Bangla talk-show indexing."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import List, Optional

import torch

from config import config
from models import FinalSegment
from engines.media import extract_audio, extract_frames
from engines.diarization import run_diarization
from engines.transcription import align_transcription_with_diarization
from engines.asr_lora import load_lora_whisper, transcribe_with_lora
from engines.vision import run_vision_pipeline
from engines.nlp import extract_speaker_names_from_intro
from engines.fusion import run_fusion_pipeline


def _get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def _write_srt(segments: List[FinalSegment], output_path: Path) -> None:
    """Write segments to SRT subtitle format."""
    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            def fmt(t: float) -> str:
                h = int(t // 3600)
                m = int((t % 3600) // 60)
                s = int(t % 60)
                ms = int((t - int(t)) * 1000)
                return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

            f.write(f"{i}\n")
            f.write(f"{fmt(seg.start)} --> {fmt(seg.end)}\n")
            f.write(f"{seg.speaker}: {seg.text}\n\n")


def _write_json(segments: List[FinalSegment], output_path: Path) -> None:
    """Write segments to JSON."""
    data = [
        {
            "start": seg.start,
            "end": seg.end,
            "speaker": seg.speaker,
            "text": seg.text,
            "confidence": seg.confidence,
        }
        for seg in segments
    ]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def run_pipeline(
    video_path: str,
    registry_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    use_lora: bool = False,
    lora_path: Optional[str] = None,
) -> List[FinalSegment]:
    """
    Run full multimodal speaker indexing pipeline.

    Args:
        video_path: Path to input video file.
        registry_dir: Optional path to face registry directory.
        output_dir: Optional output directory (defaults to config.DATA_OUTPUT_DIR).
        use_lora: Whether to use LoRA-adapted Whisper.
        lora_path: Path to LoRA adapter (required if use_lora=True).

    Returns:
        List of FinalSegment with speaker identities.
    """
    start_time = time.time()
    video = Path(video_path).resolve()

    if not video.exists():
        raise FileNotFoundError(f"Video not found: {video}")

    if registry_dir:
        config.DATA_REGISTRY_DIR = Path(registry_dir).resolve()
        config.DATA_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

    out_dir = Path(output_dir).resolve() if output_dir else config.DATA_OUTPUT_DIR
    # Update global config output dir so all engines use it
    config.DATA_OUTPUT_DIR = out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] Extracting audio from {video.name}...")
    audio_path = extract_audio(str(video))
    print(f"    Audio saved to {audio_path}")
    torch.cuda.empty_cache()

    print(f"[2/6] Running speaker diarization...")
    diarization = run_diarization(audio_path)
    print(f"    Found {len(set(s.speaker_id for s in diarization))} speakers")
    torch.cuda.empty_cache()

    print(f"[3/6] Transcribing audio...")
    if use_lora:
        if not lora_path:
            raise ValueError("lora_path required when use_lora=True")
        model, processor = load_lora_whisper(lora_path=lora_path)
        transcribed = transcribe_with_lora(model, processor, audio_path, diarization)
        del model, processor
    else:
        transcribed = align_transcription_with_diarization(audio_path, diarization)
    print(f"    Generated {len(transcribed)} transcribed segments")
    torch.cuda.empty_cache()

    print(f"[4/6] Extracting frames...")
    frame_paths = extract_frames(str(video), fps=config.VISION_FPS)
    print(f"    Extracted {len(frame_paths)} frames")
    torch.cuda.empty_cache()

    print(f"[5/6] Running vision pipeline...")
    faces = run_vision_pipeline(str(video), frame_paths=frame_paths)
    print(f"    Detected {len(faces)} face occurrences")
    torch.cuda.empty_cache()

    print(f"[6/6] Extracting speaker names from intro...")
    ordered_names = extract_speaker_names_from_intro(transcribed)
    print(f"    Found names: {ordered_names}")
    torch.cuda.empty_cache()

    print(f"[7/7] Fusing modalities...")
    final_segments = run_fusion_pipeline(diarization, transcribed, faces, ordered_names)
    print(f"    Produced {len(final_segments)} final segments")
    torch.cuda.empty_cache()

    # Write outputs
    _write_json(final_segments, out_dir / "result.json")
    _write_srt(final_segments, out_dir / "subtitles.srt")
    print(f"    Outputs written to {out_dir}")

    # Compute RTF
    duration = _get_video_duration(str(video))
    elapsed = time.time() - start_time
    rtf = elapsed / duration if duration > 0 else 0.0

    print(f"\nPipeline completed in {elapsed:.1f}s (RTF: {rtf:.2f}x)")
    return final_segments


def main():
    parser = argparse.ArgumentParser(
        description="Multimodal Bangla Talk-Show Speaker Indexing Pipeline"
    )
    parser.add_argument("--input", "-i", required=True, help="Input video file path")
    parser.add_argument("--registry", "-r", help="Face registry directory path")
    parser.add_argument("--output_dir", "-o", help="Output directory")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA-adapted Whisper")
    parser.add_argument("--lora_path", help="Path to LoRA adapter (HF Hub or local)")

    args = parser.parse_args()

    try:
        run_pipeline(
            video_path=args.input,
            registry_dir=args.registry,
            output_dir=args.output_dir,
            use_lora=args.use_lora,
            lora_path=args.lora_path,
        )
    except Exception as e:
        print(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()