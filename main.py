"""Main pipeline for multimodal Bangla talk-show indexing."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import List, Optional

from config import config
from models import FinalSegment
from runtime import describe_devices, release_gpu_memory
from engines.media import extract_audio, extract_frames
from engines.diarization import run_diarization
from engines.transcription import align_transcription_with_diarization
from engines.asr_lora import load_lora_whisper, transcribe_with_lora
from engines.nlp import extract_speaker_names_from_intro
from engines.fusion import run_fusion_pipeline

TOTAL_STAGES = 8


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


def _build_rag_index(segments: List[FinalSegment], out_dir: Path):
    """Stage 8: speaker-aware RAG index over the final transcript.

    Documented as a required post-pipeline stage (Recall@k evaluation) but
    previously unreachable — nothing in run_pipeline or the notebook ever
    constructed SpeakerAwareRAG. Imported lazily so the heavy
    sentence-transformers/chromadb import is not paid by callers that only
    want result.json.
    """
    from engines.rag import SpeakerAwareRAG

    rag = SpeakerAwareRAG(persist_dir=str(out_dir / "rag_index"))
    n_chunks = rag.ingest(segments)
    rag.close()
    return n_chunks


def run_pipeline(
    video_path: str,
    registry_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    use_lora: bool = False,
    lora_path: Optional[str] = None,
    build_rag: bool = True,
) -> List[FinalSegment]:
    """
    Run full multimodal speaker indexing pipeline.

    Args:
        video_path: Path to input video file.
        registry_dir: Optional path to face registry directory.
        output_dir: Optional output directory (defaults to config.DATA_OUTPUT_DIR).
        use_lora: Whether to use LoRA-adapted Whisper.
        lora_path: Path to LoRA adapter (required if use_lora=True).
        build_rag: Build the speaker-aware RAG index after writing outputs.

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

    print(describe_devices())
    print(f"    scratch: {config.SCRATCH_DIR}  output: {out_dir}")

    print(f"[1/{TOTAL_STAGES}] Extracting audio from {video.name}...")
    audio_path = extract_audio(str(video))
    print(f"    Audio saved to {audio_path}")

    print(f"[2/{TOTAL_STAGES}] Running speaker diarization...")
    try:
        diarization = run_diarization(audio_path)
    except Exception as e:
        # The old fallback substituted a single 0-9999s SPEAKER_00 turn, which
        # yields a structurally valid result.json, a plausible RTF, and a DER
        # that is meaningless. A gated-model 401 is the single most likely
        # Kaggle failure here and must not be reported as a successful run.
        raise RuntimeError(
            f"Diarization failed ({e.__class__.__name__}: {e}). Verify HF_TOKEN is "
            f"set and that the gated licence for {config.PYANNOTE_MODEL} has been "
            f"accepted on huggingface.co."
        ) from e
    print(f"    Found {len(set(s.speaker_id for s in diarization))} speakers "
          f"across {len(diarization)} turns")
    release_gpu_memory()

    print(f"[3/{TOTAL_STAGES}] Transcribing audio...")
    if use_lora:
        if not lora_path:
            raise ValueError("lora_path required when use_lora=True")
        model, processor = load_lora_whisper(lora_path=lora_path)
        transcribed = transcribe_with_lora(model, processor, audio_path, diarization)
        del model, processor
    else:
        transcribed = align_transcription_with_diarization(audio_path, diarization)
    print(f"    Generated {len(transcribed)} transcribed segments")
    release_gpu_memory()

    print(f"[4/{TOTAL_STAGES}] Extracting frames...")
    frame_paths = extract_frames(str(video), fps=config.VISION_FPS)
    print(f"    Extracted {len(frame_paths)} frames")

    print(f"[5/{TOTAL_STAGES}] Running vision pipeline...")
    # Imported here, not at module scope: this is the only import that pulls in
    # insightface/onnxruntime, and the audio-only path should not pay for it
    # (nor fail on it when the NumPy ABI lock is not satisfied).
    from engines.vision import run_vision_pipeline
    faces = run_vision_pipeline(str(video), frame_paths=frame_paths)
    print(f"    Detected {len(faces)} face occurrences")
    release_gpu_memory()

    print(f"[6/{TOTAL_STAGES}] Extracting speaker names from intro...")
    ordered_names = extract_speaker_names_from_intro(transcribed)
    print(f"    Found names: {ordered_names}")
    release_gpu_memory()

    print(f"[7/{TOTAL_STAGES}] Fusing modalities...")
    final_segments = run_fusion_pipeline(diarization, transcribed, faces, ordered_names)
    print(f"    Produced {len(final_segments)} final segments")

    # Write outputs
    _write_json(final_segments, out_dir / "result.json")
    _write_srt(final_segments, out_dir / "subtitles.srt")
    print(f"    Outputs written to {out_dir}")

    print(f"[8/{TOTAL_STAGES}] Building speaker-aware RAG index...")
    if build_rag:
        try:
            n_chunks = _build_rag_index(final_segments, out_dir)
            print(f"    Indexed {n_chunks} chunks -> {out_dir / 'rag_index'}")
        except Exception as e:
            # Retrieval is a post-hoc consumer of result.json, which is already
            # on disk; a RAG failure must not discard a completed transcript.
            print(f"    RAG indexing failed ({e.__class__.__name__}: {e}); "
                  f"result.json is unaffected — re-index with "
                  f"SpeakerAwareRAG().ingest('{out_dir / 'result.json'}')")
    else:
        print("    Skipped (build_rag=False)")
    release_gpu_memory()

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
    parser.add_argument("--no_rag", action="store_true",
                        help="Skip the speaker-aware RAG indexing stage")

    args = parser.parse_args()

    try:
        run_pipeline(
            video_path=args.input,
            registry_dir=args.registry,
            output_dir=args.output_dir,
            use_lora=args.use_lora,
            lora_path=args.lora_path,
            build_rag=not args.no_rag,
        )
    except Exception as e:
        print(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()