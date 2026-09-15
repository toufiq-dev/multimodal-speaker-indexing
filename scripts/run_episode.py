#!/usr/bin/env python3
"""Run the full speaker-indexing pipeline for ONE episode and record a verified result.

This is the single repeatable entry point for producing thesis evidence from a
Bengali talk-show video. It drives the same engines as ``main.run_pipeline``
(diarization, ASR, vision, NER, fusion, RAG) but with:

  * an episode-scoped output directory (``data/output/<episode_id>/``),
  * a per-video scratch dir (``config.SCRATCH_DIR/episodes/<video>``), so two
    episodes never clobber each other's cached audio/frames,
  * a reproducibility manifest (``evaluation.tracking.ExperimentRun``),
  * fusion-health regression checks (``evaluation.metrics``),
  * optional DER/JER/cpWER/speaker-name metrics when ground truth exists,
  * stage-artifact reuse: audio/frames are cached so re-running the expensive
    model stages after a fusion tweak reuses the extracted media.

Only a run whose fusion-health checks pass is recorded as valid. A failed run
leaves no ``result.json`` that could be mistaken for evidence.

Usage:
    python scripts/run_episode.py VIDEO [--registry DIR] [--id EPISODE_ID] \
        [--output-dir DIR] [--no-rag] [--force]

Examples:
    # Short validation clip first (always do this before a full episode)
    python scripts/run_episode.py data/input/global_clip_5min.mp4 \
        --registry data/registry --id global_tv_5min

    # Full 45-minute episode with ground truth available
    python scripts/run_episode.py data/input/global_tv_talkshow.mp4 \
        --registry data/registry --id global_tv_ep --gt-dir data/gt/global_tv
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

# Repo root on sys.path (works when invoked as `python scripts/run_episode.py`
# or `python -m scripts.run_episode` from the repo root).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import config  # noqa: E402
from models import DiarizationSegment, FinalSegment  # noqa: E402
from evaluation.metrics import (  # noqa: E402
    cpwer,
    der_jer,
    fusion_health_metrics,
    assert_no_regression,
    speaker_name_accuracy,
)
from evaluation.tracking import ExperimentRun  # noqa: E402

# ---------------------------------------------------------------------------
# Per-video scratch helpers
# ---------------------------------------------------------------------------


def _episode_scratch(video: Path) -> Path:
    """Scratch dir scoped to one video, so episodes never share intermediates.

    main.run_pipeline writes audio/frames to the *global* config.SCRATCH_DIR,
    so two episodes run back-to-back would silently reuse each other's cached
    audio/frames. This runner therefore drives the media stages itself into a
    per-video scratch dir, then hands the paths to the engines (which accept
    explicit audio paths / frame lists).
    """
    d = config.SCRATCH_DIR / "episodes" / video.stem
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def _run_pipeline(
    video_path: str,
    registry_dir: str | None,
    output_dir: Path,
    use_lora: bool,
    lora_path: str | None,
    build_rag: bool,
    force: bool,
) -> float:
    """Drive the full pipeline with per-video scratch + warm-cache reuse."""
    video = Path(video_path).resolve()
    scratch = _episode_scratch(video)

    # Media extraction is cheap and idempotent; reuse it when present and the
    # user did not ask for a forced re-run. Diarization + ASR are the expensive
    # stages and are always (re)run, but their inputs are stable.
    audio_path = scratch / "audio.wav"
    frames_dir = scratch / "frames"
    if (not force) and audio_path.exists() and frames_dir.exists():
        print(f"[run_episode] reusing cached media for {video.stem} from {scratch}")
    else:
        from engines.media import extract_audio, extract_frames

        print(f"[run_episode] extracting media to {scratch}")
        # extract_audio/extract_frames write into the *global* config.SCRATCH_DIR;
        # move the results into this episode's scratch so episodes can never
        # clobber each other (extract_frames clears its global output dir on
        # every call, so a later episode would delete an earlier one's frames).
        audio_path = Path(extract_audio(str(video)))
        if audio_path.parent != scratch:
            dest = scratch / audio_path.name
            if not dest.exists():
                shutil.move(str(audio_path), str(dest))
            audio_path = dest

        if frames_dir.exists():
            # Previous partial extraction in this episode's dir: clear to avoid
            # mixing old+new frames.
            shutil.rmtree(frames_dir, ignore_errors=True)
        frames_dir.mkdir(parents=True, exist_ok=True)
        frames = extract_frames(str(video), fps=config.VISION_FPS)
        for f in frames:
            shutil.move(f, frames_dir / Path(f).name)

    frame_paths = sorted(str(p) for p in frames_dir.glob("frame_*.jpg"))
    if not frame_paths:
        raise RuntimeError(
            f"no frames in {frames_dir} — media extraction produced nothing")

    t0 = time.time()
    _run_engines(
        video=video,
        audio_path=audio_path,
        frame_paths=frame_paths,
        registry_dir=registry_dir,
        output_dir=output_dir,
        use_lora=use_lora,
        lora_path=lora_path,
        build_rag=build_rag,
    )
    return time.time() - t0


def _run_engines(
    video: Path,
    audio_path: Path,
    frame_paths,
    registry_dir: str | None,
    output_dir: Path,
    use_lora: bool,
    lora_path: str | None,
    build_rag: bool,
) -> None:
    """Run diarization → ASR → vision → NER → fusion → write outputs."""
    if registry_dir:
        config.DATA_REGISTRY_DIR = Path(registry_dir).resolve()
        config.DATA_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    config.DATA_OUTPUT_DIR = output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    from engines.diarization import run_diarization
    from engines.transcription import align_transcription_with_diarization
    from engines.nlp import extract_speaker_names_from_intro
    from engines.fusion import run_fusion_pipeline
    from engines.vision import run_vision_pipeline
    from main import _write_json, _write_srt, _build_rag_index
    from runtime import release_gpu_memory

    print(f"[1] diarizing {audio_path.name}...")
    diarization = run_diarization(str(audio_path))
    print(f"    {len(set(s.speaker_id for s in diarization))} speakers, "
          f"{len(diarization)} turns")
    release_gpu_memory()

    print(f"[2] transcribing (slow stage)...")
    if use_lora:
        from engines.asr_lora import load_lora_whisper, transcribe_with_lora
        model, processor = load_lora_whisper(lora_path=lora_path)
        transcribed = transcribe_with_lora(model, processor, str(audio_path),
                                           diarization)
        del model, processor
    else:
        transcribed = align_transcription_with_diarization(str(audio_path),
                                                           diarization)
    print(f"    {len(transcribed)} transcribed segments")
    release_gpu_memory()

    print(f"[3] running vision on {len(frame_paths)} frames...")
    faces = run_vision_pipeline(str(video), frame_paths=list(frame_paths))
    print(f"    {len(faces)} face occurrences")
    release_gpu_memory()

    print(f"[4] extracting speaker names from intro...")
    ordered_names = extract_speaker_names_from_intro(transcribed)
    print(f"    names: {ordered_names}")
    release_gpu_memory()

    print(f"[5] fusing modalities...")
    final_segments = run_fusion_pipeline(
        diarization, transcribed, faces, ordered_names)
    print(f"    {len(final_segments)} final segments")

    # Persist the face evidence behind identity resolution. The registry
    # rejection gates must be set from these measured gaps, not guessed.
    try:
        from engines.fusion import fusion_diagnostics
        diag = fusion_diagnostics(diarization, faces, ordered_names)
        (output_dir / "fusion_diagnostics.json").write_text(
            json.dumps(diag, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"    fusion_diagnostics.json <- {len(diag)} speakers")
    except Exception as e:
        print(f"    diagnostics failed ({e.__class__.__name__}: {e})")

    _write_json(final_segments, output_dir / "result.json")
    _write_srt(final_segments, output_dir / "subtitles.srt")

    if build_rag:
        print(f"[6] building speaker-aware RAG index...")
        try:
            n = _build_rag_index(final_segments, output_dir)
            print(f"    indexed {n} chunks")
        except Exception as e:
            print(f"    RAG failed ({e.__class__.__name__}: {e}); "
                  f"result.json unaffected")
    release_gpu_memory()


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

def _record(
    out_dir: Path,
    segments_json: Path,
    run: ExperimentRun,
    gt_dir: Path | None,
    elapsed: float,
) -> None:
    """Write health.json + metrics.json next to result.json and print verdict."""
    raw = json.loads(segments_json.read_text(encoding="utf-8"))
    finals = [FinalSegment(**s) for s in raw]

    health = fusion_health_metrics(finals)
    assert_no_regression(health)  # raises -> run is not recorded as valid

    payload = {
        "episode": out_dir.name,
        "elapsed_sec": round(elapsed, 1),
        "health": health,
        "num_segments": len(finals),
        "speakers": sorted({s.speaker for s in finals}),
    }
    (out_dir / "health.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2)
    )

    metrics = {}
    if gt_dir and gt_dir.exists():
        # Ground-truth layout follows evaluation/dataset.py's manifest schema:
        #   gt_dir/reference.rttm + gt_dir/speaker_map.json
        from evaluation.dataset import load_ground_truth, VideoEntry

        entry = VideoEntry(id=out_dir.name, source="", media_path="")
        entry.gt_rttm = "reference.rttm"
        entry.gt_speaker_map = "speaker_map.json"
        try:
            gt = load_ground_truth(entry, gt_dir)
            hyp_turns = [
                DiarizationSegment(
                    start=s["start"], end=s["end"], speaker_id=s["speaker"])
                for s in raw
            ]
            der = der_jer(gt.turns, hyp_turns)
            name_acc = speaker_name_accuracy(gt.turns, finals)
            ref_texts = {s.get("speaker", "UNKNOWN"): s.get("text", "")
                         for s in gt.texts}
            hyp_texts = {}
            for s in finals:
                hyp_texts.setdefault(s.speaker, "")
                hyp_texts[s.speaker] += " " + s.text
            cw, _ = cpwer(ref_texts, hyp_texts) if ref_texts else (float("inf"), 0)
            metrics = {
                "DER": der["DER"], "JER": der["JER"],
                "speaker_name_accuracy": round(name_acc, 4),
                "cpWER": round(cw, 4) if cw != float("inf") else None,
            }
        except Exception as e:
            print(f"[run_episode] ground-truth metrics failed ({e.__class__.__name__}: "
                  f"{e}); recording without them")
            metrics = {"error": f"{e.__class__.__name__}: {e}"}
        (out_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2)
        )

    run.finish({**payload, "metrics": metrics})
    manifest = run.save(str(out_dir))
    print(f"\n✔ recorded: {out_dir.name}")
    print(f"  health    : dup_rate={health['duplicate_text_rate']}, "
          f"avg_cue={health['avg_cue_chars']}, "
          f"speakers={health['distinct_speakers']}")
    if metrics:
        print(f"  metrics   : DER={metrics.get('DER')}, JER={metrics.get('JER')}, "
              f"name_acc={metrics.get('speaker_name_accuracy')}, "
              f"cpWER={metrics.get('cpWER')}")
    print(f"  manifest  : {manifest}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _ensure_cudnn_on_loader_path() -> bool:
    """Export torch's bundled cuDNN 9 onto ``LD_LIBRARY_PATH``.

    CTranslate2 resolves cuDNN with ``dlopen``. glibc captures the library
    search path at **process start**, so setting ``LD_LIBRARY_PATH`` later in
    the same process has no effect — the load fails with
    ``Unable to load any of {libcudnn_cnn.so.9...}``. The only reliable fix is
    to put the directory in the environment and re-exec, so the child's loader
    sees it from the beginning.

    Returns True when the environment was changed (caller must re-exec).
    """
    try:
        from runtime import cudnn_library_dir
    except Exception:
        return False
    lib = cudnn_library_dir()
    if not lib:
        return False
    current = os.environ.get("LD_LIBRARY_PATH", "")
    if lib in current.split(os.pathsep):
        return False
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
        p for p in (lib, current) if p)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("video", help="Input video file")
    ap.add_argument("--registry", "-r", help="Face registry directory")
    ap.add_argument("--id", default=None, help="Episode id (default: video stem)")
    ap.add_argument("--output-dir", default=None, help="Override output directory")
    ap.add_argument("--gt-dir", default=None,
                    help="Ground-truth dir with reference.rttm + speaker_map.json")
    ap.add_argument("--no-rag", action="store_true", help="Skip RAG indexing")
    ap.add_argument("--use-lora", action="store_true", help="Use LoRA ASR path")
    ap.add_argument("--lora-path", help="LoRA adapter path (required with --use-lora)")
    ap.add_argument("--force", action="store_true",
                    help="Re-run even if cached audio/frames exist")
    args = ap.parse_args()

    # cuDNN 9 must be on the loader path from process start (see the docstring
    # on _ensure_cudnn_on_loader_path). Re-exec once; the guard flag prevents a
    # loop if the loader still cannot find it.
    if os.environ.get("MSI_CUDNN_REEXEC") != "1" and _ensure_cudnn_on_loader_path():
        os.environ["MSI_CUDNN_REEXEC"] = "1"
        print("[run_episode] re-exec with LD_LIBRARY_PATH for cuDNN 9")
        os.execv(sys.executable, [sys.executable] + sys.argv)

    video = Path(args.video).resolve()
    if not video.exists():
        print(f"error: video not found: {video}", file=sys.stderr)
        return 2

    # Gated models (pyannote) 401 without a token. A kernel restart clears
    # os.environ, so a pipeline launched as a subprocess can silently inherit
    # an empty environment — say so up front instead of dying mid-diarization.
    if not os.environ.get("HF_TOKEN"):
        print(
            "error: HF_TOKEN is not set in this process. Diarization uses the "
            "gated model pyannote/speaker-diarization-3.1 and will fail. Fix: "
            "run kaggle_setup.prepare_env() in the notebook before launching "
            "this script (it reloads the token from Kaggle Secrets), or re-run "
            "the bootstrap cell.",
            file=sys.stderr,
        )
        return 2

    episode_id = args.id or video.stem
    out_dir = Path(args.output_dir).resolve() if args.output_dir else (
        config.DATA_OUTPUT_DIR / episode_id
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_dir = Path(args.gt_dir).resolve() if args.gt_dir else None

    run = ExperimentRun(name=f"episode:{episode_id}", config={
        "video": str(video),
        "registry": str(Path(args.registry).resolve()) if args.registry else None,
        "episode_id": episode_id,
        "use_lora": args.use_lora,
        "lora_path": args.lora_path,
        "build_rag": not args.no_rag,
    })

    try:
        elapsed = _run_pipeline(
            str(video), args.registry, out_dir, args.use_lora, args.lora_path,
            not args.no_rag, args.force,
        )
        _record(out_dir, out_dir / "result.json", run, gt_dir, elapsed)
    except Exception as e:
        print(f"\n✗ episode {episode_id} failed: {e.__class__.__name__}: {e}",
              file=sys.stderr)
        run.finish({"error": f"{e.__class__.__name__}: {e}"})
        run.save(str(out_dir))
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
