"""Transcription engine using faster-whisper with diarization alignment."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple, Optional

from faster_whisper import WhisperModel

from config import config
from models import DiarizationSegment, TranscribedSegment, WordToken
from runtime import release_gpu_memory, with_model


def _load_model() -> WhisperModel:
    """Load faster-whisper model with appropriate compute type.

    Two preconditions are checked eagerly, because both fail later in ways
    that are hard to read:

    1. faster-whisper requires a CTranslate2 directory (``model.bin`` plus a
       CT2 ``config.json``), NOT a Transformers checkpoint. A partially
       populated CT2 directory — e.g. one whose ``model.bin`` was excluded by
       ``.gitignore`` and so never reached the Kaggle clone — fails with an
       opaque loader error.
    2. If the checkpoint was converted as English-only, faster-whisper does
       not reject ``language="bn"``: it logs a warning and forces ``"en"``,
       transcribing Bangla through an English decoder. That silent downgrade
       is far worse than a crash for a Bangla ASR benchmark.
    """
    device, compute_type = config.fw_device_and_compute()
    name = str(config.WHISPER_MODEL)

    local = Path(name)
    if local.is_dir() and not (local / "model.bin").exists():
        raise RuntimeError(
            f"{local} looks like a CTranslate2 directory but has no model.bin. "
            f"Run scripts/convert_bengali_ct2.sh and point WHISPER_MODEL at its "
            f"--output_dir (weights are gitignored, so a clone never carries them)."
        )

    model = WhisperModel(name, device=device, compute_type=compute_type)

    if not model.model.is_multilingual:
        raise RuntimeError(
            f"{name} was converted as English-only; language='bn' would be "
            f"silently downgraded to 'en'. Re-convert with the multilingual "
            f"tokenizer (ct2-transformers-converter --copy_files tokenizer.json)."
        )
    return model


def _clean_text(text: str) -> str:
    """Apply regex cleaning rules from config.

    Rules are (pattern, replacement) pairs so repeated-character runs are
    COLLAPSED to one instance (r"\1") instead of deleted entirely — the old
    behaviour erased legitimate doubled Bangla graphemes.
    """
    cleaned = text
    for pattern, replacement in config.TEXT_CLEANING_RULES:
        cleaned = re.sub(pattern, replacement, cleaned)
    return cleaned.strip()


def _restore_bengali_punct(text: str) -> str:
    """Lightweight Bengali punctuation restoration (regex heuristic, zero-dep).

    Whisper emits unpunctuated Bengali (no ।/,). Heuristics:
    - Normalize existing dandas/commas spacing
    - If long (>18 words) and no danda, insert one after first clause to aid NER

    Genuinely gated by ``config.ENABLE_PUNCTUATION_RESTORE``. The flag used to
    be dead configuration while this function ran unconditionally — which
    meant a synthetic danda was inserted into every long segment before
    WER/CER scoring, with no way to run the ablation without it.
    """
    if not config.ENABLE_PUNCTUATION_RESTORE:
        return text
    if not text:
        return text
    if not re.search(r"[\u0980-\u09FF]", text):
        return text
    # Normalize spaces around existing punctuation
    text = re.sub(r"\s*।\s*", "। ", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Heuristic insert: if >18 words and zero danda, split roughly in half
    if "।" not in text and len(text.split()) > 18:
        words = text.split()
        mid = len(words) // 2
        # try to split at a verb-like boundary; fallback to mid
        text = " ".join(words[:mid]) + "। " + " ".join(words[mid:])
    return text.strip()


def _assign_word_to_turn(
    word_start: float,
    word_end: float,
    diarization: List[DiarizationSegment],
) -> str:
    """Assign a word to a diarization turn by MIDPOINT CONTAINMENT.

    The pipeline contract is: words carry timestamps; diarization turns own
    time intervals. A word belongs to the turn whose interval contains the
    word's midpoint. Only if NO turn contains it do we fall back to the
    nearest turn boundary. This is exact for well-formed diarization output;
    IoU-based scoring is meaningless at this scale (a 0.3s word vs a 5s turn
    yields IoU < 0.1 even for perfect alignment).
    """
    if not diarization:
        return "UNKNOWN"

    word_mid = (word_start + word_end) / 2.0

    # 1) Containment: prefer the turn that contains the midpoint. If turns
    #    overlap, pick the one with the smallest duration (most specific).
    containing = [
        seg for seg in diarization
        if seg.start <= word_mid <= seg.end
    ]
    if containing:
        return min(containing, key=lambda s: s.end - s.start).speaker_id

    # 2) Fallback: nearest boundary distance.
    def _boundary_dist(seg: DiarizationSegment) -> float:
        if word_mid < seg.start:
            return seg.start - word_mid
        if word_mid > seg.end:
            return word_mid - seg.end
        return 0.0

    return min(diarization, key=_boundary_dist).speaker_id





def transcribe_audio(
    audio_path: str,
    model: Optional[WhisperModel] = None,
) -> Tuple[List[WordToken], str]:
    """
    Transcribe audio with word-level timestamps.

    Args:
        audio_path: Path to input WAV audio file.
        model: Optional pre-loaded WhisperModel. If None, loads internally.

    Returns:
        Tuple of (list of WordToken, full_text).
    """
    audio = Path(audio_path).resolve()
    if not audio.exists():
        raise FileNotFoundError(f"Audio file not found: {audio}")

    local_model = model or _load_model()

    # NOTE: transcribe() returns a generator. Only the eager setup (audio
    # decode, VAD, language validation) can raise here; decoding errors
    # surface while iterating below. The previous blanket retry with
    # language=None therefore never fired for the failures it was meant to
    # cover, and would have masked a wrong-language model if it had.
    segments, info = local_model.transcribe(
        str(audio),
        word_timestamps=True,
        vad_filter=True,
        language="bn",
    )

    print(f"[transcription] language={info.language} "
          f"(p={info.language_probability:.2f}) duration={info.duration:.1f}s")

    words: List[WordToken] = []
    full_text_parts = []

    for segment in segments:
        full_text_parts.append(segment.text)
        for word in segment.words or []:
            words.append(
                WordToken(
                    word=word.word.strip(),
                    start=word.start,
                    end=word.end,
                )
            )

    full_text = " ".join(full_text_parts).strip()

    if not words:
        print("[transcription] WARNING: no word timestamps produced; fusion "
              "will fall back to proportional text splitting.")

    if model is None:
        release_gpu_memory()
    return words, full_text


def align_transcription_with_diarization(
    audio_path: str,
    diarization: List[DiarizationSegment],
) -> List[TranscribedSegment]:
    """
    Transcribe audio and align words with diarization segments.

    Args:
        audio_path: Path to input WAV audio file.
        diarization: List of DiarizationSegment from run_diarization.

    Returns:
        List of TranscribedSegment with speaker assignments.
    """
    # Scope the model: CTranslate2 allocates outside torch's caching allocator,
    # so its VRAM is reclaimed only when the object itself is collected.
    # Calling empty_cache() while `model` was still in scope (the old code)
    # freed nothing at all.
    words, _ = with_model(
        _load_model,
        lambda model: transcribe_audio(audio_path, model=model),
        "faster-whisper",
    )

    if not words:
        return []

    for word in words:
        word.speaker_id = _assign_word_to_turn(word.start, word.end, diarization)

    segments: List[TranscribedSegment] = []
    current_words: List[WordToken] = []
    current_speaker = words[0].speaker_id

    for word in words:
        if word.speaker_id == current_speaker:
            current_words.append(word)
        else:
            if current_words:
                text = " ".join(w.word for w in current_words)
                text = _clean_text(text)
                text = _restore_bengali_punct(text)
                segments.append(
                    TranscribedSegment(
                        start=current_words[0].start,
                        end=current_words[-1].end,
                        text=text,
                        words=current_words,
                        speaker_id=current_speaker,
                    )
                )
            current_words = [word]
            current_speaker = word.speaker_id

    if current_words:
        text = " ".join(w.word for w in current_words)
        text = _clean_text(text)
        text = _restore_bengali_punct(text)
        segments.append(
            TranscribedSegment(
                start=current_words[0].start,
                end=current_words[-1].end,
                text=text,
                words=current_words,
                speaker_id=current_speaker,
            )
        )

    return segments