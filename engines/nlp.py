"""NLP engine for speaker name extraction from Bangla ASR transcripts using NER."""

from __future__ import annotations

import re
from typing import List, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

from config import Config
from models import TranscribedSegment


config = Config()


BENGALI_TITLES = {
    "ড.", "ডাঃ", "ডাঃ.", "প্রফেসর", "প্রফেসর.", "প্রোফেসর", "জনাব", "জনাব.",
    "শ্রী", "শ্রী.", "শ্রীমতী", "শ্রীমতী.", "মো.", "মোঃ", "মোহাম্মদ",
    "হাজী", "হাজী.", "আলহাজ", "আলহাজ.", "বেগম", "বেগম.", "প্রধান",
    "মন্ত্রী", "রাষ্ট্রপতি", "উপস্থাপক", "অতিথি", "সঞ্চালক",
}


def _load_ner_pipeline() -> pipeline:
    """Load NER pipeline with fallback model."""
    model_name = config.BANGLABERT_NER_MODEL
    fallback = config.BANGLABERT_NER_FALLBACK

    for name in (model_name, fallback):
        try:
            tokenizer = AutoTokenizer.from_pretrained(name)
            model = AutoModelForTokenClassification.from_pretrained(name)
            ner_pipe = pipeline(
                "ner",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple",
                device=0 if config.DEVICE == "cuda" else -1,
            )
            return ner_pipe
        except Exception:
            continue

    raise RuntimeError(f"Failed to load NER model: {model_name} or {fallback}")


def _normalize_text(text: str) -> str:
    """Normalize Bangla text for NER."""
    text = text.replace("।", ".")
    text = re.sub(r"([.!?])(\w)", r"\1 \2", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _map_subwords_to_words(
    tokens: List[str],
    offsets: List[Tuple[int, int]],
    entities: List[dict],
) -> List[Tuple[str, bool]]:
    """
    Map subword-level NER predictions to word-level.

    Returns list of (word, is_person) for each word.
    """
    word_entities = []
    current_word = ""
    current_start = None
    current_is_person = False

    for i, (token, (start, end)) in enumerate(zip(tokens, offsets)):
        if start == end:
            continue

        is_new_word = current_start is not None and start > current_start
        if is_new_word:
            word_entities.append((current_word.strip(), current_is_person))
            current_word = ""
            current_is_person = False

        current_word += token.replace("##", "").replace("▁", "")
        current_start = start

        # Check if any entity covers this token
        for ent in entities:
            if ent["start"] <= start < ent["end"] or ent["start"] < end <= ent["end"]:
                if ent["entity_group"] == "PER":
                    current_is_person = True
                break

    if current_word:
        word_entities.append((current_word.strip(), current_is_person))

    return word_entities


def _extract_names_from_entities(word_entities: List[Tuple[str, bool]]) -> List[str]:
    """Extract full person names from word-level PER tags."""
    names = []
    current_name_parts = []

    for word, is_person in word_entities:
        if not word:
            continue

        if is_person or word in BENGALI_TITLES or re.match(r"^[.\-–—]$", word):
            current_name_parts.append(word)
        else:
            if current_name_parts:
                name = " ".join(current_name_parts).strip()
                # Clean up: remove trailing dots/dashes
                name = re.sub(r"[\.\-–—]+$", "", name)
                if len(name) > 1:
                    names.append(name)
                current_name_parts = []

    if current_name_parts:
        name = " ".join(current_name_parts).strip()
        name = re.sub(r"[\.\-–—]+$", "", name)
        if len(name) > 1:
            names.append(name)

    return names


def extract_speaker_names_from_intro(segments: List[TranscribedSegment]) -> List[str]:
    """
    Extract speaker names from introduction segments (first NLP_INTRO_SECONDS).

    Args:
        segments: List of TranscribedSegment from transcription engine.

    Returns:
        Ordered list of unique person names found in intro.
    """
    intro_segments = [s for s in segments if s.start < config.NLP_INTRO_SECONDS]
    if not intro_segments:
        return []

    full_text = " ".join(s.text for s in intro_segments)
    normalized = _normalize_text(full_text)

    ner_pipe = _load_ner_pipeline()

    # Tokenize with offsets
    tokenizer = ner_pipe.tokenizer
    encoding = tokenizer(
        normalized,
        return_offsets_mapping=True,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
    offsets = encoding["offset_mapping"][0].tolist()

    # Run NER
    entities = ner_pipe(normalized)

    # Map to words
    word_entities = _map_subwords_to_words(tokens, offsets, entities)

    # Extract names
    names = _extract_names_from_entities(word_entities)

    # Deduplicate while preserving order
    seen = set()
    unique_names = []
    for name in names:
        if name not in seen:
            seen.add(name)
            unique_names.append(name)

    return unique_names