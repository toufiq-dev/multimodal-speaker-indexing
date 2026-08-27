"""Regression tests for text cleaning and ASR regex fixes."""
from __future__ import annotations

from engines.transcription import _clean_text
from engines.asr_lora import _SPECIAL_TOKEN_RE, _WHITESPACE_RE


def test_clean_text_collapses_repeated_runs_not_deletes():
    # Old behaviour deleted "আআআ" entirely. Must now keep one instance.
    assert _clean_text("আআআ ক্লাসরুম") == "আ ক্লাসরুম"


def test_clean_text_strips_special_tokens():
    text = "<|startoftranscript|><|bn|><|transcribe|> আমি ভাত খাই <|endoftranscript|>"
    assert _clean_text(text) == "আমি ভাত খাই"


def test_asr_lora_regexes_remove_not_mangle_tokens():
    text = "<|startoftranscript|><|bn|><|transcribe|> আমি ভাত খাই  বলছি<|endoftranscript|>"
    cleaned = _WHITESPACE_RE.sub(" ", _SPECIAL_TOKEN_RE.sub("", text)).strip()
    assert cleaned == "আমি ভাত খাই বলছি"
    # The old double-escaped pattern MANGLED tokens instead of removing them:
    import re
    broken = re.sub(r"<\\|[^|]+\\|>", "", text)
    assert "<|" in broken  # documents the historical bug stays dead


def test_clean_text_removes_space_before_punctuation():
    assert _clean_text("কথা বলছি ।") == "কথা বলছি।"
