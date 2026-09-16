"""Tests for rebuilding whole words from faster-whisper's sub-word tokens.

The pipeline used to ``strip()`` each token and then join tokens with a space,
which split single words apart. The observed damage in v9/v10 subtitles:
a cue beginning "ুরুর" (from "শুরুর") and inside-cue fragments like "সত ্যকে".
Because speaker assignment runs per token, a split word could also be divided
between two speakers. These tests pin the repair.
"""
from __future__ import annotations

from engines.transcription import _merge_into_words

#: Bengali dependent signs (vowel signs, virama, nukta...). No word may begin
#: with one — if one does, the word was split.
_COMBINING = set("\u09bc\u09be\u09bf\u09c0\u09c1\u09c2\u09c3\u09c4"
                 "\u09c7\u09c8\u09cb\u09cc\u09cd\u09d7\u09e2\u09e3")


class _Tok:
    """Minimal stand-in for faster_whisper's Word namedtuple."""

    def __init__(self, word: str, start: float, end: float):
        self.word, self.start, self.end = word, start, end


def test_continuation_token_merges_into_the_previous_word():
    merged = _merge_into_words([_Tok(" শু", 0.0, 1.0), _Tok("রুর", 1.0, 2.0)])
    assert [w.word for w in merged] == ["শুরুর"]
    assert (merged[0].start, merged[0].end) == (0.0, 2.0)


def test_leading_space_starts_a_new_word():
    merged = _merge_into_words([_Tok(" নির্বাচন", 0.0, 1.0),
                                _Tok(" নিয়ে", 1.0, 2.0)])
    assert [w.word for w in merged] == ["নির্বাচন", "নিয়ে"]


def test_first_token_starts_a_word_even_without_a_leading_space():
    # A segment's first token often has no leading space; it still starts a word.
    merged = _merge_into_words([_Tok("শুরু", 0.0, 1.0),
                                _Tok(" নির্বাচন", 1.0, 2.0)])
    assert [w.word for w in merged] == ["শুরু", "নির্বাচন"]


def test_observed_corruption_is_repaired():
    """'সত' + '্যকে' must become 'সত্যকে', not 'সত ্যকে'."""
    merged = _merge_into_words([_Tok(" সত", 0.0, 1.0), _Tok("্যকে", 1.0, 2.0)])
    assert " ".join(w.word for w in merged) == "সত্যকে"


def test_no_word_begins_with_a_bengali_dependent_sign():
    merged = _merge_into_words([_Tok(" শু", 0.0, 1.0),
                                _Tok("রুর", 1.0, 2.0),
                                _Tok(" করতে", 2.0, 3.0)])
    for word in merged:
        assert word.word[0] not in _COMBINING, word.word


def test_without_the_convention_nothing_is_merged():
    """If no token carries a leading space the convention is absent; fusing the
    whole stream into one word would be far worse than doing nothing."""
    merged = _merge_into_words([_Tok("শু", 0.0, 1.0), _Tok("রুর", 1.0, 2.0)])
    assert [w.word for w in merged] == ["শু", "রুর"]


def test_blank_tokens_are_dropped():
    assert _merge_into_words([]) == []
    assert _merge_into_words([_Tok("   ", 0.0, 1.0)]) == []


def test_whole_words_survive_joining_for_cue_text():
    """The cue text for a turn is built with " ".join(...) over these words."""
    merged = _merge_into_words([_Tok(" শু", 0.0, 0.4),
                                _Tok("রুর", 0.4, 0.8),
                                _Tok(" থেকে", 0.8, 1.2)])
    assert " ".join(w.word for w in merged) == "শুরুর থেকে"
