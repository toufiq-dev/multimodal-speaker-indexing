"""Tests for rebuilding whole words from faster-whisper's sub-word tokens.

The pipeline used to ``strip()`` each token and join tokens with a space, which
split single words apart. The observed damage in the v9-v13 subtitles: a cue
beginning "ুরুর" (from "শুরুর"), a cue beginning "্যাপ", and inside-cue
fragments like "সত ্যকে". Because speaker assignment runs per token, a split
word could also be divided between two speakers.

Measured on the real subtitles, the artifacts cluster at 30.00 s, 60.38 s and
other multiples of Whisper's 30-second decode window: they are words straddling
a window edge, whose tail arrives as the first token of the *next* segment. A
merge that resets at each segment boundary cannot repair them, so these tests
assert merging across segments as well as within one.
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


class _Seg:
    """Minimal stand-in for a faster_whisper segment."""

    def __init__(self, *words: _Tok):
        self.words = list(words)


def test_continuation_token_merges_into_the_previous_word():
    merged = _merge_into_words([_Seg(_Tok(" শু", 0.0, 1.0), _Tok("রুর", 1.0, 2.0))])
    assert [w.word for w in merged] == ["শুরুর"]
    assert (merged[0].start, merged[0].end) == (0.0, 2.0)


def test_leading_space_starts_a_new_word():
    merged = _merge_into_words([_Seg(_Tok(" নির্বাচন", 0.0, 1.0),
                                     _Tok(" নিয়ে", 1.0, 2.0))])
    assert [w.word for w in merged] == ["নির্বাচন", "নিয়ে"]


def test_merge_spans_segment_boundaries():
    """The 30 s window-edge case: the tail of "শুরুর" opens the next segment.
    A per-segment merge leaves it as its own word, which is the bug."""
    merged = _merge_into_words([
        _Seg(_Tok(" কিছু", 29.4, 29.8), _Tok(" শু", 29.8, 30.0)),
        _Seg(_Tok("রুর", 30.0, 30.4), _Tok(" থেকে", 30.4, 30.9)),
    ])
    assert [w.word for w in merged] == ["কিছু", "শুরুর", "থেকে"]
    assert merged[1].end == 30.4


def test_first_token_of_the_stream_starts_a_word_without_a_leading_space():
    merged = _merge_into_words([_Seg(_Tok("শুরু", 0.0, 1.0),
                                     _Tok(" নির্বাচন", 1.0, 2.0))])
    assert [w.word for w in merged] == ["শুরু", "নির্বাচন"]


def test_token_opening_with_a_combining_sign_never_starts_a_word():
    """At a 30 s window edge the tail may still carry a leading space; a
    dependent sign proves it is not a word start regardless. Real case: the
    cue at 30.00 s begins "ুরুর" because "শ" ended the previous window."""
    merged = _merge_into_words([
        _Seg(_Tok(" শ", 29.8, 30.0)),
        _Seg(_Tok(" ুরুর", 30.0, 30.4)),
    ])
    assert [w.word for w in merged] == ["শুরুর"]


def test_observed_corruption_is_repaired():
    """'সত' + '্যকে' must become 'সত্যকে', not 'সত ্যকে'."""
    merged = _merge_into_words([_Seg(_Tok(" সত", 0.0, 1.0), _Tok("্যকে", 1.0, 2.0))])
    assert " ".join(w.word for w in merged) == "সত্যকে"


def test_no_word_begins_with_a_bengali_dependent_sign():
    merged = _merge_into_words([
        _Seg(_Tok(" শু", 0.0, 1.0), _Tok("রুর", 1.0, 2.0), _Tok(" করতে", 2.0, 3.0)),
        _Seg(_Tok("া", 3.0, 3.4), _Tok(" ঠিক", 3.4, 3.9), _Tok(" না", 3.9, 4.2)),
    ])
    for word in merged:
        assert word.word[0] not in _COMBINING, word.word


def test_without_the_convention_nothing_is_merged():
    """If no token carries a leading space the convention is absent; fusing the
    whole stream into one word would be far worse than doing nothing."""
    merged = _merge_into_words([_Seg(_Tok("শু", 0.0, 1.0), _Tok("রুর", 1.0, 2.0))])
    assert [w.word for w in merged] == ["শু", "রুর"]


def test_blank_tokens_are_dropped():
    assert _merge_into_words([]) == []
    assert _merge_into_words([_Seg(_Tok("   ", 0.0, 1.0))]) == []


def test_whole_words_survive_joining_for_cue_text():
    """The cue text for a turn is built with " ".join(...) over these words."""
    merged = _merge_into_words([_Seg(_Tok(" শু", 0.0, 0.4),
                                     _Tok("রুর", 0.4, 0.8),
                                     _Tok(" থেকে", 0.8, 1.2))])
    assert " ".join(w.word for w in merged) == "শুরুর থেকে"
