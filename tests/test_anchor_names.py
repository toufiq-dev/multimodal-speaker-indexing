"""Tests for Bengali host-anchor name extraction (আমি <Name> patterns)."""
from __future__ import annotations

from engines.nlp import extract_anchor_names_from_text, extract_intro_anchor
from models import TranscribedSegment


# ── extract_anchor_names_from_text ─────────────────────────────────────

class TestExtractAnchorNames:
    """Test the greedy-regex + stopword-trim name extraction.

    The regex requires ≥2 consecutive Bengali chars after "আমি".
    Stopwords trim non-name continuations. Titles in BENGALI_TITLES are
    skipped (but only if they appear as separate words between Bengali chars).
    """

    def test_basic_three_word_name(self):
        names = extract_anchor_names_from_text(
            "আপনাদের সাথে আছে আমি রফতান আঞ্জুমান নিকোল"
        )
        assert names == ["রফতান আঞ্জুমান নিকোল"]

    def test_single_word_name(self):
        names = extract_anchor_names_from_text("আমি করিম")
        assert names == ["করিম"]

    def test_two_word_name(self):
        names = extract_anchor_names_from_text("আমি নাভিদ হোসেন")
        assert names == ["নাভিদ হোসেন"]

    def test_stopword_trims_name(self):
        # "মনে" is a stopword — extraction must stop before it
        names = extract_anchor_names_from_text("আমি রাশেদ মনে করি")
        assert names == ["রাশেদ"]

    def test_multiple_stopwords_trim(self):
        names = extract_anchor_names_from_text("আমি বলছি না কিছু")
        # "বলছি" is a stopword → trimmed immediately, nothing left
        assert names == []

    def test_title_word_skipped(self):
        # "জনাব" is a Bengali title in BENGALI_TITLES — skipped
        names = extract_anchor_names_from_text("আমি জনাব হাসান")
        assert names == ["হাসান"]

    def test_honorific_title_skipped(self):
        names = extract_anchor_names_from_text("আমি শ্রী কুমার")
        assert names == ["কুমার"]

    def test_title_with_dot_not_bengali(self):
        # "ড." — "ড" is Bengali but "." is not, so the regex can't match
        # ≥2 Bengali chars. The regex fails entirely → no match.
        names = extract_anchor_names_from_text("আমি ড. রহমান")
        assert names == []

    def test_multiple_anchors_in_text(self):
        text = "আমি রফতান আঞ্জুমান। আমি বলি এটা গুরুত্বপূর্ণ"
        names = extract_anchor_names_from_text(text)
        # First anchor: "রফতান আঞ্জুমান" (। is not Bengali, stops capture)
        assert "রফতান আঞ্জুমান" in names
        # Second anchor: "বলি" is a stopword → trimmed to nothing
        assert len(names) == 1

    def test_empty_text(self):
        assert extract_anchor_names_from_text("") == []

    def test_none_text(self):
        assert extract_anchor_names_from_text(None) == []

    def test_no_anchor_pattern(self):
        names = extract_anchor_names_from_text("শুধু কথা বলছি আজকে")
        assert names == []

    def test_four_word_limit(self):
        # "প্রথম" is a stopword, so use non-stopword ordinal-like words
        names = extract_anchor_names_from_text(
            "আমি এক দুই তিন চার পাঁচ"
        )
        # 5 words, but _MAX_NAME_WORDS = 4 → capped at 4
        assert names == ["এক দুই তিন চার"]

    def test_name_at_end_of_sentence(self):
        names = extract_anchor_names_from_text("আমি ফারহান।")
        assert names == ["ফারহান"]

    def test_name_with_english_words_ignored(self):
        # English words don't match the Bengali Unicode range, so they stop capture
        names = extract_anchor_names_from_text("আমি Hello World")
        assert names == []  # no Bengali characters after আমি

    def test_name_followed_by_non_stopword_words(self):
        # Non-stopword words are included (function doesn't know grammar)
        names = extract_anchor_names_from_text("আমি তিন বছর ধরে")
        assert names == ["তিন বছর ধরে"]

    def test_greedy_capture_across_second_ami(self):
        # The greedy regex captures everything Bengali between first আমি and
        # end of Bengali chars, so "আমি X আমি Y" → one match "X আমি Y"
        text = "আমি রাশেদ আমি নাভিদ"
        names = extract_anchor_names_from_text(text)
        assert names == ["রাশেদ আমি নাভিদ"]

    def test_punctuation_breaks_capture(self):
        # Bengali danda (।) is not in [\u0980-\u09FF ] so it stops the regex
        names = extract_anchor_names_from_text("আমি রফতান। বাকি কথা")
        assert names == ["রফতান"]

    def test_comma_breaks_capture(self):
        names = extract_anchor_names_from_text("আমি নাভিদ, হোসেন")
        assert names == ["নাভিদ"]


# ── extract_intro_anchor ───────────────────────────────────────────────

def _seg(start, end, spk, text):
    return TranscribedSegment(start=start, end=end, text=text, words=[],
                              speaker_id=spk)


class TestExtractIntroAnchor:
    """Test the intro-window anchor extraction (aggregates across segments)."""

    def test_finds_most_frequent_anchor(self):
        segments = [
            _seg(0, 10, "SPK1", "আমি রফতান আঞ্জুমান"),
            _seg(12, 25, "SPK1", "আমি রফতান বলছি"),
            _seg(30, 50, "SPK2", "আমি নাভিদ হোসেন"),
        ]
        result = extract_intro_anchor(segments)
        # "রফতান আঞ্জুমান" (1 hit), "রফতান" (1 hit from stopword trim),
        # "নাভিদ হোসেন" (1 hit) — all tied; max() returns first alphabetically
        # or by insertion order. The key point is it returns *some* anchor.
        assert result is not None
        assert "রফতান" in result or "নাভিদ" in result

    def test_returns_none_when_no_anchors(self):
        segments = [
            _seg(0, 10, "SPK1", "শুধু কথা বলছি"),
        ]
        assert extract_intro_anchor(segments) is None

    def test_respects_intro_window(self):
        from config import config
        segments = [
            _seg(0, 10, "SPK1", "আমি রফতান"),
            _seg(config.NLP_INTRO_SECONDS + 50, 200, "SPK2",
                 "আমি নাভিদ"),  # beyond intro window
        ]
        result = extract_intro_anchor(segments)
        assert result == "রফতান"

    def test_empty_segments(self):
        assert extract_intro_anchor([]) is None

    def test_all_segments_beyond_window(self):
        from config import config
        segments = [
            _seg(config.NLP_INTRO_SECONDS + 10, 200, "SPK1",
                 "আমি কেউ নই"),
        ]
        assert extract_intro_anchor(segments) is None

    def test_wins_on_count(self):
        """Name mentioned in 3 segments beats one mentioned in 1."""
        segments = [
            _seg(0, 5, "SPK1", "আমি হাসান।"),
            _seg(6, 10, "SPK1", "আমি হাসান বলছি।"),
            _seg(11, 15, "SPK1", "আমি হাসান জানি।"),
            _seg(20, 30, "SPK2", "আমি করিম।"),
        ]
        result = extract_intro_anchor(segments)
        # "হাসান" appears 3 times (segments 0,1,2), "করিম" once
        assert result == "হাসান"
