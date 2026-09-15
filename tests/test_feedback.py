"""Scoring rules, pattern encoding, and response parsing."""

from __future__ import annotations

import pytest

from wordle_solver.feedback import (
    ABSENT,
    CORRECT,
    PRESENT,
    BadResponseError,
    all_correct,
    build_api_payload,
    decode_pattern,
    encode_pattern,
    is_solved,
    parse_api_response,
    pattern_to_emoji,
    score_classic,
    score_naive,
)


def digits(scorer, guess: str, secret: str) -> list[int]:
    return decode_pattern(scorer(guess, secret), len(guess))


class TestObservedApiBehaviour:
    """Pins down what the live API was observed to do, so a change upstream shows up here.

    Recorded with:
        curl -s "https://wordle.votee.dev:8000/word/hello?guess=ooooo"
        curl -s "https://wordle.votee.dev:8000/word/hello?guess=lllll"
    """

    def test_naive_marks_every_occurrence_of_a_letter(self):
        assert digits(score_naive, "ooooo", "hello") == [PRESENT] * 4 + [CORRECT]

    def test_classic_consumes_each_letter_once(self):
        assert digits(score_classic, "ooooo", "hello") == [ABSENT] * 4 + [CORRECT]

    def test_naive_ignores_duplicate_accounting_for_l(self):
        assert digits(score_naive, "lllll", "hello") == [
            PRESENT, PRESENT, CORRECT, CORRECT, PRESENT
        ]

    def test_classic_accounts_for_duplicates_of_l(self):
        # "hello" holds two l's; both are claimed by the exact matches in slots 2 and 3.
        assert digits(score_classic, "lllll", "hello") == [
            ABSENT, ABSENT, CORRECT, CORRECT, ABSENT
        ]

    def test_rules_agree_when_the_guess_has_no_repeated_letters(self):
        # Without repeats, classic never runs out of copies, so both rules coincide.
        assert score_naive("erase", "speed") == score_classic("erase", "speed")
        assert score_naive("crane", "hello") == score_classic("crane", "hello")


class TestScoringBasics:
    @pytest.mark.parametrize("scorer", [score_naive, score_classic])
    def test_guessing_the_secret_is_all_correct(self, scorer):
        assert is_solved(scorer("hello", "hello"), 5)

    @pytest.mark.parametrize("scorer", [score_naive, score_classic])
    def test_no_shared_letters_is_all_absent(self, scorer):
        assert scorer("fgjkm", "aeiou".replace("a", "b")) == 0

    def test_classic_lets_the_exact_match_claim_the_only_copy(self):
        # "cab" holds a single "a", matched exactly at slot 1. Pass 1 claims it, so the
        # "a" the guess also spends on slot 0 has nothing left to match. Running the
        # passes in the other order would wrongly mark slot 0 present.
        assert digits(score_classic, "aab", "cab") == [ABSENT, CORRECT, CORRECT]

    def test_naive_marks_the_same_case_present(self):
        # The contrast that makes calibration necessary: same inputs, different answer.
        assert digits(score_naive, "aab", "cab") == [PRESENT, CORRECT, CORRECT]


class TestPatternEncoding:
    def test_roundtrip(self):
        pattern = encode_pattern([CORRECT, ABSENT, PRESENT, ABSENT, CORRECT])
        assert decode_pattern(pattern, 5) == [CORRECT, ABSENT, PRESENT, ABSENT, CORRECT]

    def test_slot_zero_is_the_least_significant_digit(self):
        assert encode_pattern([CORRECT, ABSENT, ABSENT]) == 2

    def test_all_correct_depends_on_size(self):
        assert all_correct(4) != all_correct(5)
        assert is_solved(all_correct(7), 7)

    def test_emoji_rendering(self):
        assert pattern_to_emoji(encode_pattern([CORRECT, PRESENT, ABSENT]), 3) == "\U0001f7e9\U0001f7e8⬛"


class TestParseApiResponse:
    def payload(self, **overrides):
        base = [
            {"slot": 0, "guess": "h", "result": "correct"},
            {"slot": 1, "guess": "o", "result": "present"},
            {"slot": 2, "guess": "o", "result": "present"},
            {"slot": 3, "guess": "l", "result": "correct"},
            {"slot": 4, "guess": "y", "result": "absent"},
        ]
        return [dict(item, **overrides) if item["slot"] == overrides.get("slot") else item for item in base]

    def test_parses_a_real_response(self):
        # Recorded from GET /word/hello?guess=hooly
        assert decode_pattern(parse_api_response(self.payload(), 5), 5) == [
            CORRECT, PRESENT, PRESENT, CORRECT, ABSENT
        ]

    def test_slot_order_is_not_trusted(self):
        shuffled = list(reversed(self.payload()))
        assert parse_api_response(shuffled, 5) == parse_api_response(self.payload(), 5)

    def test_roundtrips_with_build_api_payload(self):
        pattern = score_naive("crane", "hello")
        assert parse_api_response(build_api_payload("crane", pattern), 5) == pattern

    @pytest.mark.parametrize(
        "bad, reason",
        [
            ("not a list", "payload is a string"),
            ([], "wrong number of slots"),
            ([{"slot": 0, "guess": "h"}], "missing the result field"),
            ([{"slot": 9, "guess": "h", "result": "correct"}], "slot out of range"),
            ([{"slot": 0, "guess": "h", "result": "maybe"}], "unknown result kind"),
        ],
    )
    def test_rejects_malformed_payloads(self, bad, reason):
        with pytest.raises(BadResponseError):
            parse_api_response(bad, 1 if isinstance(bad, list) and len(bad) == 1 else 5)

    def test_rejects_duplicate_slots(self):
        duplicated = [
            {"slot": 0, "guess": "a", "result": "absent"},
            {"slot": 0, "guess": "b", "result": "absent"},
        ]
        with pytest.raises(BadResponseError):
            parse_api_response(duplicated, 2)
