"""Guess feedback: scoring rules, compact pattern encoding, and API response parsing.

A *pattern* is the per-slot feedback for one guess. It is encoded as a single base-3
integer (digit ``i`` describes slot ``i``) so that patterns are cheap to compute, cheap to
compare, and usable directly as dictionary keys when grouping candidates by outcome.

Two scoring rules live here because the Votee API does **not** implement the duplicate
letter accounting that the original Wordle uses -- see ``score_naive`` and
``score_classic``. ``calibrate_scoring`` in ``wordle_solver.api`` picks the right one at
runtime by probing the server, so the solver never has to assume.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from typing import Callable, Iterable, Mapping, Sequence

ABSENT, PRESENT, CORRECT = 0, 1, 2

#: Maps the API's ``ResultKind`` enum onto our numeric digits.
RESULT_CODES: Mapping[str, int] = {"absent": ABSENT, "present": PRESENT, "correct": CORRECT}
RESULT_NAMES: Mapping[int, str] = {code: name for name, code in RESULT_CODES.items()}
EMOJI: Mapping[int, str] = {ABSENT: "⬛", PRESENT: "\U0001f7e8", CORRECT: "\U0001f7e9"}

Scorer = Callable[[str, str], int]


class BadResponseError(Exception):
    """The API returned something that does not match the documented schema."""

    def __init__(self, message: str, payload: object = None) -> None:
        super().__init__(f"{message} -- raw payload: {payload!r}")
        self.payload = payload


def encode_pattern(digits: Iterable[int]) -> int:
    """Pack per-slot digits (slot 0 first) into one base-3 integer."""
    pattern = 0
    for power, digit in enumerate(digits):
        pattern += digit * 3**power
    return pattern


def decode_pattern(pattern: int, size: int) -> list[int]:
    """Unpack a base-3 pattern back into ``size`` per-slot digits (slot 0 first)."""
    digits = []
    for _ in range(size):
        digits.append(pattern % 3)
        pattern //= 3
    return digits


@lru_cache(maxsize=None)
def all_correct(size: int) -> int:
    """The pattern meaning "every slot is correct" for a word of this length."""
    return encode_pattern([CORRECT] * size)


def is_solved(pattern: int, size: int) -> bool:
    return pattern == all_correct(size)


@lru_cache(maxsize=None)
def _letters(word: str) -> frozenset[str]:
    """Cached set of distinct letters -- ``score_naive`` asks for this on every comparison."""
    return frozenset(word)


def score_naive(guess: str, secret: str) -> int:
    """Score a guess the way the Votee API actually does (verified against the server).

    Each slot is judged independently, with no accounting for how many times a letter
    occurs in the secret:

        correct  if guess[i] == secret[i]
        present  if guess[i] appears anywhere in secret
        absent   otherwise

    So ``score_naive("ooooo", "hello")`` marks four slots ``present`` and one ``correct``,
    even though "hello" contains a single "o".
    """
    secret_letters = _letters(secret)
    pattern = 0
    power = 1
    for guess_char, secret_char in zip(guess, secret):
        if guess_char == secret_char:
            pattern += CORRECT * power
        elif guess_char in secret_letters:
            pattern += PRESENT * power
        power *= 3
    return pattern


def score_classic(guess: str, secret: str) -> int:
    """Score a guess under the original Wordle rules, with duplicate letter accounting.

    Two passes, in this order:
      1. mark every exact positional match ``correct`` and consume that letter;
      2. mark a remaining slot ``present`` only while unconsumed copies of its letter are
         still available in the secret, otherwise ``absent``.

    The ordering matters: doing pass 2 first would let a misplaced letter steal the copy
    that an exact match is entitled to. ``score_classic("ooooo", "hello")`` marks only the
    final slot, because "hello" holds exactly one "o" and the exact match claims it.
    """
    size = len(guess)
    digits = [ABSENT] * size
    remaining: Counter[str] = Counter()

    for index in range(size):
        if guess[index] == secret[index]:
            digits[index] = CORRECT
        else:
            remaining[secret[index]] += 1

    for index in range(size):
        if digits[index] == CORRECT:
            continue
        letter = guess[index]
        if remaining[letter] > 0:
            digits[index] = PRESENT
            remaining[letter] -= 1

    return encode_pattern(digits)


#: Every scoring rule the solver knows how to mirror, keyed by the name used on the CLI.
SCORERS: Mapping[str, Scorer] = {"naive": score_naive, "classic": score_classic}


def pattern_to_emoji(pattern: int, size: int) -> str:
    return "".join(EMOJI[digit] for digit in decode_pattern(pattern, size))


def pattern_to_names(pattern: int, size: int) -> list[str]:
    return [RESULT_NAMES[digit] for digit in decode_pattern(pattern, size)]


def parse_api_response(payload: object, size: int) -> int:
    """Validate one API response body and encode it as a pattern.

    The schema promises a list of ``{"slot", "guess", "result"}`` objects. We re-order by
    ``slot`` rather than trusting the array order, and reject anything malformed loudly:
    a silently mis-parsed pattern would corrupt the candidate filter and be far harder to
    debug than an exception.
    """
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise BadResponseError("expected a JSON array of slot results", payload)
    if len(payload) != size:
        raise BadResponseError(f"expected {size} slot results, got {len(payload)}", payload)

    digits: list[int | None] = [None] * size
    for item in payload:
        if not isinstance(item, Mapping):
            raise BadResponseError("slot result is not an object", payload)
        missing = {"slot", "guess", "result"} - set(item)
        if missing:
            raise BadResponseError(f"slot result is missing {sorted(missing)}", payload)

        slot = item["slot"]
        if not isinstance(slot, int) or isinstance(slot, bool) or not 0 <= slot < size:
            raise BadResponseError(f"slot {slot!r} is out of range for size {size}", payload)
        if digits[slot] is not None:
            raise BadResponseError(f"slot {slot} appears more than once", payload)

        result = item["result"]
        if result not in RESULT_CODES:
            raise BadResponseError(f"unknown result kind {result!r}", payload)
        digits[slot] = RESULT_CODES[result]

    return encode_pattern(digits)  # type: ignore[arg-type]  -- every slot filled or we raised


def build_api_payload(guess: str, pattern: int) -> list[dict[str, object]]:
    """Inverse of ``parse_api_response``: used by the offline oracle in tests."""
    return [
        {"slot": slot, "guess": letter, "result": RESULT_NAMES[digit]}
        for slot, (letter, digit) in enumerate(zip(guess, decode_pattern(pattern, len(guess))))
    ]
