"""Opt-in fallback for secrets that are not in our dictionary.

The solver picks its guesses from a word list we bring ourselves, so a secret drawn from
the API's own (unknown) dictionary can be unreachable -- ``agnew`` from ``/random?seed=38``
is a real example. Benchmarking put that at roughly one puzzle in six.

Two verified properties of this API make those puzzles solvable anyway:

1. **Guesses are not checked against a dictionary.** ``guess=zzzzz`` is scored normally, so
   a guess can be used purely as an instrument rather than as an attempt at the answer.
2. **Under the API's naive scoring, ``absent`` means the letter occurs nowhere** in the
   secret -- far stronger than the original Wordle, where ``absent`` can just mean the
   copies are used up.

Together they let us read the secret off directly:

* **Membership phase.** Each guess carries ``size`` different letters, one per slot, and
  every slot's result independently reports whether that letter is in the secret. Five
  letters tested per guess means the whole alphabet is settled in about six guesses.
* **Enumeration phase.** Once no letter is unclassified we know the secret's exact letter
  set, so we can enumerate the strings it could be and filter them with the same
  consistency check the main solver uses -- no dictionary involved.

This only holds for ``naive`` scoring. Under the classic rule, ``absent`` carries no such
guarantee, so ``ExhaustiveRecovery`` refuses to run in that mode.
"""

from __future__ import annotations

import string
from collections import Counter
from dataclasses import dataclass, field
from itertools import product
from typing import Sequence

from .feedback import CORRECT, PRESENT, Scorer, decode_pattern, score_naive

#: Enumerating more strings than this is slower than simply probing another letter.
DEFAULT_ENUMERATION_LIMIT = 200_000

History = Sequence[tuple[str, int]]


@dataclass
class Knowledge:
    """Everything the feedback so far pins down about the secret, under naive scoring."""

    size: int
    present: set[str] = field(default_factory=set)
    absent: set[str] = field(default_factory=set)
    greens: dict[int, str] = field(default_factory=dict)
    not_at: dict[int, set[str]] = field(default_factory=dict)

    def untested(self) -> set[str]:
        return set(string.ascii_lowercase) - self.present - self.absent

    def unknown_slots(self) -> list[int]:
        return [slot for slot in range(self.size) if slot not in self.greens]


def analyse(history: History, size: int) -> Knowledge:
    """Fold the observed patterns into per-letter and per-slot facts."""
    knowledge = Knowledge(size=size)
    for guess, pattern in history:
        for slot, digit in enumerate(decode_pattern(pattern, size)):
            letter = guess[slot]
            if digit == CORRECT:
                knowledge.greens[slot] = letter
                knowledge.present.add(letter)
            elif digit == PRESENT:
                knowledge.present.add(letter)
                knowledge.not_at.setdefault(slot, set()).add(letter)
            else:
                knowledge.absent.add(letter)
    return knowledge


class ExhaustiveRecovery:
    """Drives the membership and enumeration phases described in the module docstring."""

    def __init__(
        self,
        size: int,
        words: Sequence[str] = (),
        scorer: Scorer = score_naive,
        enumeration_limit: int = DEFAULT_ENUMERATION_LIMIT,
    ) -> None:
        if scorer is not score_naive:
            raise ValueError(
                "exhaustive recovery relies on 'absent' meaning the letter occurs nowhere, "
                "which only holds under the API's naive scoring"
            )
        self.size = size
        self.scorer = scorer
        self.enumeration_limit = enumeration_limit
        # Probe common letters first: they are likelier to be in the secret, and settling a
        # letter that is present also reveals the slots it occupies.
        counts: Counter[str] = Counter(letter for word in words for letter in set(word))
        self.letter_order = sorted(
            string.ascii_lowercase, key=lambda letter: (-counts[letter], letter)
        )

    def probe_guess(self, history: History, guessed: set[str]) -> str | None:
        """Next membership probe, or ``None`` when every letter is already classified."""
        knowledge = analyse(history, self.size)
        untested = [letter for letter in self.letter_order if letter in knowledge.untested()]
        if not untested:
            return None

        filler = self._filler(knowledge)
        letters = untested[: self.size]
        guess = "".join(letters) + filler * (self.size - len(letters))

        if guess in guessed:
            # Vanishingly rare, but a repeated guess would buy no new information.
            rotated = letters[1:] + letters[:1]
            guess = "".join(rotated) + filler * (self.size - len(rotated))
        return guess

    def _filler(self, knowledge: Knowledge) -> str:
        """A letter that wastes a slot rather than muddying it.

        A known-absent letter is ideal: its slot is guaranteed to report ``absent`` and so
        cannot be mistaken for information about an untested letter.
        """
        if knowledge.absent:
            return sorted(knowledge.absent)[0]
        return self.letter_order[-1]

    def enumerate_candidates(self, history: History) -> list[str] | None:
        """Every string consistent with the feedback, or ``None`` if that is premature.

        Returns ``None`` while any letter is still unclassified (the secret could contain
        one of them) or when the space is too large to be worth materialising.
        """
        knowledge = analyse(history, self.size)
        if knowledge.untested() or not knowledge.present:
            return None

        letters = sorted(knowledge.present)
        slots = knowledge.unknown_slots()
        if len(letters) ** len(slots) > self.enumeration_limit:
            return None

        required = knowledge.present
        results = []
        for filling in product(letters, repeat=len(slots)):
            word = list(knowledge.greens.get(slot, "") for slot in range(self.size))
            for slot, letter in zip(slots, filling):
                word[slot] = letter
            candidate = "".join(word)
            # The secret uses exactly these letters: nothing else is available, and every
            # one of them was reported present at some point, so each must appear.
            if set(candidate) != required:
                continue
            if any(self.scorer(guess, candidate) != pattern for guess, pattern in history):
                continue
            results.append(candidate)
        return sorted(results)
