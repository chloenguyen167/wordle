"""Guess selection: constraint filtering plus expected information gain.

The strategy is inspired by the information-theory framing of Wordle popularised by
3Blue1Brown (see README credits); the implementation here is written from that idea rather
than ported from any existing code.

Two ideas carry the whole solver:

1. **Filtering is just replaying the scorer.** A word is still a candidate exactly when,
   if it were the secret, it would have produced every pattern the server has returned so
   far. That single rule subsumes all the usual green/yellow/grey bookkeeping -- and
   because the scorer is the one calibrated against the server, the filter stays correct
   even though this API scores duplicate letters unconventionally.

2. **Pick the guess that splits the candidates most evenly.** Scoring a guess against
   every candidate partitions them by outcome; the entropy of that partition is the
   expected number of bits the guess will reveal. Maximising it minimises how much
   uncertainty we expect to be left holding.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

from .feedback import CORRECT, PRESENT, Scorer, decode_pattern, is_solved, score_naive
from .recovery import ExhaustiveRecovery
from .wordlist import wordlist_fingerprint

DEFAULT_MAX_ROUNDS = 6
DEFAULT_GUESS_POOL_LIMIT = 500
DEFAULT_OPENER_SHORTLIST = 300
#: Below this many candidates, the solver may spend a guess on a word that cannot win
#: (see ``Solver._probe_pool``). Above it, guessing a candidate is already competitive.
DEFAULT_PROBE_THRESHOLD = 40
DEFAULT_CACHE_DIR = Path(".cache")


class SolverError(Exception):
    """The solver ran out of words it could legally guess."""


@dataclass(frozen=True)
class Round:
    index: int
    guess: str
    pattern: int
    candidates_before: int
    candidates_after: int


@dataclass
class SolveResult:
    solved: bool
    secret: str | None
    size: int
    status: str  # "solved" | "exhausted" | "no_candidates"
    scoring: str
    oracle: str
    rounds: list[Round] = field(default_factory=list)
    api_calls: int = 0
    elapsed: float = 0.0

    @property
    def round_count(self) -> int:
        return len(self.rounds)

    def to_dict(self) -> dict[str, object]:
        return {
            "solved": self.solved,
            "secret": self.secret,
            "size": self.size,
            "status": self.status,
            "scoring": self.scoring,
            "oracle": self.oracle,
            "rounds": [
                {
                    "index": r.index,
                    "guess": r.guess,
                    "pattern": r.pattern,
                    "candidates_before": r.candidates_before,
                    "candidates_after": r.candidates_after,
                }
                for r in self.rounds
            ],
            "round_count": self.round_count,
            "api_calls": self.api_calls,
            "elapsed": round(self.elapsed, 3),
        }


def expected_information(guess: str, candidates: Sequence[str], scorer: Scorer) -> float:
    """Expected bits of information from playing ``guess``, in [0, log2(len(candidates))].

    Group the candidates by the pattern ``guess`` would produce against each of them; the
    entropy of the resulting distribution is the expected information gain. A guess that
    lands every candidate in one big bucket teaches us nothing (0 bits); one that separates
    them all teaches us the most.
    """
    total = len(candidates)
    if total <= 1:
        return 0.0
    buckets: Counter[int] = Counter(scorer(guess, candidate) for candidate in candidates)
    return -sum(
        (count / total) * math.log2(count / total) for count in buckets.values()
    )


def rank_by_letter_frequency(words: Sequence[str], limit: int) -> list[str]:
    """Cheap pre-filter: the ``limit`` words made of the most common distinct letters.

    Entropy is quadratic in pool size, so on large pools we score only a promising slice.
    Distinct letters are what matter -- a word spending two slots on the same letter probes
    fewer of them.
    """
    if len(words) <= limit:
        return sorted(words)
    frequency: Counter[str] = Counter(letter for word in words for letter in set(word))
    ranked = sorted(words, key=lambda word: (-sum(frequency[c] for c in set(word)), word))
    return sorted(ranked[:limit])


def compute_opener(
    words: Sequence[str], scorer: Scorer, shortlist_size: int = DEFAULT_OPENER_SHORTLIST
) -> tuple[str, float]:
    """Pick the opening guess by measuring information gain on the actual pool.

    No hardcoded CRANE/SLATE: those are tuned for the original Wordle answer list and say
    nothing about an arbitrary dictionary or word length.

    Scoring every word against every word is O(n^2) -- roughly 72M comparisons for the
    8.5k five-letter pool, far too slow to sit through during a demo. We instead evaluate a
    frequency-ranked shortlist against the *full* pool, which keeps the measurement honest
    (the entropy is computed over every possible secret) while bounding the cost. It is an
    approximation of the argmax, not the exact one.
    """
    shortlist = rank_by_letter_frequency(words, shortlist_size)
    best_word = shortlist[0]
    best_score = -1.0
    for word in shortlist:
        score = expected_information(word, words, scorer)
        if score > best_score:
            best_word, best_score = word, score
    return best_word, best_score


class Solver:
    """Stateful solver for a single puzzle. One instance solves one secret."""

    def __init__(
        self,
        words: Sequence[str],
        size: int,
        scorer: Scorer = score_naive,
        scoring_name: str = "naive",
        guess_pool_limit: int = DEFAULT_GUESS_POOL_LIMIT,
        opener_shortlist: int = DEFAULT_OPENER_SHORTLIST,
        probe_threshold: int = DEFAULT_PROBE_THRESHOLD,
        cache_dir: Path | str | None = DEFAULT_CACHE_DIR,
        fingerprint: str | None = None,
        recovery: ExhaustiveRecovery | None = None,
    ) -> None:
        if not words:
            raise SolverError("cannot solve with an empty word list")
        self.all_words = list(words)
        self.size = size
        self.scorer = scorer
        self.scoring_name = scoring_name
        self.guess_pool_limit = guess_pool_limit
        self.opener_shortlist = opener_shortlist
        self.probe_threshold = probe_threshold
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._fingerprint = fingerprint
        # Optional: reconstructs secrets that are not in the word list at all.
        self.recovery = recovery

        self.candidates: list[str] = list(words)
        self.history: list[tuple[str, int]] = []
        self.guessed: set[str] = set()
        self.degraded = False  # set when the secret fell outside our word list

    # -- state ---------------------------------------------------------------------

    def observe(self, guess: str, pattern: int) -> None:
        """Record a scored guess and drop every candidate inconsistent with it."""
        self.candidates = [
            word for word in self.candidates if self.scorer(guess, word) == pattern
        ]
        self.history.append((guess, pattern))
        self.guessed.add(guess)

    # -- guess selection -----------------------------------------------------------

    def next_guess(self, rounds_left: int | None = None) -> str:
        if not self.history:
            return self.opening_guess()

        remaining = [word for word in self.candidates if word not in self.guessed]
        if not remaining:
            return self._recover_guess(rounds_left)
        if len(remaining) <= 2:
            # Nothing to learn that guessing one of them does not also teach us.
            return remaining[0]
        return self._pick(remaining, rounds_left)

    def _pick(self, candidates: Sequence[str], rounds_left: int | None = None) -> str:
        """Highest expected information among a bounded pool of legal guesses.

        Normally we guess a word that could still be the answer ("hard mode"): it is as
        informative as anything else while the pool is large, and it can win outright.
        Near the end that backfires on clusters like ``stood/stool/stoop/stook``, where the
        candidates differ in one slot and trying them one at a time burns a guess each. So
        when the remaining guesses cannot cover the remaining candidates, we also consider
        words that cannot win but test several of the distinguishing letters at once.
        """
        pool = set(rank_by_letter_frequency(candidates, self.guess_pool_limit))
        if rounds_left is not None and 2 <= rounds_left < len(candidates) <= self.probe_threshold:
            pool |= set(self._probe_pool(candidates))

        candidate_set = set(candidates)
        best_word = ""
        best_key: tuple[float, bool] | None = None
        for word in sorted(pool):
            # Iterating in sorted order and replacing only on a strict improvement makes
            # ties resolve to the lexicographically smallest word, so runs are repeatable.
            # A candidate outranks a non-candidate of equal information: it might just win.
            key = (expected_information(word, candidates, self.scorer), word in candidate_set)
            if best_key is None or key > best_key:
                best_word, best_key = word, key
        return best_word

    def _probe_pool(self, candidates: Sequence[str]) -> list[str]:
        """Words worth playing purely to split the candidates, even if they cannot win.

        Ranked by how many *discriminating* letters they contain -- letters held by some
        candidates but not all. A letter every candidate shares (or none has) tells us
        nothing new.
        """
        occurrences: Counter[str] = Counter(
            letter for word in candidates for letter in set(word)
        )
        total = len(candidates)
        discriminating = {
            letter for letter, count in occurrences.items() if 0 < count < total
        }
        if not discriminating:
            return []
        ranked = sorted(
            (word for word in self.all_words if word not in self.guessed),
            key=lambda word: (-len(discriminating & set(word)), word),
        )
        return ranked[: self.guess_pool_limit]

    def _recover_guess(self, rounds_left: int | None = None) -> str:
        """Fallback for when the candidate list empties out.

        That means the secret is not in our dictionary (the API has its own word list), so
        we rebuild a pool under *relaxed* constraints: keep the confirmed letter positions
        and the letters known to occur, but drop the ``absent`` constraints. Absent is the
        constraint most likely to be over-applied, and dropping it is what lets us recover
        instead of failing outright.
        """
        self.degraded = True

        if self.recovery is not None:
            guess = self._recovery_guess(rounds_left)
            if guess is not None:
                return guess

        relaxed = self._relaxed_pool()
        if relaxed:
            self.candidates = relaxed
            return self._pick(relaxed, rounds_left)

        unused = [word for word in self.all_words if word not in self.guessed]
        if not unused:
            raise SolverError("every word in the list has been guessed")
        self.candidates = []
        return self._pick(unused, rounds_left)

    def _recovery_guess(self, rounds_left: int | None) -> str | None:
        """Delegate to the dictionary-free fallback, if one was supplied.

        Enumerating comes first: once every letter is classified it produces the real
        candidate space directly. Until then we spend a guess probing more letters.
        """
        space = self.recovery.enumerate_candidates(self.history)
        if space:
            fresh = [word for word in space if word not in self.guessed]
            if fresh:
                self.candidates = fresh
                return self._pick(fresh, rounds_left)
        return self.recovery.probe_guess(self.history, self.guessed)

    def _relaxed_pool(self) -> list[str]:
        greens: dict[int, str] = {}
        present: set[str] = set()
        for guess, pattern in self.history:
            for slot, digit in enumerate(decode_pattern(pattern, self.size)):
                if digit == CORRECT:
                    greens[slot] = guess[slot]
                elif digit == PRESENT:
                    present.add(guess[slot])
        return [
            word
            for word in self.all_words
            if word not in self.guessed
            and all(word[slot] == letter for slot, letter in greens.items())
            and present.issubset(word)
        ]

    # -- opening guess, with an on-disk cache --------------------------------------

    @property
    def fingerprint(self) -> str:
        if self._fingerprint is None:
            self._fingerprint = wordlist_fingerprint(self.all_words)
        return self._fingerprint

    def _cache_path(self) -> Path | None:
        if self.cache_dir is None:
            return None
        # The scoring rule and the word list both change the answer, so both key the cache.
        name = f"opener_{self.size}_{self.scoring_name}_{self.fingerprint}.json"
        return self.cache_dir / name

    def opening_guess(self) -> str:
        """The first guess: read from cache, or measure it once and cache the result."""
        path = self._cache_path()
        if path is not None and path.exists():
            try:
                cached = json.loads(path.read_text(encoding="utf-8"))
                guess = cached["guess"]
                if isinstance(guess, str) and len(guess) == self.size:
                    return guess
            except (OSError, ValueError, KeyError):
                pass  # A damaged cache is not worth failing over; just recompute it.

        guess, entropy = compute_opener(self.all_words, self.scorer, self.opener_shortlist)
        if path is not None:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(
                        {
                            "guess": guess,
                            "entropy": round(entropy, 4),
                            "size": self.size,
                            "scoring": self.scoring_name,
                            "shortlist_size": self.opener_shortlist,
                            "pool_size": len(self.all_words),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            except OSError:
                pass  # Read-only checkout or similar: correctness does not depend on this.
        return guess

    # -- driver --------------------------------------------------------------------

    def solve(
        self,
        oracle,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
        on_round: Callable[[Round], None] | None = None,
    ) -> SolveResult:
        """Play up to ``max_rounds`` guesses against ``oracle``.

        The API is stateless and never counts attempts, so the six-guess budget is our own
        convention, borrowed from Wordle, and adjustable via ``--max-rounds``.
        """
        started = time.perf_counter()
        rounds: list[Round] = []
        solved = False
        secret: str | None = None

        for index in range(max_rounds):
            before = len(self.candidates)
            guess = self.next_guess(rounds_left=max_rounds - index)
            pattern = oracle.submit(guess)

            if is_solved(pattern, self.size):
                solved, secret = True, guess
                after = 1
            else:
                self.observe(guess, pattern)
                after = len(self.candidates)

            current = Round(index + 1, guess, pattern, before, after)
            rounds.append(current)
            if on_round is not None:
                on_round(current)
            if solved:
                break

        if solved:
            status = "solved"
        elif self.degraded or not self.candidates:
            status = "no_candidates"
        else:
            status = "exhausted"

        return SolveResult(
            solved=solved,
            secret=secret,
            size=self.size,
            status=status,
            scoring=self.scoring_name,
            oracle=getattr(oracle, "label", "unknown"),
            rounds=rounds,
            api_calls=getattr(oracle, "calls", 0),
            elapsed=time.perf_counter() - started,
        )
