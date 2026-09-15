"""The dictionary-free fallback for secrets our word list cannot represent."""

from __future__ import annotations

import pytest

from wordle_solver.api import LocalOracle
from wordle_solver.feedback import score_classic, score_naive
from wordle_solver.recovery import ExhaustiveRecovery, analyse
from wordle_solver.solver import Solver

# Not in /usr/share/dict/words in any capitalisation: the real secret behind
# /random?seed=38, and the case that motivated this module.
OUT_OF_DICTIONARY = "agnew"


def play(guesses, secret):
    return [(guess, score_naive(guess, secret)) for guess in guesses]


class TestKnowledge:
    def test_reads_letters_and_positions_off_the_feedback(self):
        knowledge = analyse(play(["annex"], OUT_OF_DICTIONARY), 5)
        assert knowledge.greens == {0: "a", 2: "n", 3: "e"}
        assert {"a", "n", "e"} <= knowledge.present
        assert "x" in knowledge.absent
        assert knowledge.unknown_slots() == [1, 4]

    def test_absent_letters_are_excluded_from_further_probing(self):
        knowledge = analyse(play(["tarie"], OUT_OF_DICTIONARY), 5)
        assert knowledge.untested().isdisjoint({"t", "r", "i", "a", "e"})


class TestProbing:
    def test_a_probe_tests_one_distinct_letter_per_slot(self, full_pool):
        recovery = ExhaustiveRecovery(5, full_pool)
        history = play(["tarie"], OUT_OF_DICTIONARY)
        probe = recovery.probe_guess(history, {"tarie"})

        assert probe is not None and len(probe) == 5
        knowledge = analyse(history, 5)
        tested = [letter for letter in probe if letter in knowledge.untested()]
        assert len(set(tested)) == len(tested) >= 1

    def test_probing_stops_once_every_letter_is_classified(self, full_pool):
        recovery = ExhaustiveRecovery(5, full_pool)
        history = play(["abcde", "fghij", "klmno", "pqrst", "uvwxy", "zzzzz"], OUT_OF_DICTIONARY)
        assert recovery.probe_guess(history, set()) is None

    def test_never_repeats_an_earlier_guess(self, full_pool):
        recovery = ExhaustiveRecovery(5, full_pool)
        history = play(["tarie"], OUT_OF_DICTIONARY)
        first = recovery.probe_guess(history, set())
        assert recovery.probe_guess(history, {first}) != first


class TestEnumeration:
    def test_waits_until_no_letter_is_unclassified(self, full_pool):
        recovery = ExhaustiveRecovery(5, full_pool)
        assert recovery.enumerate_candidates(play(["annex"], OUT_OF_DICTIONARY)) is None

    def test_reconstructs_a_secret_that_is_not_a_dictionary_word(self, full_pool):
        recovery = ExhaustiveRecovery(5, full_pool)
        history = play(["tarie", "annex"], OUT_OF_DICTIONARY)
        while (probe := recovery.probe_guess(history, {g for g, _ in history})) is not None:
            history.append((probe, score_naive(probe, OUT_OF_DICTIONARY)))

        space = recovery.enumerate_candidates(history)
        assert space is not None
        assert OUT_OF_DICTIONARY in space
        assert all(set(word) == set(OUT_OF_DICTIONARY) for word in space)

    def test_declines_a_space_too_large_to_materialise(self, full_pool):
        recovery = ExhaustiveRecovery(5, full_pool, enumeration_limit=10)
        history = play(["abcde", "fghij", "klmno", "pqrst", "uvwxy", "zzzzz"], "audio")
        assert recovery.enumerate_candidates(history) is None


class TestGuards:
    def test_refuses_classic_scoring(self, full_pool):
        # Under the classic rule an 'absent' can simply mean the copies are used up, so the
        # deduction this module is built on does not hold.
        with pytest.raises(ValueError, match="naive"):
            ExhaustiveRecovery(5, full_pool, scorer=score_classic)


class TestSolverIntegration:
    def test_solver_with_the_fallback_cracks_an_unlisted_secret(self, full_pool, cache_dir):
        assert OUT_OF_DICTIONARY not in full_pool, "fixture assumption"
        solver = Solver(
            words=full_pool,
            size=5,
            scorer=score_naive,
            scoring_name="naive",
            cache_dir=cache_dir,
            fingerprint="live-pool",
            recovery=ExhaustiveRecovery(5, full_pool),
        )
        result = solver.solve(LocalOracle(OUT_OF_DICTIONARY, score_naive), max_rounds=30)

        assert result.solved and result.secret == OUT_OF_DICTIONARY
        assert solver.degraded, "should record that it left the dictionary"

    def test_solver_without_the_fallback_still_gives_up_cleanly(self, full_pool, cache_dir):
        solver = Solver(
            words=full_pool,
            size=5,
            scorer=score_naive,
            scoring_name="naive",
            cache_dir=cache_dir,
            fingerprint="live-pool",
        )
        result = solver.solve(LocalOracle(OUT_OF_DICTIONARY, score_naive), max_rounds=8)
        assert not result.solved and result.status == "no_candidates"
