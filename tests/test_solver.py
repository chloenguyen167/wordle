"""End-to-end solver behaviour, run entirely offline against LocalOracle."""

from __future__ import annotations

import pytest

from wordle_solver.api import LocalOracle
from wordle_solver.feedback import SCORERS, is_solved, score_classic, score_naive
from wordle_solver.solver import (
    Solver,
    SolverError,
    compute_opener,
    expected_information,
    rank_by_letter_frequency,
)

MAX_ROUNDS = 6


def make_solver(pool, cache_dir, scoring_name="naive", **kwargs):
    return Solver(
        words=pool,
        size=len(pool[0]),
        scorer=SCORERS[scoring_name],
        scoring_name=scoring_name,
        opener_shortlist=kwargs.pop("opener_shortlist", 60),
        cache_dir=cache_dir,
        fingerprint=f"test-{len(pool)}",
        **kwargs,
    )


class TestSolving:
    @pytest.mark.parametrize("scoring_name", sorted(SCORERS))
    def test_solves_every_sampled_secret_within_the_budget(
        self, pool, secrets, cache_dir, scoring_name
    ):
        scorer = SCORERS[scoring_name]
        rounds_used = []
        for secret in secrets:
            solver = make_solver(pool, cache_dir, scoring_name)
            result = solver.solve(LocalOracle(secret, scorer), max_rounds=MAX_ROUNDS)
            assert result.solved, f"{scoring_name}: failed on {secret!r}"
            assert result.secret == secret
            rounds_used.append(result.round_count)

        average = sum(rounds_used) / len(rounds_used)
        assert average <= 5.0, f"{scoring_name}: averaged {average:.2f} guesses"

    def test_a_secret_guessed_first_ends_immediately(self, pool, cache_dir):
        solver = make_solver(pool, cache_dir)
        opener = solver.opening_guess()
        result = solver.solve(LocalOracle(opener, score_naive), max_rounds=MAX_ROUNDS)
        assert result.solved and result.round_count == 1


class TestInvariants:
    def test_candidates_shrink_and_always_retain_the_secret(self, pool, secrets, cache_dir):
        secret = secrets[0]
        solver = make_solver(pool, cache_dir)
        oracle = LocalOracle(secret, score_naive)

        previous = len(solver.candidates)
        for _ in range(MAX_ROUNDS):
            guess = solver.next_guess()
            pattern = oracle.submit(guess)
            if is_solved(pattern, solver.size):
                break
            solver.observe(guess, pattern)
            assert len(solver.candidates) <= previous
            assert secret in solver.candidates, "the true secret was filtered out"
            previous = len(solver.candidates)

    def test_never_repeats_a_guess(self, pool, secrets, cache_dir):
        solver = make_solver(pool, cache_dir)
        result = solver.solve(LocalOracle(secrets[1], score_naive), max_rounds=MAX_ROUNDS)
        guesses = [rnd.guess for rnd in result.rounds]
        assert len(guesses) == len(set(guesses))

    def test_selection_is_deterministic(self, pool, secrets, cache_dir):
        runs = [
            [rnd.guess for rnd in make_solver(pool, cache_dir)
             .solve(LocalOracle(secrets[2], score_naive), max_rounds=MAX_ROUNDS).rounds]
            for _ in range(2)
        ]
        assert runs[0] == runs[1]


class TestScoringMismatch:
    """Why calibration exists: mirroring the wrong rule silently discards the answer."""

    def test_classic_filter_drops_a_secret_the_api_scored_naively(self, pool, cache_dir):
        # The API scores "ooooo" against "hello" as four present + one correct. A solver
        # filtering with the classic rule expects four absent + one correct, so it rejects
        # "hello" -- the very word it is looking for.
        solver = make_solver(pool + ["hello"], cache_dir, scoring_name="classic")
        pattern = score_naive("ooooo", "hello")

        solver.observe("ooooo", pattern)

        assert "hello" not in solver.candidates

    def test_naive_filter_keeps_that_secret(self, pool, cache_dir):
        solver = make_solver(pool + ["hello"], cache_dir, scoring_name="naive")
        solver.observe("ooooo", score_naive("ooooo", "hello"))
        assert "hello" in solver.candidates

    def test_a_mismatched_solver_ends_up_degraded_or_wrong(self, pool, cache_dir):
        """Played out fully, the mismatch strands the solver with no candidates left."""
        solver = make_solver(pool + ["hello"], cache_dir, scoring_name="classic")
        solver.observe("ooooo", score_naive("ooooo", "hello"))
        result = solver.solve(LocalOracle("hello", score_naive), max_rounds=MAX_ROUNDS)
        assert solver.degraded or not result.solved


class TestDegradedModes:
    def test_secret_outside_the_word_list_does_not_crash(self, pool, cache_dir):
        solver = make_solver(pool, cache_dir)
        result = solver.solve(LocalOracle("zzzqx", score_naive), max_rounds=MAX_ROUNDS)
        assert not result.solved
        assert result.status == "no_candidates"
        assert solver.degraded

    def test_recovery_relaxes_absent_constraints(self, pool, cache_dir):
        """A secret we cannot represent still leaves the solver making legal guesses."""
        solver = make_solver(pool, cache_dir)
        result = solver.solve(LocalOracle("qxjzv", score_naive), max_rounds=MAX_ROUNDS)
        assert len(result.rounds) == MAX_ROUNDS
        assert all(len(rnd.guess) == 5 for rnd in result.rounds)

    def test_empty_word_list_is_rejected_up_front(self):
        with pytest.raises(SolverError):
            Solver(words=[], size=5)


class TestInformationMetrics:
    def test_entropy_is_zero_when_nothing_can_be_distinguished(self):
        assert expected_information("aaaaa", ["aaaaa"], score_naive) == 0.0

    def test_entropy_is_one_bit_for_an_even_two_way_split(self):
        # "ab" scores differently against these two secrets, splitting them 1:1.
        assert expected_information("ab", ["ab", "cd"], score_naive) == pytest.approx(1.0)

    def test_entropy_is_bounded_by_the_pool_size(self, pool):
        import math

        score = expected_information(pool[0], pool, score_naive)
        assert 0.0 <= score <= math.log2(len(pool))

    def test_frequency_ranking_bounds_the_pool(self, pool):
        ranked = rank_by_letter_frequency(pool, 50)
        assert len(ranked) == 50
        assert ranked == sorted(ranked)

    def test_opener_is_measured_not_hardcoded(self, pool):
        """The opening guess must come from the pool in play, not a Wordle-specific word."""
        word, entropy = compute_opener(pool, score_naive, shortlist_size=40)
        assert word in pool
        assert entropy > 0

        subset = [w for w in pool if w.startswith(("a", "b", "c"))]
        other, _ = compute_opener(subset, score_naive, shortlist_size=40)
        assert other in subset


class TestOpenerCache:
    def test_second_solver_reuses_the_cached_opener(self, pool, tmp_path):
        first = make_solver(pool, str(tmp_path))
        opener = first.opening_guess()
        cached = list(tmp_path.glob("opener_*.json"))
        assert len(cached) == 1

        second = make_solver(pool, str(tmp_path))
        assert second.opening_guess() == opener

    def test_a_corrupt_cache_is_recomputed_not_fatal(self, pool, tmp_path):
        solver = make_solver(pool, str(tmp_path))
        opener = solver.opening_guess()
        for path in tmp_path.glob("opener_*.json"):
            path.write_text("{not json", encoding="utf-8")
        assert make_solver(pool, str(tmp_path)).opening_guess() == opener

    def test_cache_key_separates_scoring_rules(self, pool, tmp_path):
        make_solver(pool, str(tmp_path), scoring_name="naive").opening_guess()
        make_solver(pool, str(tmp_path), scoring_name="classic").opening_guess()
        assert len(list(tmp_path.glob("opener_*.json"))) == 2


class TestWordListErrors:
    def test_a_length_with_no_words_fails_cleanly(self):
        """A missing optional dependency must not escape as a raw ImportError."""
        from wordle_solver.wordlist import WordListError, load_words

        with pytest.raises(WordListError, match="no 99-letter words"):
            load_words(99)

    def test_a_missing_word_list_file_is_reported(self, tmp_path):
        from wordle_solver.wordlist import WordListError, load_words

        with pytest.raises(WordListError, match="not found"):
            load_words(5, tmp_path / "nope.txt")
