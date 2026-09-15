"""Integration tests against the real Votee API.

Deselected by default (see pytest.ini). Run them with:

    pytest -m live

They need network access and deliberately make real requests, so they are kept out of the
normal suite: a red build should mean our logic broke, not that the wifi dropped.
"""

from __future__ import annotations

import pytest

from wordle_solver.api import (
    ApiError,
    CALIBRATION_GUESS,
    CALIBRATION_SECRET,
    DailyOracle,
    RandomOracle,
    VoteeClient,
    WordOracle,
    calibrate_scoring,
)
from wordle_solver.feedback import parse_api_response, score_classic, score_naive
from wordle_solver.solver import Solver
from wordle_solver.wordlist import load_words

pytestmark = pytest.mark.live

MAX_ROUNDS = 6
SEEDS = [1, 7, 42, 123, 2024]


@pytest.fixture(scope="module")
def client():
    with VoteeClient() as api:
        yield api


@pytest.fixture(scope="module")
def scoring(client):
    return calibrate_scoring(client)


def solve(oracle, scoring, wordlist_cache={}):
    name, scorer = scoring
    if oracle.size not in wordlist_cache:
        wordlist_cache[oracle.size] = load_words(oracle.size)
    words = wordlist_cache[oracle.size]
    solver = Solver(
        words=words.words,
        size=oracle.size,
        scorer=scorer,
        scoring_name=name,
        fingerprint=words.fingerprint,
    )
    return solver.solve(oracle, max_rounds=MAX_ROUNDS)


class TestApiContract:
    def test_scoring_rule_is_still_naive(self, client):
        """Pins the behaviour the solver is built around; goes red if Votee changes it."""
        payload = client.guess_word(CALIBRATION_SECRET, CALIBRATION_GUESS)
        observed = parse_api_response(payload, len(CALIBRATION_SECRET))
        assert observed == score_naive(CALIBRATION_GUESS, CALIBRATION_SECRET)
        assert observed != score_classic(CALIBRATION_GUESS, CALIBRATION_SECRET)

    def test_calibration_detects_it(self, scoring):
        assert scoring[0] == "naive"

    def test_response_shape_matches_the_documented_schema(self, client):
        payload = client.guess_word("hello", "crane")
        assert isinstance(payload, list) and len(payload) == 5
        assert set(payload[0]) == {"slot", "guess", "result"}
        assert {item["result"] for item in payload} <= {"absent", "present", "correct"}

    def test_length_mismatch_is_reported_not_retried(self, client):
        # The API answers 400 with a plain-text body, which must not be mistaken for JSON.
        with pytest.raises(ApiError) as excinfo:
            client.guess_word("hello", "hi")
        assert "400" in str(excinfo.value)


class TestKnownAnswers:
    """/word/{word} gives us a known-answer test: we choose the secret ourselves."""

    @pytest.mark.parametrize("secret", ["hello", "crane", "fuzzy"])
    def test_solves_a_word_we_picked(self, client, scoring, secret):
        result = solve(WordOracle(client, secret), scoring)
        assert result.solved, f"{secret}: {result.status}"
        assert result.secret == secret
        assert result.round_count <= MAX_ROUNDS

    def test_solves_a_six_letter_word(self, client, scoring):
        result = solve(WordOracle(client, "puzzle"), scoring)
        assert result.solved and result.secret == "puzzle"


class TestRandomWords:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_solves_a_seeded_random_word(self, client, scoring, seed):
        result = solve(RandomOracle(client, size=5, seed=seed), scoring)
        assert result.solved, f"seed {seed}: {result.status}"

    def test_the_same_seed_gives_the_same_secret(self, client, scoring):
        first = solve(RandomOracle(client, size=5, seed=99), scoring)
        second = solve(RandomOracle(client, size=5, seed=99), scoring)
        assert first.secret == second.secret

    @pytest.mark.parametrize("seed", [7, 2024])
    def test_handles_a_non_default_word_length(self, client, scoring, seed):
        """Nothing about the solver is tied to five letters."""
        result = solve(RandomOracle(client, size=6, seed=seed), scoring)
        assert result.size == 6
        assert result.solved, f"seed {seed} (size 6): {result.status}"


class TestDaily:
    def test_solves_todays_puzzle(self, client, scoring):
        result = solve(DailyOracle(client, size=5), scoring)
        assert result.solved, result.status
