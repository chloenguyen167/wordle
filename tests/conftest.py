"""Shared fixtures.

The offline tests deliberately run against a reduced pool: they check correctness and
termination, not guess quality. Real performance numbers come from scripts/benchmark.py.
"""

from __future__ import annotations

import random

import pytest

from wordle_solver.wordlist import load_words

SAMPLE_SIZE = 1500
WORD_SIZE = 5


@pytest.fixture(scope="session")
def full_pool() -> list[str]:
    return load_words(WORD_SIZE).words


@pytest.fixture(scope="session")
def pool(full_pool: list[str]) -> list[str]:
    """A deterministic slice of the real word list, small enough to keep tests quick."""
    return sorted(random.Random(0).sample(full_pool, SAMPLE_SIZE))


@pytest.fixture(scope="session")
def cache_dir(tmp_path_factory) -> str:
    """Session-scoped cache so the opening guess is measured once for the whole run."""
    return str(tmp_path_factory.mktemp("opener-cache"))


@pytest.fixture(scope="session")
def secrets(pool: list[str]) -> list[str]:
    return random.Random(1).sample(pool, 20)
