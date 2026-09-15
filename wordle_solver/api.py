"""HTTP client for the Votee Wordle API, plus the oracle abstraction the solver talks to.

The solver never touches HTTP directly. It is handed an ``Oracle`` -- something that turns
a guess into a pattern -- which lets the exact same solving code run against the live API
or against a purely local secret in tests.
"""

from __future__ import annotations

import time
from typing import Any, Protocol

import requests

from .feedback import (
    SCORERS,
    BadResponseError,
    Scorer,
    build_api_payload,
    parse_api_response,
    score_naive,
)

DEFAULT_BASE_URL = "https://wordle.votee.dev:8000"

#: Probe used by ``calibrate_scoring``. The two scoring rules disagree on this exact pair,
#: which is what makes one request enough to tell them apart:
#:   naive   -> present, present, present, present, correct
#:   classic -> absent,  absent,  absent,  absent,  correct
CALIBRATION_SECRET = "hello"
CALIBRATION_GUESS = "ooooo"


class ApiError(Exception):
    """The API could not be reached, or answered with an error status."""


class VoteeClient:
    """Thin ``requests`` wrapper over the three documented endpoints.

    Retries connection errors, timeouts, 429s and 5xx with exponential backoff; treats any
    other 4xx as a bug in our request and fails immediately rather than hammering the API.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 10.0,
        max_retries: int = 3,
        backoff: float = 0.5,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self.session = session or requests.Session()
        self.calls = 0

    # -- endpoints ----------------------------------------------------------------

    def guess_daily(self, guess: str, size: int | None = None) -> list[dict[str, Any]]:
        return self._get("/daily", {"guess": guess, "size": size})

    def guess_random(
        self, guess: str, size: int | None = None, seed: int | None = None
    ) -> list[dict[str, Any]]:
        return self._get("/random", {"guess": guess, "size": size, "seed": seed})

    def guess_word(self, word: str, guess: str) -> list[dict[str, Any]]:
        # /word/{word} takes no size parameter -- the length comes from the path segment.
        return self._get(f"/word/{word}", {"guess": guess})

    # -- plumbing -----------------------------------------------------------------

    def _get(self, path: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        url = f"{self.base_url}{path}"
        query = {key: value for key, value in params.items() if value is not None}
        last_error: str = "request never completed"

        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.get(url, params=query, timeout=self.timeout)
            except requests.RequestException as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                if response.status_code == 200:
                    self.calls += 1
                    try:
                        return response.json()
                    except ValueError as exc:
                        raise BadResponseError(f"response was not JSON ({exc})", response.text[:200])
                if response.status_code == 429 or response.status_code >= 500:
                    last_error = f"HTTP {response.status_code}: {_error_detail(response)}"
                else:
                    raise ApiError(
                        f"GET {path} failed with HTTP {response.status_code}: {_error_detail(response)}"
                    )

            if attempt < self.max_retries:
                time.sleep(self.backoff * 2**attempt)

        raise ApiError(f"GET {path} failed after {self.max_retries + 1} attempts -- {last_error}")

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> VoteeClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _error_detail(response: requests.Response) -> str:
    """Extract a human-readable reason from an error response.

    Worth the care: a 422 comes back as FastAPI's JSON validation envelope, while a 400
    (for example a guess whose length does not match the word) comes back as plain text,
    so ``response.json()`` would raise and mask the real message.
    """
    try:
        body = response.json()
    except ValueError:
        return response.text.strip()[:200] or "<empty body>"
    if isinstance(body, dict) and "detail" in body:
        return str(body["detail"])[:200]
    return str(body)[:200]


def calibrate_scoring(client: VoteeClient) -> tuple[str, Scorer]:
    """Ask the server which scoring rule it uses, using a single probe request.

    Returns ``(name, scorer)``. Falling back to ``naive`` on an unrecognised answer is
    deliberate: that is what the API was observed to do, so it is the safer default if the
    probe ever stops being decisive.
    """
    payload = client.guess_word(CALIBRATION_SECRET, CALIBRATION_GUESS)
    observed = parse_api_response(payload, len(CALIBRATION_SECRET))
    for name, scorer in SCORERS.items():
        if scorer(CALIBRATION_GUESS, CALIBRATION_SECRET) == observed:
            return name, scorer
    return "naive", score_naive


def resolve_scorer(choice: str, client: VoteeClient | None) -> tuple[str, Scorer]:
    """Turn the ``--scoring`` CLI value into a concrete scorer."""
    if choice != "auto":
        return choice, SCORERS[choice]
    if client is None:
        raise ApiError("--scoring auto needs an API client to probe")
    return calibrate_scoring(client)


# -- oracles ----------------------------------------------------------------------


class Oracle(Protocol):
    """Anything that can score a guess for one hidden secret."""

    size: int
    label: str
    calls: int

    def submit(self, guess: str) -> int: ...


class _BaseOracle:
    def __init__(self, size: int, label: str) -> None:
        self.size = size
        self.label = label
        self.calls = 0

    def submit(self, guess: str) -> int:
        if len(guess) != self.size:
            raise ValueError(f"guess {guess!r} is not {self.size} letters long")
        payload = self._request(guess)
        self.calls += 1
        return parse_api_response(payload, self.size)

    def _request(self, guess: str) -> list[dict[str, Any]]:
        raise NotImplementedError


class DailyOracle(_BaseOracle):
    def __init__(self, client: VoteeClient, size: int = 5) -> None:
        super().__init__(size, "daily")
        self.client = client

    def _request(self, guess: str) -> list[dict[str, Any]]:
        return self.client.guess_daily(guess, self.size)


class RandomOracle(_BaseOracle):
    def __init__(self, client: VoteeClient, size: int = 5, seed: int | None = None) -> None:
        super().__init__(size, f"random(seed={seed})" if seed is not None else "random")
        self.client = client
        self.seed = seed

    def _request(self, guess: str) -> list[dict[str, Any]]:
        return self.client.guess_random(guess, self.size, self.seed)


class WordOracle(_BaseOracle):
    def __init__(self, client: VoteeClient, word: str) -> None:
        super().__init__(len(word), f"word({word})")
        self.client = client
        self.word = word

    def _request(self, guess: str) -> list[dict[str, Any]]:
        return self.client.guess_word(self.word, guess)


class LocalOracle(_BaseOracle):
    """Offline stand-in that scores against a known secret.

    It deliberately round-trips through the same JSON shape the API returns, so offline
    tests exercise ``parse_api_response`` as well as the solver itself.
    """

    def __init__(self, secret: str, scorer: Scorer = score_naive) -> None:
        super().__init__(len(secret), f"local({secret})")
        self.secret = secret
        self.scorer = scorer

    def _request(self, guess: str) -> list[dict[str, Any]]:
        return build_api_payload(guess, self.scorer(guess, self.secret))
