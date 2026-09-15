"""Terminal rendering. Kept apart from the solver so the algorithm stays I/O free."""

from __future__ import annotations

from .feedback import pattern_to_emoji
from .solver import Round, SolveResult

STATUS_NOTES = {
    "exhausted": "ran out of guesses -- try --max-rounds",
    "no_candidates": "the secret is not in our word list (the API uses its own dictionary)",
}


def render_header(oracle_label: str, size: int, pool: int, source: str, scoring: str) -> str:
    return (
        f"puzzle   : {oracle_label}  (size {size})\n"
        f"word list: {pool:,} candidates from {source}\n"
        f"scoring  : {scoring}"
    )


def render_round(rnd: Round, size: int) -> str:
    spaced = " ".join(rnd.guess)
    emoji = pattern_to_emoji(rnd.pattern, size)
    return f"  {rnd.index}  {spaced}   {emoji}   candidates left: {rnd.candidates_after:,}"


def render_summary(result: SolveResult) -> str:
    if result.solved:
        headline = f"solved '{result.secret}' in {result.round_count} guess(es)"
    else:
        note = STATUS_NOTES.get(result.status, result.status)
        headline = f"not solved after {result.round_count} guess(es) -- {note}"
    return f"\n{headline}\n{result.api_calls} API call(s), {result.elapsed:.2f}s"
