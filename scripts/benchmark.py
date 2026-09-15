#!/usr/bin/env python3
"""Measure solver performance over a range of /random seeds.

    python scripts/benchmark.py --n 30 --size 5

Reports the guess distribution and, deliberately, the number of puzzles whose secret was
not in our word list -- that failure mode is a property of bringing our own dictionary and
is worth showing rather than hiding.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wordle_solver.api import (  # noqa: E402
    ApiError,
    RandomOracle,
    VoteeClient,
    resolve_scorer,
)
from wordle_solver.feedback import BadResponseError, score_naive  # noqa: E402
from wordle_solver.recovery import ExhaustiveRecovery  # noqa: E402
from wordle_solver.solver import DEFAULT_MAX_ROUNDS, Solver  # noqa: E402
from wordle_solver.wordlist import load_words  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=30, help="how many seeds to run")
    parser.add_argument("--size", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--max-rounds", type=int, default=DEFAULT_MAX_ROUNDS)
    parser.add_argument("--scoring", choices=["auto", "naive", "classic"], default="auto")
    parser.add_argument("--wordlist", default=None)
    parser.add_argument(
        "--exhaustive-fallback",
        action="store_true",
        help="reconstruct secrets that are not in the word list (naive scoring only)",
    )
    args = parser.parse_args(argv)

    word_list = load_words(args.size, args.wordlist)
    seeds = range(args.seed_start, args.seed_start + args.n)
    started = time.perf_counter()

    with VoteeClient() as client:
        scoring_name, scorer = resolve_scorer(args.scoring, client)
        print(f"word list : {len(word_list):,} words of {args.size} letters ({word_list.source})")
        print(f"scoring   : {scoring_name}")
        recovery = (
            ExhaustiveRecovery(args.size, word_list.words, scorer)
            if args.exhaustive_fallback and scorer is score_naive
            else None
        )
        print(f"budget    : {args.max_rounds} guesses")
        print(f"fallback  : {'on' if recovery else 'off'}\n")
        print(f"{'seed':>6}  {'result':<8} {'rounds':>6}  {'calls':>5}  secret")
        print("-" * 46)

        results = []
        for seed in seeds:
            oracle = RandomOracle(client, size=args.size, seed=seed)
            solver = Solver(
                words=word_list.words,
                size=args.size,
                scorer=scorer,
                scoring_name=scoring_name,
                fingerprint=word_list.fingerprint,
                recovery=recovery,
            )
            try:
                result = solver.solve(oracle, max_rounds=args.max_rounds)
            except (ApiError, BadResponseError) as exc:
                print(f"{seed:>6}  {'error':<8} {'-':>6}  {oracle.calls:>5}  {exc}")
                continue

            results.append(result)
            verdict = "solved" if result.solved else result.status
            print(
                f"{seed:>6}  {verdict:<8} {result.round_count:>6}  "
                f"{result.api_calls:>5}  {result.secret or '-'}"
            )

    if not results:
        print("\nno puzzles completed")
        return 1

    solved = [r for r in results if r.solved]
    rounds = [r.round_count for r in solved]
    missing = sum(1 for r in results if r.status == "no_candidates")
    elapsed = time.perf_counter() - started

    print("\n" + "=" * 46)
    print(f"puzzles          : {len(results)}")
    print(f"solved           : {len(solved)} ({len(solved) / len(results):.0%})")
    if rounds:
        print(f"guesses mean     : {statistics.mean(rounds):.2f}")
        print(f"guesses median   : {statistics.median(rounds):.1f}")
        print(f"guesses max      : {max(rounds)}")
        spread = Counter(rounds)
        print("distribution     : " + "  ".join(f"{k}:{spread[k]}" for k in sorted(spread)))
    print(f"secret not in list: {missing}")
    print(f"total API calls  : {sum(r.api_calls for r in results)}")
    print(f"wall time        : {elapsed:.1f}s  ({elapsed / len(results):.2f}s per puzzle)")
    return 0 if len(solved) == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
