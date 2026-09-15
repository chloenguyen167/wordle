"""Command line entry point.

    python -m wordle_solver.cli daily
    python -m wordle_solver.cli random --seed 42 --size 6
    python -m wordle_solver.cli word hello
    python -m wordle_solver.cli warm-cache --sizes 4 5 6 7

Exit codes: 0 solved, 1 not solved within the guess budget, 2 configuration or API error.
"""

from __future__ import annotations

import argparse
import json
import sys

from .api import (
    DEFAULT_BASE_URL,
    ApiError,
    DailyOracle,
    RandomOracle,
    VoteeClient,
    WordOracle,
    resolve_scorer,
)
from .display import render_header, render_round, render_summary
from .feedback import BadResponseError, score_naive
from .recovery import ExhaustiveRecovery
from .solver import (
    DEFAULT_GUESS_POOL_LIMIT,
    DEFAULT_MAX_ROUNDS,
    DEFAULT_OPENER_SHORTLIST,
    Solver,
    SolverError,
)
from .wordlist import WordListError, load_words


#: Applied after parsing. The shared options use ``argparse.SUPPRESS`` so that they can be
#: given on either side of the subcommand without the subparser's defaults clobbering a
#: value the user already supplied before it.
DEFAULTS = {
    "base_url": DEFAULT_BASE_URL,
    "wordlist": None,
    "scoring": "auto",
    "max_rounds": DEFAULT_MAX_ROUNDS,
    "guess_pool_limit": DEFAULT_GUESS_POOL_LIMIT,
    "opener_shortlist": DEFAULT_OPENER_SHORTLIST,
    "no_cache": False,
    "json": False,
    "quiet": False,
    "exhaustive_fallback": False,
    "size": 5,
    "seed": None,
    "sizes": [4, 5, 6, 7],
}


#: The fallback trades guesses for coverage, so it needs more room than Wordle's six.
FALLBACK_MAX_ROUNDS = 30


def _common_options() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False, argument_default=argparse.SUPPRESS)
    common.add_argument("--base-url", help=f"API base URL (default: {DEFAULT_BASE_URL})")
    common.add_argument("--wordlist", help="path to a custom word list")
    common.add_argument(
        "--scoring",
        choices=["auto", "naive", "classic"],
        help="feedback rule to mirror; 'auto' probes the API once to detect it (default)",
    )
    common.add_argument("--max-rounds", type=int, help="guess budget (default: 6)")
    common.add_argument("--guess-pool-limit", type=int, help="max guesses scored per round")
    common.add_argument("--opener-shortlist", type=int, help="words considered for the opener")
    common.add_argument("--no-cache", action="store_true", help="ignore the cached opening guess")
    common.add_argument("--json", action="store_true", help="print the result as JSON")
    common.add_argument("--quiet", action="store_true", help="only print the summary")
    common.add_argument(
        "--exhaustive-fallback",
        action="store_true",
        help="if the secret is not in the word list, reconstruct it by probing letters "
        "(naive scoring only; needs a larger --max-rounds, defaulted to "
        f"{FALLBACK_MAX_ROUNDS})",
    )
    return common


def build_parser() -> argparse.ArgumentParser:
    common = _common_options()
    parser = argparse.ArgumentParser(
        prog="wordle_solver",
        description="Automatically solve the Votee Wordle-like puzzle API.",
        parents=[common],
    )
    sub = parser.add_subparsers(dest="command", required=True)

    daily = sub.add_parser(
        "daily",
        help="solve today's puzzle",
        parents=[common],
        argument_default=argparse.SUPPRESS,
    )
    daily.add_argument("--size", type=int, help="word length (default: 5)")

    random_cmd = sub.add_parser(
        "random",
        help="solve a random word",
        parents=[common],
        argument_default=argparse.SUPPRESS,
    )
    random_cmd.add_argument("--size", type=int, help="word length (default: 5)")
    random_cmd.add_argument("--seed", type=int, help="reuse a seed to replay the same secret")

    word = sub.add_parser(
        "word",
        help="solve a word you choose (a known-answer test)",
        parents=[common],
        argument_default=argparse.SUPPRESS,
    )
    word.add_argument("target", help="the secret word the API should score against")

    warm = sub.add_parser(
        "warm-cache",
        help="precompute opening guesses so later runs start instantly",
        parents=[common],
        argument_default=argparse.SUPPRESS,
    )
    warm.add_argument("--sizes", type=int, nargs="+", help="word lengths to precompute")

    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    supplied = set(vars(args))
    for name, value in DEFAULTS.items():
        if not hasattr(args, name):
            setattr(args, name, value)
    # The fallback is useless inside a six-guess budget, so raise the ceiling for it --
    # but never override a budget the user asked for explicitly.
    if args.exhaustive_fallback and "max_rounds" not in supplied:
        args.max_rounds = FALLBACK_MAX_ROUNDS
    return args


def _build_recovery(args, scoring_name: str, scorer, words):
    """Build the dictionary-free fallback, or explain why it is unavailable."""
    if not args.exhaustive_fallback:
        return None
    if scorer is not score_naive:
        print(
            f"warning: --exhaustive-fallback needs naive scoring, but the API scores "
            f"'{scoring_name}'; continuing without it",
            file=sys.stderr,
        )
        return None
    return ExhaustiveRecovery(len(words[0]), words, scorer)


def _make_oracle(args: argparse.Namespace, client: VoteeClient):
    if args.command == "daily":
        return DailyOracle(client, args.size)
    if args.command == "random":
        return RandomOracle(client, args.size, args.seed)
    if args.command == "word":
        return WordOracle(client, args.target.strip().lower())
    raise ValueError(f"unknown command {args.command!r}")


def _warm_cache(args: argparse.Namespace) -> int:
    """Precompute and cache the opening guess for each size, under every scoring rule."""
    from .feedback import SCORERS  # local import keeps the hot path lean

    for size in args.sizes:
        word_list = load_words(size, args.wordlist)
        for name, scorer in SCORERS.items():
            solver = Solver(
                words=word_list.words,
                size=size,
                scorer=scorer,
                scoring_name=name,
                opener_shortlist=args.opener_shortlist,
                fingerprint=word_list.fingerprint,
            )
            guess = solver.opening_guess()
            print(f"size {size} / {name:<7} -> {guess}   ({len(word_list):,} words)")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        if args.command == "warm-cache":
            return _warm_cache(args)

        with VoteeClient(base_url=args.base_url) as client:
            oracle = _make_oracle(args, client)
            word_list = load_words(oracle.size, args.wordlist)
            scoring_name, scorer = resolve_scorer(args.scoring, client)
            recovery = _build_recovery(args, scoring_name, scorer, word_list.words)

            solver = Solver(
                words=word_list.words,
                size=oracle.size,
                scorer=scorer,
                scoring_name=scoring_name,
                guess_pool_limit=args.guess_pool_limit,
                opener_shortlist=args.opener_shortlist,
                cache_dir=None if args.no_cache else ".cache",
                fingerprint=word_list.fingerprint,
                recovery=recovery,
            )

            show = not (args.quiet or args.json)
            if show:
                print(
                    render_header(
                        oracle.label, oracle.size, len(word_list), word_list.source, scoring_name
                    )
                )
                print()

            result = solver.solve(
                oracle,
                max_rounds=args.max_rounds,
                on_round=(lambda rnd: print(render_round(rnd, oracle.size))) if show else None,
            )

        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(render_summary(result))
        return 0 if result.solved else 1

    except (ApiError, BadResponseError, WordListError, SolverError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
