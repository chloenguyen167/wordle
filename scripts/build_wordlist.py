#!/usr/bin/env python3
"""Regenerate the bundled English word list at ``data/words_en.txt``.

This is a one-off build step, not part of the solver runtime: the generated file is
committed so that reviewers can run the solver on any OS without downloading anything.

Source preference:
  1. ``/usr/share/dict/words`` -- the ``web2`` list shipped with macOS/BSD, derived from
     Webster's Second International Dictionary (1934), which is in the public domain.
  2. NLTK's ``words`` corpus, if NLTK happens to be installed.

Only entries that are already lowercase ASCII letters are kept, which drops proper nouns
(``Aachen``, ``Zulu``) without needing a separate name list.

Usage:
    python scripts/build_wordlist.py [--out data/words_en.txt] [--min-len 3] [--max-len 12]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SYSTEM_DICT = Path("/usr/share/dict/words")


def read_system_dict(path: Path) -> tuple[str, list[str]]:
    """Return (source label, raw entries) from a newline-delimited dictionary file."""
    with path.open(encoding="utf-8", errors="ignore") as handle:
        return str(path), [line.strip() for line in handle]


def read_nltk() -> tuple[str, list[str]]:
    """Return (source label, raw entries) from the NLTK ``words`` corpus."""
    import nltk  # noqa: PLC0415 -- optional dependency, imported only when needed
    from nltk.corpus import words as nltk_words  # noqa: PLC0415

    try:
        entries = nltk_words.words()
    except LookupError:
        nltk.download("words")
        entries = nltk_words.words()
    return "nltk.corpus.words", list(entries)


def collect_entries() -> tuple[str, list[str]]:
    if SYSTEM_DICT.exists():
        return read_system_dict(SYSTEM_DICT)
    try:
        return read_nltk()
    except ImportError:
        raise SystemExit(
            f"No word source available: {SYSTEM_DICT} is missing and NLTK is not installed.\n"
            "Install NLTK (pip install nltk) or point --out at a list you supply yourself."
        )


def normalise(entries: list[str], min_len: int, max_len: int) -> list[str]:
    """Keep lowercase ASCII words within the length range, deduplicated and sorted."""
    kept = {
        word
        for word in entries
        if min_len <= len(word) <= max_len and word.isascii() and word.isalpha() and word.islower()
    }
    return sorted(kept)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("data/words_en.txt"))
    parser.add_argument("--min-len", type=int, default=3)
    parser.add_argument("--max-len", type=int, default=12)
    args = parser.parse_args(argv)

    source, entries = collect_entries()
    words = normalise(entries, args.min_len, args.max_len)
    if not words:
        raise SystemExit(f"No usable words found in {source}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(words) + "\n", encoding="utf-8")

    print(f"source : {source}")
    print(f"output : {args.out} ({len(words):,} words, {args.out.stat().st_size / 1024:.0f} KiB)")
    print("counts by length:")
    for size in range(args.min_len, args.max_len + 1):
        count = sum(1 for word in words if len(word) == size)
        print(f"  {size:2d} -> {count:6,d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
