"""Loading the candidate word pool.

The API never tells us which dictionary its secrets come from, so the solver brings its
own. Sources are tried in order of reproducibility: an explicit path beats the list
bundled in this repo, which beats whatever the host OS happens to ship.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

PACKAGE_ROOT = Path(__file__).resolve().parent
BUNDLED_WORDS = PACKAGE_ROOT.parent / "data" / "words_en.txt"
SYSTEM_DICT = Path("/usr/share/dict/words")


class WordListError(Exception):
    """No usable word source, or no words of the requested length."""


@dataclass(frozen=True)
class WordList:
    """A length-filtered, deduplicated, sorted pool plus the source it came from."""

    size: int
    source: str
    words: list[str]

    def __len__(self) -> int:
        return len(self.words)

    @property
    def fingerprint(self) -> str:
        return wordlist_fingerprint(self.words)


def wordlist_fingerprint(words: list[str]) -> str:
    """Short digest of the pool, used to invalidate the cached opening guess."""
    digest = hashlib.sha1("\n".join(words).encode("utf-8"))
    return digest.hexdigest()[:8]


def _read_lines(path: Path) -> Iterator[str]:
    with path.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            yield line.strip()


def _read_nltk() -> Iterator[str]:
    """Deliberately not a generator: the import must fail here, while we can still catch it."""
    from nltk.corpus import words as nltk_words  # noqa: PLC0415 -- optional dependency

    return iter(nltk_words.words())


def _candidate_sources(path: str | Path | None) -> Iterator[tuple[str, Iterator[str]]]:
    """Yield ``(label, entries)`` for each source worth trying, best first."""
    if path is not None:
        explicit = Path(path)
        if not explicit.exists():
            raise WordListError(f"word list not found: {explicit}")
        yield str(explicit), _read_lines(explicit)
        return

    if BUNDLED_WORDS.exists():
        yield f"bundled:{BUNDLED_WORDS.name}", _read_lines(BUNDLED_WORDS)
    if SYSTEM_DICT.exists():
        yield str(SYSTEM_DICT), _read_lines(SYSTEM_DICT)
    try:
        entries = _read_nltk()
    except (ImportError, LookupError):
        pass  # NLTK missing, or installed without its 'words' corpus downloaded.
    else:
        yield "nltk.corpus.words", entries


def load_words(size: int, path: str | Path | None = None) -> WordList:
    """Load every lowercase ASCII word of exactly ``size`` letters.

    Raises ``WordListError`` if no source yields any word of that length -- an empty pool
    would otherwise surface much later as a confusing "no candidates" failure.
    """
    if size < 1:
        raise WordListError(f"word size must be positive, got {size}")

    tried: list[str] = []
    for label, entries in _candidate_sources(path):
        tried.append(label)
        words = sorted(
            {
                word
                for raw in entries
                for word in (raw.strip().lower(),)
                if len(word) == size and word.isascii() and word.isalpha()
            }
        )
        if words:
            return WordList(size=size, source=label, words=words)

    if not tried:
        raise WordListError(
            "no word source available: the bundled list is missing, /usr/share/dict/words "
            "does not exist, and NLTK is not installed. Regenerate the bundled list with "
            "`python scripts/build_wordlist.py` or pass --wordlist PATH."
        )
    raise WordListError(f"no {size}-letter words found in any of: {', '.join(tried)}")
