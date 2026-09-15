# Votee Wordle Solver

A Python program that automatically solves the Wordle-like puzzle served by
`https://wordle.votee.dev:8000`, for any word length, by combining constraint filtering
with expected-information-gain guess selection.

```
$ python -m wordle_solver.cli daily

puzzle   : daily  (size 5)
word list: 8,506 candidates from bundled:words_en.txt
scoring  : naive

  1  t a r i e   🟨🟩⬛🟩⬛   candidates left: 10
  2  c h o l d   ⬛🟨⬛⬛⬛   candidates left: 1
  3  h a b i t   🟩🟩🟩🟩🟩   candidates left: 1

solved 'habit' in 3 guess(es)
3 API call(s), 0.42s
```

## Quickstart

Tested on Python 3.12.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m wordle_solver.cli daily                     # today's puzzle
python -m wordle_solver.cli random --seed 42          # a seeded random word
python -m wordle_solver.cli random --seed 7 --size 6  # any word length
python -m wordle_solver.cli word hello                # a word you choose
```

Useful options (accepted before or after the subcommand):

| Option | Purpose |
| --- | --- |
| `--size N` | Word length. Nothing in the solver assumes 5. |
| `--max-rounds N` | Guess budget (default 6). |
| `--scoring auto\|naive\|classic` | Which feedback rule to mirror. `auto` (default) probes the API. |
| `--exhaustive-fallback` | Solve secrets that are not in the word list. See [Coverage](#coverage-and-the-exhaustive-fallback). |
| `--wordlist PATH` | Use your own word list. |
| `--json` | Emit the result as JSON. |
| `warm-cache --sizes 4 5 6 7` | Precompute opening guesses so later runs start instantly. |

Exit codes: `0` solved, `1` not solved within the budget, `2` configuration or API error.

## How it works

**Filtering is just replaying the scorer.** A word is still a candidate exactly when, if it
were the secret, it would have produced every pattern the server has returned so far:

```python
self.candidates = [w for w in self.candidates if self.scorer(guess, w) == pattern]
```

That one rule subsumes the usual green/yellow/grey bookkeeping. It also stays correct under
an unconventional scoring rule, as long as `scorer` matches the server — which is why
calibration (below) matters so much.

**Guesses are chosen by expected information gain.** Scoring a candidate guess against
every remaining candidate partitions them by outcome; the entropy of that partition is the
number of bits the guess is expected to reveal, and we play the guess that maximises it.
A guess that lumps every candidate into one bucket teaches nothing; one that spreads them
evenly teaches the most. This framing is the information-theoretic approach to Wordle
popularised by 3Blue1Brown (see [Credits](#credits)).

Supporting details:

- **Patterns are base-3 integers.** `absent=0, present=1, correct=2`, slot 0 least
  significant. Cheap to compute, cheap to compare, usable directly as dictionary keys when
  grouping candidates.
- **The opening guess is measured, not hardcoded.** `CRANE`/`SLATE` are tuned for the
  original Wordle answer list and say nothing about an arbitrary dictionary or word length.
  Instead the opener is computed from the pool actually in play and cached in `.cache/`,
  keyed by word length, scoring rule, and a fingerprint of the word list, so changing any of
  them invalidates it. For the 8,506-word five-letter pool the measured opener is `tarie`
  (6.02 bits); for seven letters it is `saltine`.
- **Cost is bounded deliberately.** Scoring every word against every word is quadratic —
  about 72M comparisons for the five-letter pool. The opener therefore evaluates a
  frequency-ranked shortlist (300 words) against the *full* pool: the entropy is still
  measured over every possible secret, but the search for the argmax is approximate. Later
  rounds cap the guess pool the same way. This is a deliberate accuracy-for-speed trade.
- **Mostly hard mode, with an endgame exception.** Normally the solver only guesses words
  that could still be the answer: equally informative while the pool is large, and able to
  win outright. That backfires on clusters like `stood / stool / stoop / stook`, where
  trying them one at a time burns a guess each. So when the remaining guesses cannot cover
  the remaining candidates, the solver will also consider a word that cannot win but tests
  several of the distinguishing letters at once. On `/random?seed=16` (`stoop`) this turned
  a loss into a 4-guess win: round 3 played `pylon`, which is not a candidate, and cut 16
  candidates to 1.

## API finding: this API does not score duplicate letters like Wordle

The single most important thing discovered while building this. The API's feedback rule is
**not** the original Wordle rule:

```
$ curl -s "https://wordle.votee.dev:8000/word/hello?guess=ooooo"
present, present, present, present, correct

$ curl -s "https://wordle.votee.dev:8000/word/hello?guess=lllll"
present, present, correct, correct, present
```

Under Wordle's rules `hello` contains exactly one `o`, claimed by the exact match in the
last slot, so the first four slots should be `absent`. This API instead judges each slot
independently:

```
correct  if guess[i] == secret[i]
present  if guess[i] occurs anywhere in secret
absent   otherwise
```

This matters because a solver that filters with the textbook two-pass rule would compute a
different pattern for the true secret than the server returned, **discard the answer**, and
then fail with an empty candidate list. `tests/test_solver.py::TestScoringMismatch` pins
that failure down explicitly.

So the solver does not assume. On startup it sends one probe — `/word/hello?guess=ooooo`,
chosen because the two rules disagree on it — and selects whichever scoring function
reproduces the server's answer:

```
$ python -m wordle_solver.cli word hello
scoring  : naive
```

Both rules are implemented (`score_naive`, `score_classic`), so the solver keeps working if
Votee ever fixes this; `--scoring classic` forces the Wordle rule.
`tests/test_live_api.py::TestApiContract::test_scoring_rule_is_still_naive` goes red if the
server's behaviour changes.

Other verified API behaviour:

- Guesses are **not** validated against a dictionary — `guess=zzzzz` is scored normally.
- A guess whose length differs from the word returns HTTP 400 with a **plain-text** body,
  not JSON, so `response.json()` raises and must be guarded.
- Uppercase is accepted for both the guess and the `/word/{word}` path segment.
- The API is stateless and never counts attempts. The six-guess budget is our own
  convention, borrowed from Wordle, and adjustable with `--max-rounds`.
- `/word/{word}` takes no `size` parameter; the length comes from the path.

## Word list

The API never reveals which dictionary its secrets come from, so the solver brings its own.
`data/words_en.txt` is generated from `/usr/share/dict/words` — the `web2` list shipped with
macOS and BSD, derived from Webster's Second International Dictionary (1934), which is in
the public domain. Only entries that are already lowercase ASCII letters are kept, which
drops proper nouns without needing a separate name list.

It is committed to the repository so the solver runs anywhere with no download step.
Regenerate it with:

```bash
python scripts/build_wordlist.py
```

| Length | Words | | Length | Words |
| ---: | ---: | --- | ---: | ---: |
| 3 | 1,142 | | 8 | 26,446 |
| 4 | 4,360 | | 9 | 28,841 |
| 5 | 8,506 | | 10 | 27,928 |
| 6 | 15,073 | | 11 | 23,778 |
| 7 | 20,562 | | 12 | 18,843 |

At runtime the loader falls back through: `--wordlist PATH` → the bundled file →
`/usr/share/dict/words` → NLTK's `words` corpus if it happens to be installed. NLTK is not
a dependency.

`web2` is broad rather than curated, so it carries plenty of archaic words. That makes the
pool larger than necessary — slightly slower, slightly diluted entropy — but improves the
chance of containing the secret, which is the failure mode that actually costs games.

## Coverage and the exhaustive fallback

Bringing our own dictionary has one real consequence: **the secret may not be in it.**
Benchmarking 60 seeds found 10 such puzzles. `/random?seed=38` is `agnew`, a surname that
does not appear in `web2` under any capitalisation, so no dictionary-based solver using this
word list can ever guess it. The benchmark reports these separately rather than hiding them,
and the solver exits with the explicit status `no_candidates`.

`--exhaustive-fallback` (off by default) solves them anyway, by exploiting two verified
properties of the API together:

1. guesses are not checked against a dictionary, so a guess can be used as a measuring
   instrument rather than an attempt at the answer; and
2. under this API's naive scoring, `absent` means the letter occurs **nowhere** in the
   secret — a much stronger fact than in real Wordle.

When the candidate list empties, the solver switches to reading the secret off directly.
Each probe guess carries a different letter in every slot, and each slot independently
reports whether that letter is in the secret, so five letters are tested per guess and the
whole alphabet is settled in about six. Once no letter is unclassified, the secret's exact
letter set is known, and the strings it could be are enumerated and filtered with the same
consistency check the main solver uses — no dictionary involved:

```
$ python -m wordle_solver.cli --exhaustive-fallback random --seed 38
  4  a p n e a   🟩⬛🟩🟩🟨   candidates left: 0     <- dictionary exhausted
  5  o s u y c   ⬛⬛⬛⬛⬛   candidates left: 0     <- probes from here on
  6  h m b g k   ⬛⬛⬛🟨⬛   candidates left: 0
  7  w f v z j   🟨⬛⬛⬛⬛   candidates left: 0
  8  q b b b b   ⬛⬛⬛⬛⬛   candidates left: 0
  9  a g n e w   🟩🟩🟩🟩🟩   candidates left: 1

solved 'agnew' in 9 guess(es)
```

It is off by default and needs a larger budget (`--max-rounds`, defaulted to 30 when the
flag is set) because those guesses buy information rather than chances to win, which is not
how Wordle is meant to be played. It also only applies to `naive` scoring —
`ExhaustiveRecovery` refuses to run under the classic rule, where `absent` can merely mean
the copies are used up.

## Benchmark results

`scripts/benchmark.py` runs a range of `/random` seeds and reports the guess distribution.
Measured on the bundled word list, against the live API:

```bash
python scripts/benchmark.py --n 60 --size 5
python scripts/benchmark.py --n 60 --size 5 --exhaustive-fallback --max-rounds 30
```

| Run | Solved | Mean | Median | Max | Secret not in word list |
| --- | --- | --- | --- | --- | --- |
| 5 letters, 60 seeds, 6-guess budget | 50/60 (83%) | 3.84 | 4 | 6 | 10 |
| 5 letters, 60 seeds, `--exhaustive-fallback` | 60/60 (100%) | 4.63 | 4 | 10 | 0 |
| 6 letters, 25 seeds, 6-guess budget | 22/25 (88%) | 3.59 | 3.5 | 5 | 3 |

Read them together: **every puzzle whose secret was in the word list was solved**, in 3.84
guesses on average. The 83% headline is entirely dictionary coverage, not search quality,
and the fallback run confirms it. Means are taken over solved puzzles; the fallback's
higher mean is the price of the probing guesses.

## Testing

```bash
pytest            # 57 offline tests, no network
pytest -m live    # 17 integration tests against the real API
```

Live tests are deselected by default (`pytest.ini`) so that a red build means the logic
broke, not that the network did.

- `tests/test_feedback.py` — both scoring rules, pattern encoding, and response parsing.
  The duplicate-letter cases recorded from the live API are frozen here as assertions.
- `tests/test_solver.py` — full solves against an offline oracle, plus invariants: the
  candidate list only shrinks, the true secret is never filtered out, no guess repeats, and
  runs are deterministic. Includes the scoring-mismatch regression described above.
- `tests/test_recovery.py` — the probing and enumeration phases, and the guard that refuses
  classic scoring.
- `tests/test_live_api.py` — known answers via `/word/{word}`, five seeded `/random` words,
  non-default word lengths, `/daily`, and the API contract checks.

The offline tests run against a deterministic 1,500-word slice of the real list: they check
correctness and termination, not guess quality. Performance numbers come from the benchmark.

## Design decisions and limitations

- **Standard library only, plus `requests`.** The entropy loop is plain Python. NumPy would
  make it several times faster, but the bottleneck is already handled by caching the opener
  and bounding the guess pool, and the readable version is the one worth defending.
- **The opener is an approximate argmax**, measured over a 300-word shortlist rather than
  all 8,506 candidates. Widening it with `--opener-shortlist` costs seconds, not minutes.
- **Coverage, not search, is the limiting factor.** See above.
- **The six-guess budget is a convention, not a rule the API enforces.**
- **No persistence.** Each solve is a self-contained process; the only thing written to disk
  is the cached opening guess, which is a pure function of the word list and scoring rule.

## Project layout

```
wordle_solver/
  feedback.py   scoring rules, base-3 pattern encoding, response validation
  wordlist.py   word source resolution and length filtering
  api.py        HTTP client, scoring calibration, and the oracle abstraction
  solver.py     constraint filtering and expected-information guess selection
  recovery.py   opt-in dictionary-free fallback
  display.py    terminal rendering
  cli.py        argument parsing and wiring
scripts/
  build_wordlist.py   regenerates data/words_en.txt
  benchmark.py        runs many seeds and reports the guess distribution
```

The solver never touches HTTP directly: it is handed an `Oracle`, which turns a guess into a
pattern. `LocalOracle` implements the same interface against a known secret, which is what
makes the offline test suite possible.

## Credits

- The guess-selection strategy is inspired by the information-theory approach to Wordle
  popularised by 3Blue1Brown — *"Solving Wordle using information theory"*,
  <https://www.youtube.com/watch?v=v68zYyaEmEA>. No code was taken from that project; the
  entropy computation here is written from the idea.
- The bundled word list is derived from `/usr/share/dict/words` (the BSD/macOS `web2` list),
  which comes from Webster's Second International Dictionary and is in the public domain.
  NLTK's `words` corpus is supported as an optional fallback source.
- Claude Code (Anthropic) was used as an AI pair-programming assistant while building this,
  for planning, review, and drafting. The approach, the API investigation, and the final
  code were directed and reviewed by me.
