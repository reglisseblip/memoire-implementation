<div align="center">

# Multi-agent LLM reading of financial news, measured against a HAR volatility baseline

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.2.3-150458?logo=pandas&logoColor=white)
![statsmodels](https://img.shields.io/badge/statsmodels-0.14.6-4051B5)
![pydantic](https://img.shields.io/badge/pydantic-2.12.5-E92063?logo=pydantic&logoColor=white)

![Tests](https://img.shields.io/badge/Tests-50%20passed-brightgreen)

Nine language-model calls read the news of one trading session and return three numbers. A fixed
arithmetic rule, never a model, turns those numbers into variables. An HAR regression then decides
whether they add anything to what the prices already said.

This is the code behind an M2 MIAGE Informatique Decisionnelle dissertation, Universite Paris
Dauphine-PSL.

</div>

## The question

> To what extent can the analysis of textual information by large language models complement
> quantitative models in better anticipating market dynamics during periods of financial stress?

The scope is fixed narrowly on purpose. Anticipating market dynamics means forecasting the
magnitude of the next session, measured by its realised volatility, and not its direction. A
signal judged useful announces, before the session opens, that it will be agitated, without
saying whether the price will rise or fall. Directional strategies are out of scope, and so is
any use of prices by the readers: the language model never sees a price, a volume or a market
indicator.

Two hypotheses follow from it:

- **H1**, the text block improves the forecast of realised volatility over the HAR alone.
- **H2**, that contribution is larger under stress than in a calm regime.

## Prerequisites

- Python 3.11 or later
- An API key for any OpenAI-compatible inference endpoint (the campaign ran against MiniMax-M2)
- A price table and a news corpus, described under [Data](#data). Neither is in this repository.

## Installation

```bash
git clone https://github.com/reglisseblip/memoire-implementation.git
cd memoire-implementation
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                # then put your key in it
```

The test suite needs no key and no data:

```bash
python -m pytest tests/ -q
```

## Usage

The four commands below are the experiment, in order. Every call is keyed by the bytes that
produced it and cached on disk, so an interrupted run restarts from its own cache and pays only
for what it had not reached. `--dry-run` prices a campaign against that cache without emitting
anything.

```bash
# 1. Draw the sampling frame, before any reading exists. Deterministic given the seed.
python -m textagents.analysis.sample --per-ticker 189

# 2. The nine-call chain over that perimeter, with a spend cap.
python -m textagents.cli.run_campaign --perimeter results/perimeter.csv --tag main --budget 25

# 3. The control arm: one call, same digests, same model, same temperature.
python -m textagents.cli.run_single --perimeter results/perimeter.csv --tag control

# 4. The verdict.
python -m textagents.analysis.har_contrast --signals main    # in sample, N0 / N1 / N2
python -m textagents.analysis.oos --signals main             # out of sample, Clark-West
```

Each run writes three files under `results/`: `raw_<tag>.parquet` with one row per replicate,
`signals_<tag>.parquet` with the median across replicates, and `dropped_<tag>.csv` with the units
that did not have exactly K usable replicates and were refused rather than aggregated over what
survived.

## How it works

Four steps. Only the third calls a model.

**Step 1, collection.** Out of scope for this repository. A source is kept only if it timestamps
to the minute and covers the whole window.

**Step 2, from documents to one digest** (`textagents/corpus.py`). A document informs session *t*
only if it was public strictly before the forecast origin of that session, fixed at the 17:30
Paris close. The origin is built in local time and compared in UTC, so the rule survives daylight
saving, and the comparison is strict: a timestamp equal to the origin belongs to the next window.
Documents under 200 characters are dropped, near-duplicates are removed on word trigrams at 85 %
overlap, and what remains is assembled into a single digest. Every reader of a unit receives that
digest byte for byte, which is what makes a gap between two readers a difference in reading rather
than in coverage.

**Step 3, the reading** (`textagents/chain/`). Nine calls per unit, in four tiers:

| Tier | Calls | What it does |
| -- | -- | -- |
| 1 | 3 analysts | Read the same digest under three imposed angles: fundamentals, external shocks, tone |
| 2 | 2 contradictors | Argue the favourable and the unfavourable case from the three reports |
| 3 | 1 forecaster | Settle on an expected magnitude for the next session, never a direction |
| 4 | 3 magnitude reviewers | Revise or confirm that magnitude under opposed instructions |

Calls within a tier are independent: none receives another's answer, and all start from the same
text or the same reports. That independence is what makes their disagreement interpretable. The
route between tiers is decided by `textagents/chain/routing.py`, not by any model, so the bill of
a campaign is known before it starts.

**Step 4, the aggregation** (`textagents/chain/aggregate.py`). The nine objects are reduced to
three numbers by a formula written in advance: tone and ambiguity are relevance-weighted means,
the expected amplitude is the median of the three reviewers. The median, not the mean, because one
reviewer is instructed to defend the extreme case and a mean would let that instruction pull the
published estimate by construction. Putting a language model at this point would have introduced
an unlogged decision exactly where the result is formed.

**The control arm** (`textagents/cli/run_single.py`) receives the same digests, the same model and
the same temperature, and fills the same three variables in one call instead of nine. The
comparison is therefore about the organisation of the reading and nothing else.

**The verdict** (`textagents/analysis/`). Three nested models, estimated by OLS with instrument
fixed effects and Newey-West standard errors:

```
N0   log rv[t+1] = a_i + b_d log rv_d + b_w log rv_w + b_m log rv_m      prices only
N1   N0 + (tone, ambiguity, amplitude)                                   H1
N2   N1 + (text x stressed regime)                                       H2
```

A Wald test asks whether the added block is jointly zero. Out of sample, coefficients are
re-estimated as the window expands and the two forecast series are compared with the Clark-West
(2007) adjusted-MSPE statistic, which corrects the handicap a nested model carries under the null.
Kish's effective sample size then reports how many observations the result actually rests on, and
the test is replayed without the best five and the best ten forecasts.

## Repository layout

```
textagents/
├── config.py          every setting of a run, overridable as TEXTAGENTS_<FIELD>
├── corpus.py          step 2: the alignment rule, deduplication, the digest
├── regime.py          the calm / stressed split, defined once
├── chain/             step 3 and step 4
│   ├── schemas.py     the pydantic object each reader must return
│   ├── state.py       what one unit carries, one named slot per producer
│   ├── nodes.py       one node per reader
│   ├── routing.py     where the chain goes next, and when it stops
│   ├── pipeline.py    one unit through the four tiers
│   ├── runner.py      the campaign: concurrency, replicates, the median
│   └── aggregate.py   the fixed arithmetic rule
├── llm/               cache, ledger, spend cap, structured output, rate governor
├── analysis/          sample.py, har_contrast.py, oos.py
├── cli/               run_campaign.py (nine calls), run_single.py (one call)
└── prompts/           the eleven instruction files, with their SHA-256 fingerprints
tests/                 50 tests on the deterministic half: alignment, aggregation, routing, regime
```

## Configuration

Runtime secrets live in `.env` at the root of the working tree, which is git-ignored. See
`.env.example`.

| Variable | Default | What it is |
| -- | -- | -- |
| `LLM_BASE_URL` | none | Any OpenAI-compatible endpoint |
| `LLM_API_KEY` | none | Required before the first call, never at import time |
| `LLM_MIN_INTERVAL_S` | `0.05` | Floor on the delay between two calls |

Every field of `textagents.config.Config` can be overridden as `TEXTAGENTS_<FIELD>`, coerced to
the type of the committed default so a misspelt value fails at startup rather than misconfiguring
a run. The committed defaults are MiniMax-M2 at temperature 0, three replicates per unit, nine
calls per unit, five workers, and a 25 USD cap re-read from disk before every call.

## Data

Neither the price table nor the news corpus is in this repository. The corpus is copyrighted press
material, and the code regenerates every derived table from these two inputs.

`data/processed/panel.parquet`, one row per instrument and session:

| Column | Type | What it is |
| -- | -- | -- |
| `ticker` | str | Euronext Paris ticker, for example `TTE.PA` |
| `session_date` | date | The trading session |
| `y` | float | `log` realised volatility of session *t+1*, the forecast target |
| `ln_rv_d`, `ln_rv_w`, `ln_rv_m` | float | The three HAR components, in logs |
| `rv_gk` | float | Garman-Klass realised variance, which the regime split reads |

`data/processed/news_*.parquet`, one row per document, in the streams named by
`textagents/corpus.py`:

| Column | Type | What it is |
| -- | -- | -- |
| `doc_id` | str | Stable identifier |
| `ticker` | str | The instrument the document is attached to |
| `published_utc` | timestamp, tz-aware | Refused if naive: guessing the zone moves a document across the session boundary for half the year |
| `title`, `body` | str | The text handed to the readers |
| `source` | str | Named in the digest |

The dissertation reads three Paris-listed stocks between 4 January 2021 and 31 July 2025:
TotalEnergies, Atos and Air Liquide.

## What the campaign found

Reported here so the code can be read against its own results, not as a claim this repository
proves on its own.

| | Value |
| -- | -- |
| Units read | 564, three times each, 1 692 readings, all completed |
| Calls, all campaigns | 51 193, of which 99.58 % respected the imposed format |
| Total spend | 55.13 USD |
| In sample, N1 vs N0 | R2 0.7095 to 0.7245, gain +0.0150, p = 0.0002 |
| Out of sample, 324 forecasts | Clark-West +2.50, p = 0.0062 |
| Effective sample size (Kish) | 2.7 out of 324 |

H1 is supported and H2 is not: the interaction with the stressed regime adds nothing measurable,
and estimated separately, the chain's contribution shows up in the calm regime while the
single-call control's shows up under stress. The out-of-sample gain is carried by five forecasts
out of 324, which is what the effective sample size reports, and 144 of the 324 remain worse than
the HAR alone. Drop the ten best and the result no longer holds.
