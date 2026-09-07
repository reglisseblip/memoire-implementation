"""Run the reading chain over a set of instruments and sessions and write what it produced.

This is the entry point for steps 3 and 4 of the chain: nine calls per unit, then the fixed
aggregation formula. Every call is keyed by the bytes that produced it, so an interrupted run
restarts from its own cache and pays only for what it had not reached.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import time
from pathlib import Path

import pandas as pd

from textagents import corpus as corpus_module
from textagents.chain.routing import Routing
from textagents.chain.runner import Campaign, combine_replicates, dropped_units
from textagents.chain.schemas import AnalystReport
from textagents.config import PROJECT_ROOT, PROMPTS_DIR, Config
from textagents.llm.client import Client
from textagents.llm.governor import Limits

INSTRUMENTS = ("TTE.PA", "ATO.PA", "AI.PA")


def load_env(path: Path | str | None = None) -> dict[str, str]:
    """Read `KEY = value` lines from a .env file into the environment, and return what it set.

    A variable already present in the environment wins over the file.
    """
    target = Path(path) if path else PROJECT_ROOT / ".env"
    applied: dict[str, str] = {}
    if not target.exists():
        return applied
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and not os.environ.get(key):
            os.environ[key] = value
            applied[key] = value
    return applied


def sessions_for(panel: pd.DataFrame, ticker: str, start: dt.date | None,
                 end: dt.date | None) -> list[dt.date]:
    """Sessions available for one instrument, optionally bounded by start and end."""
    subset = panel[panel["ticker"] == ticker]
    days = sorted(subset["session_date"].unique())
    if start:
        days = [d for d in days if d >= start]
    if end:
        days = [d for d in days if d <= end]
    return days


def stride_sample(days: list[dt.date], limit: int | None) -> list[dt.date]:
    """A fixed stride across the whole window, never the first N sessions.

    The first sessions of the window are quiet, so a pilot drawn from them would price the
    campaign on its cheapest units.
    """
    if not limit or limit >= len(days):
        return days
    step = len(days) / limit
    return [days[min(len(days) - 1, int(i * step))] for i in range(limit)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="sessions per instrument, sampled on a stride over the whole window")
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--tickers", type=str, default=",".join(INSTRUMENTS))
    parser.add_argument("--no-macro", action="store_true",
                        help="issuer and regulatory streams only")
    parser.add_argument("--budget", type=float, default=None, help="spend cap for this run, USD")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--replicates", type=int, default=None)
    parser.add_argument("--tag", type=str, default="run", help="names the output files")
    parser.add_argument("--perimeter", type=str, default=None,
                        help="CSV of (ticker, session_date) fixed in advance by analysis.sample. "
                             "Overrides --start, --end and --limit.")
    parser.add_argument("--dry-run", action="store_true",
                        help="build the units, price them against the cache, emit nothing")
    args = parser.parse_args()

    load_env()
    config = Config()
    if args.budget is not None:
        config.budget_usd = args.budget
    if args.workers is not None:
        config.workers = args.workers
    if args.replicates is not None:
        config.replicates = args.replicates

    streams = (corpus_module.ISSUER_STREAMS if args.no_macro
               else corpus_module.DEFAULT_STREAMS)
    corpus = corpus_module.load_corpus(config.data_dir, streams)

    panel = pd.read_parquet(config.data_dir / "panel.parquet")
    panel["session_date"] = pd.to_datetime(panel["session_date"]).dt.date
    start = dt.date.fromisoformat(args.start) if args.start else None
    end = dt.date.fromisoformat(args.end) if args.end else None
    tickers = tuple(t.strip() for t in args.tickers.split(",") if t.strip())
    if args.perimeter:
        # The perimeter is read back verbatim and its fingerprint recomputed, so a run can be
        # shown to have covered the sample it declared.
        from textagents.analysis.sample import fingerprint
        drawn = pd.read_csv(PROJECT_ROOT / args.perimeter)
        drawn["session_date"] = pd.to_datetime(drawn["session_date"])
        print(f"perimeter  : {args.perimeter}, {len(drawn):,} units, "
              f"fingerprint {fingerprint(drawn)[:32]}")
        tickers = tuple(sorted(drawn["ticker"].unique()))
        sessions = {t: sorted(g["session_date"].dt.date.unique())
                    for t, g in drawn.groupby("ticker")}
    else:
        sessions = {t: stride_sample(sessions_for(panel, t, start, end), args.limit)
                    for t in tickers}

    Client.configure_limits(Limits(tokens_per_minute=config.tokens_per_minute,
                                   requests_per_minute=config.requests_per_minute))
    client = Client(
        model=config.model, cache_dir=config.cache_dir, ledger_path=config.ledger_path,
        mandates_dir=PROMPTS_DIR, prior_ledgers=config.prior_ledgers,
        budget_usd=config.budget_usd, temperature=config.temperature,
        max_output_tokens=config.max_output_tokens,
        min_interval_s=float(os.environ.get("LLM_MIN_INTERVAL_S", 0.05)),
    )

    campaign = Campaign(config=config, client=client,
                        routing=Routing(max_debate_rounds=config.max_debate_rounds,
                                        max_panel_rounds=config.max_panel_rounds))

    print(f"corpus     : {len(corpus):,} documents, streams {streams}")
    print(f"config     : {config.describe()}")
    print(f"paid so far: {client.total_cost():.4f} USD on this project")

    work = [unit for ticker in tickers for unit in campaign.units(corpus, ticker,
                                                                 sessions[ticker],
                                                                 corpus_module.DAILY_ORIGIN)]
    with_document = [u for u in work if u[5] > 0]
    print(f"units      : {len(work):,}, of which {len(with_document):,} with a document "
          f"({len(with_document) / max(1, len(work)):.1%})")

    # Priced against the cache before anything is emitted, so a resumed run is not stopped by its
    # own budget guard for money it does not need.
    to_pay = sum(1 for u in with_document
                 if not client.in_cache(u[3], "analyst_fundamentals", u[2], AnalystReport))
    print(f"to pay     : {to_pay:,} units outside the cache, "
          f"{to_pay * config.calls_per_unit():,} calls")
    digests = [len(u[3]) for u in with_document]
    if digests:
        median = sorted(digests)[len(digests) // 2]
        print(f"digest     : median {median:,} chars, max {max(digests):,} chars")

    if args.dry_run:
        return

    started = time.time()

    def heartbeat(done: int, total: int) -> None:
        if done % 50 and done != total:
            return
        rate = done / max(time.time() - started, 1)
        left = (total - done) / rate / 60 if rate else 0
        print(f"  {done:>6,}/{total:,} units | {rate * 60:5.1f} u/min | "
              f"{left:5.1f} min left | {client.total_cost():7.3f} USD", flush=True)

    rows = campaign.run(corpus, tickers, sessions, corpus_module.DAILY_ORIGIN,
                        work=work, progress=heartbeat)
    elapsed = time.time() - started

    results = config.results_dir
    results.mkdir(parents=True, exist_ok=True)
    if rows.empty:
        print("\nNO unit produced. Nothing is written.")
        return

    rows.to_parquet(results / f"raw_{args.tag}.parquet", index=False)
    combined = combine_replicates(rows, config.replicates)
    combined.to_parquet(results / f"signals_{args.tag}.parquet", index=False)
    dropped = dropped_units(rows, config.replicates)
    if not dropped.empty:
        dropped.to_csv(results / f"dropped_{args.tag}.csv", index=False)

    calls = int(rows["calls"].sum())
    spent = float(rows["cost_usd"].sum())
    print(f"\ncampaign finished in {elapsed / 60:.1f} min")
    print(f"raw rows           : {len(rows):,}")
    print(f"combined units     : {len(combined):,}  (K={config.replicates} required)")
    print(f"rejected units     : {len(dropped):,}")
    print(f"billed calls       : {calls:,}")
    print(f"cost of this run   : {spent:.4f} USD")
    print(f"cost per unit      : {spent / max(1, len(combined)):.5f} USD")
    print(f"throughput         : {calls / max(elapsed, 1) * 60:.0f} calls/min")
    print(f"project total      : {client.total_cost():.4f} USD")
    print(f"written to         : results/signals_{args.tag}.parquet")


if __name__ == "__main__":
    main()
