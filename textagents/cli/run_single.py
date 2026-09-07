"""The control arm: one call to the same model, on the same digest, at the same temperature.

The digest handed to this reader is built by the same function, from the same corpus, under the
same alignment rule as the nine-call chain, with the same replicate count and the same median
across replicates. The only difference is how many times the model is asked.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from textagents import corpus as corpus_module
from textagents.chain.aggregate import aggregate_single, empty_unit
from textagents.chain.runner import combine_replicates, dropped_units
from textagents.chain.schemas import SingleReading
from textagents.cli.run_campaign import (INSTRUMENTS, load_env, sessions_for,
                                         stride_sample)
from textagents.config import PROJECT_ROOT, PROMPTS_DIR, Config
from textagents.llm.client import BudgetExceeded, Client, ProviderError, sha256
from textagents.llm.governor import Limits

MANDATE = "single_reader"


def run_one(client: Client, ticker: str, session: dt.date, replicate: int,
            digest: str, fingerprint: str, n_documents: int) -> dict:
    """One (ticker, session, replicate) through a single call."""
    row = {"ticker": ticker, "session": session.isoformat(), "replicate": replicate,
           "has_document": n_documents > 0, "n_documents": n_documents,
           "digest_sha256": fingerprint[:16], "calls": 0, "cost_usd": 0.0, "n_failures": 0}
    if n_documents == 0:
        row.update(empty_unit())
        return row

    result = client.call(digest, MANDATE, SingleReading, replicate=replicate,
                         ticker=ticker, session=session.isoformat())
    if not result.cache_hit:
        row["calls"] = 1
        row["cost_usd"] = result.cost_usd
    if not result.parse_ok:
        row["n_failures"] = 1
    row.update(aggregate_single(result.parsed))
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--tickers", type=str, default=",".join(INSTRUMENTS))
    parser.add_argument("--no-macro", action="store_true")
    parser.add_argument("--budget", type=float, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--replicates", type=int, default=None)
    parser.add_argument("--tag", type=str, default="unique")
    parser.add_argument("--perimeter", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
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

    Client.configure_limits(Limits(tokens_per_minute=config.tokens_per_minute,
                                   requests_per_minute=config.requests_per_minute))
    client = Client(
        model=config.model, cache_dir=config.cache_dir, ledger_path=config.ledger_path,
        mandates_dir=PROMPTS_DIR, prior_ledgers=config.prior_ledgers,
        budget_usd=config.budget_usd, temperature=config.temperature,
        max_output_tokens=config.max_output_tokens,
        min_interval_s=float(os.environ.get("LLM_MIN_INTERVAL_S", 0.05)),
    )

    # Digests are built once per (ticker, session) and shared across replicates, exactly as the
    # nine-call chain does, so both arms read the same bytes.
    perimeter = None
    if args.perimeter:
        from textagents.analysis.sample import fingerprint
        drawn = pd.read_csv(PROJECT_ROOT / args.perimeter)
        drawn["session_date"] = pd.to_datetime(drawn["session_date"])
        print(f"perimeter  : {len(drawn):,} units, fingerprint {fingerprint(drawn)[:32]}")
        perimeter = {t: sorted(g["session_date"].dt.date.unique())
                     for t, g in drawn.groupby("ticker")}
        tickers = tuple(sorted(perimeter))

    work = []
    for ticker in tickers:
        days = (perimeter[ticker] if perimeter
                else stride_sample(sessions_for(panel, ticker, start, end), args.limit))
        for session in days:
            documents = corpus_module.deduplicate(
                corpus_module.eligible_documents(corpus, ticker, session,
                                                 corpus_module.DAILY_ORIGIN)
            )
            digest = corpus_module.build_digest(documents, ticker, session)
            fingerprint = sha256(digest)
            for replicate in range(config.replicates):
                work.append((ticker, session, replicate, digest, fingerprint, len(documents)))

    with_document = [u for u in work if u[5] > 0]
    to_pay = sum(1 for u in with_document
                 if not client.in_cache(u[3], MANDATE, u[2], SingleReading))
    print(f"corpus     : {len(corpus):,} documents")
    print(f"units      : {len(work):,}, of which {len(with_document):,} with a document")
    print(f"to pay     : {to_pay:,} calls outside the cache")
    print(f"paid so far: {client.total_cost():.4f} USD on this project")
    if args.dry_run:
        return

    rows, lock, stop = [], threading.Lock(), threading.Event()

    def task(unit):
        if stop.is_set():
            return None
        try:
            return run_one(client, *unit)
        except (BudgetExceeded, ProviderError):
            stop.set()
            return None

    started = time.time()
    with ThreadPoolExecutor(max_workers=config.workers) as pool:
        futures = [pool.submit(task, unit) for unit in work]
        for future in as_completed(futures):
            row = future.result()
            if row is not None:
                with lock:
                    rows.append(row)
    elapsed = time.time() - started

    frame = pd.DataFrame(rows)
    results = config.results_dir
    results.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        print("\nNO unit produced. Nothing is written.")
        return

    frame.to_parquet(results / f"raw_{args.tag}.parquet", index=False)
    combined = combine_replicates(frame, config.replicates)
    combined.to_parquet(results / f"signals_{args.tag}.parquet", index=False)
    dropped = dropped_units(frame, config.replicates)
    if not dropped.empty:
        dropped.to_csv(results / f"dropped_{args.tag}.csv", index=False)

    spent = float(frame["cost_usd"].sum())
    calls = int(frame["calls"].sum())
    print(f"\nsingle reader finished in {elapsed / 60:.1f} min")
    print(f"raw rows         : {len(frame):,}")
    print(f"combined units   : {len(combined):,}")
    print(f"rejected units   : {len(dropped):,}")
    print(f"billed calls     : {calls:,}")
    print(f"cost of this run : {spent:.4f} USD")
    print(f"project total    : {client.total_cost():.4f} USD")
    print(f"written to       : results/signals_{args.tag}.parquet")


if __name__ == "__main__":
    main()
