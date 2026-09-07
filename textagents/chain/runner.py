"""The campaign: build the units, run them concurrently, take the median over replicates.

The unit is the grain of parallelism: tiers inside a unit are sequential, units are independent.
A single governor is shared by the whole process because the provider quota belongs to the API
key rather than to a worker. A unit that does not have exactly K usable replicates is dropped
rather than aggregated over what survived.
"""

from __future__ import annotations

import datetime as dt
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from textagents import corpus as corpus_module
from textagents.chain import nodes
from textagents.chain.pipeline import run_unit
from textagents.chain.routing import Routing
from textagents.chain.state import new_state
from textagents.config import Config
from textagents.llm.client import BudgetExceeded, Client, ProviderError, sha256

#: Fields whose replicates are combined by a median. Everything else is either a count, which is
#: identical across replicates, or a label, which is not aggregated at all.
MEDIAN_FIELDS = (
    "tone", "ambiguity", "analyst_spread", "exogeneity", "novelty",
    "mandate_gap", "signed_cancellation", "conviction_gap",
    "amplitude", "panel_dispersion", "measured_confidence", "interval_width",
    "forecast_amplitude", "amplitude_shift", "amplitude_multiple",
)


@dataclass
class Campaign:
    """One run over a set of instruments and sessions."""

    config: Config
    client: Client
    routing: Routing = field(default_factory=Routing)
    stop: threading.Event = field(default_factory=threading.Event)
    rows: list[dict[str, Any]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def units(self, corpus: pd.DataFrame, ticker: str, sessions: list[dt.date],
              origin: dt.time) -> list[tuple[str, dt.date, int, str, str, int]]:
        """Every (ticker, session, replicate) to run, with its digest already built.

        The digest is built once per session and reused across replicates, so the cache key
        differs only by the replicate index.
        """
        out = []
        for session in sessions:
            documents = corpus_module.deduplicate(
                corpus_module.eligible_documents(corpus, ticker, session, origin)
            )
            digest = corpus_module.build_digest(documents, ticker, session)
            fingerprint = sha256(digest)
            for replicate in range(self.config.replicates):
                out.append((ticker, session, replicate, digest, fingerprint, len(documents)))
        return out

    def _run_one(self, unit) -> dict[str, Any] | None:
        ticker, session, replicate, digest, fingerprint, n_documents = unit
        if self.stop.is_set():
            return None
        state = new_state(ticker, session.isoformat(), digest, fingerprint,
                          n_documents, replicate)
        try:
            run_unit(state, self.client, self.routing,
                     self.config.analyst_mandates)
        except BudgetExceeded:
            # The cap is a hard stop, not a slow-down: every worker is told to finish.
            self.stop.set()
            return None
        except ProviderError:
            self.stop.set()
            return None
        return nodes.summarise(state)

    def run(self, corpus: pd.DataFrame, tickers: tuple[str, ...],
            sessions: dict[str, list[dt.date]], origin: dt.time,
            work: list | None = None, progress=None) -> pd.DataFrame:
        """Run the whole campaign concurrently, resumable through the cache.

        `work` may be supplied by a caller that has already built the units, since building them
        scans the corpus once per session. `progress` is called with the number of units finished
        and the number expected.
        """
        if work is None:
            work = [unit for ticker in tickers
                    for unit in self.units(corpus, ticker, sessions[ticker], origin)]

        done = 0
        with ThreadPoolExecutor(max_workers=self.config.workers) as pool:
            futures = [pool.submit(self._run_one, unit) for unit in work]
            for future in as_completed(futures):
                row = future.result()
                done += 1
                if progress is not None:
                    progress(done, len(work))
                if row is None:
                    continue
                with self._lock:
                    self.rows.append(row)

        return pd.DataFrame(self.rows)


def combine_replicates(rows: pd.DataFrame, k_required: int) -> pd.DataFrame:
    """Median across replicates, refusing any unit that does not have exactly K usable ones.

    A median over two values is a different estimator from a median over three, so a unit is
    either complete or dropped.
    """
    if rows.empty:
        return rows

    usable = rows[rows["n_failures"] == 0]
    counts = usable.groupby(["ticker", "session"]).size()
    complete = set(counts[counts == k_required].index)

    combined = []
    for (ticker, session), group in usable.groupby(["ticker", "session"]):
        if (ticker, session) not in complete:
            continue
        row: dict[str, Any] = {
            "ticker": ticker, "session": session,
            "has_document": bool(group["has_document"].iloc[0]),
            "n_documents": int(group["n_documents"].iloc[0]),
            "replicates_used": len(group),
            "calls": int(group["calls"].sum()),
            "cost_usd": float(group["cost_usd"].sum()),
        }
        for column in MEDIAN_FIELDS:
            if column in group:
                row[column] = float(statistics.median(group[column].astype(float)))
        combined.append(row)

    out = pd.DataFrame(combined)
    if not out.empty:
        out = out.sort_values(["ticker", "session"]).reset_index(drop=True)
    return out


def dropped_units(rows: pd.DataFrame, k_required: int) -> pd.DataFrame:
    """Units the campaign refused, so the loss is reported rather than merely absent."""
    if rows.empty:
        return rows
    grouped = rows.groupby(["ticker", "session"]).agg(
        replicates=("replicate", "count"),
        failures=("n_failures", "sum"),
        documents=("n_documents", "first"),
    ).reset_index()
    return grouped[(grouped["replicates"] != k_required) | (grouped["failures"] > 0)]
