"""Draw the sampling frame: the sessions read by the chain, fixed before any reading exists.

Sessions are drawn per instrument, stratified on the regime and spread across every year present.
Stratifying on the regime is not selection on the outcome: the regime is computed from the
twenty-one preceding sessions and lagged one, so it is known before the session opens. It only
changes the marginal frequency of the sample, which is stated when the sample is described.

The draw is deterministic given the seed, and the resulting session list is fingerprinted so the
perimeter can be shown to predate the readings.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from textagents import regime as regime_module
from textagents.config import PROJECT_ROOT

#: Written down so the draw is reproducible and cannot be re-rolled until it flatters.
SEED = 20260808


def draw(panel: pd.DataFrame, per_ticker: int, balance: float = 0.5) -> pd.DataFrame:
    """`per_ticker` sessions per instrument, half stressed, spread across every year present.

    Years are covered explicitly rather than left to a uniform draw, which would concentrate the
    stressed cell in the two or three periods where stressed sessions cluster and turn the regime
    effect into a period effect.
    """
    rng = np.random.default_rng(SEED)
    chosen = []
    for ticker, group in panel.groupby("ticker"):
        marked = regime_module.classify(group).dropna(subset=["stressed"])
        marked["year"] = marked["session_date"].dt.year
        years = sorted(marked["year"].unique())
        for regime, share in ((1.0, balance), (0.0, 1.0 - balance)):
            want = int(round(per_ticker * share))
            pool = marked[marked["stressed"] == regime]
            per_year = max(1, want // max(1, len(years)))
            picked = []
            for year in years:
                candidates = pool[pool["year"] == year]
                if candidates.empty:
                    continue
                take = min(per_year, len(candidates))
                picked.append(candidates.sample(take, random_state=int(rng.integers(1e9))))
            taken = pd.concat(picked) if picked else pool.head(0)
            # Top up from whatever remains, so a year that could not supply its share does not
            # silently shrink the sample.
            if len(taken) < want:
                rest = pool.drop(taken.index)
                if len(rest):
                    taken = pd.concat([taken, rest.sample(min(want - len(taken), len(rest)),
                                                          random_state=int(rng.integers(1e9)))])
            chosen.append(taken)
    out = pd.concat(chosen).sort_values(["ticker", "session_date"]).reset_index(drop=True)
    return out[["ticker", "session_date", "trailing_vol", "stressed"]]


def fingerprint(sample: pd.DataFrame) -> str:
    """The perimeter's identity, so it can be shown to predate the readings."""
    rows = [f"{t}|{d.date()}" for t, d in zip(sample["ticker"], sample["session_date"])]
    return hashlib.sha256("\n".join(sorted(rows)).encode()).hexdigest()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-ticker", type=int, default=189)
    parser.add_argument("--balance", type=float, default=0.5)
    parser.add_argument("--out", type=str, default="results/perimeter.csv")
    args = parser.parse_args()

    panel = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "panel.parquet")
    panel["session_date"] = pd.to_datetime(panel["session_date"])
    sample = draw(panel, args.per_ticker, args.balance)

    print(f"units drawn            : {len(sample):,}")
    print(f"  of which stressed    : {int(sample['stressed'].sum()):,}")
    print(f"  of which calm        : {int((sample['stressed'] == 0).sum()):,}")
    print(f"perimeter fingerprint  : {fingerprint(sample)[:32]}")
    print(f"seed                   : {SEED}\n")
    table = sample.assign(year=sample["session_date"].dt.year).groupby(
        ["ticker", "year"])["stressed"].agg(["size", "sum"]).rename(
        columns={"size": "sessions", "sum": "stressed"})
    print(table.unstack("year").fillna(0).astype(int).to_string())

    target = PROJECT_ROOT / args.out
    target.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(target, index=False)
    (target.parent / "perimeter.json").write_text(json.dumps({
        "seed": SEED, "per_ticker": args.per_ticker, "balance": args.balance,
        "lookback": regime_module.LOOKBACK,
        "stress_quantile": regime_module.STRESS_QUANTILE,
        "regime_estimator": "garman_klass",
        "n": len(sample), "sha256": fingerprint(sample),
    }, indent=2), encoding="utf-8")
    print(f"\nwritten to {target}")


if __name__ == "__main__":
    main()
