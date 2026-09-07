"""In-sample contrast of three nested models, the reference being the HAR of Corsi (2009).

    N0   y = a + b1 rv_d + b2 rv_w + b3 rv_m       price-only reference
    N1   N0 + the reading block                    does the text add anything
    N2   N1 + (reading x stressed regime)          does it add more under stress

N2 carries the hypothesis: the added value of the reading is tested as an interaction with the
regime, not as a level effect. Estimation is OLS with instrument fixed effects and Newey-West
standard errors, since realised volatility is persistent.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import statsmodels.api as sm

from textagents import regime as regime_module
from textagents.config import PROJECT_ROOT

RESULTS = PROJECT_ROOT / "results"

LOOKBACK = 21
STRESS_QUANTILE = 2 / 3

#: The price-only reference. `ln_vix` is left out on purpose: it is itself a forecast, and adding
#: it would turn the baseline into a forecast combination.
HAR = ["ln_rv_d", "ln_rv_w", "ln_rv_m"]

#: The three columns produced by the reading chain.
TEXT = ["tone", "ambiguity", "amplitude"]


def load_signals(tags: list[str]) -> pd.DataFrame:
    frames = [pd.read_parquet(RESULTS / f"signals_{t}.parquet") for t in tags
              if (RESULTS / f"signals_{t}.parquet").exists()]
    if not frames:
        raise FileNotFoundError(f"no signals for {tags}")
    out = pd.concat(frames, ignore_index=True)
    out["session_date"] = pd.to_datetime(out["session"])
    return out


def build(signals: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Join the readings to the price panel and label the regime of each session."""
    panel = panel.copy()
    panel["session_date"] = pd.to_datetime(panel["session_date"])
    panel = regime_module.classify_panel(panel)

    # The suffix goes on the panel: `tone` exists on both sides, and the first suffix applies to
    # the left frame, so suffixing the signals would leave TEXT estimating the panel's column.
    merged = signals.merge(panel, on=["ticker", "session_date"], suffixes=("", "_panel"))
    merged = merged.dropna(subset=["y", *HAR, "trailing_vol", "stressed", *TEXT])
    merged["stressed"] = merged["stressed"].astype(float)
    return merged


def fit(frame: pd.DataFrame, columns: list[str], lags: int = 5):
    """OLS with instrument fixed effects and Newey-West standard errors."""
    design = frame[columns].copy()
    for ticker in sorted(frame["ticker"].unique())[1:]:
        design[f"fe_{ticker}"] = (frame["ticker"] == ticker).astype(float)
    design = sm.add_constant(design, has_constant="add")
    return sm.OLS(frame["y"].values, design.values).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags})


def nested_test(frame: pd.DataFrame, base: list[str], full: list[str], label: str) -> dict:
    """Wald test that the added block is jointly zero, under HAC covariance."""
    small, big = fit(frame, base), fit(frame, full)
    added = [c for c in full if c not in base]
    k = len(added)
    # The added columns sit first in the design, so their indices are 1..k after the constant.
    restriction = np.zeros((k, len(big.params)))
    for i in range(k):
        restriction[i, full.index(added[i]) + 1] = 1.0
    wald = big.wald_test(restriction, scalar=True)
    return {
        "contrast": label,
        "columns added": k,
        "R2 base": round(float(small.rsquared), 4),
        "R2 full": round(float(big.rsquared), 4),
        "R2 gain": round(float(big.rsquared - small.rsquared), 4),
        # Under cov_type='HAC' statsmodels returns the quadratic form and reads its p-value off a
        # chi2(k), so the statistic is a chi-square and not an F.
        "Wald chi2": round(float(wald.statistic), 3),
        "p (HAC)": round(float(wald.pvalue), 4),
        "n": int(len(frame)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signals", nargs="+", required=True)
    args = parser.parse_args()

    signals = load_signals(args.signals)
    panel = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "panel.parquet")
    frame = build(signals, panel)

    print("=" * 82)
    print("DOES THE TEXT ADD ANYTHING TO THE HAR, AND MORE SO UNDER STRESS?")
    print("=" * 82)
    print("target     : y = log realised volatility of the session")
    print(f"reference  : HAR of Corsi (2009), {' + '.join(HAR)}")
    print(f"text       : {' + '.join(TEXT)}")
    print(f"regime     : volatility of the {LOOKBACK} preceding sessions, lagged one session,")
    print("             top third per instrument")
    print("std errors : Newey-West, 5 lags, instrument fixed effects\n")
    print(frame.groupby(["ticker", "stressed"]).size().unstack(fill_value=0)
          .rename(columns={0.0: "calm", 1.0: "stressed"}).to_string())

    interactions = []
    for column in TEXT:
        name = f"{column}_x_stressed"
        frame[name] = frame[column] * frame["stressed"]
        interactions.append(name)

    tests = [
        nested_test(frame, HAR, HAR + TEXT, "N1 - N0 : does the text add anything?"),
        nested_test(frame, HAR + ["stressed"], HAR + ["stressed"] + TEXT,
                    "the same, controlling for the regime"),
        nested_test(frame, HAR + ["stressed"] + TEXT,
                    HAR + ["stressed"] + TEXT + interactions,
                    "N2 - N1 : DOES IT ADD MORE UNDER STRESS?"),
    ]
    print("\n" + pd.DataFrame(tests).to_string(index=False))

    print("\n-- the interaction coefficients, one by one --")
    model = fit(frame, HAR + ["stressed"] + TEXT + interactions)
    names = ["const"] + HAR + ["stressed"] + TEXT + interactions
    rows = []
    for i, name in enumerate(names):
        if name in TEXT or name in interactions:
            rows.append({"term": name, "coef": round(float(model.params[i]), 4),
                         "t (HAC)": round(float(model.tvalues[i]), 2),
                         "p": round(float(model.pvalues[i]), 4)})
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n-- the same contrast, estimated separately within each regime --")
    for regime, sub in frame.groupby("stressed"):
        label = "stressed" if regime else "calm"
        if len(sub) < 40:
            print(f"  {label}: {len(sub)} observations, too few to estimate")
            continue
        result = nested_test(sub, HAR, HAR + TEXT, f"text, {label} regime")
        print(f"  {label:8s} n={result['n']:4d} | R2 gain {result['R2 gain']:+.4f} | "
              f"chi2={result['Wald chi2']:6.3f} | p={result['p (HAC)']:.4f}")


if __name__ == "__main__":
    main()
