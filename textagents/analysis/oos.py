"""Out-of-sample verdict on the reading block, and how few observations carry it.

The two models are nested, so a Diebold-Mariano test would be biased against the larger one:
under the null its extra coefficients are zero and estimating them only adds noise. Clark and
West (2007) correct the loss differential by the squared gap between the two forecasts, which is
exactly the term estimation noise contributes.

The concentration of the gain is read from Kish's effective sample size for unequal weights,
(sum w)^2 / sum(w^2), applied to each observation's contribution, and confirmed by dropping the
best few observations and re-running the test.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from textagents.analysis.har_contrast import HAR, build, fit, load_signals, nested_test
from textagents.config import PROJECT_ROOT

BLOCK = ["amplitude"]

#: Default block of `out_of_sample`. The in-sample contrast adds the three text columns, but the
#: expected amplitude alone carries almost all of the gain, so the curve shows that column.
OUT_OF_SAMPLE_BLOCK = ["amplitude"]


def out_of_sample(frame: pd.DataFrame, min_train: int = 80,
                  block: list[str] | None = None) -> pd.DataFrame:
    """One row per out-of-sample prediction, with both models' squared errors.

    Rows are ordered by date across instruments. Both predictions are returned, not only their
    errors, because the Clark-West statistic needs the gap between the two forecasts.
    """
    block = OUT_OF_SAMPLE_BLOCK if block is None else block
    rows = []
    for ticker, group in frame.groupby("ticker"):
        group = group.sort_values("session_date").reset_index(drop=True)
        for i in range(min_train, len(group)):
            train, test = group.iloc[:i], group.iloc[i:i + 1]
            actual = test["y"].values[0]
            prediction = {}
            for label, columns in (("har", HAR), ("text", HAR + block)):
                design = sm.add_constant(train[columns].values, has_constant="add")
                beta = np.linalg.lstsq(design, train["y"].values, rcond=None)[0]
                prediction[label] = float(np.r_[1.0, test[columns].values[0]] @ beta)
            rows.append({
                "date": test["session_date"].values[0],
                "ticker": ticker,
                "stressed": float(test["stressed"].values[0]),
                "e2_har": (actual - prediction["har"]) ** 2,
                "e2_text": (actual - prediction["text"]) ** 2,
                "p_har": prediction["har"],
                "p_text": prediction["text"],
            })
    # Stable sort with the ticker as second key: the instruments share their session dates, so
    # sorting on the date alone would leave the order of ties to the sort implementation.
    return (pd.DataFrame(rows).sort_values(["date", "ticker"], kind="mergesort")
            .reset_index(drop=True))


def clark_west(oos: pd.DataFrame) -> tuple[float, float, int]:
    """Adjusted-MSPE statistic, one-sided, against the standard normal.

    The adjustment is the squared gap between the two forecasts, so the test asks whether the
    larger model wins by more than the cost of estimating coefficients the null says are zero.
    """
    f = (oos["e2_har"] - oos["e2_text"]) + (oos["p_har"] - oos["p_text"]) ** 2
    n = len(f)
    statistic = float(f.mean() / (f.std(ddof=1) / np.sqrt(n)))
    return statistic, float(1.0 - stats.norm.cdf(statistic)), n


def effective_n(contributions) -> float:
    """Kish (1965) for unequal weights: how many observations the result rests on.

    Fed the per-observation contribution to the gain, negatives included. A result spread evenly
    over n observations returns n, one carried by a single session returns 1.
    """
    w = np.asarray(contributions, dtype=float)
    total = w.sum()
    if total == 0:
        return 0.0
    return float(total ** 2 / np.square(w).sum())


def in_sample_contributions(frame: pd.DataFrame) -> np.ndarray:
    """Squared error each unit saves in sample, which is what the R-squared gain is made of."""
    residual_har = frame["y"].values - fit(frame, HAR).fittedvalues
    residual_text = frame["y"].values - fit(frame, HAR + BLOCK).fittedvalues
    return residual_har ** 2 - residual_text ** 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signals", nargs="+", required=True)
    parser.add_argument("--min-train", type=int, default=80)
    args = parser.parse_args()

    panel = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "panel.parquet")
    frame = build(load_signals(args.signals), panel)
    oos = out_of_sample(frame, min_train=args.min_train, block=BLOCK)

    print("=" * 78)
    print("OUT OF SAMPLE: THE EXPECTED AMPLITUDE AGAINST THE HAR ALONE")
    print("=" * 78)
    print(f"block added : {' + '.join(BLOCK)}")
    print(f"window      : expanding, {args.min_train} units minimum before the first forecast")
    print("test        : Clark-West (2007), one-sided\n")

    rows = []
    for label, sub in (("all", oos),
                       ("calm regime", oos[oos["stressed"] == 0.0]),
                       ("stressed regime", oos[oos["stressed"] == 1.0])):
        statistic, pvalue, n = clark_west(sub)
        rows.append({"": label, "observations": n,
                     "statistic": round(statistic, 2), "p one-sided": round(pvalue, 4)})
    print(pd.DataFrame(rows).to_string(index=False))

    gain = oos["e2_har"] - oos["e2_text"]
    total = gain.sum()
    print("\n-- how many observations carry this gain --")
    print(f"effective sample size (Kish)         : {effective_n(gain):.1f} out of {len(gain)}")
    print(f"share carried by the best five       : {gain.nlargest(5).sum() / total:.1%}")
    print(f"forecasts worse than the HAR alone   : {(gain < 0).sum()} out of {len(gain)}")
    for k in (5, 10):
        remaining = oos.drop(gain.nlargest(k).index)
        _, pvalue, _ = clark_west(remaining)
        verdict = "still significant" if pvalue < 0.05 else "no longer significant"
        print(f"without the {k:2d} best forecasts       : p = {pvalue:.4f}, {verdict}")

    print("\n-- the same examination on the IN-sample contrast --")
    contributions = in_sample_contributions(frame)
    print(f"effective sample size (Kish)         : "
          f"{effective_n(contributions):.1f} out of {len(frame)}")
    order = pd.Series(contributions, index=frame.index)
    for k in (10,):
        remaining = frame.drop(order.nlargest(k).index)
        result = nested_test(remaining, HAR, HAR + BLOCK, f"without {k}")
        verdict = "still significant" if result["p (HAC)"] < 0.05 else "no longer significant"
        print(f"without the {k:2d} best units           : gain {result['R2 gain']:+.4f}, "
              f"p = {result['p (HAC)']:.4f}, {verdict}")


if __name__ == "__main__":
    main()
