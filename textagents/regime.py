"""The calm / stressed split, defined once so every call site uses the same labels.

The measure is Garman-Klass, built from the open, high, low and close of the same session. It is
therefore invariant to a share consolidation the price series does not adjust for, and it is the
same estimator as the forecast target, so an interaction between regime and target is not an
artefact of two different measurements. The trailing average is taken over the preceding sessions
and lagged one further session, so a session is never classified from its own volatility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: One trading month, the horizon the amplitude scale is anchored to.
LOOKBACK = 21

#: Above this quantile of trailing volatility a session is stressed. Two thirds is a compromise:
#: a higher cut leaves too few stressed sessions, a lower one dilutes the contrast.
STRESS_QUANTILE = 2 / 3

# Label values as written in the result files. `stressed` below carries the same split as a
# float, and that is the column the estimation scripts read; these strings are for the eye.
CALM, STRESSED, UNKNOWN = "calm", "stressed", "unknown"


def trailing_volatility(prices: pd.DataFrame, lookback: int = LOOKBACK) -> pd.Series:
    """Annualised volatility of the `lookback` sessions before each one, from Garman-Klass.

    Raises when `rv_gk` is absent rather than falling back to close-to-close, which on this
    sample misreads an unadjusted share consolidation as extreme volatility.
    """
    if "rv_gk" not in prices:
        raise KeyError(
            "rv_gk absent from the price frame. The close-to-close fallback is not applied "
            "silently: on this sample it reports 20,957 % annualised volatility for Atos across "
            "twenty-one sessions, from a share consolidation the price series does not adjust."
        )
    daily = np.sqrt(prices["rv_gk"].astype(float))
    return (daily.rolling(lookback).mean() * np.sqrt(252)).shift(1)


def classify(prices: pd.DataFrame, quantile: float = STRESS_QUANTILE,
             lookback: int = LOOKBACK) -> pd.DataFrame:
    """Add `trailing_vol` and `regime` to a single instrument's price frame, sorted by session."""
    out = prices.sort_values("session_date").copy()
    out["trailing_vol"] = trailing_volatility(out, lookback)
    threshold = out["trailing_vol"].quantile(quantile)
    out["regime"] = np.where(out["trailing_vol"] >= threshold, STRESSED, CALM)
    out.loc[out["trailing_vol"].isna(), "regime"] = UNKNOWN
    out["stressed"] = np.where(out["regime"] == STRESSED, 1.0,
                               np.where(out["regime"] == CALM, 0.0, np.nan))
    return out


def classify_panel(panel: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Classify each instrument on its own, so a quiet one is not labelled calm throughout
    because another is more violent."""
    panel = panel.copy()
    panel["session_date"] = pd.to_datetime(panel["session_date"])
    return pd.concat([classify(group, **kwargs) for _, group in panel.groupby("ticker")],
                     ignore_index=True)
