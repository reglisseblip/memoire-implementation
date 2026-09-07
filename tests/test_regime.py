"""The calm / stressed split, which must be knowable before the session it labels.

If the label leaked the session's own volatility, the interaction the third model tests would be
an artefact of the measurement rather than a property of the market.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from textagents.regime import (
    CALM,
    LOOKBACK,
    STRESSED,
    UNKNOWN,
    classify,
    classify_panel,
    trailing_volatility,
)


def prices(rv: list[float], ticker: str = "TTE.PA") -> pd.DataFrame:
    return pd.DataFrame({
        "ticker": ticker,
        "session_date": pd.date_range("2021-01-04", periods=len(rv), freq="B"),
        "rv_gk": rv,
    })


def test_the_label_never_sees_the_session_it_labels():
    """One violent session in an otherwise quiet window must not label itself stressed."""
    quiet = [0.0001] * (LOOKBACK + 5)
    frame = prices(quiet + [1.0] + quiet[:5])
    out = classify(frame)
    spike = out.index[LOOKBACK + 5]
    assert out.loc[spike, "regime"] != STRESSED
    # The session after it can be, since by then the spike is in the trailing window.
    assert out.loc[spike + 1, "trailing_vol"] > out.loc[spike, "trailing_vol"]


def test_the_first_sessions_have_no_history_and_are_left_unlabelled():
    """They are `unknown`, not calm: an unfilled window is not evidence of a quiet market."""
    out = classify(prices([0.0002] * (LOOKBACK + 10)))
    assert (out.head(LOOKBACK)["regime"] == UNKNOWN).all()
    assert out["stressed"].head(LOOKBACK).isna().all()


def test_every_labelled_session_is_calm_or_stressed_and_nothing_else():
    out = classify(prices(list(np.linspace(0.0001, 0.01, LOOKBACK + 40))))
    labelled = out[out["regime"] != UNKNOWN]
    assert set(labelled["regime"]) <= {CALM, STRESSED}
    assert set(labelled["stressed"].unique()) <= {0.0, 1.0}


def test_the_stressed_third_is_the_top_third_of_the_instruments_own_history():
    out = classify(prices(list(np.linspace(0.0001, 0.05, LOOKBACK + 90))))
    labelled = out[out["regime"] != UNKNOWN]
    share = (labelled["regime"] == STRESSED).mean()
    assert 0.25 < share < 0.42


def test_a_quiet_instrument_is_not_labelled_calm_throughout_because_another_is_violent():
    """Each instrument is split on its own history, so a defensive stock still has its own
    stressed sessions and a distressed one still has its own quiet ones."""
    defensive = prices(list(np.linspace(0.00005, 0.0002, LOOKBACK + 60)), ticker="AI.PA")
    distressed = prices(list(np.linspace(0.01, 0.2, LOOKBACK + 60)), ticker="ATO.PA")
    out = classify_panel(pd.concat([defensive, distressed], ignore_index=True))
    labelled = out[out["regime"] != UNKNOWN]

    for ticker in ("AI.PA", "ATO.PA"):
        seen = set(labelled[labelled["ticker"] == ticker]["regime"])
        assert seen == {CALM, STRESSED}, f"{ticker} was labelled {seen}"

    # And the split is relative, not absolute: the defensive stock's stressed sessions are
    # quieter than the distressed one's calm sessions.
    quiet_stressed = labelled[(labelled["ticker"] == "AI.PA")
                              & (labelled["regime"] == STRESSED)]["trailing_vol"].max()
    loud_calm = labelled[(labelled["ticker"] == "ATO.PA")
                         & (labelled["regime"] == CALM)]["trailing_vol"].min()
    assert quiet_stressed < loud_calm


def test_the_close_to_close_fallback_is_refused_rather_than_applied_silently():
    """On this sample it reads an unadjusted share consolidation as extreme volatility."""
    frame = prices([0.0002] * 30).drop(columns=["rv_gk"])
    with pytest.raises(KeyError, match="rv_gk"):
        trailing_volatility(frame)
