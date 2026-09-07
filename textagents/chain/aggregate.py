"""Step 4 of the chain: the fixed aggregation rule.

Turns the committee readings into the text variables used by the econometric
models. Every formula here is arithmetic and deterministic: no language model
is involved at this stage.
"""

from __future__ import annotations

import math
import statistics
from typing import Any

from textagents.chain.schemas import (
    AmplitudeView,
    AnalystReport,
    DebatePosition,
    ShockKind,
    SingleReading,
)

#: Shock kinds whose cause lies outside the issuer.
EXOGENOUS_KINDS = frozenset({
    ShockKind.MACRO_POLICY, ShockKind.GEOPOLITICAL, ShockKind.SECTOR_WIDE,
})

#: Full width of the amplitude scale, used to normalise dispersion into [0, 1].
AMPLITUDE_SCALE_WIDTH = 3.0 - (-1.6)


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    """Relevance-weighted mean, falling back to a plain mean when every weight is zero."""
    total = sum(weights)
    if total <= 0.0:
        return statistics.fmean(values) if values else 0.0
    return sum(v * w for v, w in zip(values, weights)) / total


def aggregate_analysts(reports: list[AnalystReport]) -> dict[str, float]:
    """Level and spread across the analyst tier.

    Tone is weighted by relevance and intensity, ambiguity by relevance only.
    """
    if not reports:
        return {"tone": 0.0, "ambiguity": 0.0, "analyst_spread": 0.0,
                "exogeneity": 0.0, "novelty": 0.0, "n_analysts": 0.0}

    relevance = [r.relevance for r in reports]
    signed = [r.polarity * r.intensity for r in reports]
    polarities = [r.polarity for r in reports]

    return {
        "tone": _weighted_mean(signed, relevance),
        "ambiguity": _weighted_mean([r.ambiguity for r in reports], relevance),
        # Population standard deviation: the readers are the whole committee, not a sample.
        "analyst_spread": statistics.pstdev(polarities) if len(polarities) > 1 else 0.0,
        "exogeneity": sum(1 for r in reports if r.shock_kind in EXOGENOUS_KINDS) / len(reports),
        "novelty": sum(1 for r in reports if r.is_new_information) / len(reports),
        "n_analysts": float(len(reports)),
    }


def aggregate_debate(supportive: DebatePosition | None,
                     adverse: DebatePosition | None) -> dict[str, float]:
    """Distance and sum of the two opposed positions on the same text.

    The gap measures how much of the output comes from the mandate; the sum
    departs from zero only when the evidence overrides a mandate.
    """
    if supportive is None or adverse is None:
        return {"mandate_gap": 0.0, "signed_cancellation": 0.0,
                "conviction_gap": 0.0, "debate_complete": 0.0}

    return {
        "mandate_gap": abs(supportive.polarity - adverse.polarity),
        "signed_cancellation": supportive.polarity + adverse.polarity,
        "conviction_gap": supportive.conviction - adverse.conviction,
        "debate_complete": 1.0,
    }


def aggregate_panel(views: list[AmplitudeView]) -> dict[str, float]:
    """Expected magnitude and the spread around it.

    The magnitude is the median of the reviewer seats, since one seat is
    mandated to defend an extreme case. Dispersion is normalised by the
    amplitude scale, and measured confidence is its complement.
    """
    if not views:
        return {"amplitude": 0.0, "panel_dispersion": 0.0, "measured_confidence": 0.0,
                "interval_width": 0.0, "n_seats": 0.0}

    levels = [v.log_amplitude for v in views]
    spread = statistics.pstdev(levels) if len(levels) > 1 else 0.0
    normalised = min(1.0, spread / (AMPLITUDE_SCALE_WIDTH / 2.0))

    return {
        "amplitude": statistics.median(levels),
        "panel_dispersion": spread,
        "measured_confidence": 1.0 - normalised,
        "interval_width": statistics.fmean([v.upper - v.lower for v in views]),
        "n_seats": float(len(views)),
    }


def aggregate_unit(reports: list[AnalystReport],
                   supportive: DebatePosition | None,
                   adverse: DebatePosition | None,
                   forecast: AmplitudeView | None,
                   panel: list[AmplitudeView]) -> dict[str, Any]:
    """Everything one unit contributes, as one flat mapping of columns.

    `amplitude_shift` records how far the reviewer seats moved the
    forecaster's own estimate.
    """
    out: dict[str, Any] = {}
    out.update(aggregate_analysts(reports))
    out.update(aggregate_debate(supportive, adverse))
    out.update(aggregate_panel(panel))

    if forecast is not None:
        out["forecast_amplitude"] = forecast.log_amplitude
        out["amplitude_shift"] = out["amplitude"] - forecast.log_amplitude if panel else 0.0
    else:
        out["forecast_amplitude"] = 0.0
        out["amplitude_shift"] = 0.0

    # Readable multiple, alongside the logarithm the model works in.
    out["amplitude_multiple"] = math.exp(out["amplitude"])
    return out


def aggregate_single(reading: "SingleReading | None") -> dict[str, Any]:
    """The single-call control, mapped onto the same columns as the committee.

    The arithmetic is the same: tone is polarity times intensity, and the
    amplitude is the reader's own log estimate. Spread columns are zero,
    since one call cannot disagree with itself.
    """
    if reading is None:
        return empty_unit()

    out: dict[str, Any] = {
        "tone": reading.polarity * reading.intensity,
        "ambiguity": reading.ambiguity,
        "analyst_spread": 0.0,
        "exogeneity": 1.0 if reading.shock_kind in EXOGENOUS_KINDS else 0.0,
        "novelty": 1.0 if reading.is_new_information else 0.0,
        "n_analysts": 1.0,
        "mandate_gap": 0.0, "signed_cancellation": 0.0,
        "conviction_gap": 0.0, "debate_complete": 0.0,
        "amplitude": reading.log_amplitude,
        "panel_dispersion": 0.0,
        "measured_confidence": 1.0,
        "interval_width": 0.0,
        "n_seats": 0.0,
        "forecast_amplitude": reading.log_amplitude,
        "amplitude_shift": 0.0,
    }
    out["amplitude_multiple"] = math.exp(out["amplitude"])
    return out


def empty_unit() -> dict[str, Any]:
    """The reading of a session with no eligible document: every field is zero."""
    return aggregate_unit([], None, None, None, [])
