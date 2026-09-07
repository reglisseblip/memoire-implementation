"""Nodes of the reading chain (step 3 of the protocol).

Each node reads the state, makes one call, and writes its own slot. A node never
fetches a price, never chooses the next node, and never writes into another
node's slot. Each tier receives a compact rendering of typed objects rather than
accumulated prose: the analysts see the digest only, the debaters see the three
analyst reports, the forecaster sees the reports and both debate positions, and
the reviewer tier sees all of the above plus the forecaster's estimate.
"""

from __future__ import annotations

from typing import Any

from textagents.chain.schemas import AmplitudeView, AnalystReport, DebatePosition
from textagents.chain.state import AnalystSlot, ReadingState
from textagents.llm.client import CallResult, Client

#: Analyst tier as the first campaign ran it. Kept because its readings are in
#: the cache and in the published results.
ANALYST_MANDATES_V1 = ("analyst_fundamentals", "analyst_exogeneity", "analyst_tone")

#: Default from 2026-08-08 on. The only change is the exogeneity seat, whose
#: original wording returned a relevance of exactly zero on 91.5 % of readings
#: over 53 sessions holding nothing but a market briefing, against 7.6 % for the
#: revised wording.
ANALYST_MANDATES = ("analyst_fundamentals", "analyst_exogeneity_v2", "analyst_tone")


def _slot(result: CallResult, mandate: str) -> AnalystSlot:
    return AnalystSlot(
        mandate=mandate, report=result.parsed, parse_ok=result.parse_ok,
        cache_hit=result.cache_hit, tokens_in=result.tokens_in, tokens_out=result.tokens_out,
        latency_ms=result.latency_ms, cost_usd=result.cost_usd,
    )


def _account(state: ReadingState, result: CallResult, tier: str) -> None:
    if not result.cache_hit:
        state["calls"] += 1
        state["cost_usd"] += result.cost_usd
    if not result.parse_ok:
        state["failures"].append(tier)


def render_reports(state: ReadingState) -> str:
    """Render the analyst tier for the tiers that read it, one compact line per report."""
    lines = ["ANALYST REPORTS"]
    for mandate, slot in state["analysts"].items():
        report: AnalystReport | None = slot.get("report")
        if report is None:
            lines.append(f"- {mandate}: reading unavailable")
            continue
        lines.append(
            f"- {mandate}: relevance {report.relevance:.2f}, polarity {report.polarity:+.2f}, "
            f"intensity {report.intensity:.2f}, ambiguity {report.ambiguity:.2f}, "
            f"cause {report.shock_kind.value}, "
            f"new information {'yes' if report.is_new_information else 'no'}"
        )
        if report.evidence:
            lines.append(f'  quotation: "{report.evidence}"')
    return "\n".join(lines)


def render_debate(state: ReadingState) -> str:
    """Render both debate positions for the tiers below."""
    lines = ["CONTRADICTORY DEBATE"]
    for side, key in (("favourable", "supportive"), ("unfavourable", "adverse")):
        position: DebatePosition | None = state["debate"].get(key)
        if position is None:
            lines.append(f"- {side}: position unavailable")
            continue
        lines.append(
            f"- {side}: polarity {position.polarity:+.2f}, conviction {position.conviction:.2f}"
        )
        lines.append(f"  strongest point: {position.strongest_point}")
        lines.append(f"  concession: {position.concession or 'none'}")
    return "\n".join(lines)


def render_forecast(state: ReadingState) -> str:
    """Render the forecaster's amplitude estimate for the reviewer tier."""
    view: AmplitudeView | None = state.get("forecast")
    if view is None:
        return "AMPLITUDE FORECAST: unavailable"
    return (
        f"AMPLITUDE FORECAST\n"
        f"- log amplitude {view.log_amplitude:+.2f} "
        f"(range {view.lower:+.2f} to {view.upper:+.2f})\n"
        f"- driver: {view.driver}"
    )


# Tier one


def run_analyst(state: ReadingState, client: Client, mandate: str) -> ReadingState:
    """One analyst on the digest. All three receive identical bytes."""
    result = client.call(state["digest"], mandate, AnalystReport,
                         replicate=state["replicate"], ticker=state["ticker"],
                         session=state["session"])
    state["analysts"][mandate] = _slot(result, mandate)
    _account(state, result, mandate)
    return state


# Tier two


def run_debater(state: ReadingState, client: Client, side: str) -> ReadingState:
    """One side of the contradictory reading, over the analyst reports."""
    mandate = f"debate_{side}"
    context = render_reports(state)
    result = client.call(context, mandate, DebatePosition, replicate=state["replicate"],
                         ticker=state["ticker"], session=state["session"])
    state["debate"][side] = result.parsed
    state["debate"]["rounds"] += 1
    state["debate"]["last_speaker"] = side
    _account(state, result, mandate)
    return state


# Tier three


def run_forecaster(state: ReadingState, client: Client) -> ReadingState:
    """The forecaster settles on one amplitude estimate from the reports and the debate."""
    context = f"{render_reports(state)}\n\n{render_debate(state)}"
    result = client.call(context, "forecaster", AmplitudeView, replicate=state["replicate"],
                         ticker=state["ticker"], session=state["session"])
    state["forecast"] = result.parsed
    _account(state, result, "forecaster")
    return state


# Tier four


def run_panel_seat(state: ReadingState, client: Client, seat: str) -> ReadingState:
    """One amplitude reviewer, who revises or confirms the forecaster's estimate."""
    mandate = f"panel_{seat}"
    context = (f"{render_reports(state)}\n\n{render_debate(state)}\n\n"
               f"{render_forecast(state)}")
    result = client.call(context, mandate, AmplitudeView, replicate=state["replicate"],
                         ticker=state["ticker"], session=state["session"])
    state["panel"][seat] = result.parsed
    state["panel"]["rounds"] += 1
    state["panel"]["last_speaker"] = seat
    _account(state, result, mandate)
    return state


# Tier five


def run_aggregator(state: ReadingState) -> ReadingState:
    """Apply the fixed aggregation rule (step 4). No call, no model, no discretion."""
    from textagents.chain.aggregate import aggregate_unit

    reports = [slot["report"] for slot in state["analysts"].values()
               if slot.get("report") is not None]
    panel = [view for view in (state["panel"].get("extreme"), state["panel"].get("central"),
                               state["panel"].get("conservative")) if view is not None]
    state["aggregate"] = aggregate_unit(
        reports, state["debate"].get("supportive"), state["debate"].get("adverse"),
        state.get("forecast"), panel,
    )
    return state


def summarise(state: ReadingState) -> dict[str, Any]:
    """One flat row per unit, ready for the results table."""
    row: dict[str, Any] = {
        "ticker": state["ticker"], "session": state["session"],
        "replicate": state["replicate"], "has_document": not state["is_empty"],
        "n_documents": state["n_documents"], "digest_sha256": state["digest_sha256"][:16],
        "calls": state["calls"], "cost_usd": state["cost_usd"],
        "n_failures": len(state["failures"]),
    }
    row.update(state["aggregate"])
    return row
