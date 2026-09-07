"""Step 3 of the chain: run one unit through the four tiers, in the order the router gives.

Routing is deterministic, so the number of calls a unit makes is known before the run starts.
A unit with no document is aggregated straight to zeros and never calls the model.
"""

from __future__ import annotations

from textagents.chain import nodes
from textagents.chain.routing import (
    AGGREGATOR,
    DEBATE_SUPPORTIVE,
    FORECASTER,
    PANEL_CENTRAL,
    PANEL_CONSERVATIVE,
    PANEL_EXTREME,
    Routing,
)
from textagents.chain.state import ReadingState
from textagents.llm.client import Client

#: Loop guard. A unit takes a couple of dozen turns at most, so reaching this ceiling is a defect.
MAX_TURNS = 200

class RunawayChain(RuntimeError):
    """Raised when the router never reached the aggregator."""


def run_unit(state: ReadingState, client: Client, routing: Routing | None = None,
             analyst_mandates: tuple[str, ...] | None = None) -> ReadingState:
    """Run one (ticker, session, replicate) through every tier.

    `analyst_mandates` is a parameter rather than a module constant so that a run can state
    which wording of the analyst seats produced it.
    """
    routing = routing or Routing()

    if state["is_empty"]:
        return nodes.run_aggregator(state)

    for mandate in (analyst_mandates or nodes.ANALYST_MANDATES):
        nodes.run_analyst(state, client, mandate)

    turns = 0
    while True:
        turns += 1
        if turns > MAX_TURNS:
            raise RunawayChain(
                f"the debate did not terminate in {MAX_TURNS} turns for "
                f"{state['ticker']} {state['session']}. The router is meant to make this "
                "impossible, so this is a defect and not a slow unit."
            )
        nxt = routing.after_debate_turn(state)
        if nxt == FORECASTER:
            break
        nodes.run_debater(state, client,
                          "supportive" if nxt == DEBATE_SUPPORTIVE else "adverse")

    nodes.run_forecaster(state, client)

    seats = {PANEL_EXTREME: "extreme", PANEL_CENTRAL: "central",
             PANEL_CONSERVATIVE: "conservative"}
    turns = 0
    while True:
        turns += 1
        if turns > MAX_TURNS:
            raise RunawayChain(
                f"the panel did not terminate in {MAX_TURNS} turns for "
                f"{state['ticker']} {state['session']}."
            )
        nxt = routing.after_panel_turn(state)
        if nxt == AGGREGATOR:
            break
        nodes.run_panel_seat(state, client, seats[nxt])

    return nodes.run_aggregator(state)
