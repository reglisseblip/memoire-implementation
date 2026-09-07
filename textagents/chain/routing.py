"""Where the chain goes next, and when it stops.

The route is decided by the code, not by the model: a two-speaker exchange stops at
`2 * max_debate_rounds` turns and a three-seat panel at `3 * max_panel_rounds`, so the number of
calls a unit will emit is known before the run starts. With both budgets at 1, one unit costs
3 analysts + 2 debaters + 1 forecaster + 3 panel seats = 9 calls, plus 1 for the language-model
adjudicator on the descriptive branch.
"""

from __future__ import annotations

from textagents.chain.state import ReadingState

#: Node names. Kept as constants because they appear both in the graph wiring and in the router's
#: return values, and a typo in either place produces a graph that never reaches a node.
ANALYST_NODES = ("analyst_fundamentals", "analyst_exogeneity", "analyst_tone")
DEBATE_SUPPORTIVE = "debate_supportive"
DEBATE_ADVERSE = "debate_adverse"
FORECASTER = "forecaster"
PANEL_EXTREME = "panel_extreme"
PANEL_CENTRAL = "panel_central"
PANEL_CONSERVATIVE = "panel_conservative"
AGGREGATOR = "aggregator"


class Routing:
    """Deterministic routing, parameterised by the two round budgets."""

    def __init__(self, max_debate_rounds: int = 1, max_panel_rounds: int = 1) -> None:
        if max_debate_rounds < 1 or max_panel_rounds < 1:
            raise ValueError(
                f"round budgets must be at least 1, got debate={max_debate_rounds}, "
                f"panel={max_panel_rounds}. A budget of 0 would skip a tier entirely and the "
                "resulting state would be indistinguishable from a tier that failed."
            )
        self.max_debate_rounds = max_debate_rounds
        self.max_panel_rounds = max_panel_rounds

    # tier two

    def after_debate_turn(self, state: ReadingState) -> str:
        """Alternate between the two mandates, then hand over to the forecaster.

        Two speakers, so the budget is `2 * max_debate_rounds` turns.
        """
        debate = state["debate"]
        if debate["rounds"] >= 2 * self.max_debate_rounds:
            return FORECASTER
        return DEBATE_ADVERSE if debate["last_speaker"] == "supportive" else DEBATE_SUPPORTIVE

    # tier four

    def after_panel_turn(self, state: ReadingState) -> str:
        """Cycle the three seats, then hand over to the aggregator.

        Three seats, so the budget is `3 * max_panel_rounds` turns. The cycle order is fixed
        (extreme, conservative, central) and never depends on what was said.
        """
        panel = state["panel"]
        if panel["rounds"] >= 3 * self.max_panel_rounds:
            return AGGREGATOR
        last = panel["last_speaker"]
        if last == "extreme":
            return PANEL_CONSERVATIVE
        if last == "conservative":
            return PANEL_CENTRAL
        return PANEL_EXTREME

    # accounting

    def calls_per_unit(self, with_adjudicator: bool = False) -> int:
        """Exact number of calls a full unit will emit, used by the spend guard before the run."""
        return (
            len(ANALYST_NODES)
            + 2 * self.max_debate_rounds
            + 1
            + 3 * self.max_panel_rounds
            + (1 if with_adjudicator else 0)
        )
