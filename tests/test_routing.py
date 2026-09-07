"""The route is decided by the code, never by the model.

That is what makes the bill of a campaign known before it starts, and what makes a disagreement
between two readers attributable to the document or the instruction rather than to one of them
having been given more turns than the other.
"""

from __future__ import annotations

import pytest

from textagents.chain.routing import (
    AGGREGATOR,
    ANALYST_NODES,
    DEBATE_ADVERSE,
    DEBATE_SUPPORTIVE,
    FORECASTER,
    PANEL_CENTRAL,
    PANEL_CONSERVATIVE,
    PANEL_EXTREME,
    Routing,
)
from textagents.chain.state import new_state


def state(debate_rounds=0, last_debater="", panel_rounds=0, last_seat=""):
    out = new_state("TTE.PA", "2025-01-15", "digest", "f" * 64, n_documents=1)
    out["debate"]["rounds"] = debate_rounds
    out["debate"]["last_speaker"] = last_debater
    out["panel"]["rounds"] = panel_rounds
    out["panel"]["last_speaker"] = last_seat
    return out


# The bill, known in advance


def test_a_unit_costs_nine_calls_and_the_number_is_known_before_the_run():
    """Three analysts, two contradictors, one forecaster, three magnitude reviewers."""
    assert Routing().calls_per_unit() == 9
    assert len(ANALYST_NODES) == 3


def test_the_adjudicator_is_a_tenth_call_and_is_off_by_default():
    assert Routing().calls_per_unit(with_adjudicator=True) == 10


def test_raising_a_round_budget_raises_the_bill_by_a_known_amount():
    assert Routing(max_debate_rounds=2).calls_per_unit() == 11
    assert Routing(max_panel_rounds=2).calls_per_unit() == 12


def test_a_budget_of_zero_is_refused_rather_than_silently_skipping_a_tier():
    """A skipped tier and a failed tier would leave states that cannot be told apart."""
    with pytest.raises(ValueError, match="at least 1"):
        Routing(max_debate_rounds=0)
    with pytest.raises(ValueError, match="at least 1"):
        Routing(max_panel_rounds=0)


# The contradictors


def test_the_two_contradictors_alternate():
    routing = Routing()
    assert routing.after_debate_turn(state(debate_rounds=1, last_debater="supportive")) \
        == DEBATE_ADVERSE
    assert routing.after_debate_turn(state(debate_rounds=1, last_debater="adverse")) \
        == DEBATE_SUPPORTIVE


def test_the_exchange_stops_at_its_budget_and_hands_over_to_the_forecaster():
    routing = Routing()
    assert routing.after_debate_turn(state(debate_rounds=2, last_debater="adverse")) == FORECASTER


# The magnitude reviewers


def test_the_three_seats_cycle_in_a_fixed_order():
    """Extreme, then conservative, then central. The order never depends on what was said."""
    routing = Routing()
    assert routing.after_panel_turn(state(panel_rounds=1, last_seat="extreme")) \
        == PANEL_CONSERVATIVE
    assert routing.after_panel_turn(state(panel_rounds=2, last_seat="conservative")) \
        == PANEL_CENTRAL


def test_the_first_seat_to_speak_is_the_extreme_one():
    assert Routing().after_panel_turn(state(panel_rounds=0, last_seat="")) == PANEL_EXTREME


def test_the_seats_stop_at_their_budget_and_hand_over_to_the_aggregator():
    """Nothing follows the reviewers except arithmetic."""
    assert Routing().after_panel_turn(state(panel_rounds=3, last_seat="central")) == AGGREGATOR


def test_routing_never_reads_the_content_of_a_reading():
    """Two states that differ only in what the readers said route identically."""
    routing = Routing()
    quiet = state(panel_rounds=1, last_seat="extreme")
    loud = state(panel_rounds=1, last_seat="extreme")
    loud["digest"] = "a completely different document"
    assert routing.after_panel_turn(quiet) == routing.after_panel_turn(loud)


# A unit at rest


def test_a_new_unit_has_every_container_and_no_reading():
    """A missing slot downstream therefore means a node did not run, not that it ran empty."""
    unit = new_state("ATO.PA", "2024-09-17", "digest", "a" * 64, n_documents=2)
    assert unit["analysts"] == {} and unit["forecast"] is None
    assert unit["debate"]["rounds"] == 0 and unit["panel"]["rounds"] == 0
    assert unit["calls"] == 0 and unit["cost_usd"] == 0.0
    assert unit["is_empty"] is False


def test_a_session_with_no_document_is_marked_before_any_call():
    assert new_state("ATO.PA", "2024-09-17", "", "0" * 64, n_documents=0)["is_empty"] is True
