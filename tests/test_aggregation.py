"""The fixed rule that turns nine readings into three numbers.

These formulas produce the variables that are then tested statistically, so a mistake here does
not crash: it publishes a number. Every rule that could plausibly have been written another way
is pinned by a test that says why it was written this way.
"""

from __future__ import annotations

import math

from textagents.chain.aggregate import (
    aggregate_analysts,
    aggregate_debate,
    aggregate_panel,
    aggregate_single,
    aggregate_unit,
    empty_unit,
)
from textagents.chain.schemas import (
    AmplitudeView,
    AnalystReport,
    DebatePosition,
    ShockKind,
    SingleReading,
    Stance,
)


def report(relevance=0.8, polarity=0.5, intensity=0.6, ambiguity=0.3, forward_looking=0.5,
           is_new_information=True, shock_kind=ShockKind.ISSUER_RESULTS,
           stance=Stance.NEUTRAL) -> AnalystReport:
    return AnalystReport(
        relevance=relevance, polarity=polarity, intensity=intensity, ambiguity=ambiguity,
        forward_looking=forward_looking, is_new_information=is_new_information,
        shock_kind=shock_kind, stance=stance, evidence="quoted from the document",
    )


def position(polarity=0.5, conviction=0.6) -> DebatePosition:
    return DebatePosition(polarity=polarity, conviction=conviction,
                          strongest_point="the strongest point", concession="the concession")


def view(level=0.7, lower=None, upper=None) -> AmplitudeView:
    """One seat's magnitude, with a plausible range that stays inside the scale."""
    return AmplitudeView(log_amplitude=level,
                         lower=max(-1.6, level - 0.3) if lower is None else lower,
                         upper=min(3.0, level + 0.3) if upper is None else upper,
                         driver="what moves it")


# The analyst tier


def test_tone_is_weighted_by_relevance():
    """A document a reader judges irrelevant must not pull the tone of the session."""
    weighted = aggregate_analysts([report(relevance=1.0, polarity=1.0, intensity=1.0),
                                   report(relevance=0.0, polarity=-1.0, intensity=1.0)])
    assert weighted["tone"] == 1.0


def test_tone_is_polarity_times_intensity_not_polarity_alone():
    """A clearly adverse but routine announcement is not the same signal as a clearly adverse
    one that changes the issuer's situation."""
    routine = aggregate_analysts([report(relevance=1.0, polarity=-1.0, intensity=0.1)])
    material = aggregate_analysts([report(relevance=1.0, polarity=-1.0, intensity=0.9)])
    assert routine["tone"] > material["tone"]
    assert math.isclose(routine["tone"], -0.1)


def test_a_reading_with_no_relevance_anywhere_is_neutral_not_missing():
    """Zero weight everywhere falls back to the plain mean rather than dividing by zero."""
    out = aggregate_analysts([report(relevance=0.0, polarity=0.4, intensity=1.0),
                              report(relevance=0.0, polarity=-0.4, intensity=1.0)])
    assert math.isclose(out["tone"], 0.0)


def test_analyst_spread_is_the_population_deviation():
    """The three seats are the whole committee, not a sample drawn from a larger one."""
    out = aggregate_analysts([report(polarity=-1.0), report(polarity=1.0)])
    assert math.isclose(out["analyst_spread"], 1.0)


def test_one_analyst_cannot_disagree_with_itself():
    assert aggregate_analysts([report()])["analyst_spread"] == 0.0


def test_exogeneity_counts_only_causes_outside_the_issuer():
    out = aggregate_analysts([report(shock_kind=ShockKind.GEOPOLITICAL),
                              report(shock_kind=ShockKind.SECTOR_WIDE),
                              report(shock_kind=ShockKind.ISSUER_GOVERNANCE),
                              report(shock_kind=ShockKind.ISSUER_CAPITAL)])
    assert math.isclose(out["exogeneity"], 0.5)


# The debate tier


def test_the_mandate_gap_measures_how_much_comes_from_the_instruction():
    out = aggregate_debate(position(polarity=0.9), position(polarity=-0.9))
    assert math.isclose(out["mandate_gap"], 1.8)
    assert math.isclose(out["signed_cancellation"], 0.0)


def test_a_speaker_that_concludes_against_its_mandate_leaves_a_trace():
    """Both sides landing on the same sign is the evidence overriding the instruction."""
    out = aggregate_debate(position(polarity=0.4), position(polarity=0.6))
    assert out["signed_cancellation"] > 0.0


def test_half_a_debate_is_no_debate():
    out = aggregate_debate(position(), None)
    assert out["debate_complete"] == 0.0
    assert out["mandate_gap"] == 0.0


# The magnitude reviewers


def test_the_amplitude_is_the_median_of_the_seats_not_their_mean():
    """One seat is instructed to defend the extreme case, so a mean would let that instruction
    pull the published estimate by construction."""
    seats = [view(0.2), view(0.3), view(3.0)]
    assert math.isclose(aggregate_panel(seats)["amplitude"], 0.3)
    assert aggregate_panel(seats)["amplitude"] < sum(v.log_amplitude for v in seats) / 3


def test_agreement_between_the_seats_reads_as_confidence():
    agreed = aggregate_panel([view(0.5), view(0.5), view(0.5)])
    split = aggregate_panel([view(-1.0), view(0.5), view(2.5)])
    assert agreed["measured_confidence"] == 1.0
    assert split["measured_confidence"] < agreed["measured_confidence"]


# The unit


def test_a_session_with_no_eligible_document_is_zero_on_every_column():
    """It is a silent session, not a missing one: the row exists and carries no signal."""
    out = empty_unit()
    assert out["tone"] == 0.0 and out["ambiguity"] == 0.0 and out["amplitude"] == 0.0
    assert math.isclose(out["amplitude_multiple"], 1.0)


def test_the_shift_records_how_far_the_reviewers_moved_the_forecaster():
    out = aggregate_unit([report()], position(), position(-0.4), view(0.5),
                         [view(1.0), view(1.0), view(1.0)])
    assert math.isclose(out["forecast_amplitude"], 0.5)
    assert math.isclose(out["amplitude_shift"], 0.5)


def test_a_forecast_no_reviewer_saw_is_not_reported_as_a_shift():
    out = aggregate_unit([report()], position(), position(-0.4), view(0.5), [])
    assert out["amplitude_shift"] == 0.0


# The control arm


def test_the_control_fills_exactly_the_same_columns_as_the_chain():
    """The two branches are compared column by column, so their shapes cannot drift apart."""
    chain = aggregate_unit([report()], position(), position(-0.4), view(0.5), [view(0.5)])
    control = aggregate_single(SingleReading(
        relevance=0.8, polarity=0.5, intensity=0.6, ambiguity=0.3, log_amplitude=0.5,
        shock_kind=ShockKind.ISSUER_RESULTS, is_new_information=True, evidence="quoted"))
    assert set(chain) == set(control)


def test_the_control_computes_tone_the_same_way_the_chain_does():
    control = aggregate_single(SingleReading(
        relevance=1.0, polarity=-1.0, intensity=0.1, ambiguity=0.3, log_amplitude=0.0,
        shock_kind=ShockKind.NONE, is_new_information=False, evidence="quoted"))
    assert math.isclose(control["tone"], -0.1)


def test_one_call_cannot_disagree_with_itself():
    control = aggregate_single(SingleReading(
        relevance=1.0, polarity=0.0, intensity=0.0, ambiguity=0.0, log_amplitude=0.0,
        shock_kind=ShockKind.NONE, is_new_information=False, evidence="quoted"))
    assert control["analyst_spread"] == 0.0
    assert control["panel_dispersion"] == 0.0
    assert control["debate_complete"] == 0.0
