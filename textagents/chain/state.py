"""Typed state carried by one (ticker, session) unit through the reading chain.

One named slot per producer and no message history: every reader is a pure function of the
digest it is handed and of the mandate it is given, which keeps a run reproducible and makes the
cache key the hash of those two inputs. Readers never see prices, volumes or market indicators.
"""

from __future__ import annotations

from typing import Annotated, Any

from typing_extensions import TypedDict

from textagents.chain.schemas import AmplitudeView, AnalystReport, DebatePosition


class AnalystSlot(TypedDict, total=False):
    """One analyst's output, plus the accounting needed to audit and price the call."""

    mandate: Annotated[str, "identifier of the prompt this reader was given"]
    report: Annotated[AnalystReport | None, "None when the call failed the schema"]
    parse_ok: Annotated[bool, "False if the object did not validate after the single retry"]
    cache_hit: Annotated[bool, "True if the response came from disk, so it cost nothing"]
    tokens_in: Annotated[int, "prompt tokens billed"]
    tokens_out: Annotated[int, "completion tokens billed, reasoning block included"]
    latency_ms: Annotated[int, "wall clock of the call"]
    cost_usd: Annotated[float, "billed cost of this call alone"]


class DebateState(TypedDict, total=False):
    """Step 3, tier two: the two contradictors. `rounds` is what stops the exchange."""

    supportive: Annotated[DebatePosition | None, "the reader mandated to argue for the issuer"]
    adverse: Annotated[DebatePosition | None, "the reader mandated to argue against it"]
    rounds: Annotated[int, "exchanges completed; the router compares it to 2 * max_debate_rounds"]
    last_speaker: Annotated[str, "'supportive' or 'adverse', so the router alternates"]


class PanelState(TypedDict, total=False):
    """Step 3, tier four: the three amplitude reviewers, who argue about magnitude."""

    extreme: Annotated[AmplitudeView | None, "the seat that must defend the tail case"]
    central: Annotated[AmplitudeView | None, "the seat that must defend the modal case"]
    conservative: Annotated[AmplitudeView | None, "the seat that must defend the muted case"]
    rounds: Annotated[int, "exchanges completed; compared to 3 * max_panel_rounds"]
    last_speaker: Annotated[str, "seat that spoke last, so the router cycles"]


class ReadingState(TypedDict, total=False):
    """Everything one (ticker, session) unit carries through the graph."""

    # identity of the unit, fixed before the first call
    ticker: Annotated[str, "Euronext Paris ticker, e.g. TTE.PA"]
    session: Annotated[str, "ISO date of the session being forecast"]
    replicate: Annotated[int, "which of the K readings this is; the median is taken downstream"]

    # the single input every reader sees, byte for byte
    digest: Annotated[str, "the user message itself, identical across all mandates"]
    digest_sha256: Annotated[str, "fingerprint of the digest; proves the perimeter never moved"]
    n_documents: Annotated[int, "eligible documents in the digest, 0 for a silent session"]
    is_empty: Annotated[bool, "True when no eligible document exists; no call is ever emitted"]

    # tier one
    analysts: Annotated[dict[str, AnalystSlot], "mandate identifier -> that reader's slot"]

    # tier two
    debate: Annotated[DebateState, "the contradictory reading"]

    # tier three
    forecast: Annotated[AmplitudeView | None, "the forecaster's magnitude, before the panel"]

    # tier four
    panel: Annotated[PanelState, "the three seats arguing about magnitude"]

    # tier five, two adjudications on the same state
    aggregate: Annotated[dict[str, Any], "fixed-rule aggregation; this is the tested path"]
    verdict: Annotated[Any, "language-model adjudication; DESCRIPTIVE branch only"]

    # accounting
    calls: Annotated[int, "calls emitted for this unit, cache hits excluded"]
    cost_usd: Annotated[float, "billed cost of this unit"]
    failures: Annotated[list[str], "tiers whose object did not validate after the retry"]


def new_state(ticker: str, session: str, digest: str, digest_sha256: str,
              n_documents: int, replicate: int = 0) -> ReadingState:
    """Return a unit at rest, before any call.

    Every container is created here rather than lazily, so a missing slot downstream means a node
    did not run.
    """
    return ReadingState(
        ticker=ticker,
        session=session,
        replicate=replicate,
        digest=digest,
        digest_sha256=digest_sha256,
        n_documents=n_documents,
        is_empty=n_documents == 0,
        analysts={},
        debate=DebateState(supportive=None, adverse=None, rounds=0, last_speaker=""),
        forecast=None,
        panel=PanelState(extreme=None, central=None, conservative=None, rounds=0,
                         last_speaker=""),
        aggregate={},
        verdict=None,
        calls=0,
        cost_usd=0.0,
        failures=[],
    )
