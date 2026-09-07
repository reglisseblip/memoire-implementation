"""Output schema imposed on the readings, step 3 of the chain.

One pydantic model per role. Each field description carries the instruction the model needs,
because strict structured-output mode drops the numeric bounds from the wire schema; pydantic
re-imposes them on receipt.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# Placeholder strings a model writes instead of leaving a field empty. Coerced to None.
_NULLISH = {"", "none", "n/a", "na", "null", "nil", "-", "unknown", "inconnu", "non applicable"}


def _nullish_to_none(value):
    if isinstance(value, str) and value.strip().lower() in _NULLISH:
        return None
    return value


def _clip(limit: int):
    """Cut an over-long quotation to its cap instead of refusing the whole reading.

    Applies to string fields only, which serve as audit anchors: a prefix of a literal
    quotation is still a literal quotation. No numeric field is ever repaired.
    """
    def clip(value):
        if isinstance(value, str) and len(value) > limit:
            return value[:limit]
        return value
    return clip


class Stance(str, Enum):
    """Direction a reader attributes to a document. Descriptive only, never entered in a model."""

    SUPPORTIVE = "supportive"
    NEUTRAL = "neutral"
    ADVERSE = "adverse"


class ShockKind(str, Enum):
    """Where the document locates the cause of the shock, if any."""

    MACRO_POLICY = "macro_policy"
    GEOPOLITICAL = "geopolitical"
    SECTOR_WIDE = "sector_wide"
    ISSUER_GOVERNANCE = "issuer_governance"
    ISSUER_RESULTS = "issuer_results"
    ISSUER_CAPITAL = "issuer_capital"
    NONE = "none"


class AnalystReport(BaseModel):
    """What one analyst extracts from the documents of a single (ticker, session).

    Every analyst returns this same shape whatever its mandate, so their outputs are directly
    comparable and their disagreement is measurable.
    """

    relevance: float = Field(
        ge=0.0, le=1.0,
        description=(
            "How much this document bears on the risk of THIS issuer, from 0.0 to 1.0. "
            "0.0 means the text is unrelated to the issuer's risk and the other fields should "
            "be left at their neutral values. Judge relevance to risk, not topical similarity."
        ),
    )
    polarity: float = Field(
        ge=-1.0, le=1.0,
        description=(
            "Signed reading of the document under your mandate, from -1.0 to +1.0. "
            "-1.0 clearly adverse for the issuer, 0.0 neutral or balanced, +1.0 clearly "
            "supportive. State what the text implies, not what you would like it to imply: a "
            "mandate tells you where to look, never what to conclude."
        ),
    )
    intensity: float = Field(
        ge=0.0, le=1.0,
        description=(
            "Magnitude of the implication regardless of its sign, from 0.0 to 1.0. "
            "A routine announcement is near 0.1; an event that changes the issuer's situation "
            "is above 0.7. Independent of polarity: a strongly good and a strongly bad piece of "
            "news both score high."
        ),
    )
    ambiguity: float = Field(
        ge=0.0, le=1.0,
        description=(
            "How far this text admits opposed readings, from 0.0 to 1.0. "
            "0.0 means any careful reader would draw the same conclusion; 1.0 means the text "
            "genuinely supports contradictory interpretations. Judge the text, not your own "
            "hesitation."
        ),
    )
    forward_looking: float = Field(
        ge=0.0, le=1.0,
        description=(
            "Share of the content that concerns the future rather than the past, from 0.0 to "
            "1.0. A results release is near 0.2; guidance or a plan is above 0.7."
        ),
    )
    is_new_information: bool = Field(
        description=(
            "True if this text carries information not already public, false if it restates an "
            "agency dispatch or an earlier release. The corpus is largely agency copy, so an "
            "honest false here is as useful as a true."
        ),
    )
    shock_kind: ShockKind = Field(
        description=(
            "Where the document locates the cause. Use the issuer_* values when the text is "
            "about what the firm does or decides, and macro_policy / geopolitical / sector_wide "
            "only when the text itself speaks of a cause outside the firm. Use none when the "
            "text describes no shock at all."
        ),
    )
    stance: Stance = Field(
        description=(
            "Discrete summary of polarity: supportive, neutral or adverse. Must agree in sign "
            "with the polarity field."
        ),
    )
    evidence: str = Field(
        max_length=240,
        description=(
            "A LITERAL quotation from the source, copied word for word, at most 240 characters, "
            "that carries your reading. No paraphrase, no translation, no ellipsis of your own. "
            "If nothing in the text supports a reading, return an empty string."
        ),
    )

    @field_validator("evidence", mode="before")
    @classmethod
    def _blank_if_nullish(cls, v):
        return "" if _nullish_to_none(v) is None else _clip(240)(v)


class DebatePosition(BaseModel):
    """One side of the contradictory reading, argued from the analyst reports only.

    Each side returns a bounded position rather than free prose, and the adjudication that
    follows is arithmetic, so the same inputs always give the same verdict.
    """

    polarity: float = Field(
        ge=-1.0, le=1.0,
        description=(
            "Your position on the next session's direction, from -1.0 to +1.0, after reading "
            "the analyst reports. Argue your mandate honestly: if the evidence does not support "
            "your side, say so with a number that crosses zero. A mandated reader that still "
            "concludes against its mandate is the strongest evidence this system can produce."
        ),
    )
    conviction: float = Field(
        ge=0.0, le=1.0,
        description=(
            "How much of the analyst evidence supports your position, from 0.0 to 1.0. This is "
            "a property of the EVIDENCE, not of you: 0.0 means nothing in the reports supports "
            "your side, 1.0 means the reports point your way unambiguously."
        ),
    )
    strongest_point: str = Field(
        max_length=300,
        description="The single strongest element of your case, one sentence, at most 300 characters.",
    )
    concession: str = Field(
        max_length=300,
        description=(
            "The strongest element AGAINST your position that you nonetheless accept, one "
            "sentence. An empty string is allowed only if the reports contain nothing against "
            "you, which is rare and should make you doubt your reading."
        ),
    )

    _clip_prose = field_validator("strongest_point", "concession", mode="before")(_clip(300))


class AmplitudeView(BaseModel):
    """An expected magnitude, on a log scale anchored to an observable multiple.

    Returned by the forecaster and by each amplitude reviewer, so their spread is computable
    without rescaling.
    """

    log_amplitude: float = Field(
        ge=-1.6, le=3.0,
        description=(
            "Natural logarithm of the expected volatility multiple for the next session, "
            "relative to this issuer's average volatility over the last 22 sessions. "
            "Anchors: 0.0 an ordinary session; +0.7 twice ordinary; +1.6 five times; "
            "+2.3 ten times; +3.0 twenty times; -0.7 half of ordinary. "
            "Default to 0.0 and depart from it only when the documents justify it. "
            "A 20x session is rare but it happens: do not compress the top of the scale."
        ),
    )
    lower: float = Field(
        ge=-1.6, le=3.0,
        description="Lower bound of your plausible range on the same log scale. Must be <= log_amplitude.",
    )
    upper: float = Field(
        ge=-1.6, le=3.0,
        description="Upper bound of your plausible range on the same log scale. Must be >= log_amplitude.",
    )
    driver: str = Field(
        max_length=300,
        description="One sentence naming what in the documents drives this magnitude.",
    )

    _clip_driver = field_validator("driver", mode="before")(_clip(300))

    @field_validator("upper")
    @classmethod
    def _ordered(cls, upper, info):
        low = info.data.get("lower")
        mid = info.data.get("log_amplitude")
        if low is not None and mid is not None and not (low <= mid <= upper):
            raise ValueError(
                f"bounds out of order: lower={low}, log_amplitude={mid}, upper={upper}. "
                "An interval that does not contain its own point estimate is not an interval."
            )
        return upper


class SingleReading(BaseModel):
    """The control arm: one reader, one call, on the same digest as the committee.

    It returns exactly the quantities the committee's fixed rule emits, with the same names,
    bounds and anchors, so the two arms have the same dimension. A lone reader returns a level
    and no spread.
    """

    relevance: float = Field(
        ge=0.0, le=1.0,
        description=(
            "How much these documents bear on the risk of THIS issuer, from 0.0 to 1.0. "
            "0.0 means the texts are unrelated to the issuer's risk and the other fields should "
            "be left at their neutral values. Judge relevance to risk, not topical similarity."
        ),
    )
    polarity: float = Field(
        ge=-1.0, le=1.0,
        description=(
            "Signed reading of the documents, from -1.0 to +1.0. -1.0 clearly adverse for the "
            "issuer, 0.0 neutral or balanced, +1.0 clearly supportive. State what the text "
            "implies, not what you would like it to imply."
        ),
    )
    intensity: float = Field(
        ge=0.0, le=1.0,
        description=(
            "Magnitude of the implication regardless of its sign, from 0.0 to 1.0. "
            "A routine announcement is near 0.1; an event that changes the issuer's situation "
            "is above 0.7. Independent of polarity: strongly good and strongly bad news both "
            "score high."
        ),
    )
    ambiguity: float = Field(
        ge=0.0, le=1.0,
        description=(
            "How far these texts admit opposed readings, from 0.0 to 1.0. "
            "0.0 means any careful reader would draw the same conclusion; 1.0 means the text "
            "genuinely supports contradictory interpretations. Judge the text, not your own "
            "hesitation."
        ),
    )
    log_amplitude: float = Field(
        ge=-1.6, le=3.0,
        description=(
            "Natural logarithm of the expected volatility multiple for the next session, "
            "relative to this issuer's average volatility over the last 22 sessions. "
            "Anchors: 0.0 an ordinary session; +0.7 twice ordinary; +1.6 five times; "
            "+2.3 ten times; +3.0 twenty times; -0.7 half of ordinary. "
            "Default to 0.0 and depart from it only when the documents justify it. "
            "A 20x session is rare but it happens: do not compress the top of the scale."
        ),
    )
    shock_kind: ShockKind = Field(
        description=(
            "Where the documents locate the cause. Use the issuer_* values when the text is "
            "about what the firm does or decides, and macro_policy / geopolitical / sector_wide "
            "only when the text itself speaks of a cause outside the firm. Use none when the "
            "text describes no shock at all."
        ),
    )
    is_new_information: bool = Field(
        description=(
            "True if these texts carry information not already public, false if they restate an "
            "agency dispatch or an earlier release."
        ),
    )
    evidence: str = Field(
        max_length=240,
        description=(
            "A LITERAL quotation from the source, copied word for word, at most 240 characters, "
            "that carries your reading. No paraphrase, no translation. If nothing in the text "
            "supports a reading, return an empty string."
        ),
    )

    @field_validator("evidence", mode="before")
    @classmethod
    def _blank_if_nullish(cls, v):
        return "" if _nullish_to_none(v) is None else _clip(240)(v)


class PanelRole(str, Enum):
    """The three seats of the amplitude review, each mandated to defend one region of the scale."""

    EXTREME = "extreme"
    CENTRAL = "central"
    CONSERVATIVE = "conservative"


class Verdict(BaseModel):
    """What the language-model adjudicator produces, on the descriptive branch only.

    The tested path aggregates by a fixed rule and never by the model. This schema exists so
    both adjudications can run on the same state and their gap be published.
    """

    direction: Stance = Field(description="Direction the debate supports, once weighed.")
    log_amplitude: float = Field(
        ge=-1.6, le=3.0,
        description="Magnitude retained after the panel, on the anchored log scale.",
    )
    position: Literal[-1, 0, 1] = Field(
        description="Position implied by this verdict: -1 short, 0 flat, +1 long.",
    )
    rationale: str = Field(
        max_length=600,
        description="Why the debate resolves this way, in two or three sentences.",
    )

    _clip_rationale = field_validator("rationale", mode="before")(_clip(600))
