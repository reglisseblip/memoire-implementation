"""The look-ahead rule, which is the one mistake that produces spectacular and worthless results.

Attaching a document to a session it could not have informed does not raise, is invisible in
every downstream number, and inflates every result. So the boundary is tested from both sides,
in winter and in summer, and the digest budget is tested for the truncation it is allowed to do.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from textagents import corpus as corpus_module
from textagents.corpus import (
    DAILY_ORIGIN,
    EXCERPT_MARKER,
    MIN_DOCUMENT_CHARS,
    build_digest,
    deduplicate,
    eligible_documents,
    is_eligible,
    load_corpus,
    origin_utc,
)


def document(doc_id: str, ticker: str, published: str, body: str, title: str = "t",
             source: str = "s") -> dict:
    return {"doc_id": doc_id, "ticker": ticker, "published_utc": pd.Timestamp(published),
            "title": title, "body": body, "source": source}


def frame(rows: list[dict]) -> pd.DataFrame:
    out = pd.DataFrame(rows)
    out["published_utc"] = pd.to_datetime(out["published_utc"], utc=True)
    return out


# The forecast origin


def test_origin_is_paris_local_time_not_utc_clock_time():
    """17:30 in Paris is 16:30 UTC in winter and 15:30 UTC in summer."""
    assert origin_utc(dt.date(2025, 1, 15), DAILY_ORIGIN).hour == 16
    assert origin_utc(dt.date(2025, 7, 15), DAILY_ORIGIN).hour == 15


def test_a_timestamp_equal_to_the_origin_belongs_to_the_later_window():
    """Strictly before, never at. Equality is the only boundary a reader could be handed twice."""
    origin = origin_utc(dt.date(2025, 1, 15), DAILY_ORIGIN)
    assert is_eligible(origin - pd.Timedelta(seconds=1), origin)
    assert not is_eligible(origin, origin)
    assert not is_eligible(origin + pd.Timedelta(seconds=1), origin)


# The eligibility window


def test_a_document_published_after_the_close_is_held_for_the_next_session():
    session = dt.date(2025, 1, 15)
    corpus = frame([
        document("before", "TTE.PA", "2025-01-15T16:29:00Z", "b" * 300),
        document("after", "TTE.PA", "2025-01-15T16:31:00Z", "a" * 300),
    ])
    kept = eligible_documents(corpus, "TTE.PA", session, DAILY_ORIGIN)
    assert [d.doc_id for d in kept] == ["before"]


def test_a_document_older_than_the_lookback_is_not_news_for_this_session():
    """Without this bound a silent session inherits the previous week and looks informed."""
    session = dt.date(2025, 1, 15)
    corpus = frame([
        document("yesterday", "TTE.PA", "2025-01-14T17:00:00Z", "y" * 300),
        document("last_week", "TTE.PA", "2025-01-08T17:00:00Z", "w" * 300),
    ])
    kept = eligible_documents(corpus, "TTE.PA", session, DAILY_ORIGIN)
    assert [d.doc_id for d in kept] == ["yesterday"]


def test_a_document_shorter_than_the_floor_carries_only_a_headline():
    session = dt.date(2025, 1, 15)
    corpus = frame([
        document("stub", "TTE.PA", "2025-01-15T09:00:00Z", "x" * (MIN_DOCUMENT_CHARS - 1)),
        document("real", "TTE.PA", "2025-01-15T09:00:00Z", "x" * MIN_DOCUMENT_CHARS),
    ])
    kept = eligible_documents(corpus, "TTE.PA", session, DAILY_ORIGIN)
    assert [d.doc_id for d in kept] == ["real"]


def test_another_issuers_document_is_never_handed_to_this_reader():
    session = dt.date(2025, 1, 15)
    corpus = frame([
        document("mine", "TTE.PA", "2025-01-15T09:00:00Z", "m" * 300),
        document("theirs", "ATO.PA", "2025-01-15T09:00:00Z", "t" * 300),
    ])
    kept = eligible_documents(corpus, "TTE.PA", session, DAILY_ORIGIN)
    assert [d.doc_id for d in kept] == ["mine"]


def test_a_naive_timestamp_is_refused_rather_than_assumed_to_be_utc(tmp_path):
    """Guessing the zone moves a document across the session boundary for half the year."""
    naive = pd.DataFrame([{
        "doc_id": "d", "ticker": "TTE.PA", "title": "t", "source": "s", "body": "b" * 300,
        "published_utc": pd.Timestamp("2025-01-15T16:00:00"),
    }])
    naive.to_parquet(tmp_path / "news_amf.parquet", index=False)
    with pytest.raises(ValueError, match="naive"):
        load_corpus(tmp_path, ("news_amf",))


def test_a_missing_corpus_raises_instead_of_returning_an_empty_frame(tmp_path):
    """An empty frame would read as a window with no news, which is a different claim."""
    with pytest.raises(FileNotFoundError):
        load_corpus(tmp_path, ("news_amf",))


# Deduplication


def test_a_wire_story_republished_by_ten_sites_counts_once():
    body = " ".join(f"word{i}" for i in range(200))
    documents = eligible_from([
        document("wire", "TTE.PA", "2025-01-15T09:00:00Z", body),
        document("copy", "TTE.PA", "2025-01-15T10:00:00Z", body),
    ])
    assert [d.doc_id for d in deduplicate(documents)] == ["wire"]


def test_deduplication_only_looks_backwards():
    """Nothing later removes something earlier, so the kept document is always the first."""
    body = " ".join(f"word{i}" for i in range(200))
    documents = eligible_from([
        document("first", "TTE.PA", "2025-01-15T08:00:00Z", body),
        document("second", "TTE.PA", "2025-01-15T09:00:00Z", body),
        document("third", "TTE.PA", "2025-01-15T10:00:00Z", body),
    ])
    assert [d.doc_id for d in deduplicate(documents)] == ["first"]


def test_two_genuinely_different_documents_both_survive():
    documents = eligible_from([
        document("a", "TTE.PA", "2025-01-15T09:00:00Z",
                 " ".join(f"alpha{i}" for i in range(200))),
        document("b", "TTE.PA", "2025-01-15T10:00:00Z",
                 " ".join(f"beta{i}" for i in range(200))),
    ])
    assert len(deduplicate(documents)) == 2


def eligible_from(rows: list[dict]):
    return eligible_documents(frame(rows), rows[0]["ticker"], dt.date(2025, 1, 15), DAILY_ORIGIN)


# The digest


def test_every_reader_of_a_unit_receives_the_same_bytes():
    """A gap measured between two readers has to be a difference in reading, not in coverage."""
    documents = eligible_from([document("a", "TTE.PA", "2025-01-15T09:00:00Z", "a" * 400)])
    first = build_digest(documents, "TTE.PA", dt.date(2025, 1, 15))
    second = build_digest(documents, "TTE.PA", dt.date(2025, 1, 15))
    assert first == second


def test_a_unit_with_no_document_yields_a_header_and_nothing_else():
    digest = build_digest([], "TTE.PA", dt.date(2025, 1, 15))
    assert "TTE.PA" in digest and "2025-01-15" in digest
    assert "DOCUMENT" not in digest


def test_the_digest_stays_within_its_budget_and_says_when_it_cut():
    """A document larger than the whole budget is excerpted, and the cut is stated in the text."""
    budget_chars = 100 * corpus_module.CHARS_PER_TOKEN
    documents = eligible_from([document("huge", "TTE.PA", "2025-01-15T09:00:00Z", "x" * 10_000)])
    digest = build_digest(documents, "TTE.PA", dt.date(2025, 1, 15), max_tokens=100)
    assert len(digest) <= budget_chars
    assert EXCERPT_MARKER.strip() in digest


def test_documents_are_rendered_oldest_first():
    documents = eligible_from([
        document("early", "TTE.PA", "2025-01-15T08:00:00Z",
                 " ".join(f"alpha{i}" for i in range(60))),
        document("late", "TTE.PA", "2025-01-15T10:00:00Z",
                 " ".join(f"beta{i}" for i in range(60))),
    ])
    digest = build_digest(documents, "TTE.PA", dt.date(2025, 1, 15))
    assert digest.index("alpha0") < digest.index("beta0")
