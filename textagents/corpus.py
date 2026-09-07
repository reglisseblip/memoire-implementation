"""Step 2 of the chain: select the documents a unit may see and assemble them into one digest.

A document informs session t only if it was public strictly before the forecast origin of that
session. The origin is built in Paris local time and compared in UTC, so the rule survives
daylight saving; the comparison is strict, a timestamp equal to the origin belongs to the later
window.
"""

from __future__ import annotations

import datetime as dt
import zoneinfo
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PARIS = zoneinfo.ZoneInfo("Europe/Paris")

#: Close of continuous trading on Euronext Paris. The closing auction is excluded on purpose.
DAILY_ORIGIN = dt.time(17, 30, 0)

#: One second before the opening. Not 08:45, where a cluster of filings is stamped exactly.
OVERNIGHT_ORIGIN = dt.time(8, 59, 59)

#: Below this length a document carries only a headline and boilerplate.
MIN_DOCUMENT_CHARS = 200

#: Characters per token, used to hold a digest inside its budget. Crude on purpose, so the
#: perimeter does not depend on the tokeniser of any particular model.
CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class Document:
    """One collected text, with what is needed to place it in time."""

    doc_id: str
    ticker: str
    published_utc: pd.Timestamp
    title: str
    body: str
    source: str

    @property
    def n_chars(self) -> int:
        return len(self.body)


#: Issuer and regulatory streams.
ISSUER_STREAMS = ("news_boursorama", "news_amf")

#: Macro stream, kept apart so a run can be made with and without it and the difference measured.
MACRO_STREAMS = ("news_macro_bfm",)

DEFAULT_STREAMS = ISSUER_STREAMS + MACRO_STREAMS


def load_corpus(processed_dir: Path | str,
                streams: tuple[str, ...] = DEFAULT_STREAMS) -> pd.DataFrame:
    """The requested corpora, concatenated, with timestamps converted to UTC.

    A naive timestamp is refused rather than assumed to be UTC, since guessing the zone moves a
    document across the session boundary for half the year.
    """
    processed = Path(processed_dir)
    frames = []
    for name in streams:
        path = processed / f"{name}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
    if not frames:
        raise FileNotFoundError(
            f"no corpus under {processed} for streams {streams}. Nothing is fabricated in "
            "their absence."
        )

    corpus = pd.concat(frames, ignore_index=True)
    stamps = pd.to_datetime(corpus["published_utc"], utc=False)
    if getattr(stamps.dt, "tz", None) is None:
        raise ValueError(
            "published_utc carries naive timestamps. Their zone must be recorded, not guessed: "
            "Paris is UTC+1 in winter and UTC+2 in summer, so assuming a zone puts half the "
            "year on the wrong side of the session boundary."
        )
    corpus["published_utc"] = stamps.dt.tz_convert("UTC")
    return corpus


def origin_utc(session: dt.date, origin: dt.time) -> pd.Timestamp:
    """The forecast origin of a session as an instant in UTC, built from Paris local time."""
    local = dt.datetime.combine(session, origin, tzinfo=PARIS)
    return pd.Timestamp(local).tz_convert("UTC")


def is_eligible(published_utc: pd.Timestamp, origin: pd.Timestamp) -> bool:
    """Strictly before the origin. Equality belongs to the later window."""
    return bool(published_utc < origin)


def eligible_documents(corpus: pd.DataFrame, ticker: str, session: dt.date,
                       origin: dt.time, lookback_days: int = 1) -> list[Document]:
    """Documents this ticker's readers may see when forecasting `session`.

    `lookback_days` bounds how far back the window reaches; one day, so a stale document does not
    count as information on a silent session. Documents shorter than MIN_DOCUMENT_CHARS are
    dropped.
    """
    end = origin_utc(session, origin)
    start = end - pd.Timedelta(days=lookback_days)

    subset = corpus[corpus["ticker"] == ticker]
    out: list[Document] = []
    for _, row in subset.iterrows():
        stamp = row["published_utc"]
        if not (start <= stamp) or not is_eligible(stamp, end):
            continue
        body = str(row.get("body") or "")
        if len(body) < MIN_DOCUMENT_CHARS:
            continue
        out.append(Document(
            doc_id=str(row["doc_id"]), ticker=ticker, published_utc=stamp,
            title=str(row.get("title") or ""), body=body, source=str(row.get("source") or ""),
        ))
    out.sort(key=lambda d: (d.published_utc, d.doc_id))
    return out


def deduplicate(documents: list[Document], threshold: float = 0.85) -> list[Document]:
    """Drop documents that mostly restate an earlier one.

    Comparison is on the set of word trigrams, which is insensitive to a reordered lede and to a
    changed headline, and it only looks backwards, so nothing later removes something earlier.
    """
    def trigrams(text: str) -> set[tuple[str, ...]]:
        words = text.lower().split()
        return {tuple(words[i:i + 3]) for i in range(max(0, len(words) - 2))}

    kept: list[Document] = []
    seen: list[set[tuple[str, ...]]] = []
    for document in documents:
        grams = trigrams(document.body)
        if not grams:
            continue
        duplicate = False
        for earlier in seen:
            overlap = len(grams & earlier) / max(1, min(len(grams), len(earlier)))
            if overlap >= threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(document)
            seen.append(grams)
    return kept


def build_digest(documents: list[Document], ticker: str, session: dt.date,
                 max_tokens: int = 3000) -> str:
    """The single user message every reader of this unit receives, byte for byte.

    It is identical across mandates, so a gap measured between two readers is a difference in
    reading and not in coverage; the mandate is prepended by the client as a system message.
    Truncation is by whole documents, most recent first, and a document larger than the whole
    budget is excerpted with the cut stated in the text. The scaffolding is in English, like the
    mandates; document bodies are reproduced in the language they were published in.
    """
    header = f"Instrument: {ticker}\nSession: {session.isoformat()}\n"
    # The newline joining the header to the blocks counts against the budget too.
    budget = max_tokens * CHARS_PER_TOKEN - len(header) - 1

    blocks: list[str] = []
    used = 0
    for rank, document in enumerate(reversed(documents), start=1):
        preamble = (
            f"[DOCUMENT {rank}]\n"
            f"Source: {document.source}\n"
            f"Published: {document.published_utc.isoformat()}\n"
            f"Title: {document.title}\n\n"
        )
        block = f"{preamble}{document.body}\n"
        if used + len(block) > budget:
            if blocks:
                break
            block = preamble + _excerpt(document.body, budget - len(preamble))
        blocks.append(block)
        used += len(block)

    return header + "\n" + "\n".join(reversed(blocks)) if blocks else header


#: Stated in the digest itself when a document had to be cut, so the reader knows it is holding
#: an excerpt.
EXCERPT_MARKER = "\n\n[...] Document truncated: only the beginning is provided."


def _excerpt(body: str, budget: int) -> str:
    """The head of an over-long document, cut at a paragraph break where one is near the limit."""
    room = max(0, budget - len(EXCERPT_MARKER))
    head = body[:room]
    cut = head.rfind("\n\n")
    # A paragraph boundary is honoured only if it keeps most of what the budget allowed.
    if cut > room * 0.6:
        head = head[:cut]
    return head + EXCERPT_MARKER
