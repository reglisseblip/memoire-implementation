"""Inference client used by step 3 of the chain: cache, ledger, spend cap, structured output.

It talks to any OpenAI-compatible endpoint through `base_url`. Four properties matter. The cache
key includes the fingerprint of the mandate text, so an edited mandate is a new key and cannot
serve answers produced under the previous wording. The spend cap is re-read from disk before
every call, so two processes writing the same ledger cannot cross it together. The model
identifier returned by the provider is recorded, not the alias that was requested, which is what
makes a silent substitution detectable afterwards. A response that does not validate is refused:
there is no free-text fallback, because the units that fail are the longest and most ambiguous
documents rather than a random sample.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from textagents.llm.governor import Limits, RateGovernor
from textagents.llm.structured import schema_contract, strict_json_schema, validate_payload

#: Ceiling that no configuration can raise. Changing it means editing this line.
ABSOLUTE_HARD_CAP_USD = 120.0

#: Tags wrapping the scratchpad a reasoning model emits before its answer. Stripping it is not
#: cosmetic: the first balanced JSON object of a raw response is sometimes a draft the model then
#: corrected.
REASONING_TAGS = ("think", "thinking", "reasoning", "reflection", "analysis", "scratchpad")

LEDGER_COLUMNS = [
    "timestamp_utc", "event", "ticker", "session", "mandate", "replicate", "attempt",
    "model_requested", "model_returned", "mandate_sha256", "context_sha256", "cache_key",
    "structure_mode", "tokens_in", "tokens_in_cached", "tokens_out", "latency_ms", "cost_usd",
    "status", "parse_ok", "error",
]

#: Ordered degradation of the wire-level request; the mode reached is written to the ledger. The
#: provider used here accepts both `json_schema` and `json_object` and honours neither, so the
#: shape is in fact imposed by `schema_contract` in the system message. The ledger therefore
#: records `<mode>|contract` rather than implying the wire channel was in force.
STRUCTURE_MODES = ("json_schema", "json_object", "tolerant")

#: Attempts before a unit is refused. Most failures are a busy provider rather than an unreadable
#: text, which is why the figure is well above two.
MAX_ATTEMPTS = 6

#: Exponential backoff with jitter, in seconds, capped. The cap matters: a sleeping worker holds
#: a quota reservation and a budget reservation the whole time.
BACKOFF_BASE_S = 2.0
BACKOFF_CAP_S = 45.0

#: Errors that mean "not now" rather than "not ever". They are kept out of the parse-failure
#: rate, which the dissertation publishes.
TRANSIENT_ERRORS = (
    "RateLimitError", "APITimeoutError", "APIConnectionError", "InternalServerError",
    "APIStatusError", "ConnectionError", "ReadTimeout", "Timeout", "ServiceUnavailable",
)

#: The one condition that justifies stepping down the structured-output cascade: the endpoint
#: says the request itself is malformed.
SCHEMA_REJECTIONS = ("BadRequestError", "UnprocessableEntityError")

#: Statuses no amount of waiting repairs (no credit, no key, no permission). They stop the
#: campaign instead of being retried on every remaining unit.
TERMINAL_STATUS = (401, 402, 403)


def _error_name(exc: BaseException) -> str:
    return type(exc).__name__


def _status_of(exc: BaseException) -> int | None:
    """The HTTP status, from the attribute if the SDK set one, else from the message.

    The message fallback is needed because the OpenAI SDK raises a generic `APIStatusError` for
    statuses it has no dedicated class for, 402 among them.
    """
    status = getattr(exc, "status_code", None)
    if status is not None:
        return int(status)
    match = re.search(r"Error code:\s*(\d{3})", str(exc))
    return int(match.group(1)) if match else None


def _is_terminal(exc: BaseException) -> bool:
    """No credit, no key, no permission. Waiting does not fix any of the three."""
    return _status_of(exc) in TERMINAL_STATUS


def _is_transient(exc: BaseException) -> bool:
    if _is_terminal(exc):
        return False
    name = _error_name(exc)
    if name in SCHEMA_REJECTIONS:
        return False
    status = _status_of(exc)
    if status is not None:
        return status == 429 or status >= 500
    return name in TRANSIENT_ERRORS


def _rejects_schema(exc: BaseException) -> bool:
    if _error_name(exc) in SCHEMA_REJECTIONS:
        return True
    return getattr(exc, "status_code", None) == 400


class BudgetExceeded(RuntimeError):
    """Raised before a call that would cross the cap. Nothing is emitted, nothing is truncated."""


class ProviderError(RuntimeError):
    """Raised when the endpoint refuses durably. The caller stops rather than degrading."""


def _now() -> str:
    """UTC timestamp for the ledger, to the second."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(payload: bytes | str, length: int | None = None) -> str:
    raw = payload.encode("utf-8") if isinstance(payload, str) else payload
    digest = hashlib.sha256(raw).hexdigest()
    return digest[:length] if length else digest


def strip_reasoning(text: str | None) -> str:
    """Remove the scratchpad a reasoning model emits before its answer.

    An unclosed tag means the generation was truncated by the token limit; everything after the
    opening is then dropped, which yields an empty string and an outright parse failure rather
    than promoting unfinished reasoning to the rank of an answer.
    """
    if not text:
        return ""
    out = text
    for tag in REASONING_TAGS:
        out = re.sub(rf"<{tag}>.*?</{tag}>", " ", out, flags=re.S | re.I)
    lowered = out.lower()
    for tag in REASONING_TAGS:
        opening = f"<{tag}>"
        if opening in lowered:
            out = out[: lowered.index(opening)]
            lowered = out.lower()
    return out.strip()


def extract_json_object(text: str) -> Any:
    """Last balanced JSON object of a response, after the scratchpad is removed.

    The last and not the first: if a draft object escapes the stripper, taking the first reads
    the sketch while taking the last reads the conclusion.
    """
    if not isinstance(text, str):
        return None
    body = strip_reasoning(text).strip()
    if body.startswith("```"):
        body = body.split("\n", 1)[-1] if "\n" in body else body
        if body.rstrip().endswith("```"):
            body = body.rstrip()[:-3]
    body = body.strip()

    try:
        return json.loads(body)
    except json.JSONDecodeError:
        pass

    found = None
    start = body.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for pos in range(start, len(body)):
            char = body[pos]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        found = json.loads(body[start:pos + 1])
                    except json.JSONDecodeError:
                        pass
                    break
        start = body.find("{", start + 1)
    return found


@dataclass
class CallResult:
    """One call, whether it was billed, served from cache, or refused."""

    parsed: BaseModel | None
    parse_ok: bool
    cache_hit: bool
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0
    cost_usd: float = 0.0
    model_returned: str = ""
    structure_mode: str = ""
    error: str = ""


class Client:
    """A bounded producer of typed objects."""

    # Class level on purpose: the provider quota, the spend cap and the ledger file are shared by
    # every instance in the process, so the locks that protect them must be too.
    _rate_lock = threading.Lock()
    _budget_lock = threading.Lock()
    _ledger_lock = threading.Lock()

    #: Class level too. A per-instance timestamp would let each worker respect the interval on
    #: its own while together they exceed it N-fold.
    _last_call = 0.0

    _governor: "RateGovernor | None" = None

    @classmethod
    def _shared_governor(cls) -> RateGovernor:
        with cls._rate_lock:
            if cls._governor is None:
                cls._governor = RateGovernor(Limits())
            return cls._governor

    @classmethod
    def configure_limits(cls, limits: Limits) -> RateGovernor:
        """Install the provider's real ceilings, once, before a campaign starts.

        The defaults are deliberately modest: a campaign that has not measured its quota should
        run slowly rather than be throttled halfway through.
        """
        with cls._rate_lock:
            cls._governor = RateGovernor(limits)
            return cls._governor

    def __init__(
        self,
        model: str,
        cache_dir: Path | str,
        ledger_path: Path | str,
        mandates_dir: Path | str,
        base_url: str | None = None,
        api_key: str | None = None,
        prior_ledgers: tuple[Path | str, ...] = (),
        budget_usd: float = 20.0,
        temperature: float = 0.0,
        price_in_per_m: float = 0.30,
        price_out_per_m: float = 1.20,
        min_interval_s: float = 0.8,
        max_output_tokens: int = 900,
        governor: RateGovernor | None = None,
    ) -> None:
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.ledger_path = Path(ledger_path)
        self.prior_ledgers = tuple(Path(p) for p in prior_ledgers)
        self.mandates_dir = Path(mandates_dir)
        self.base_url = base_url or os.environ.get("LLM_BASE_URL")
        self.api_key = api_key or os.environ.get("LLM_API_KEY")
        self.budget_usd = min(float(budget_usd), ABSOLUTE_HARD_CAP_USD)
        self.temperature = float(temperature)
        self.price_in_per_m = price_in_per_m
        self.price_out_per_m = price_out_per_m
        self.min_interval_s = min_interval_s
        # Read by `_request`, by the spend estimate and by the quota reservation, so it lives on
        # the instance: three readers of the same ceiling that could disagree would make the
        # budget guard and the actual request describe two different campaigns.
        self.max_output_tokens = int(max_output_tokens)
        # One governor for the whole process: the quota belongs to the API key, not to a worker.
        self.governor = governor if governor is not None else Client._shared_governor()
        self._reserved = 0.0
        self._mandate_cache: dict[str, tuple[str, str]] = {}
        self._ledger_state: dict[str, dict[str, Any]] = {}
        self._client = None
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)

    # -- mandates

    def mandate(self, mandate_id: str) -> tuple[str, str]:
        """Text and fingerprint of a mandate file, newlines normalised to LF.

        Normalisation keeps the hash a property of the content rather than of the machine the
        file was checked out on.
        """
        if mandate_id not in self._mandate_cache:
            path = self.mandates_dir / f"{mandate_id}.txt"
            if not path.exists():
                raise FileNotFoundError(
                    f"mandate {mandate_id!r} not found at {path}. A missing mandate stops the "
                    "run rather than falling back to another one."
                )
            raw = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
            self._mandate_cache[mandate_id] = (raw.decode("utf-8"), sha256(raw))
        return self._mandate_cache[mandate_id]

    # -- cache

    def system_prompt(self, mandate_id: str, schema: type[BaseModel]) -> tuple[str, str, str]:
        """The exact system message sent, its mandate fingerprint, and its contract fingerprint.

        The two fingerprints are kept apart: the mandate is what the experiment manipulates, the
        contract is a property of the tier's schema. Hashing them together would make a schema
        edit look like a mandate edit in the ledger.
        """
        mandate_text, mandate_sha = self.mandate(mandate_id)
        contract = schema_contract(schema)
        return f"{mandate_text}\n\n{contract}", mandate_sha, sha256(contract)

    def cache_key(self, context: str, mandate_id: str, replicate: int,
                  schema: type[BaseModel] | None = None) -> str:
        """Everything that can change the answer, and nothing that cannot.

        The mandate's content is in the key, not only its name, so editing a mandate file
        invalidates its entries. The output contract is in the key for the same reason: since the
        provider ignores `response_format`, a schema edit changes the bytes the model receives.
        """
        _, mandate_sha = self.mandate(mandate_id)
        seed = [self.model, mandate_id, mandate_sha, str(replicate),
                f"{self.temperature:.4f}", sha256(context)]
        if schema is not None:
            seed.append(sha256(schema_contract(schema)))
        return sha256("|".join(seed))

    def _cache_path(self, key: str) -> Path:
        # Two levels: a single directory holding tens of thousands of files is slow on Windows.
        return self.cache_dir / key[:2] / f"{key}.json"

    def in_cache(self, context: str, mandate_id: str, replicate: int,
                 schema: type[BaseModel] | None = None) -> bool:
        """Read-only, no side effect, no call. Used to price a resumption honestly.

        The schema must be passed, since `call` puts the contract fingerprint in the key: a check
        that omits it computes a different key and reports every entry as absent.
        """
        return self._cache_path(
            self.cache_key(context, mandate_id, replicate, schema)).exists()

    def _read_cache(self, key: str) -> dict[str, Any] | None:
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def _write_cache(self, key: str, entry: dict[str, Any]) -> None:
        path = self._cache_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(entry, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)   # atomic: a crash mid-write never leaves a half entry

    # -- ledger and cap

    #: Cost column of a ledger written before this client existed. Reading only the new column
    #: would restart the count at zero, so the cap would authorise past spending a second time.
    LEGACY_COST_COLUMNS = ("cout_usd",)

    @staticmethod
    def _row_cost(row: dict[str, str]) -> float:
        raw = row.get("cost_usd")
        if raw in (None, ""):
            for legacy in Client.LEGACY_COST_COLUMNS:
                if row.get(legacy) not in (None, ""):
                    raw = row[legacy]
                    break
        try:
            return float(raw or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _sum_ledger(self, path: Path) -> float:
        """Cost of a ledger, reading only the bytes added since the last look.

        The file is still opened and read before every call, which is what the guarantee says,
        but only from the offset already consumed. If the file shrank, or its modification time
        moved backwards, the cached state is discarded and the whole file is read again.
        """
        if not path.exists():
            return 0.0
        stat = path.stat()
        key = str(path.resolve())
        cached = self._ledger_state.get(key)

        if cached and stat.st_size >= cached["size"] and stat.st_mtime_ns >= cached["mtime"]:
            offset, total, header = cached["offset"], cached["total"], cached["header"]
        else:
            offset, total, header = 0, 0.0, None

        with path.open("r", encoding="utf-8", newline="") as handle:
            if header is None:
                reader = csv.DictReader(handle)
                for row in reader:
                    total += self._row_cost(row)
                header = reader.fieldnames
            else:
                handle.seek(offset)
                for row in csv.DictReader(handle, fieldnames=header):
                    total += self._row_cost(row)
            offset = handle.tell()

        self._ledger_state[key] = {
            "size": stat.st_size, "mtime": stat.st_mtime_ns,
            "offset": offset, "total": total, "header": header,
        }
        return total

    def total_cost(self) -> float:
        """Everything this project has ever been billed, re-read from disk.

        Deliberately not memoised: a figure held in memory cannot see the calls of another
        process writing to the same ledger. Sums the current ledger and every prior ledger
        declared in `prior_ledgers`, so the cap applies to total project spend.
        """
        total = self._sum_ledger(self.ledger_path)
        for prior in self.prior_ledgers:
            if prior.resolve() != self.ledger_path.resolve():
                total += self._sum_ledger(prior)
        return total

    def _estimate(self, context_chars: int, mandate_chars: int, max_out_tokens: int) -> float:
        """Deliberately pessimistic: 3 characters per token in, the full output budget out."""
        tokens_in = (context_chars + mandate_chars) / 3.0
        return (tokens_in / 1e6) * self.price_in_per_m + (max_out_tokens / 1e6) * self.price_out_per_m

    def _check_budget(self, estimate: float) -> None:
        with self._budget_lock:
            spent = self.total_cost()
            if spent + self._reserved + estimate > self.budget_usd:
                raise BudgetExceeded(
                    f"refusing to call: {spent:.4f} USD already spent, {self._reserved:.4f} "
                    f"reserved in flight, {estimate:.4f} estimated for this call, against a cap "
                    f"of {self.budget_usd:.2f}."
                )
            self._reserved += estimate

    def _release(self, estimate: float) -> None:
        with self._budget_lock:
            self._reserved = max(0.0, self._reserved - estimate)

    def _append_ledger(self, **row: Any) -> None:
        """Append one row, under a lock, flushed to the operating system before returning.

        Two workers writing a CSV row at once interleave their bytes and produce a line with
        fields from two different calls, which parses without error. The flush matters for the
        same reason the cap is re-read from disk: another process must see this row.
        """
        with self._ledger_lock:
            exists = self.ledger_path.exists()
            with self.ledger_path.open("a", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=LEDGER_COLUMNS,
                                        extrasaction="ignore")
                if not exists:
                    writer.writeheader()
                writer.writerow({col: row.get(col, "") for col in LEDGER_COLUMNS})
                handle.flush()
                os.fsync(handle.fileno())

    # -- rate

    def _backoff(self, attempt: int) -> None:
        """Sleep before retrying a transient refusal, with jitter.

        Without the jitter, workers throttled by the same window all wake at the same instant and
        reproduce the burst that caused the refusal.
        """
        delay = min(BACKOFF_CAP_S, BACKOFF_BASE_S * (2 ** (attempt - 1)))
        time.sleep(delay * (0.5 + random.random()))

    def _wait_rate_limit(self) -> None:
        """Minimum gap between two requests. The real regulator is the governor.

        A fixed interval assumes every call costs the same, which the measured token spread
        contradicts. This remains as a floor so a burst of cache misses cannot hammer the
        endpoint.
        """
        with Client._rate_lock:
            gap = time.time() - Client._last_call
            if gap < self.min_interval_s:
                time.sleep(self.min_interval_s - gap)
            # Written on the class, not on self: an instance attribute would shadow the shared
            # one and restore the per-instance behaviour this guards against.
            Client._last_call = time.time()

    # -- the call

    @property
    def endpoint(self):
        """The provider client, built on first use so offline tests never need a key."""
        if self._client is None:
            from openai import OpenAI            # lazy: importing must not require a key

            if not self.api_key:
                raise ProviderError(
                    "no API key. Set LLM_API_KEY or pass api_key."
                )
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        return self._client

    def _request(self, system: str, user: str, schema: type[BaseModel], mode: str):
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
        }
        if mode == "json_schema":
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "reading", "strict": True,
                                "schema": strict_json_schema(schema)},
            }
        elif mode == "json_object":
            payload["response_format"] = {"type": "json_object"}
        return self.endpoint.chat.completions.create(**payload)

    def call(self, context: str, mandate_id: str, schema: type[BaseModel],
             replicate: int = 0, ticker: str = "", session: str = "") -> CallResult:
        """One reading: cache, spend guard, quota, request, parse, ledger.

        The order is deliberate. The cache is consulted before anything is reserved, so a resumed
        run holds no quota and spends no budget on work already done. The spend guard then runs
        before the rate governor, because refusing on money is instant while waiting on tokens is
        not. On failure the unit is refused: `parse_ok` is false and nothing is recovered from
        prose.
        """
        key = self.cache_key(context, mandate_id, replicate, schema)
        cached = self._read_cache(key)
        if cached is not None:
            parsed, error = validate_payload(extract_json_object(cached.get("raw", "")), schema)
            return CallResult(parsed=parsed, parse_ok=parsed is not None, cache_hit=True,
                              model_returned=cached.get("model_returned", ""),
                              structure_mode=cached.get("structure_mode", ""), error=error)

        system, mandate_sha, _ = self.system_prompt(mandate_id, schema)
        estimate = self._estimate(len(context), len(system), self.max_output_tokens)
        self._check_budget(estimate)
        reservation = self.admit(context, mandate_id, self.max_output_tokens, schema)

        last_error = ""
        mode_index = 0
        try:
            for attempt in range(1, MAX_ATTEMPTS + 1):
                mode_used = ""
                response = None
                try:
                    self._wait_rate_limit()
                    started = time.time()
                    mode = STRUCTURE_MODES[min(mode_index, len(STRUCTURE_MODES) - 1)]
                    response = self._request(system, context, schema, mode)
                    # The wire mode that was accepted, plus the fact that the shape was in truth
                    # imposed by the prompt contract.
                    mode_used = f"{mode}|contract"
                except Exception as exc:      # noqa: BLE001 - recorded, never swallowed
                    last_error = f"{type(exc).__name__}: {exc}"
                    response = None
                    if _is_terminal(exc):
                        # Every remaining unit would fail the same way.
                        self._release(estimate)
                        self.governor.release(reservation)
                        self._append_ledger(
                            timestamp_utc=_now(), event="stopped", ticker=ticker,
                            session=session, mandate=mandate_id, replicate=replicate,
                            attempt=attempt, model_requested=self.model, cache_key=key[:16],
                            status="terminal", parse_ok=0, cost_usd="0",
                            error=last_error[:200])
                        raise ProviderError(
                            f"the endpoint refuses durably and no retry can help: "
                            f"{last_error[:180]}. The campaign stops here; what is already in "
                            "the cache is intact and a resumed run pays only for what it had "
                            "not reached."
                        ) from exc
                    if _rejects_schema(exc):
                        # The endpoint refuses this response_format, and only this justifies
                        # stepping down: a busy provider says nothing about the schema.
                        mode_index += 1
                    elif _is_transient(exc):
                        # A queue, not a refusal. Without a pause the retry arrives inside the
                        # same exhausted window and fails for the same reason.
                        self._backoff(attempt)
                    else:
                        break
                if response is None:
                    continue

                latency_ms = int((time.time() - started) * 1000)
                message = getattr(response.choices[0], "message", None)
                raw = getattr(message, "content", "") or ""
                usage = getattr(response, "usage", None)
                tokens_in = int(getattr(usage, "prompt_tokens", 0) or 0)
                tokens_out = int(getattr(usage, "completion_tokens", 0) or 0)
                cached_in = int(getattr(getattr(usage, "prompt_tokens_details", None),
                                        "cached_tokens", 0) or 0)
                cost = ((tokens_in / 1e6) * self.price_in_per_m
                        + (tokens_out / 1e6) * self.price_out_per_m)
                # The model that answered, not the alias asked for.
                returned = getattr(response, "model", "") or ""

                self.governor.settle(reservation, tokens_in + tokens_out)
                parsed, error = validate_payload(extract_json_object(raw), schema)

                self._append_ledger(
                    timestamp_utc=_now(), event="call", ticker=ticker, session=session,
                    mandate=mandate_id, replicate=replicate, attempt=attempt,
                    model_requested=self.model, model_returned=returned,
                    mandate_sha256=mandate_sha[:16], context_sha256=sha256(context, 16),
                    cache_key=key[:16], structure_mode=mode_used, tokens_in=tokens_in,
                    tokens_in_cached=cached_in, tokens_out=tokens_out, latency_ms=latency_ms,
                    cost_usd=f"{cost:.8f}", status="ok", parse_ok=int(parsed is not None),
                    error=error[:200],
                )
                if parsed is not None:
                    self._write_cache(key, {"raw": raw, "model_returned": returned,
                                            "structure_mode": mode_used, "cost_usd": cost})
                    return CallResult(parsed=parsed, parse_ok=True, cache_hit=False,
                                      tokens_in=tokens_in, tokens_out=tokens_out,
                                      latency_ms=latency_ms, cost_usd=cost,
                                      model_returned=returned, structure_mode=mode_used)
                last_error = error
        finally:
            self._release(estimate)

        self.governor.release(reservation)
        # Which refusal, because the two mean opposite things. `refused_transport` is the
        # provider declining to answer after every retry; `refused_parse` is an answer that would
        # not validate, which is the quantity the dissertation publishes.
        transport = any(name in last_error for name in TRANSIENT_ERRORS)
        self._append_ledger(timestamp_utc=_now(), event="refused", ticker=ticker,
                            session=session, mandate=mandate_id, replicate=replicate,
                            attempt=MAX_ATTEMPTS, model_requested=self.model,
                            cache_key=key[:16],
                            status="refused_transport" if transport else "refused_parse",
                            parse_ok=0, cost_usd="0", error=last_error[:200])
        return CallResult(parsed=None, parse_ok=False, cache_hit=False, error=last_error)

    def admit(self, context: str, mandate_id: str, max_out_tokens: int = 900,
              schema: type[BaseModel] | None = None) -> int:
        """Block until the provider's quota can take this call, and hold its budget.

        Returns the reservation, which the caller must settle with the billed usage or release if
        the call never happened. The output contract counts towards the estimate: it is sent on
        every call, and leaving it out would understate the reservation on the short tiers, which
        are also the most frequent.
        """
        mandate_text, _ = self.mandate(mandate_id)
        system_chars = len(mandate_text)
        if schema is not None:
            system_chars += len(schema_contract(schema))
        estimated = (len(context) + system_chars) / 3.0 + max_out_tokens
        return self.governor.reserve(estimated)
