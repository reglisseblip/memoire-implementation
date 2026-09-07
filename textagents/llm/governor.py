"""Rate governor: a sliding window over tokens per minute and requests per minute.

A fixed interval between calls assumes every call costs the same, which the measured token spread
of this workload contradicts. Both provider ceilings are therefore enforced over a rolling
minute. The cost of a call is estimated and reserved before it is sent, then replaced by the
billed usage when the response comes back, so a call in flight is never invisible to the
accounting.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

WINDOW_SECONDS = 60.0


@dataclass(frozen=True)
class Limits:
    """The two ceilings a provider enforces, plus the safety margin applied to both.

    `margin` exists because the provider's tokeniser is not ours, and a request rejected for
    crossing a quota still costs a round trip.
    """

    tokens_per_minute: int = 100_000
    requests_per_minute: int = 60
    margin: float = 0.90

    def token_budget(self) -> float:
        return self.tokens_per_minute * self.margin

    def request_budget(self) -> float:
        return self.requests_per_minute * self.margin


class RateGovernor:
    """Admits calls so that neither ceiling is crossed over any rolling minute.

    One instance is shared by every worker in the process: the quota belongs to the API key, not
    to a thread, so a per-worker accounting would let N workers exceed the ceiling together.
    """

    def __init__(self, limits: Limits | None = None) -> None:
        self.limits = limits or Limits()
        self._lock = threading.Condition()
        #: (timestamp, tokens, reservation_id or None)
        self._events: deque[tuple[float, float, int | None]] = deque()
        self._next_reservation = 0
        self._admitted = 0
        self._waited_s = 0.0

    # -- window bookkeeping

    def _evict(self, now: float) -> None:
        horizon = now - WINDOW_SECONDS
        while self._events and self._events[0][0] < horizon:
            self._events.popleft()

    def _used(self) -> tuple[float, int]:
        tokens = sum(event[1] for event in self._events)
        return tokens, len(self._events)

    def _wait_for(self, need_tokens: float, now: float) -> float:
        """Seconds until enough of the window has aged out to admit `need_tokens`.

        Returns 0.0 when the call fits immediately, and -1.0 when even an empty window could not
        hold it. The wait is computed rather than polled, since the moment each recorded event
        leaves the window is known.
        """
        tokens, requests = self._used()
        token_budget = self.limits.token_budget()
        request_budget = self.limits.request_budget()

        if tokens + need_tokens <= token_budget and requests + 1 <= request_budget:
            return 0.0

        released_tokens = 0.0
        released_requests = 0
        for timestamp, event_tokens, _ in self._events:
            released_tokens += event_tokens
            released_requests += 1
            token_ok = tokens - released_tokens + need_tokens <= token_budget
            request_ok = requests - released_requests + 1 <= request_budget
            if token_ok and request_ok:
                return max(0.0, timestamp + WINDOW_SECONDS - now)
        # Even an empty window cannot hold this call: it is larger than the whole budget.
        return -1.0

    # -- public

    def reserve(self, estimated_tokens: float) -> int:
        """Block until the call fits, then hold `estimated_tokens` against the window.

        Raises when a single call is larger than an entire minute of budget, because waiting
        would never help and truncating the input would change what was measured.
        """
        estimated_tokens = max(1.0, float(estimated_tokens))
        with self._lock:
            if estimated_tokens > self.limits.token_budget():
                raise ValueError(
                    f"a single call needs {estimated_tokens:.0f} tokens, more than the "
                    f"{self.limits.token_budget():.0f} available in a whole minute at "
                    f"{self.limits.tokens_per_minute} TPM. Waiting cannot help: raise the "
                    "limit or shorten the input, but do not truncate silently."
                )
            while True:
                now = time.monotonic()
                self._evict(now)
                delay = self._wait_for(estimated_tokens, now)
                if delay <= 0.0:
                    break
                self._waited_s += delay
                self._lock.wait(timeout=delay)

            self._next_reservation += 1
            reservation = self._next_reservation
            self._events.append((time.monotonic(), estimated_tokens, reservation))
            self._admitted += 1
            return reservation

    def settle(self, reservation: int, actual_tokens: float) -> None:
        """Replace an estimate by what the provider actually billed.

        The timestamp is kept, so a long call cannot slide out of the window early.
        """
        with self._lock:
            for index, (timestamp, _tokens, held) in enumerate(self._events):
                if held == reservation:
                    self._events[index] = (timestamp, max(0.0, float(actual_tokens)), None)
                    break
            self._lock.notify_all()

    def release(self, reservation: int) -> None:
        """Drop a reservation whose call never happened, so its budget is not held for a minute."""
        with self._lock:
            for index, (_timestamp, _tokens, held) in enumerate(self._events):
                if held == reservation:
                    del self._events[index]
                    break
            self._lock.notify_all()

    def snapshot(self) -> dict[str, float]:
        """Current occupancy, for logging and for the campaign report."""
        with self._lock:
            now = time.monotonic()
            self._evict(now)
            tokens, requests = self._used()
            return {
                "tokens_in_window": tokens,
                "requests_in_window": float(requests),
                "token_budget": self.limits.token_budget(),
                "request_budget": self.limits.request_budget(),
                "token_utilisation": tokens / max(self.limits.token_budget(), 1.0),
                "admitted": float(self._admitted),
                "waited_seconds": self._waited_s,
            }

    @staticmethod
    def useful_workers(limits: Limits, tokens_per_call: float, latency_s: float) -> float:
        """How many workers the ceilings can actually keep busy.

        Beyond this figure, workers queue inside the governor rather than talking to the provider.
        Reported before a campaign starts, so the concurrency is chosen from the quota.
        """
        by_tokens = limits.token_budget() / max(tokens_per_call, 1.0) / 60.0
        by_requests = limits.request_budget() / 60.0
        return min(by_tokens, by_requests) * latency_s
