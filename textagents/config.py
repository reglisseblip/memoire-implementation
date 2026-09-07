"""Every setting of a run, in one place, and the three paths the rest of the code resolves from.

Each field can be overridden from the environment through TEXTAGENTS_<FIELD>, coerced to the type
of the committed default so a misspelt value fails at startup instead of misconfiguring a run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path

#: The package directory. Everything shipped with the code is resolved from here.
PACKAGE_ROOT = Path(__file__).resolve().parent

#: The working tree. Everything a run reads or writes is resolved from here, so a clone that has
#: not been installed and one that has both find the same `data/`, `runs/` and `results/`.
PROJECT_ROOT = PACKAGE_ROOT.parent

#: The eleven instruction files. Shipped with the package because their bytes are hashed into
#: every cache key: they are code, not configuration.
PROMPTS_DIR = PACKAGE_ROOT / "prompts"

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _coerce(raw: str, reference):
    if isinstance(reference, bool):
        low = raw.strip().lower()
        if low in _TRUE:
            return True
        if low in _FALSE:
            return False
        raise ValueError(f"expected a boolean, got {raw!r}")
    if isinstance(reference, int) and not isinstance(reference, bool):
        return int(raw)
    if isinstance(reference, float):
        return float(raw)
    if isinstance(reference, Path):
        return Path(raw)
    return raw


@dataclass
class Config:
    """The whole configuration of a run."""

    # paths
    data_dir: Path = PROJECT_ROOT / "data" / "processed"
    cache_dir: Path = PROJECT_ROOT / "runs" / "cache"
    ledger_path: Path = PROJECT_ROOT / "runs" / "ledger_agents.csv"
    mandates_dir: Path = PROMPTS_DIR
    results_dir: Path = PROJECT_ROOT / "results"

    #: Ledgers written by earlier runs. Their spend counts against the budget cap too.
    prior_ledgers: tuple[Path, ...] = (PROJECT_ROOT / "runs" / "ledger.csv",)

    # model
    model: str = "MiniMax-M2"
    temperature: float = 0.0
    max_output_tokens: int = 900

    # provider ceilings
    tokens_per_minute: int = 100_000
    requests_per_minute: int = 60

    # spend
    budget_usd: float = 25.0

    # the chain
    #: The three analyst seats of tier one. Named here rather than in a module constant so a run
    #: records which wording of the exogeneity seat it used.
    analyst_mandates: tuple[str, ...] = (
        "analyst_fundamentals", "analyst_exogeneity_v2", "analyst_tone")
    max_debate_rounds: int = 1
    max_panel_rounds: int = 1
    #: Readings per unit. The aggregation takes the median over them.
    replicates: int = 3

    # concurrency
    #: Follows from the token ceiling, not from the number of cores: at 100,000 tokens a minute
    #: and about 3,150 tokens a call, five workers can be kept busy.
    workers: int = 5

    def __post_init__(self) -> None:
        for spec in fields(self):
            env = os.environ.get(f"TEXTAGENTS_{spec.name.upper()}")
            if env in (None, ""):
                continue
            current = getattr(self, spec.name)
            if isinstance(current, tuple):
                continue
            try:
                setattr(self, spec.name, _coerce(env, current))
            except ValueError as exc:
                raise ValueError(f"TEXTAGENTS_{spec.name.upper()}: {exc}") from exc

    def calls_per_unit(self) -> int:
        """Nine calls per unit with the committed rounds, known before the run starts."""
        return 3 + 2 * self.max_debate_rounds + 1 + 3 * self.max_panel_rounds

    def describe(self) -> str:
        return (
            f"model={self.model} T={self.temperature} K={self.replicates} "
            f"calls/unit={self.calls_per_unit()} workers={self.workers} "
            f"cap={self.budget_usd:.2f}USD tpm={self.tokens_per_minute} "
            f"analysts={'+'.join(m.replace('analyst_', '') for m in self.analyst_mandates)}"
        )
