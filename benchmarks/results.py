from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from .harness import Stats


@dataclass(frozen=True, slots=True)
class BaselineResult:
    """One baseline's latency plus correctness agreement with CPUAttn."""

    backend: str
    wall: Stats
    max_abs_err: float | None
    speedup: float | None


class ThreadLevel(NamedTuple):
    """One worker count's best plan, measured natively."""

    workers: int
    native_median_ms: float
    wall_median_ms: float
    plan: str


@dataclass(frozen=True, slots=True)
class CaseResult:
    """Everything measured for one workload; error short-circuits the rest."""

    kind: str
    name: str
    params: dict[str, object]
    plan: str | None = None
    first_call_ms: float | None = None
    wall: Stats | None = None
    native: Stats | None = None
    baselines: tuple[BaselineResult, ...] = ()
    max_abs_err: float | None = None
    numeric_ok: bool | None = None
    tune_steps: int | None = None
    final_skv: int | None = None
    thread_levels: tuple[ThreadLevel, ...] | None = None
    diagnostics: dict[str, object] | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class Environment:
    """Run-level provenance recorded with every report."""

    timestamp_utc: str
    platform: str
    cpu: str
    architecture: str
    physical_cores: int
    numpy_version: str
    torch_version: str | None
    git_commit: str | None
    warmup: int
    runs: int
    stream_warmup: int
    suite: str
