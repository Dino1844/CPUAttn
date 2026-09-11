from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable


def timed_ns(function: Callable[[], object]) -> int:
    started = time.perf_counter_ns()
    function()
    return time.perf_counter_ns() - started


def collect(function: Callable[[], object], warmup: int, runs: int) -> list[int]:
    for _ in range(warmup):
        function()
    return [timed_ns(function) for _ in range(runs)]


def collect_with_probe(
    function: Callable[[], object],
    probe: Callable[[object], int],
    warmup: int,
    runs: int,
) -> tuple[list[int], list[int]]:
    """Time each run and capture one companion value derived from its result."""
    for _ in range(warmup):
        function()
    walls: list[int] = []
    companions: list[int] = []
    for _ in range(runs):
        started = time.perf_counter_ns()
        result = function()
        walls.append(time.perf_counter_ns() - started)
        companions.append(probe(result))
    return walls, companions


@dataclass(frozen=True, slots=True)
class Stats:
    """Distribution summary of a latency sample set, in nanoseconds."""

    samples: int
    median: float
    p25: float
    p75: float
    minimum: float
    mean: float

    def as_ms(self) -> dict[str, float]:
        return {
            "median_ms": self.median / 1e6,
            "p25_ms": self.p25 / 1e6,
            "p75_ms": self.p75 / 1e6,
            "min_ms": self.minimum / 1e6,
            "mean_ms": self.mean / 1e6,
        }


def _percentile(ordered: list[float], fraction: float) -> float:
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(samples_ns: list[int]) -> Stats:
    if not samples_ns:
        raise ValueError("cannot summarize an empty sample set")
    ordered = sorted(float(value) for value in samples_ns)
    count = len(ordered)
    middle = count // 2
    median = ordered[middle] if count % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0
    return Stats(
        samples=count,
        median=median,
        p25=_percentile(ordered, 0.25),
        p75=_percentile(ordered, 0.75),
        minimum=ordered[0],
        mean=math.fsum(ordered) / count,
    )
