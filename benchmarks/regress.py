"""Gate a benchmark run against a stored baseline.

    python -m benchmarks.regress \
        --baseline artifacts/benchmarks/quiet-baseline-x86_64.json \
        --candidate artifacts/benchmarks/latest.json --tolerance 1.25

Exits non-zero when any baseline case is slower than the baseline by more than
the tolerance, or is missing/unavailable in the candidate (a crashed workload
carries an ``error`` and no per-case stat, so treating it as "absent" is what
makes the gate fail closed).

The default statistic is each case's per-run minimum (``--stat min``). On a
shared machine a contention spike during tuning can cache a pathological plan
and inflate a case's median by orders of magnitude while its minimum stays
clean, so the minimum is the more stable cross-run signal. ``--stat median``
restores the old behaviour; latency is still a distribution, so this is a coarse
gate, best compared on the same machine in the same quiet window. Environment
differences are reported but not fatal.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

_STATS = {"min": "min_ms", "median": "median_ms"}
_ENVIRONMENT_KEYS = (
    "cpu",
    "architecture",
    "physical_cores",
    "platform",
    "numpy",
    "torch",
    "git_commit",
    "suite",
    "warmup",
    "runs",
    "stream_warmup",
)

Report = dict[str, dict[str, float]]


@dataclass(frozen=True, slots=True)
class Delta:
    """One case's candidate/baseline ratio for a chosen metric."""

    name: str
    metric: str
    baseline_ms: float
    candidate_ms: float

    @property
    def ratio(self) -> float:
        return self.candidate_ms / self.baseline_ms


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def load_report(
    path: str | Path, stat: str = "min_ms"
) -> tuple[Report, dict[str, object]]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path} is not valid JSON: {error}") from error
    if not isinstance(data, dict) or not isinstance(data.get("cases"), list):
        raise ValueError(f"{path} is not a CPUAttn benchmark report")
    report: Report = {}
    for case in data["cases"]:
        if not isinstance(case, dict):
            continue
        name = case.get("name")
        if not isinstance(name, str):
            continue
        metrics: dict[str, float] = {}
        for source in ("native", "wall"):
            stats = case.get(source)
            if isinstance(stats, dict):
                value = _number(stats.get(stat))
                if value is not None:
                    metrics[source] = value
        report[name] = metrics
    environment = data.get("environment")
    return report, environment if isinstance(environment, dict) else {}


def compare(baseline: Report, candidate: Report, metric: str) -> tuple[Delta, ...]:
    """Comparable cases: both stats present and the baseline is positive."""
    deltas = []
    for name, base in baseline.items():
        base_ms = base.get(metric)
        candidate_ms = candidate.get(name, {}).get(metric)
        if base_ms is None or candidate_ms is None or base_ms <= 0.0:
            continue
        deltas.append(Delta(name, metric, base_ms, candidate_ms))
    return tuple(deltas)


def merge_reports(reports: list[Report]) -> Report:
    """Per-case, per-source minimum across runs (min-of-mins)."""
    merged: Report = {}
    for report in reports:
        for name, metrics in report.items():
            target = merged.setdefault(name, {})
            for source, value in metrics.items():
                if source not in target or value < target[source]:
                    target[source] = value
    return merged


def unavailable(baseline: Report, candidate: Report, metric: str) -> tuple[str, ...]:
    """Baseline cases that produced no comparable stat in the candidate."""
    names = []
    for name, base in baseline.items():
        base_ms = base.get(metric)
        if base_ms is None:
            continue
        candidate_ms = candidate.get(name, {}).get(metric)
        if candidate_ms is None or base_ms <= 0.0:
            names.append(name)
    return tuple(sorted(names))


def regressions(deltas: tuple[Delta, ...], tolerance: float) -> tuple[Delta, ...]:
    return tuple(delta for delta in deltas if delta.ratio > tolerance)


def environment_differences(
    baseline: dict[str, object], candidate: dict[str, object]
) -> tuple[str, ...]:
    return tuple(
        key
        for key in _ENVIRONMENT_KEYS
        if baseline.get(key) != candidate.get(key)
    )


def _status(delta: Delta, tolerance: float) -> str:
    if delta.ratio > tolerance:
        return "REGRESS"
    if delta.ratio < 1.0 / tolerance:
        return "faster"
    return "ok"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark regression gate")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        type=Path,
        action="append",
        required=True,
        help="candidate report; repeat to merge runs per-case by minimum",
    )
    parser.add_argument("--metric", choices=("native", "wall"), default="native")
    parser.add_argument("--stat", choices=tuple(_STATS), default="min")
    parser.add_argument("--tolerance", type=float, default=1.25)
    arguments = parser.parse_args(argv)
    if arguments.tolerance < 1.0:
        parser.error("--tolerance must be >= 1.0")

    stat = _STATS[arguments.stat]
    baseline, base_env = load_report(arguments.baseline, stat)
    candidates = [load_report(path, stat) for path in arguments.candidate]
    candidate = merge_reports([report for report, _ in candidates])
    cand_env = candidates[-1][1]
    differences = environment_differences(base_env, cand_env)
    if differences:
        print(f"warning: environment differs on {', '.join(differences)}")
    deltas = compare(baseline, candidate, arguments.metric)
    if not deltas:
        print("no comparable cases between the two reports")
        return 1

    print(f"{'case':34} {'baseline':>10} {'candidate':>10} {'ratio':>7} {'status':>8}")
    for delta in sorted(deltas, key=lambda item: -item.ratio):
        print(
            f"{delta.name:34} {delta.baseline_ms:10.4f} {delta.candidate_ms:10.4f} "
            f"{delta.ratio:7.3f} {_status(delta, arguments.tolerance):>8}"
        )
    missing = unavailable(baseline, candidate, arguments.metric)
    if missing:
        print(f"unavailable in candidate: {', '.join(missing)}")
    bad = regressions(deltas, arguments.tolerance)
    print(
        f"{len(bad)} regression(s) over {arguments.tolerance:.2f}x, "
        f"{len(missing)} unavailable, across {len(deltas)} comparable cases "
        f"(metric={arguments.metric}, stat={arguments.stat})"
    )
    return 1 if bad or missing else 0


if __name__ == "__main__":
    sys.exit(main())
