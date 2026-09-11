from __future__ import annotations

import json
from pathlib import Path

from cpuattn.hardware.host import Host

from .harness import Stats
from .results import CaseResult, Environment, ThreadLevel


def _git_commit() -> str | None:
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            capture_output=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() or None


def build_environment(
    runtime_host: Host, suite: str, warmup: int, runs: int, stream_warmup: int
) -> Environment:
    import datetime
    import platform as platform_module

    import numpy as np

    from .baselines import torch_available

    torch_version: str | None = None
    if torch_available():
        import torch

        torch_version = torch.__version__
    return Environment(
        timestamp_utc=datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        platform=platform_module.platform(),
        cpu=f"{runtime_host.vendor} {runtime_host.model}",
        architecture=runtime_host.architecture,
        physical_cores=runtime_host.physical_cores,
        numpy_version=np.__version__,
        torch_version=torch_version,
        git_commit=_git_commit(),
        warmup=warmup,
        runs=runs,
        stream_warmup=stream_warmup,
        suite=suite,
    )


def _stats_json(stats: Stats) -> dict[str, object]:
    payload: dict[str, object] = {"samples": stats.samples}
    payload.update(stats.as_ms())
    return payload


def _thread_level_json(level: ThreadLevel) -> dict[str, object]:
    return {
        "workers": level.workers,
        "native_median_ms": level.native_median_ms,
        "wall_median_ms": level.wall_median_ms,
        "plan": level.plan,
    }


def case_to_json(result: CaseResult) -> dict[str, object]:
    payload: dict[str, object] = {
        "kind": result.kind,
        "name": result.name,
        "params": result.params,
    }
    if result.error is not None:
        payload["error"] = result.error
        return payload
    payload["plan"] = result.plan
    if result.first_call_ms is not None:
        payload["first_call_ms"] = result.first_call_ms
    if result.wall is not None:
        payload["wall"] = _stats_json(result.wall)
    if result.native is not None:
        payload["native"] = _stats_json(result.native)
    payload["baselines"] = [
        {
            "backend": baseline.backend,
            "wall": _stats_json(baseline.wall),
            "max_abs_err": baseline.max_abs_err,
            "speedup": baseline.speedup,
        }
        for baseline in result.baselines
    ]
    payload["max_abs_err"] = result.max_abs_err
    payload["numeric_ok"] = result.numeric_ok
    if result.tune_steps is not None:
        payload["tune_steps"] = result.tune_steps
    if result.final_skv is not None:
        payload["final_skv"] = result.final_skv
    if result.thread_levels is not None:
        payload["thread_levels"] = [_thread_level_json(level) for level in result.thread_levels]
    return payload


def report_to_json(environment: Environment, results: list[CaseResult]) -> dict[str, object]:
    return {
        "environment": {
            "timestamp_utc": environment.timestamp_utc,
            "platform": environment.platform,
            "cpu": environment.cpu,
            "architecture": environment.architecture,
            "physical_cores": environment.physical_cores,
            "numpy": environment.numpy_version,
            "torch": environment.torch_version,
            "git_commit": environment.git_commit,
            "suite": environment.suite,
            "warmup": environment.warmup,
            "runs": environment.runs,
            "stream_warmup": environment.stream_warmup,
        },
        "cases": [case_to_json(result) for result in results],
    }


def write_report(payload: dict[str, object], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output


def _baseline_speedups(result: CaseResult) -> dict[str, float | None]:
    return {baseline.backend.split("@")[0]: baseline.speedup for baseline in result.baselines}


def _speedup_text(speedup: float | None) -> str:
    return "-" if speedup is None else f"{speedup:.2f}x"


def failed_cases(results: list[CaseResult]) -> list[CaseResult]:
    """Cases that either crashed or produced numerically unacceptable output."""
    return [
        result
        for result in results
        if result.error is not None or result.numeric_ok is False
    ]


def print_console(results: list[CaseResult], environment: Environment) -> None:
    print("CPUAttn benchmark suite")
    print(f"cpu: {environment.cpu} ({environment.physical_cores} physical cores)")
    print(f"platform: {environment.platform}")
    print(
        f"suite: {environment.suite}, warmup={environment.warmup}, "
        f"runs={environment.runs}, stream_warmup={environment.stream_warmup}"
    )
    print()
    header = (
        f"{'case':<24} {'kind':<17} {'first ms':>10} {'wall med':>10} "
        f"{'native med':>10} {'vs numpy':>9} {'vs torch':>9}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        if result.error is not None:
            print(f"{result.name:<24} {result.kind:<17} ERROR: {result.error[:80]}")
            continue
        first = "-" if result.first_call_ms is None else f"{result.first_call_ms:.1f}"
        wall = "-" if result.wall is None else f"{result.wall.median / 1e6:.3f}"
        native = "-" if result.native is None else f"{result.native.median / 1e6:.3f}"
        speedups = _baseline_speedups(result)
        vs_numpy = _speedup_text(speedups.get("numpy_blas"))
        vs_torch = _speedup_text(speedups.get("torch_sdpa"))
        marker = "" if result.numeric_ok is not False else "  NUMERIC-FAIL"
        print(
            f"{result.name:<24} {result.kind:<17} {first:>10} {wall:>10} "
            f"{native:>10} {vs_numpy:>9} {vs_torch:>9}{marker}"
        )
    for result in results:
        if result.thread_levels is None:
            continue
        print()
        print(f"thread scaling: {result.name}")
        for level in result.thread_levels:
            print(
                f"  workers={level.workers:>3}  native {level.native_median_ms:9.3f} ms   "
                f"wall {level.wall_median_ms:9.3f} ms   best={level.plan}"
            )
    print()
    failed = failed_cases(results)
    if failed:
        print(f"{len(failed)} case(s) failed (execution error or numeric mismatch)")
