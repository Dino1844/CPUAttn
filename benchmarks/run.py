from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from cpuattn.hardware.host import detect_host

from .report import build_environment, failed_cases, print_console, report_to_json, write_report
from .results import CaseResult
from .runners import run_attention_case, run_linear_case, run_thread_scaling_case
from .workloads import (
    AttentionWorkload,
    LinearWorkload,
    ThreadScalingWorkload,
    select,
)


def _run_case(
    workload: AttentionWorkload | LinearWorkload | ThreadScalingWorkload,
    *,
    warmup: int,
    runs: int,
    stream_warmup: int,
    with_torch: bool,
) -> CaseResult:
    if isinstance(workload, AttentionWorkload):
        return run_attention_case(
            workload,
            warmup=warmup,
            runs=runs,
            stream_warmup=stream_warmup,
            with_torch=with_torch,
        )
    if isinstance(workload, LinearWorkload):
        return run_linear_case(workload, warmup=warmup, runs=runs)
    assert isinstance(workload, ThreadScalingWorkload)
    return run_thread_scaling_case(workload, warmup=warmup, runs=runs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CPUAttn benchmark suite")
    parser.add_argument("--suite", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--filter", default=None, help="regex on workload name")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=15)
    parser.add_argument("--stream-warmup", type=int, default=8)
    parser.add_argument(
        "--no-torch", action="store_true", help="skip the PyTorch SDPA baseline"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON report path (default: artifacts/benchmarks/<timestamp>.json)",
    )
    arguments = parser.parse_args(argv)
    if arguments.runs < 1 or arguments.warmup < 0:
        parser.error("--runs must be >= 1 and --warmup must be >= 0")
    logging.getLogger("cpuattn").setLevel(logging.WARNING)
    workloads = select(arguments.suite, arguments.filter)
    host = detect_host()
    environment = build_environment(
        host,
        arguments.suite,
        arguments.warmup,
        arguments.runs,
        arguments.stream_warmup,
    )
    results: list[CaseResult] = []
    for index, workload in enumerate(workloads, start=1):
        print(
            f"[{index}/{len(workloads)}] {workload.name} ...",
            flush=True,
        )
        try:
            results.append(
                _run_case(
                    workload,
                    warmup=arguments.warmup,
                    runs=arguments.runs,
                    stream_warmup=arguments.stream_warmup,
                    with_torch=not arguments.no_torch,
                )
            )
        except Exception as error:  # one bad workload must not sink the suite
            results.append(
                CaseResult(
                    kind=getattr(workload, "kind", type(workload).__name__.lower()),
                    name=workload.name,
                    params={},
                    error=f"{type(error).__name__}: {error}",
                )
            )
    failed = failed_cases(results)
    print_console(results, environment)
    output = arguments.output
    if output is None:
        stamp = environment.timestamp_utc.replace(":", "").replace("-", "").replace("+", "-")
        host_label = host.architecture
        output = Path("artifacts") / "benchmarks" / f"{stamp}-{host_label}.json"
    payload = report_to_json(environment, results)
    written = write_report(payload, output)
    print(f"report: {written}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
