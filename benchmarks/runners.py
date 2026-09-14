from __future__ import annotations

import functools
import tempfile
import time
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from cpuattn import Linear, Parallel, Runtime, diagnostics, expr, transition
from cpuattn.core.tensor import Axis, TensorArgSpec
from cpuattn.core.validate import ValidatedCall, validate_parallel_call
from cpuattn.schedule.plan import ExecutionPlan

from .baselines import (
    attention_numpy,
    attention_torch,
    torch,
    kda_numpy,
    linear_numpy,
    torch_available,
)
from .harness import collect, collect_with_probe, summarize, timed_ns
from .results import BaselineResult, CaseResult, ThreadLevel
from .workloads import AttentionWorkload, LinearWorkload, ThreadScalingWorkload

_BASELINE_WARMUP = 1
_BASELINE_MAX_RUNS = 15
_NUMERIC_TOLERANCE = 5e-3


def _rng_for(name: str) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32(name.encode("utf-8")))


def _attention_operator(d: int, causal: bool) -> Parallel:
    return Parallel(
        score_mod=expr.identity() * (1.0 / d**0.5),
        mask_mod=expr.causal() if causal else expr.as_expr(True),
    )


def _attention_params(workload: AttentionWorkload | ThreadScalingWorkload) -> dict[str, object]:
    """Shape facts shared by both attention workload types."""
    return {
        "b": workload.b,
        "hq": workload.hq,
        "hkv": workload.hkv,
        "sq": workload.sq,
        "skv": workload.skv,
        "d": workload.d,
        "dv": workload.dv,
        "causal": workload.causal,
    }


def _plan_name(runtime: Runtime) -> str:
    selection = runtime.last_selection
    assert selection is not None
    return diagnostics.plan_label(selection.winner.plan)


def _winner_native_ns(runtime: Runtime) -> int:
    selection = runtime.last_selection
    assert selection is not None
    return selection.winner.latency_ns


def _diagnostics(runtime: Runtime, max_abs_err: float | None) -> dict[str, object] | None:
    """Embed the runtime's structured record, with the harness's numeric check."""
    record = runtime.explain()
    if record is None:
        return None
    if max_abs_err is not None:
        record = record.with_numeric(
            {
                "max_abs_err": max_abs_err,
                "tolerance": _NUMERIC_TOLERANCE,
                "numeric_ok": max_abs_err <= _NUMERIC_TOLERANCE,
            }
        )
    return record.as_json()


def _baseline(
    backend: str,
    samples: list[int],
    max_abs_err: float | None,
    ours_median_ns: float,
) -> BaselineResult:
    stats = summarize(samples)
    return BaselineResult(
        backend=backend,
        wall=stats,
        max_abs_err=max_abs_err,
        speedup=stats.median / max(ours_median_ns, 1.0),
    )


def _torch_baseline(
    call: Callable[[], np.ndarray],
    *,
    ours_median_ns: float,
    runs: int,
    worker_counts: tuple[int, int],
) -> BaselineResult:
    """Best torch SDPA across CPUAttn's own worker count and the full core set.

    Threads are configured outside the timed region and restored afterwards.
    """
    original_threads = torch.get_num_threads()
    best_samples: list[int] = []
    best_threads = 0
    try:
        for threads in sorted(set(worker_counts)):
            torch.set_num_threads(threads)
            samples = collect(call, warmup=_BASELINE_WARMUP, runs=runs)
            if not best_samples or summarize(samples).median < summarize(best_samples).median:
                best_samples, best_threads = samples, threads
    finally:
        torch.set_num_threads(original_threads)
    return _baseline(f"torch_sdpa@{best_threads}t", best_samples, None, ours_median_ns)


def _attention_baselines(
    numpy_call: Callable[[], np.ndarray],
    torch_call: Callable[[int], np.ndarray] | None,
    *,
    ours_median_ns: float,
    runs: int,
    workers: int,
    cores: int,
    max_abs_err: float,
) -> tuple[BaselineResult, ...]:
    """NumPy/BLAS always; torch SDPA only when requested and importable."""
    baselines = [
        _baseline(
            "numpy_blas",
            collect(numpy_call, warmup=_BASELINE_WARMUP, runs=runs),
            max_abs_err,
            ours_median_ns,
        )
    ]
    if torch_call is not None and torch_available():
        baselines.append(
            _torch_baseline(
                torch_call,
                ours_median_ns=ours_median_ns,
                runs=runs,
                worker_counts=(workers, cores),
            )
        )
    return tuple(baselines)


def run_attention_case(
    workload: AttentionWorkload,
    *,
    warmup: int,
    runs: int,
    stream_warmup: int,
    with_torch: bool,
) -> CaseResult:
    if workload.kv_cache:
        return _run_decode_stream(workload, warmup=stream_warmup, with_torch=with_torch)
    return _run_prefill(workload, warmup=warmup, runs=runs, with_torch=with_torch)


def _run_prefill(
    workload: AttentionWorkload, *, warmup: int, runs: int, with_torch: bool
) -> CaseResult:
    rng = _rng_for(workload.name)
    operator = _attention_operator(workload.d, workload.causal)
    scale = 1.0 / workload.d**0.5
    b, hq, hkv, sq, skv, d, dv = workload.shape()
    q = rng.normal(size=(b, hq, sq, d)).astype(np.float32)
    k = rng.normal(size=(b, hkv, skv, d)).astype(np.float32)
    v = rng.normal(size=(b, hkv, skv, dv)).astype(np.float32)
    with tempfile.TemporaryDirectory(prefix="cpuattn-bench-") as cache_dir:
        runtime = Runtime(cache_dir=cache_dir)
        first_call_ms = timed_ns(lambda: runtime.run(operator, q=q, k=k, v=v)) / 1e6
        expected = attention_numpy(q, k, v, causal=workload.causal, scale=scale)
        max_abs_err = float(np.abs(runtime.run(operator, q=q, k=k, v=v) - expected).max())
        wall_samples, native_samples = collect_with_probe(
            lambda: runtime.run(operator, q=q, k=k, v=v),
            lambda result: _winner_native_ns(runtime),
            warmup=warmup,
            runs=runs,
        )
        ours_median = summarize(wall_samples).median
        baselines = _attention_baselines(
            lambda: attention_numpy(q, k, v, causal=workload.causal, scale=scale),
            (
                lambda: attention_torch(
                    q, k, v, causal=workload.causal, query_offset=0
                )
            )
            if with_torch
            else None,
            ours_median_ns=ours_median,
            runs=runs,
            workers=runtime.last_selection.winner.plan.launch.workers,
            cores=runtime.host.physical_cores,
            max_abs_err=max_abs_err,
        )
        return CaseResult(
            kind="attention_prefill",
            name=workload.name,
            params=_attention_params(workload),
            plan=_plan_name(runtime),
            first_call_ms=first_call_ms,
            wall=summarize(wall_samples),
            native=summarize(native_samples),
            baselines=baselines,
            max_abs_err=max_abs_err,
            numeric_ok=max_abs_err <= _NUMERIC_TOLERANCE,
            diagnostics=_diagnostics(runtime, max_abs_err),
        )


class _StreamStep(NamedTuple):
    """One decode-stream step's latency record."""

    skv: int
    wall_ns: int
    native_ns: int
    mode: str


def _run_decode_stream(
    workload: AttentionWorkload, *, warmup: int, with_torch: bool
) -> CaseResult:
    rng = _rng_for(workload.name)
    operator = _attention_operator(workload.d, causal=True)
    scale = 1.0 / workload.d**0.5
    total = warmup + workload.stream_steps
    final_skv = workload.start_skv + total
    b, hq, hkv, d, dv = workload.b, workload.hq, workload.hkv, workload.d, workload.dv
    k_buffer = rng.normal(size=(b, hkv, final_skv, d)).astype(np.float32)
    v_buffer = rng.normal(size=(b, hkv, final_skv, dv)).astype(np.float32)
    q_tokens = rng.normal(size=(total, b, hq, 1, d)).astype(np.float32)
    with tempfile.TemporaryDirectory(prefix="cpuattn-bench-") as cache_dir:
        runtime = Runtime(cache_dir=cache_dir)
        records: list[_StreamStep] = []
        for step in range(total):
            skv = workload.start_skv + step + 1
            q = q_tokens[step]
            started = time.perf_counter_ns()
            output = runtime.run(
                operator, q=q, k=k_buffer[:, :, :skv], v=v_buffer[:, :, :skv], kv_cache=True
            )
            records.append(
                _StreamStep(
                    skv=skv,
                    wall_ns=time.perf_counter_ns() - started,
                    native_ns=_winner_native_ns(runtime),
                    mode=runtime.last_selection.mode,
                )
            )
        first_steady_skv = workload.start_skv + warmup + 1
        steady = [step for step in records if step.mode != "tune" and step.skv >= first_steady_skv]
        tune_steps = sum(1 for step in records if step.mode == "tune")
        expected = attention_numpy(
            q_tokens[-1],
            k_buffer[:, :, :final_skv],
            v_buffer[:, :, :final_skv],
            causal=True,
            scale=scale,
            query_offset=final_skv - 1,
        )
        max_abs_err = float(np.abs(output - expected).max())
        params = _attention_params(workload)
        del params["skv"]
        params.update({"stream": True, "start_skv": workload.start_skv, "steps": workload.stream_steps})
        if not steady:
            return CaseResult(
                kind="attention_decode",
                name=workload.name,
                params=params,
                plan=_plan_name(runtime),
                first_call_ms=records[0].wall_ns / 1e6,
                max_abs_err=max_abs_err,
                numeric_ok=max_abs_err <= _NUMERIC_TOLERANCE,
                diagnostics=_diagnostics(runtime, max_abs_err),
                tune_steps=tune_steps,
                final_skv=final_skv,
            )
        walls = [step.wall_ns for step in steady]
        natives = [step.native_ns for step in steady]
        # Baselines priced at the median stream position, matching steady steps.
        median_step = warmup + workload.stream_steps // 2
        median_skv = workload.start_skv + median_step
        baselines = _attention_baselines(
            lambda: attention_numpy(
                q_tokens[median_step - 1],
                k_buffer[:, :, :median_skv],
                v_buffer[:, :, :median_skv],
                causal=True,
                scale=scale,
                query_offset=median_skv - 1,
            ),
            (
                lambda: attention_torch(
                    q_tokens[median_step - 1],
                    k_buffer[:, :, :median_skv],
                    v_buffer[:, :, :median_skv],
                    causal=True,
                    query_offset=median_skv - 1,
                )
            )
            if with_torch
            else None,
            ours_median_ns=summarize(walls).median,
            runs=min(len(walls), _BASELINE_MAX_RUNS),
            workers=runtime.last_selection.winner.plan.launch.workers,
            cores=runtime.host.physical_cores,
            max_abs_err=max_abs_err,
        )
        return CaseResult(
            kind="attention_decode",
            name=workload.name,
            params=params,
            plan=_plan_name(runtime),
            first_call_ms=records[0].wall_ns / 1e6,
            wall=summarize(walls),
            native=summarize(natives),
            baselines=baselines,
            max_abs_err=max_abs_err,
            numeric_ok=max_abs_err <= _NUMERIC_TOLERANCE,
            diagnostics=_diagnostics(runtime, max_abs_err),
            tune_steps=tune_steps,
            final_skv=final_skv,
        )


@dataclass(frozen=True, slots=True)
class _LinearVariant:
    """A linear recurrence: operator plus its fixed arguments and baseline."""

    operator: Linear
    arguments: dict[str, np.ndarray] | None
    baseline: Callable[[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
    normalize_keys: bool = False


def _linear_variant(workload: LinearWorkload) -> _LinearVariant:
    b, groups, heads, sequence, d, dv = workload.shape()
    argument_rng = _rng_for(workload.name + "-args")
    if workload.kind == "standard":
        operator = Linear(
            transition=transition.program(
                transition.scale(0.95),
                transition.outer(expr.var("k"), expr.var("v")),
            ),
            readout=transition.readout(timing="after"),
        )
        return _LinearVariant(
            operator, None, lambda q, k, v: linear_numpy(q, k, v, decay=0.95)
        )
    if workload.kind == "mamba2":
        decay = argument_rng.uniform(0.85, 0.99, size=(b, sequence)).astype(np.float32)
        operator = Linear(
            transition=transition.program(
                transition.scale(expr.argument("a")),
                transition.outer(expr.var("k"), expr.var("v")),
            ),
            readout=transition.readout(timing="after"),
            arguments=(TensorArgSpec("a", (Axis.BATCH, Axis.SEQUENCE)),),
        )
        return _LinearVariant(
            operator,
            {"a": decay},
            lambda q, k, v: linear_numpy(q, k, v, decay=decay),
        )
    if workload.kind == "kda":
        assert groups == heads, "the KDA baseline indexes K per state head"
        gate = argument_rng.uniform(0.9, 1.0, size=(b, sequence)).astype(np.float32)
        beta = argument_rng.uniform(0.05, 0.95, size=(b, sequence)).astype(np.float32)
        operator = Linear(
            transition=transition.program(
                transition.scale(expr.argument("gate")),
                transition.rank1(
                    -expr.argument("beta") * expr.var("k"), expr.var("k")
                ),
                transition.outer(
                    expr.var("k"), expr.argument("beta") * expr.var("v")
                ),
            ),
            readout=transition.readout(timing="after"),
            arguments=(
                TensorArgSpec("gate", (Axis.BATCH, Axis.SEQUENCE)),
                TensorArgSpec("beta", (Axis.BATCH, Axis.SEQUENCE)),
            ),
        )
        return _LinearVariant(
            operator,
            {"gate": gate, "beta": beta},
            lambda q, k, v: kda_numpy(q, k, v, gate=gate, beta=beta),
            normalize_keys=True,
        )
    raise ValueError(f"unknown linear workload kind {workload.kind!r}")


def _linear_tensors(
    rng: np.random.Generator, workload: LinearWorkload
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    b, groups, heads, sequence, d, dv = workload.shape()
    q = rng.normal(size=(b, groups, sequence, d)).astype(np.float32)
    k = rng.normal(size=(b, groups, sequence, d)).astype(np.float32)
    v = rng.normal(size=(b, heads, sequence, dv)).astype(np.float32)
    return q, k, v


def run_linear_case(workload: LinearWorkload, *, warmup: int, runs: int) -> CaseResult:
    rng = _rng_for(workload.name)
    variant = _linear_variant(workload)
    q, k, v = _linear_tensors(rng, workload)
    if variant.normalize_keys:
        # Delta-rule stability requires unit-norm keys, as in real KDA
        # projections; the operator itself cannot express a norm.
        k = (k / np.linalg.norm(k, axis=-1, keepdims=True)).astype(np.float32)
    with tempfile.TemporaryDirectory(prefix="cpuattn-bench-") as cache_dir:
        runtime = Runtime(cache_dir=cache_dir)

        def run() -> object:
            return runtime.run(
                operator=variant.operator, q=q, k=k, v=v, arguments=variant.arguments
            )

        first_call_ms = timed_ns(run) / 1e6
        expected, _ = variant.baseline(q, k, v)
        max_abs_err = float(np.abs(run().output - expected).max())
        wall_samples, native_samples = collect_with_probe(
            run,
            lambda result: _winner_native_ns(runtime),
            warmup=warmup,
            runs=runs,
        )
        baseline = _baseline(
            "numpy_blas",
            collect(
                lambda: variant.baseline(q, k, v),
                warmup=_BASELINE_WARMUP,
                runs=runs,
            ),
            max_abs_err,
            summarize(wall_samples).median,
        )
        return CaseResult(
            kind=f"linear_{workload.kind}",
            name=workload.name,
            params={
                "b": workload.b,
                "groups": workload.groups,
                "heads": workload.heads,
                "sequence": workload.sequence,
                "d": workload.d,
                "dv": workload.dv,
            },
            plan=_plan_name(runtime),
            first_call_ms=first_call_ms,
            wall=summarize(wall_samples),
            native=summarize(native_samples),
            baselines=(baseline,),
            max_abs_err=max_abs_err,
            numeric_ok=max_abs_err <= _NUMERIC_TOLERANCE,
            diagnostics=_diagnostics(runtime, max_abs_err),
        )


def run_thread_scaling_case(
    workload: ThreadScalingWorkload, *, warmup: int, runs: int
) -> CaseResult:
    rng = _rng_for(workload.name)
    operator = _attention_operator(workload.d, workload.causal)
    q = rng.normal(size=(workload.b, workload.hq, workload.sq, workload.d)).astype(np.float32)
    k = rng.normal(size=(workload.b, workload.hkv, workload.skv, workload.d)).astype(np.float32)
    v = rng.normal(size=(workload.b, workload.hkv, workload.skv, workload.dv)).astype(np.float32)
    with tempfile.TemporaryDirectory(prefix="cpuattn-bench-") as cache_dir:
        runtime = Runtime(cache_dir=cache_dir)
        runtime.run(operator, q=q, k=k, v=v)
        call = validate_parallel_call(operator, q=q, k=k, v=v)
        # Thread scaling deliberately enumerates every legal plan so each worker
        # count can be forced and priced.
        plans = runtime.planner.build_all(
            runtime.backend.enumerate_code_plans(operator, call, runtime.host),
            operator,
            call,
        )
        levels: list[ThreadLevel] = []
        for workers in workload.workers:
            candidates = [plan for plan in plans if plan.launch.workers == workers]
            if not candidates:
                continue
            levels.append(
                _best_plan_level(
                    runtime, operator, call, workers, candidates, warmup, runs
                )
            )
        return CaseResult(
            kind="thread_scaling",
            name=workload.name,
            params=_attention_params(workload),
            plan=_plan_name(runtime),
            thread_levels=tuple(levels),
            diagnostics=_diagnostics(runtime, None),
        )


def _best_plan_level(
    runtime: Runtime,
    operator: Parallel,
    call: ValidatedCall,
    workers: int,
    candidates: list[ExecutionPlan],
    warmup: int,
    runs: int,
) -> ThreadLevel:
    """Best native median among all plans sharing one worker count."""
    best: ThreadLevel | None = None
    for plan in candidates:
        code = plan.code
        label = f"{code.lowering.value}/{code.packing.value}"
        kernel = runtime.compiler.load(
            runtime.compiler.compile(operator, call, code, runtime.backend)
        )
        runtime.executor.prepare_launch(kernel, plan.launch)
        walls, natives = collect_with_probe(
            functools.partial(runtime.executor.run, kernel, plan, call),
            lambda timed: timed.latency_ns,
            warmup=warmup,
            runs=runs,
        )
        level = ThreadLevel(
            workers=workers,
            native_median_ms=summarize(natives).median / 1e6,
            wall_median_ms=summarize(walls).median / 1e6,
            plan=label,
        )
        if best is None or level.native_median_ms < best.native_median_ms:
            best = level
    assert best is not None
    return best
