from dataclasses import dataclass, replace

import numpy as np
import pytest

from cpuattn import Linear, Parallel, expr, rownorm, transition
from cpuattn.core.validate import validate_linear_call, validate_parallel_call
from cpuattn.hardware.host import Host, LogicalCpu
from cpuattn.native.backends.x86 import X86_AVX2
from cpuattn.schedule.plan import (
    CodePlan,
    DecompositionKind,
    ExecutionPlan,
    FusionKind,
    LaunchPlan,
    LoweringKind,
    MemoryPlan,
    MemoryRegion,
    MergeKind,
    MicrokernelSpec,
    PackingKind,
    PlacementKind,
    TileShape,
    WorkerGroup,
    WorkspaceABI,
)
from cpuattn.schedule.planning import PlanBuilder
from cpuattn.tuning import (
    SingleCoreTuner,
    Tuner,
    TuningContext,
    WorkerCountTuner,
)


def _host(architecture: str = "x86_64", workers: int = 8) -> Host:
    features = {"avx2", "fma"} if architecture == "x86_64" else {"asimd"}
    return Host(
        architecture,
        "fixture",
        "fixture",
        frozenset(features),
        tuple(LogicalCpu(cpu, cpu, 0, 0) for cpu in range(workers)),
    )


def _code(
    tile_q: int = 4,
    packing: PackingKind = PackingKind.NONE,
    tile_dv: int = 8,
) -> CodePlan:
    regions = (
        ("scratch",)
        if packing is PackingKind.NONE
        else ("packed_k", "scratch")
    )
    return CodePlan(
        "fake",
        LoweringKind.PARALLEL_BLOCKED,
        MicrokernelSpec(f"test_m{tile_q}", 2),
        TileShape(tile_q, 16, 8, tile_dv),
        32,
        DecompositionKind.QUERY_BLOCKS,
        packing,
        MergeKind.NONE,
        FusionKind.PATTERN_OWNED,
        WorkspaceABI(regions),
    )


def _plan(
    workers: int,
    *,
    tile_q: int = 4,
    packing: PackingKind = PackingKind.NONE,
    cpu_id: int | None = None,
    tile_dv: int = 8,
) -> ExecutionPlan:
    cpus = (
        (cpu_id,)
        if cpu_id is not None
        else tuple(range(workers))
    )
    return ExecutionPlan(
        _code(tile_q, packing, tile_dv),
        LaunchPlan(cpus, (WorkerGroup(0, cpus),)),
        MemoryPlan(
            (
                MemoryRegion(
                    "scratch",
                    0,
                    workers * 64,
                    64,
                    PlacementKind.WORKER_LOCAL,
                    (0,),
                ),
            ),
            workers * 64,
        ),
    )


def _context(
    plans: tuple[ExecutionPlan, ...],
    *,
    architecture: str = "x86_64",
    pattern: str = "parallel",
) -> TuningContext:
    host_workers = max(
        cpu_id for plan in plans for cpu_id in plan.launch.cpu_ids
    ) + 1
    operator = (
        Parallel().canonical()
        if pattern == "parallel"
        else Linear(
            transition=transition.program(
                transition.outer(expr.var("k"), expr.var("v"))
            )
        ).canonical()
    )
    return TuningContext(
        host=_host(architecture, host_workers),
        operator=operator,
        workload={
            "pattern": pattern,
            "shape": {
                "batch": 1,
                "query": 32,
                "key": 32,
                "query_head": 2,
                "kv_head": 1,
                "d": 8,
                "dv": 8,
            },
            "kv_cache": False,
            "dtype": "fp32",
            "layout": "bhsd-contiguous",
        },
        candidates=plans,
    )


@dataclass(frozen=True, slots=True)
class OrderedTuner(Tuner):
    arch = "x86_64"
    pattern = "parallel"
    version = 3

    reverse: bool = False

    def choose_next(self, context, history):
        measured = {item.plan.identity for item in history}
        candidates = (
            reversed(context.candidates)
            if self.reverse
            else context.candidates
        )
        return next(
            (plan for plan in candidates if plan.identity not in measured), None
        )


@dataclass(frozen=True, slots=True)
class RemeasureTuner(Tuner):
    def choose_next(self, context, history):
        if not history:
            return context.candidates[0]
        if len(history[0].samples_ns) < 2:
            return context.candidates[0]
        return None


def test1() -> None:
    """The base class owns budget, repeat, history, and winner selection."""
    plans = tuple(_plan(workers) for workers in range(1, 5))
    samples = {
        plans[0].identity: iter((90, 80, 100)),
        plans[1].identity: iter((50, 60, 40)),
        plans[2].identity: iter((1, 1, 1)),
        plans[3].identity: iter((1, 1, 1)),
    }
    calls: list[str] = []

    def measure(plan):
        calls.append(plan.identity)
        return next(samples[plan.identity])

    result = OrderedTuner(maxnum=2, repeat=3).select(_context(plans), measure)

    assert result.winner == plans[1]
    assert calls == [plans[0].identity] * 3 + [plans[1].identity] * 3
    assert tuple(item.latency_ns for item in result.measurements) == (90, 50)
    assert tuple(item.samples_ns for item in result.measurements) == (
        (90, 80, 100),
        (50, 60, 40),
    )

    values = iter((20, 10))
    repeated = RemeasureTuner(repeat=1).select(
        _context((plans[0],)), lambda plan: next(values)
    )
    assert repeated.measurements[0].samples_ns == (20, 10)
    assert repeated.measurements[0].latency_ns == 15


def test2() -> None:
    """Compatibility, legal-plan membership, and cache identity are enforced."""
    plans = (_plan(1), _plan(2))
    tuner = OrderedTuner(maxnum=None, repeat=1, reverse=True)
    identity = tuner.cache_identity()

    assert identity["class"].endswith("OrderedTuner")
    assert identity["arch"] == "x86_64"
    assert identity["pattern"] == "parallel"
    assert identity["version"] == 3
    assert identity["config"] == {
        "maxnum": None,
        "repeat": 1,
        "reverse": True,
    }
    assert OrderedTuner(reverse=False).cache_identity() != identity

    with pytest.raises(ValueError, match="architecture"):
        tuner.select(_context(plans, architecture="aarch64"), lambda plan: 1)
    with pytest.raises(ValueError, match="pattern"):
        tuner.select(_context(plans, pattern="linear"), lambda plan: 1)

    @dataclass(frozen=True, slots=True)
    class ForeignPlanTuner(Tuner):
        def choose_next(self, context, history):
            return replace(context.candidates[0], code=_code(6))

    with pytest.raises(ValueError, match="legal candidate"):
        ForeignPlanTuner().select(_context(plans), lambda plan: 1)


def test3() -> None:
    """WorkerCountTuner selects deterministic representatives from full plans."""
    plans = tuple(
        plan
        for workers in range(1, 9)
        for plan in (_plan(workers, tile_q=2), _plan(workers, tile_q=4))
    )
    calls: list[ExecutionPlan] = []

    def measure(plan):
        calls.append(plan)
        return 100 - plan.launch.workers

    result = WorkerCountTuner().select(_context(tuple(reversed(plans))), measure)

    assert len(calls) == 6
    assert len({plan.launch.workers for plan in calls}) == 6
    assert all(plan.code.tile.q == 4 for plan in calls)
    assert result.winner.launch.workers == max(
        plan.launch.workers for plan in calls
    )

    all_calls: list[ExecutionPlan] = []
    WorkerCountTuner(maxnum=None).select(
        _context(plans), lambda plan: all_calls.append(plan) or 1
    )
    assert {plan.launch.workers for plan in all_calls} == set(range(1, 9))


def test4() -> None:
    """SingleCoreTuner predicts a single-core shortlist, then measures it."""
    plans = tuple(
        _plan(1, tile_q=tile, packing=packing, cpu_id=cpu)
        for cpu in (0, 4)
        for packing in (PackingKind.NONE, PackingKind.K_TRANSPOSED)
        for tile in (2, 4, 6)
    ) + (
        _plan(
            1,
            tile_q=4,
            packing=PackingKind.K_TRANSPOSED,
            cpu_id=4,
            tile_dv=16,
        ),
        _plan(2),
    )
    calls: list[ExecutionPlan] = []

    def measure(plan):
        calls.append(plan)
        packed = plan.code.packing is PackingKind.K_TRANSPOSED
        return (10 if packed else 100) + abs(plan.code.tile.q - 4)

    context = _context(tuple(reversed(plans)))
    tuner = SingleCoreTuner(maxnum=4)
    predictions = tuner.predictions(context)
    result = tuner.select(
        context,
        measure,
    )

    assert len(calls) == 4
    assert all(plan.launch.cpu_ids == (0,) for plan in calls)
    assert [plan.identity for plan in calls] == [
        item.plan.identity for item in predictions[:4]
    ]
    assert all(item.predicted_cycles > 0 for item in predictions)
    assert all(item.arithmetic_intensity > 0 for item in predictions)
    assert list(predictions) == sorted(
        predictions,
        key=lambda item: (item.predicted_cycles, item.plan.identity),
    )
    assert result.winner.code.packing is PackingKind.K_TRANSPOSED

    default_calls: list[ExecutionPlan] = []
    SingleCoreTuner().select(
        context,
        lambda plan: default_calls.append(plan) or 1,
    )
    assert len(default_calls) == 3

    exhaustive: list[ExecutionPlan] = []
    SingleCoreTuner(maxnum=None).select(
        _context(plans),
        lambda plan: exhaustive.append(plan) or 1,
    )
    assert len(exhaustive) == 6
    assert all(plan.launch.cpu_ids == (0,) for plan in exhaustive)

def test5() -> None:
    """Single-core estimates expose roofline work for Parallel and Linear."""
    host = _host(workers=1)
    operator = Linear(
        transition=transition.program(
            transition.scale(0.95),
            transition.outer(expr.var("k"), expr.var("v")),
        )
    )
    q = np.ones((1, 1, 32, 16), dtype=np.float32)
    v = np.ones((1, 1, 32, 16), dtype=np.float32)
    call = validate_linear_call(operator, q=q, k=q, v=v)
    codes = X86_AVX2.enumerate_code_plans(operator, call, host)
    plans = PlanBuilder(host, page_size=4096).build_all(codes, operator, call)
    context = TuningContext(host, operator.canonical(), call.canonical(), plans)

    predictions = SingleCoreTuner().predictions(context)

    assert len(predictions) == 6
    assert {item.plan.code.lowering for item in predictions} == {
        LoweringKind.LINEAR_SCAN,
        LoweringKind.LINEAR_CHUNKED,
    }
    assert all(item.plan.launch.cpu_ids == (0,) for item in predictions)

    parallel_input = np.ones((1, 1, 4, 16), dtype=np.float32)
    parallel = Parallel(row_norm=rownorm.identity())
    validated = validate_parallel_call(
        parallel, q=parallel_input, k=parallel_input, v=parallel_input
    )
    code_plans = X86_AVX2.enumerate_code_plans(parallel, validated, host)
    execution_plans = PlanBuilder(host, page_size=4096).build_all(
        code_plans, parallel, validated
    )
    parallel_predictions = SingleCoreTuner().predictions(
        TuningContext(
            host,
            parallel.canonical(),
            validated.canonical(),
            execution_plans,
        )
    )
    assert all(item.memory_bytes > 0 for item in parallel_predictions)
    assert all(item.working_set_bytes > 0 for item in parallel_predictions)


def test6() -> None:
    """Packed key padding is charged as work, not as a second tail penalty."""
    host = _host(workers=1)
    operator = Parallel(row_norm=rownorm.identity())
    q = np.ones((1, 1, 2, 16), dtype=np.float32)

    def packed_compute_cycles(key_size: int) -> float:
        k = np.ones((1, 1, key_size, 16), dtype=np.float32)
        v = np.ones((1, 1, key_size, 8), dtype=np.float32)
        call = validate_parallel_call(operator, q=q, k=k, v=v)
        codes = X86_AVX2.enumerate_code_plans(operator, call, host)
        plans = PlanBuilder(host, page_size=4096).build_all(
            codes, operator, call
        )
        context = TuningContext(
            host, operator.canonical(), call.canonical(), plans
        )
        estimate = next(
            item
            for item in SingleCoreTuner().predictions(context)
            if item.plan.code.packing is PackingKind.K_TRANSPOSED
            and item.plan.code.tile.q == 4
            and item.plan.code.tile.k == 16
            and item.plan.code.tile.dv == 8
        )
        return estimate.compute_cycles

    assert packed_compute_cycles(9) <= packed_compute_cycles(16)
