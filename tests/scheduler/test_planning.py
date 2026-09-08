import numpy as np

from cpuattn import Parallel
from cpuattn.core.validate import validate_parallel_call
from cpuattn.hardware.host import Host, LogicalCpu
from cpuattn.native.backends import select_backend
from cpuattn.schedule.plan import LoweringKind, PackingKind, PlacementKind
from cpuattn.schedule.planning import PlanBuilder


def _dual_numa_host(cores_per_node: int = 4) -> Host:
    cpus = tuple(
        LogicalCpu(cpu, cpu % cores_per_node, cpu // cores_per_node, cpu // cores_per_node)
        for cpu in range(2 * cores_per_node)
    )
    return Host(
        "x86_64", "fixture", "dual", frozenset({"avx2", "fma"}), cpus
    )


def _parallel_call(sequence: int = 8):
    operator = Parallel()
    call = validate_parallel_call(
        operator,
        q=np.zeros((1, 8, sequence, 16), np.float32),
        k=np.zeros((1, 4, 65, 16), np.float32),
        v=np.zeros((1, 4, 65, 16), np.float32),
    )
    return operator, call


def test1() -> None:
    """Parallel validation plans retain every lowering and packing form."""
    host = _dual_numa_host()
    operator, call = _parallel_call()
    codes = select_backend(host).enumerate_code_plans(operator, call, host)
    by_lowering = {
        lowering: {code.packing for code in codes if code.lowering is lowering}
        for lowering in (
            LoweringKind.PARALLEL_BLOCKED,
            LoweringKind.PARALLEL_SPLIT_K,
            LoweringKind.PARALLEL_2D,
        )
    }

    assert all(
        packings == {PackingKind.NONE, PackingKind.K_TRANSPOSED}
        for packings in by_lowering.values()
    )


def test2() -> None:
    """Validation plans preserve packed NUMA ownership and direct layouts."""
    host = _dual_numa_host()
    operator, call = _parallel_call()
    backend = select_backend(host)
    codes = backend.enumerate_code_plans(operator, call, host)
    packed = next(
        code
        for code in codes
        if code.lowering is LoweringKind.PARALLEL_BLOCKED
        and code.packing is PackingKind.K_TRANSPOSED
    )
    plans = PlanBuilder(host, page_size=4096).build_all((packed,), operator, call)
    cross_numa = [plan for plan in plans if plan.launch.crosses_numa]

    assert cross_numa
    for launch_ids in {plan.launch.cpu_ids for plan in cross_numa}:
        matching = [plan for plan in cross_numa if plan.launch.cpu_ids == launch_ids]
        regions = [
            next(region for region in plan.memory.regions if region.name == "packed_k")
            for plan in matching
        ]
        assert {region.owner_groups for region in regions} == {(0,), (1,), (0, 1)}
        assert all(region.placement is PlacementKind.NUMA_LOCAL for region in regions)

    direct = next(
        code
        for code in codes
        if code.lowering is LoweringKind.PARALLEL_BLOCKED
        and code.packing is PackingKind.NONE
    )
    direct_plans = PlanBuilder(host, page_size=4096).build_all(
        (direct,), operator, call
    )
    assert direct_plans
    assert all(
        tuple(region.name for region in plan.memory.regions) == ("scratch",)
        for plan in direct_plans
    )


def test3() -> None:
    """Planning exposes every legal plan without worker-policy preselection."""
    host = _dual_numa_host(10)
    operator, call = _parallel_call(64)
    backend = select_backend(host)
    codes = backend.enumerate_code_plans(operator, call, host)
    builder = PlanBuilder(host, page_size=4096)

    first = builder.build_all(codes, operator, call)
    second = builder.build_all(tuple(reversed(codes)), operator, call)
    workers = {plan.launch.workers for plan in first}

    assert len(first) > len(workers)
    assert {1, 2, 4, 8, 10, 16, 20} <= workers
    assert tuple(plan.identity for plan in first) == tuple(
        plan.identity for plan in second
    )
    assert all(
        any(
            plan.launch.workers == workers and not plan.launch.crosses_numa
            for plan in first
        )
        for workers in {1, 2, 4, 8, 10}
    )
    assert any(
        plan.launch.workers <= 10 and plan.launch.crosses_numa for plan in first
    )
    assert all(
        plan.launch.crosses_numa for plan in first if plan.launch.workers > 10
    )
