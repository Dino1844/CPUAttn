from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from ..schedule.plan import ExecutionPlan, LoweringKind, PackingKind, ScheduleKind
from .tuner import Measurement, Tuner, TuningContext, _shape


@dataclass(frozen=True, slots=True)
class WorkerCountTuner(Tuner):
    """Measure one statically chosen plan for each selected worker count."""

    maxnum: int | None = 6
    version = 7

    def choose_next(
        self,
        context: TuningContext,
        history: tuple[Measurement, ...],
    ) -> ExecutionPlan | None:
        measured = {item.plan.identity for item in history}
        return next(
            (
                plan
                for plan in _representatives(context, self.maxnum)
                if plan.identity not in measured
            ),
            None,
        )


def _representatives(
    context: TuningContext,
    limit: int | None,
) -> tuple[ExecutionPlan, ...]:
    by_key: dict[tuple[int, PackingKind], list[ExecutionPlan]] = {}
    for plan in context.candidates:
        by_key.setdefault((plan.launch.workers, plan.code.packing), []).append(plan)
    anchors = _worker_anchors({workers for workers, _ in by_key}, context, limit)
    representatives: list[ExecutionPlan] = []
    for workers in anchors:
        for packing in PackingKind:
            group = by_key.get((workers, packing))
            if group:
                representatives.append(
                    min(
                        group,
                        key=lambda plan: _plan_preference(plan, context),
                    )
                )
    return tuple(representatives)


def _worker_anchors(
    legal_workers: Iterable[int],
    context: TuningContext,
    limit: int | None,
) -> tuple[int, ...]:
    legal = set(legal_workers)
    if not legal:
        return ()
    if limit is None or len(legal) <= limit:
        return tuple(sorted(legal, reverse=True))

    physical_by_node: dict[int, set[tuple[int, int]]] = {}
    for cpu in context.host.cpus:
        physical_by_node.setdefault(cpu.numa_node, set()).add(
            (cpu.socket_id, cpu.core_id)
        )
    topology = {max(legal)}
    topology.update(len(cores) for cores in physical_by_node.values())
    priority = sorted(topology & legal, reverse=True)
    priority.extend(
        workers
        for workers in sorted(legal, reverse=True)
        if _is_power_of_two(workers) and workers not in topology
    )
    priority.extend(
        workers for workers in sorted(legal, reverse=True) if workers not in priority
    )
    return tuple(priority[:limit])


def _plan_preference(
    plan: ExecutionPlan,
    context: TuningContext,
) -> tuple[object, ...]:
    code = plan.code
    lanes = code.vector_bytes // 4
    shape = _shape(context.workload)
    if context.pattern == "parallel":
        lowering = {
            LoweringKind.PARALLEL_BLOCKED: 0,
            LoweringKind.PARALLEL_2D: 1,
            LoweringKind.PARALLEL_SPLIT_K: 2,
        }[code.lowering]
        query_reuse = shape["query"] * (
            shape["query_head"] // shape["kv_head"]
        )
        prefer_packed = bool(context.workload.get("kv_cache")) and query_reuse >= 32
        # Long causal queries have a per-item cost ramp; measuring the dynamic
        # representative lets the tuner amortize the ramp against scheduling cost.
        prefer_dynamic = shape["query"] >= 256
        code_key = (
            lowering,
            (code.schedule is ScheduleKind.DYNAMIC) != prefer_dynamic,
            (code.packing is PackingKind.K_TRANSPOSED) != prefer_packed,
            abs(code.tile.q - (8 if lanes >= 16 else 4)),
            abs(code.microkernel.qk_vectors - 2),
            abs(code.tile.k - ((4 if lanes >= 16 else 2) * lanes)),
            abs(code.tile.dv - ((2 if lanes >= 16 else 1) * lanes)),
        )
    else:
        lowering = {
            LoweringKind.LINEAR_CHUNKED: 0,
            LoweringKind.LINEAR_DELTA: 0,
            LoweringKind.LINEAR_SCAN: 1,
            LoweringKind.LINEAR_2D: 2,
        }[code.lowering]
        # The delta block amortizes a triangular solve, so it prefers the
        # largest block; the plain chunked form prefers a small block.
        block_key = (
            -code.tile.q
            if code.lowering is LoweringKind.LINEAR_DELTA
            else abs(code.tile.q - min(4, shape["sequence"]))
        )
        code_key = (
            lowering,
            block_key,
            abs(code.tile.d - lanes),
            abs(code.tile.dv - 2 * lanes),
        )

    packed_region = next(
        (region for region in plan.memory.regions if region.name == "packed_k"),
        None,
    )
    replicas = 0 if packed_region is None else packed_region.replica_count
    placement_key = (
        plan.launch.crosses_numa,
        -replicas,
        plan.memory.total_bytes,
        plan.launch.cpu_ids,
    )
    return (*code_key, *placement_key, plan.identity)


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


__all__ = ["WorkerCountTuner"]
