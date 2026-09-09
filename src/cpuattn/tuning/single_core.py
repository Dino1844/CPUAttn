from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ..hardware.host import Host
from ..schedule.plan import ExecutionPlan, LoweringKind, PackingKind
from .tuner import Measurement, Tuner, TuningContext, _shape


_FP32_BYTES = 4


@dataclass(frozen=True, slots=True)
class SingleCoreEstimate:
    """Small roofline report used to explain and order a tile search."""

    plan: ExecutionPlan
    predicted_cycles: float
    compute_cycles: float
    data_cycles: float
    packing_cycles: float
    overhead_cycles: float
    memory_bytes: int
    working_set_bytes: int
    arithmetic_intensity: float


@dataclass(frozen=True, slots=True)
class _Work:
    flops: int
    memory_bytes: int
    packing_elements: int
    tile_visits: int
    working_set_bytes: int
    fma_chains: int
    register_vectors: int


@dataclass(frozen=True, slots=True)
class SingleCoreTuner(Tuner):
    """Predict a small tile shortlist on one physical CPU, then measure it."""

    maxnum: int | None = 3

    def choose_next(
        self,
        context: TuningContext,
        history: tuple[Measurement, ...],
    ) -> ExecutionPlan | None:
        measured = {item.plan.identity for item in history}
        return next(
            (
                estimate.plan
                for estimate in self.predictions(context)
                if estimate.plan.identity not in measured
            ),
            None,
        )

    def predictions(
        self,
        context: TuningContext,
    ) -> tuple[SingleCoreEstimate, ...]:
        """Rank legal one-worker plans without compiling or running them."""
        self._validate_context(context)
        plans = tuple(
            plan for plan in context.candidates if plan.launch.workers == 1
        )
        if not plans:
            raise ValueError("SingleCoreTuner requires a one-worker plan")

        available = {plan.launch.cpu_ids[0] for plan in plans}
        cpu_id = next(
            (cpu for cpu in context.host.allowed_cpu_ids if cpu in available),
            min(available),
        )
        fixed_cpu_plans = tuple(
            plan for plan in plans if plan.launch.cpu_ids == (cpu_id,)
        )
        return rank_single_core_plans(fixed_cpu_plans, context, cpu_id)


def rank_single_core_plans(
    plans: tuple[ExecutionPlan, ...], context: TuningContext, cpu_id: int
) -> tuple[SingleCoreEstimate, ...]:
    estimates = tuple(_estimate(plan, context, cpu_id) for plan in plans)
    return tuple(
        sorted(estimates, key=lambda item: (item.predicted_cycles, item.plan.identity))
    )


def _estimate(
    plan: ExecutionPlan, context: TuningContext, cpu_id: int
) -> SingleCoreEstimate:
    shape = _shape(context.workload)
    work = (
        _parallel_work(plan, shape)
        if context.pattern == "parallel"
        else _linear_work(plan, shape)
    )
    vector_bytes = plan.code.vector_bytes
    lanes = vector_bytes // _FP32_BYTES
    fma_pipes = 2 if context.host.features & {"fma", "asimd", "sve"} else 1
    peak_flops = 2 * lanes * fma_pipes
    utilization = min(1.0, work.fma_chains / 4)
    compute_cycles = work.flops / (peak_flops * utilization)

    bandwidth = _bandwidth(context.host, cpu_id, work.working_set_bytes, vector_bytes)
    data_cycles = work.memory_bytes / bandwidth
    packing_cycles = work.packing_elements / lanes
    registers = 32 if context.host.architecture == "aarch64" or lanes >= 16 else 16
    spills = max(0, work.register_vectors - (registers - 6))
    overhead_cycles = work.tile_visits * (4 + spills)
    predicted = max(compute_cycles, data_cycles) + packing_cycles + overhead_cycles
    return SingleCoreEstimate(
        plan=plan,
        predicted_cycles=predicted,
        compute_cycles=compute_cycles,
        data_cycles=data_cycles,
        packing_cycles=packing_cycles,
        overhead_cycles=overhead_cycles,
        memory_bytes=work.memory_bytes,
        working_set_bytes=work.working_set_bytes,
        arithmetic_intensity=work.flops / max(1, work.memory_bytes),
    )


def _parallel_work(plan: ExecutionPlan, shape: Mapping[str, int]) -> _Work:
    code, tile = plan.code, plan.code.tile
    batch, heads = shape["batch"], shape["query_head"]
    kv_heads, query, key = shape["kv_head"], shape["query"], shape["key"]
    d_size, dv_size = shape["d"], shape["dv"]
    lanes = code.vector_bytes // _FP32_BYTES
    rows = batch * heads
    q_tiles, k_tiles = _ceil_div(query, tile.q), _ceil_div(key, tile.k)
    packed = code.packing is PackingKind.K_TRANSPOSED
    executed_key = k_tiles * tile.k if packed else key

    qk_flops = 2 * rows * query * executed_key * _round_up(d_size, lanes)
    pv_flops = 2 * rows * query * key * _round_up(dv_size, lanes)
    tensor_elements = rows * query * (d_size + dv_size)
    tensor_elements += batch * kv_heads * key * (d_size + dv_size)
    streamed_elements = rows * q_tiles * key * (d_size + dv_size)
    packing_elements = batch * kv_heads * key * d_size if packed else 0
    memory_bytes = (tensor_elements + streamed_elements) * _FP32_BYTES
    memory_bytes += 2 * packing_elements * _FP32_BYTES
    working_set = (
        tile.q * d_size + tile.k * (d_size + dv_size) + tile.q * dv_size
    ) * _FP32_BYTES
    fma_chains = tile.q * code.microkernel.qk_vectors if packed else 1
    registers = (
        tile.q * code.microkernel.qk_vectors + tile.q + 3 if packed else 5
    )
    return _Work(
        qk_flops + pv_flops,
        memory_bytes,
        packing_elements,
        rows * q_tiles * k_tiles,
        working_set,
        fma_chains,
        registers,
    )


def _linear_work(plan: ExecutionPlan, shape: Mapping[str, int]) -> _Work:
    tile = plan.code.tile
    rows = shape["batch"] * shape["state_head"]
    sequence, d_size, dv_size = shape["sequence"], shape["d"], shape["dv"]
    state_elements = d_size * dv_size
    block = tile.q if plan.code.lowering is LoweringKind.LINEAR_CHUNKED else 1
    padded_sequence = _ceil_div(sequence, block) * block
    flops = 4 * rows * padded_sequence * state_elements
    if block > 1:
        flops += rows * padded_sequence * block * (d_size + dv_size)
    input_elements = rows * padded_sequence * (2 * d_size + 2 * dv_size)
    if plan.code.lowering is LoweringKind.LINEAR_2D:
        input_elements *= _ceil_div(dv_size, tile.dv)
    memory_bytes = (input_elements + rows * state_elements) * _FP32_BYTES
    return _Work(
        flops,
        memory_bytes,
        0,
        rows * _ceil_div(sequence, block),
        state_elements * _FP32_BYTES + plan.memory.total_bytes,
        _ceil_div(dv_size, plan.code.vector_bytes // _FP32_BYTES),
        block + 4,
    )


def _bandwidth(
    host: Host, cpu_id: int, working_set: int, vector_bytes: int
) -> float:
    caches = sorted(
        (
            cache for cache in host.caches
            if cpu_id in cache.shared_cpus
            and cache.kind.lower() in {"data", "unified"}
        ),
        key=lambda cache: cache.level,
    )
    level = next(
        (cache.level for cache in caches if working_set <= cache.size_bytes),
        4,
    )
    return max(8.0, vector_bytes * {1: 2.0, 2: 1.0, 3: 0.5}.get(level, 0.25))


def _round_up(value: int, multiple: int) -> int:
    return _ceil_div(value, multiple) * multiple


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


__all__ = ["SingleCoreEstimate", "SingleCoreTuner"]
