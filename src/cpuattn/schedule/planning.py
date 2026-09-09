from __future__ import annotations

import os

from ..hardware.host import Host
from ..core.operator import Linear, Operator, Parallel
from .plan import (
    CodePlan,
    DecompositionKind,
    ExecutionPlan,
    LaunchPlan,
    LoweringKind,
    MemoryPlan,
    MemoryRegion,
    PackingKind,
    PlacementKind,
    WorkerGroup,
)
from ..core.validate import LinearCall, ParallelCall, ValidatedCall


class PlanBuilder:
    def __init__(self, host: Host, *, page_size: int | None = None) -> None:
        self.host = host
        self.page_size = page_size or os.sysconf("SC_PAGE_SIZE")
        if self.page_size <= 0 or self.page_size & (self.page_size - 1):
            raise ValueError("system page size must be a positive power of two")
        self._by_id = {cpu.cpu_id: cpu for cpu in host.cpus}

    def build_all(
        self,
        code_plans: tuple[CodePlan, ...],
        operator: Operator,
        call: ValidatedCall,
    ) -> tuple[ExecutionPlan, ...]:
        plans = [
            ExecutionPlan(code, launch, memory)
            for code in code_plans
            for launch in self.launch_plans(code, operator, call)
            for memory in self._memory_plans(code, launch, operator, call)
        ]
        unique = {plan.identity: plan for plan in plans}
        return tuple(unique[key] for key in sorted(unique))

    def launch_plans(
        self,
        code: CodePlan,
        operator: Operator,
        call: ValidatedCall,
    ) -> tuple[LaunchPlan, ...]:
        maximum = self.maximum_parallelism(code, operator, call)
        if maximum <= 0:
            return ()
        by_node = self._physical_cpus_by_node()
        physical_count = sum(len(cpus) for cpus in by_node.values())
        limit = min(maximum, physical_count)
        sizes = {1, limit}
        size = 2
        while size <= limit:
            sizes.add(size)
            size *= 2
        sizes.update(min(limit, len(cpus)) for cpus in by_node.values())

        launches: dict[tuple[int, ...], LaunchPlan] = {}
        for workers in sorted(value for value in sizes if 0 < value <= limit):
            for node, cpus in sorted(by_node.items()):
                if workers <= len(cpus):
                    launch = self._launch(tuple(cpus[:workers]))
                    launches[launch.cpu_ids] = launch
            if len(by_node) > 1 and workers >= 2:
                balanced = self._balanced_cpus(by_node, workers)
                if len({self._by_id[cpu].numa_node for cpu in balanced}) > 1:
                    launch = self._launch(balanced)
                    launches[launch.cpu_ids] = launch

        result = tuple(launches[key] for key in sorted(launches, key=lambda ids: (len(ids), ids)))
        if code.decomposition is DecompositionKind.SPLIT_KEY:
            result = tuple(launch for launch in result if launch.workers > 1)
        elif code.decomposition is DecompositionKind.QUERY_GROUPS_KEY_PARTS:
            rows, key_tiles = self._parallel_grid(call, code)
            result = tuple(
                launch
                for launch in result
                if launch.workers >= 4
                and _has_nontrivial_factorization(launch.workers)
                and rows >= 2
                and key_tiles >= 2
            )
        elif code.decomposition is DecompositionKind.STATE_HEADS_DV_BLOCKS:
            assert isinstance(call, LinearCall)
            state_heads = call.v.shape[0] * call.v.shape[1]
            result = tuple(
                launch
                for launch in result
                if launch.workers > 1
                and (call.v.shape[3] + code.tile.dv - 1) // code.tile.dv > 1
                and launch.workers <= self.maximum_parallelism(code, operator, call)
                and self.maximum_parallelism(code, operator, call) > state_heads
            )
        return result

    def maximum_parallelism(
        self,
        code: CodePlan,
        operator: Operator,
        call: ValidatedCall,
    ) -> int:
        if isinstance(operator, Parallel):
            assert isinstance(call, ParallelCall)
            rows, key_tiles = self._parallel_grid(call, code)
            if code.decomposition is DecompositionKind.QUERY_BLOCKS:
                return rows
            if code.decomposition is DecompositionKind.SPLIT_KEY:
                return key_tiles
            return rows * key_tiles
        assert isinstance(operator, Linear) and isinstance(call, LinearCall)
        state_heads = call.v.shape[0] * call.v.shape[1]
        if code.decomposition is DecompositionKind.STATE_HEADS:
            return state_heads
        dv_blocks = (call.v.shape[3] + code.tile.dv - 1) // code.tile.dv
        return state_heads * dv_blocks

    def _parallel_grid(self, call: ValidatedCall, code: CodePlan) -> tuple[int, int]:
        assert isinstance(call, ParallelCall)
        b, hq, sq, _ = call.q.shape
        rows = b * hq * ((sq + code.tile.q - 1) // code.tile.q)
        key_tiles = (call.k.shape[2] + code.tile.k - 1) // code.tile.k
        return rows, key_tiles

    def _memory_plans(
        self,
        code: CodePlan,
        launch: LaunchPlan,
        operator: Operator,
        call: ValidatedCall,
    ) -> tuple[MemoryPlan, ...]:
        owners = tuple(range(len(launch.groups)))
        private_alignment = self.page_size

        if isinstance(operator, Parallel):
            assert isinstance(call, ParallelCall)
            b, _, _, d = call.q.shape
            _, hkv, skv, _ = call.k.shape
            if code.lowering is LoweringKind.PARALLEL_BLOCKED:
                floats = code.tile.q * code.tile.k * 2 + code.tile.q * call.v.shape[3]
                worker_stride = _align(floats * 4, private_alignment)
                size = launch.workers * worker_stride
                name = "scratch"
            else:
                rows = b * call.q.shape[1] * call.q.shape[2]
                part_floats = call.v.shape[3] + len(operator.row_norm.states)
                worker_stride = _align(rows * part_floats * 4, private_alignment)
                size = launch.workers * worker_stride
                name = "partials"
            packed_owners = ((),)
            packed_copy_bytes = 0
            if code.packing is PackingKind.K_TRANSPOSED:
                packed_skv = _align(skv, code.tile.k)
                packed_copy_bytes = _align(
                    b * hkv * d * packed_skv * 4,
                    self.page_size,
                )
                packed_owners = (
                    tuple((group,)) for group in range(len(launch.groups))
                )
                if launch.crosses_numa:
                    packed_owners = (*packed_owners, owners)

            plans = []
            for packed_owner in packed_owners:
                regions: list[MemoryRegion] = []
                offset = 0
                if packed_owner:
                    packed_size = packed_copy_bytes * len(packed_owner)
                    regions.append(
                        MemoryRegion(
                            "packed_k",
                            offset,
                            packed_size,
                            self.page_size,
                            PlacementKind.NUMA_LOCAL,
                            packed_owner,
                        )
                    )
                    offset += packed_size
                regions.append(
                    MemoryRegion(
                        name,
                        offset,
                        size,
                        private_alignment,
                        PlacementKind.WORKER_LOCAL,
                        owners,
                    )
                )
                offset += size
                memory = MemoryPlan(tuple(regions), offset)
                self._validate_memory(code, launch, memory)
                plans.append(memory)
            return tuple(plans)
        else:
            assert isinstance(operator, Linear) and isinstance(call, LinearCall)
            d = call.q.shape[3]
            dv = call.v.shape[3]
            if code.lowering is LoweringKind.LINEAR_CHUNKED:
                block = code.tile.q
                floats = 2 * block * d + 2 * block * dv + block * block + 3 * block
            else:
                floats = 2 * d + 2 * dv
            worker_stride = _align(floats * 4, private_alignment)
            size = launch.workers * worker_stride
            regions = [
                MemoryRegion(
                    "scratch",
                    0,
                    size,
                    private_alignment,
                    PlacementKind.WORKER_LOCAL,
                    owners,
                )
            ]
            memory = MemoryPlan(tuple(regions), size)
            self._validate_memory(code, launch, memory)
            return (memory,)

    @staticmethod
    def _validate_memory(
        code: CodePlan,
        launch: LaunchPlan,
        memory: MemoryPlan,
    ) -> None:
        if tuple(region.name for region in memory.regions) != code.workspace_abi.region_names:
            raise ValueError("memory plan does not satisfy the code workspace ABI")
        for region in memory.regions:
            if any(owner < 0 or owner >= len(launch.groups) for owner in region.owner_groups):
                raise ValueError("memory plan refers to an unknown worker group")

    def _physical_cpus_by_node(self) -> dict[int, list[int]]:
        result: dict[int, list[int]] = {}
        seen: set[tuple[int, int]] = set()
        for cpu in self.host.cpus:
            core = (cpu.socket_id, cpu.core_id)
            if core in seen:
                continue
            seen.add(core)
            result.setdefault(cpu.numa_node, []).append(cpu.cpu_id)
        return result

    def _balanced_cpus(
        self,
        by_node: dict[int, list[int]],
        workers: int,
    ) -> tuple[int, ...]:
        result: list[int] = []
        nodes = sorted(by_node)
        offset = 0
        while len(result) < workers:
            progressed = False
            for node in nodes:
                cpus = by_node[node]
                if offset < len(cpus):
                    result.append(cpus[offset])
                    progressed = True
                    if len(result) == workers:
                        break
            if not progressed:
                break
            offset += 1
        return tuple(result)

    def _launch(self, cpu_ids: tuple[int, ...]) -> LaunchPlan:
        grouped: dict[int, list[int]] = {}
        for cpu_id in cpu_ids:
            grouped.setdefault(self._by_id[cpu_id].numa_node, []).append(cpu_id)
        groups = tuple(
            WorkerGroup(node, tuple(cpus)) for node, cpus in sorted(grouped.items())
        )
        return LaunchPlan(cpu_ids, groups)


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) & -alignment


def _has_nontrivial_factorization(value: int) -> bool:
    return any(value % divisor == 0 for divisor in range(2, value // 2 + 1))


__all__ = ["PlanBuilder"]
