from __future__ import annotations

import ctypes
import mmap

import numpy as np

from .compiler import NativeKernel
from ..core.operator import LinearResult
from ..schedule.plan import ExecutionPlan, LaunchPlan, TimedPlan
from ..core.validate import LinearCall, ParallelCall, ValidatedCall


class Executor:
    """Drive generated kernels through their packed flat-ABI entry point.

    Each plan gets two reusable arrays: a pointer array (tensors, then
    workspace/cpu_ids/worker_groups/packed owners) and a scalar array (dims and
    runtime ABI values). Filling them in place replaces per-argument ctypes
    marshalling, and the generated ``cpuattn_execute_packed`` expands them back
    into the kernel's flat ``cpuattn_run`` argument list.
    """

    def __init__(self) -> None:
        self._arena: mmap.mmap | None = None
        self._arena_view: np.ndarray | None = None
        self._arena_key: str | None = None
        self._prepared: set[tuple[str, tuple[int, ...]]] = set()
        self._launch_arrays: dict[
            str, tuple[np.ndarray, np.ndarray, np.ndarray]
        ] = {}
        self._packed: dict[
            str, tuple[np.ndarray, np.ndarray, ctypes.c_uint64]
        ] = {}

    def _workspace_for(self, plan: ExecutionPlan) -> np.ndarray:
        if plan.identity != self._arena_key:
            self.close()
            self._arena = mmap.mmap(-1, plan.memory.total_bytes, access=mmap.ACCESS_WRITE)
            self._arena_view = np.frombuffer(
                self._arena, dtype=np.uint8, count=plan.memory.total_bytes
            )
            self._arena_key = plan.identity
        assert self._arena_view is not None
        return self._arena_view

    def _launch_arrays_for(
        self, plan: ExecutionPlan
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        arrays = self._launch_arrays.get(plan.identity)
        if arrays is None:
            group_by_cpu = {
                cpu_id: group_index
                for group_index, group in enumerate(plan.launch.groups)
                for cpu_id in group.cpu_ids
            }
            worker_groups = np.asarray(
                [group_by_cpu[cpu_id] for cpu_id in plan.launch.cpu_ids],
                dtype=np.int32,
            )
            packed_region = next(
                (region for region in plan.memory.regions if region.name == "packed_k"),
                None,
            )
            packed_owners = np.asarray(
                () if packed_region is None else packed_region.owner_groups,
                dtype=np.int32,
            )
            cpu_ids = np.asarray(plan.launch.cpu_ids, dtype=np.int32)
            arrays = (cpu_ids, worker_groups, packed_owners)
            self._launch_arrays[plan.identity] = arrays
        return arrays

    def close(self) -> None:
        self._arena_view = None
        if self._arena is not None:
            self._arena.close()
        self._arena = None
        self._arena_key = None
        self._packed.clear()

    def prepare_launch(self, kernel: NativeKernel, launch: LaunchPlan) -> None:
        token = (kernel.compiled.artifact_key, launch.cpu_ids)
        if token in self._prepared:
            return
        cpu_ids = np.asarray(launch.cpu_ids, dtype=np.int32)
        status = kernel._prepare_launch(
            ctypes.c_void_p(cpu_ids.ctypes.data),
            ctypes.c_int(launch.workers),
        )
        if status != 0:
            raise RuntimeError(f"native launch preconditioning failed with status {status}")
        self._prepared.add(token)

    def _packed_template(
        self,
        kernel: NativeKernel,
        plan: ExecutionPlan,
        call: ValidatedCall,
    ) -> tuple[np.ndarray, np.ndarray, ctypes.c_uint64]:
        """Prebuild the pointer/scalar arrays for one artifact and plan.

        Slots hold tensors in call order followed by workspace, ``cpu_ids``,
        ``worker_groups`` and packed owner groups; scalars hold the Parallel
        dims and runtime ABI values in the order documented in the generated
        ``cpuattn_execute_packed``. Only the per-call slots are rewritten.
        """
        token = f"{kernel.compiled.artifact_key}:{plan.identity}"
        template = self._packed.get(token)
        if template is not None:
            return template
        workspace = self._workspace_for(plan)
        cpu_ids, worker_groups, packed_owners = self._launch_arrays_for(plan)
        argument_count = len(call.arguments)
        if isinstance(call, ParallelCall):
            slots = np.empty(8 + argument_count, dtype=np.uint64)
            scalars = np.empty(15, dtype=np.int64)
            slots[4 + argument_count] = workspace.ctypes.data
            slots[5 + argument_count] = cpu_ids.ctypes.data
            slots[6 + argument_count] = worker_groups.ctypes.data
            slots[7 + argument_count] = (
                packed_owners.ctypes.data if packed_owners.size else 0
            )
            scalars[11] = plan.memory.total_bytes
            scalars[13] = len(plan.launch.groups)
            scalars[14] = packed_owners.size
        else:
            slots = np.empty(7 + argument_count, dtype=np.uint64)
            scalars = np.empty(8, dtype=np.int64)
            slots[5 + argument_count] = workspace.ctypes.data
            slots[6 + argument_count] = cpu_ids.ctypes.data
            scalars[6] = plan.memory.total_bytes
        elapsed = ctypes.c_uint64()
        template = (slots, scalars, elapsed)
        self._packed[token] = template
        return template

    def run(
        self,
        kernel: NativeKernel,
        plan: ExecutionPlan,
        call: ValidatedCall,
        packed_prefix: int = 0,
    ) -> TimedPlan:
        if kernel.compiled.code != plan.code:
            raise ValueError("compiled code and execution plan do not match")
        slots, scalars, elapsed = self._packed_template(kernel, plan, call)
        if isinstance(call, ParallelCall):
            b, hq, sq, d = call.q.shape
            _, hkv, skv, _ = call.k.shape
            dv = call.v.shape[3]
            output = np.empty((b, hq, sq, dv), dtype=np.float32)
            values = (
                call.q,
                call.k,
                call.v,
                output,
                *(value for _, value in call.arguments),
            )
            scalars[0] = call.query_offset
            scalars[1] = packed_prefix
            scalars[2] = call.k_pitch or skv * d
            scalars[3] = call.v_pitch or skv * dv
            scalars[4:11] = (b, hq, hkv, sq, skv, d, dv)
            scalars[12] = plan.launch.workers
            result_value: object = output
        else:
            assert isinstance(call, LinearCall)
            b, groups, sequence, d = call.q.shape
            heads = call.v.shape[1]
            dv = call.v.shape[3]
            state = (
                np.zeros((b, heads, d, dv), dtype=np.float32)
                if call.state is None
                else call.state.copy()
            )
            output = np.empty((b, heads, sequence, dv), dtype=np.float32)
            values = (
                call.q,
                call.k,
                call.v,
                state,
                output,
                *(value for _, value in call.arguments),
            )
            scalars[0:6] = (b, groups, heads, sequence, d, dv)
            scalars[7] = plan.launch.workers
            result_value = LinearResult(output, state)
        for index, value in enumerate(values):
            slots[index] = value.ctypes.data
        status = kernel._execute_packed(
            ctypes.c_void_p(slots.ctypes.data),
            ctypes.c_void_p(scalars.ctypes.data),
            ctypes.byref(elapsed),
        )
        if status != 0:
            raise RuntimeError(f"native execution failed with status {status}")
        return TimedPlan(plan, kernel.compiled, int(elapsed.value), result_value)


__all__ = ["Executor"]
