from __future__ import annotations

import ctypes
import mmap

import numpy as np

from .compiler import NativeKernel
from ..core.operator import LinearResult
from ..schedule.plan import ExecutionPlan, LaunchPlan, TimedPlan
from ..core.validate import LinearCall, ParallelCall, ValidatedCall


class Executor:
    def __init__(self) -> None:
        self._arena: mmap.mmap | None = None
        self._arena_view: np.ndarray | None = None
        self._arena_key: str | None = None
        self._prepared: set[tuple[str, tuple[int, ...]]] = set()
        self._launch_arrays: dict[
            str, tuple[np.ndarray, np.ndarray, np.ndarray]
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

    def run(
        self,
        kernel: NativeKernel,
        plan: ExecutionPlan,
        call: ValidatedCall,
    ) -> TimedPlan:
        if kernel.compiled.code != plan.code:
            raise ValueError("compiled code and execution plan do not match")
        cpu_ids, worker_groups, packed_owners = self._launch_arrays_for(plan)
        workspace = self._workspace_for(plan)
        elapsed = ctypes.c_uint64()
        if isinstance(call, ParallelCall):
            b, hq, sq, d = call.q.shape
            _, hkv, skv, _ = call.k.shape
            dv = call.v.shape[3]
            output = np.empty((b, hq, sq, dv), dtype=np.float32)
            values = [call.q, call.k, call.v, output, *(value for _, value in call.arguments)]
            dimensions = (b, hq, hkv, sq, skv, d, dv)
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
            values = [
                call.q,
                call.k,
                call.v,
                state,
                output,
                *(value for _, value in call.arguments),
            ]
            dimensions = (b, groups, heads, sequence, d, dv)
            result_value = LinearResult(output, state)

        arguments: list[object] = [ctypes.c_void_p(value.ctypes.data) for value in values]
        arguments.extend(ctypes.c_int64(value) for value in dimensions)
        arguments.extend((
            ctypes.c_void_p(workspace.ctypes.data),
            ctypes.c_size_t(plan.memory.total_bytes),
            ctypes.c_void_p(cpu_ids.ctypes.data),
            ctypes.c_int(plan.launch.workers),
        ))
        if isinstance(call, ParallelCall):
            owner_pointer = (
                ctypes.c_void_p(packed_owners.ctypes.data)
                if packed_owners.size
                else ctypes.c_void_p()
            )
            arguments.extend((
                ctypes.c_void_p(worker_groups.ctypes.data),
                ctypes.c_int(len(plan.launch.groups)),
                owner_pointer,
                ctypes.c_int(packed_owners.size),
            ))
        arguments.append(ctypes.byref(elapsed))
        status = kernel._execute(*arguments)
        if status != 0:
            raise RuntimeError(f"native execution failed with status {status}")
        return TimedPlan(plan, kernel.compiled, int(elapsed.value), result_value)


__all__ = ["Executor"]
