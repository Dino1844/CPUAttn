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
        self._arg_templates: dict[str, list[object]] = {}
        self._elapsed_slots: dict[str, ctypes.c_uint64] = {}

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
        self._arg_templates.clear()
        self._elapsed_slots.clear()

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

    def _argument_template(
        self,
        kernel: NativeKernel,
        plan: ExecutionPlan,
        call: ValidatedCall,
    ) -> list[object]:
        """Prebuild the complete, call-ready argument list for a plan."""
        token = f"{kernel.compiled.artifact_key}:{plan.identity}"
        template = self._arg_templates.get(token)
        if template is not None:
            return template
        workspace = self._workspace_for(plan)
        cpu_ids, worker_groups, packed_owners = self._launch_arrays_for(plan)
        argument_count = len(call.arguments)
        parallel = isinstance(call, ParallelCall)
        pointer_count = (4 if parallel else 5) + argument_count
        arguments: list[object] = [
            ctypes.c_void_p() for _ in range(pointer_count)
        ]
        offset_slot: ctypes.c_int64 | None = None
        if parallel:
            b, hq, sq, d = call.q.shape
            _, hkv, skv, _ = call.k.shape
            dv = call.v.shape[3]
            dimensions = (b, hq, hkv, sq, skv, d, dv)
            # Per-call slot: baking it into the source would make one artifact per length.
            offset_slot = ctypes.c_int64()
            arguments.append(offset_slot)
        else:
            b, groups, sequence, d = call.q.shape
            heads = call.v.shape[1]
            dv = call.v.shape[3]
            dimensions = (b, groups, heads, sequence, d, dv)
        arguments.extend(ctypes.c_int64(value) for value in dimensions)
        arguments.extend((
            ctypes.c_void_p(workspace.ctypes.data),
            ctypes.c_size_t(plan.memory.total_bytes),
            ctypes.c_void_p(cpu_ids.ctypes.data),
            ctypes.c_int(plan.launch.workers),
        ))
        if parallel:
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
        elapsed = ctypes.c_uint64()
        arguments.append(ctypes.byref(elapsed))
        kernel._execute.argtypes = (
            [ctypes.c_void_p] * pointer_count
            + [ctypes.c_int64] * (len(dimensions) + (1 if parallel else 0))
            + [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_int]
            + (
                [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
                if parallel
                else []
            )
            + [ctypes.POINTER(ctypes.c_uint64)]
        )
        self._arg_templates[token] = arguments
        self._elapsed_slots[token] = elapsed
        return arguments

    def run(
        self,
        kernel: NativeKernel,
        plan: ExecutionPlan,
        call: ValidatedCall,
    ) -> TimedPlan:
        if kernel.compiled.code != plan.code:
            raise ValueError("compiled code and execution plan do not match")
        if isinstance(call, ParallelCall):
            b, hq, sq, d = call.q.shape
            output = np.empty((b, hq, sq, call.v.shape[3]), dtype=np.float32)
            values = [call.q, call.k, call.v, output, *(value for _, value in call.arguments)]
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
            result_value = LinearResult(output, state)

        token = f"{kernel.compiled.artifact_key}:{plan.identity}"
        arguments = self._argument_template(kernel, plan, call)
        elapsed = self._elapsed_slots[token]
        for slot, value in zip(arguments, values):
            slot.value = value.ctypes.data
        if isinstance(call, ParallelCall):
            b, hq, sq, d = call.q.shape
            _, hkv, skv, _ = call.k.shape
            dv = call.v.shape[3]
            dims = (call.query_offset, b, hq, hkv, sq, skv, d, dv)
        else:
            assert isinstance(call, LinearCall)
            dims = (b, groups, heads, sequence, d, dv)
        scalar_base = len(values)
        for slot, value in zip(arguments[scalar_base : scalar_base + len(dims)], dims):
            slot.value = value
        status = kernel._execute(*arguments)
        if status != 0:
            raise RuntimeError(f"native execution failed with status {status}")
        return TimedPlan(plan, kernel.compiled, int(elapsed.value), result_value)


__all__ = ["Executor"]
