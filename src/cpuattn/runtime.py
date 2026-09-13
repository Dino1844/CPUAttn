from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from threading import Lock
from typing import Any, Mapping, NamedTuple

import numpy as np

from .core.operator import Linear, Operator, Parallel
from .core.tensor import Axis
from .core.validate import (
    ParallelCall,
    ValidatedCall,
    validate_linear_call,
    validate_parallel_call,
)
from . import diagnostics
from .hardware.host import Host, detect_host
from .log import event, startup
from .native.backends import select_backend
from .native.backends.base import Backend
from .native.compiler import Compiler, NativeKernel
from .native.executor import Executor
from .schedule.plan import ExecutionPlan, TimedPlan
from .schedule.planning import PlanBuilder
from .tuning.cache import SelectionCache, selection_key
from .tuning.tuner import PlanRecord, Selection, Tuner, TuningContext
from .tuning.worker_count import WorkerCountTuner


class _PlanEntry(NamedTuple):
    """Cached plan set with the components of its selection key."""

    plans: tuple[ExecutionPlan, ...]
    candidates_digest: str
    workload: str
    workload_canonical: dict[str, object]


class _PackedKState(NamedTuple):
    """Packed-K stream state, trusted only while it is the latest native run."""

    key: tuple[Any, ...]
    skv: int
    seq: int
    k_view: np.ndarray


_WARM_BUDGET_NS = 2_000_000

_KV_BUCKET_UNIT = 64


def _kv_bucket(key_length: int) -> int:
    """Bucket a growing KV length so decode tunes once per bucket, not per step."""
    if key_length <= _KV_BUCKET_UNIT:
        return _KV_BUCKET_UNIT
    return _KV_BUCKET_UNIT << ((key_length - 1) // _KV_BUCKET_UNIT).bit_length()


def _packed_skv(key_length: int, tile_k: int) -> int:
    """Key length padded to the kernel tile — the packed layout's row stride."""
    return -(-key_length // tile_k) * tile_k


def _warm_host() -> None:
    """Brief busy-spin so frequency governors ramp before sampling."""
    deadline = time.perf_counter_ns() + _WARM_BUDGET_NS
    accumulator = 0.0
    while time.perf_counter_ns() < deadline:
        for _ in range(256):
            accumulator += 1.0
    if accumulator < 0.0:  # keep the loop observably live
        raise AssertionError


class Runtime:
    def __init__(
        self,
        *,
        host: Host | None = None,
        backend: Backend | None = None,
        cache_dir: str | Path | None = None,
        tuner: Tuner | None = None,
    ) -> None:
        self.host = detect_host() if host is None else host
        self.backend = select_backend(self.host) if backend is None else backend
        if not self.backend.supports(self.host):
            raise ValueError(
                f"backend {self.backend.backend_id} is incompatible with supplied Host"
            )
        startup(self.host, self.backend)
        self.compiler = Compiler(cache_dir)
        self.planner = PlanBuilder(self.host)
        self.executor = Executor()
        self.tuner = WorkerCountTuner() if tuner is None else tuner
        self._selection_cache = SelectionCache(self.compiler.cache_dir)
        self._plan_cache: dict[tuple[str, str, str], _PlanEntry] = {}
        self._selection_keys: dict[tuple[str, str], str] = {}
        self._winner_kernels: dict[tuple[str, str], NativeKernel] = {}
        self._tuner_json: str | None = None
        self._tuner_json_owner: Tuner | None = None
        self._workload_json: dict[tuple[Any, ...], str] = {}
        self._bucket_calls: dict[tuple[str, int], ParallelCall] = {}
        # Bumped on every native run; packed-K state is trusted only while unchanged.
        self._arena_seq = 0
        # Single slot: only the latest native run's state can satisfy the seq gate,
        # and pinning the K view keeps its buffer address from being reused.
        self._kv_packed: _PackedKState | None = None
        self.last_selection: Selection | None = None
        self._last_entry: _PlanEntry | None = None
        self._execution_lock = Lock()
        event("runtime ready tuner=%s", type(self.tuner).__name__)

    def run(
        self,
        operator: Operator,
        *,
        q: Any,
        k: Any,
        v: Any,
        state: Any | None = None,
        arguments: Mapping[str, Any] | None = None,
        kv_cache: bool = False,
    ):
        call = self._validated_call(
            operator,
            q=q,
            k=k,
            v=v,
            state=state,
            arguments=arguments,
            kv_cache=kv_cache,
        )
        with self._execution_lock:
            tune_call = self._tune_call(operator, call)
            entry = self._plans(operator, tune_call)
            self._last_entry = entry
            selection = self._select(operator, call, tune_call, entry)
            self.last_selection = selection
            # Cache-mode winners are replayed verbatim; logging every call
            # would only duplicate the tune-time record.
            if selection.mode == "tune":
                event(
                    "selection complete mode=%s plan=%s latency_ns=%d",
                    selection.mode,
                    selection.winner.plan.identity[:12],
                    selection.winner.latency_ns,
                )
                record = self.explain()
                if record is not None:
                    diagnostics.log_selection(record)
            return selection.winner.result

    def explain(self) -> diagnostics.SelectionDiagnostics | None:
        """Structured record of the last selection, or None before any run.

        Rebuilt on demand from data the runtime already holds. The reference-free
        runtime leaves ``numeric`` empty; a test or benchmark can attach one with
        ``SelectionDiagnostics.with_numeric``.
        """
        selection = self.last_selection
        entry = self._last_entry
        if selection is None or entry is None:
            return None
        return diagnostics.build_selection_diagnostics(
            selection=selection,
            host=self.host,
            backend=self.backend,
            workload=entry.workload_canonical,
            plans=entry.plans,
            tuner=self.tuner,
            compiler_stats=self.compiler.stats,
        )

    def _validated_call(
        self,
        operator: Operator,
        *,
        q: Any,
        k: Any,
        v: Any,
        state: Any | None,
        arguments: Mapping[str, Any] | None,
        kv_cache: bool,
    ) -> ValidatedCall:
        if isinstance(operator, Parallel):
            if state is not None:
                raise ValueError("Parallel does not accept a recurrent state")
            return validate_parallel_call(
                operator,
                q=q,
                k=k,
                v=v,
                arguments=arguments,
                kv_cache=kv_cache,
            )
        if isinstance(operator, Linear):
            if kv_cache:
                raise ValueError("kv_cache is a Parallel execution property")
            return validate_linear_call(
                operator,
                q=q,
                k=k,
                v=v,
                state=state,
                arguments=arguments,
            )
        raise TypeError(f"unsupported operator type {type(operator).__name__}")

    def _tune_call(self, operator: Operator, call: ValidatedCall) -> ValidatedCall:
        """Tune growing-KV decode at the bucket edge so each bucket tunes once."""
        if not isinstance(call, ParallelCall) or not call.kv_cache:
            return call
        if any(Axis.KEY in argument.axes for argument in operator.arguments):
            return call
        edge = _kv_bucket(call.k.shape[2])
        token = (operator.fingerprint, edge)
        bucket_call = self._bucket_calls.get(token)
        if bucket_call is None:
            b, hkv = call.q.shape[0], call.k.shape[1]
            kd, dv = call.k.shape[3], call.v.shape[3]
            # Validate the bucket tensors so k/v pitches match what real calls
            # report; hand-built calls would index every head at panel zero.
            bucket_call = validate_parallel_call(
                operator,
                q=np.zeros(call.q.shape, dtype=np.float32),
                k=np.zeros((b, hkv, edge, kd), dtype=np.float32),
                v=np.zeros((b, hkv, edge, dv), dtype=np.float32),
                arguments=call.argument_map,
                kv_cache=True,
            )
            self._bucket_calls[token] = bucket_call
            event(
                "kv_cache bucket edge=%d operator=%s",
                edge,
                operator.fingerprint[:12],
            )
        return bucket_call

    def _plans(
        self,
        operator: Operator,
        call: ValidatedCall,
    ) -> _PlanEntry:
        workload = self._workload_json_for(operator, call)
        key = (operator.fingerprint, workload, self.backend.backend_id)
        cached = self._plan_cache.get(key)
        if cached is None:
            code_plans = self.backend.enumerate_code_plans(
                operator,
                call,
                self.host,
            )
            plans = self.planner.build_all(code_plans, operator, call)
            candidates = "\n".join(sorted(plan.identity for plan in plans))
            cached = _PlanEntry(
                plans=plans,
                candidates_digest=hashlib.sha256(
                    candidates.encode("utf-8")
                ).hexdigest(),
                workload=workload,
                workload_canonical=call.canonical(),
            )
            self._plan_cache[key] = cached
        return cached

    def _workload_json_for(
        self,
        operator: Operator,
        call: ValidatedCall,
    ) -> str:
        """Memoize the canonical-workload JSON on the shape facts it derives from."""
        if isinstance(call, ParallelCall):
            structure: tuple[Any, ...] = (
                "parallel",
                call.q.shape,
                call.k.shape,
                call.v.shape,
                call.q.dtype.str,
                call.kv_cache,
            )
        else:
            structure = (
                "linear",
                call.q.shape,
                call.k.shape,
                call.v.shape,
                call.q.dtype.str,
                call.state is not None,
            )
        key = (operator.fingerprint, structure)
        workload = self._workload_json.get(key)
        if workload is None:
            workload = json.dumps(
                call.canonical(),
                sort_keys=True,
                separators=(",", ":"),
            )
            self._workload_json[key] = workload
        return workload

    def _select(
        self,
        operator: Operator,
        call: ValidatedCall,
        tune_call: ValidatedCall,
        entry: _PlanEntry,
    ) -> Selection:
        plans = entry.plans
        candidates_digest = entry.candidates_digest
        workload_json = entry.workload
        # Key on identity so replacing runtime.tuner invalidates the memo.
        if self._tuner_json_owner is not self.tuner:
            self._tuner_json = json.dumps(
                self.tuner.cache_identity(),
                sort_keys=True,
                separators=(",", ":"),
            )
            self._tuner_json_owner = self.tuner
        tuner_json = self._tuner_json
        memo_key = (operator.fingerprint, workload_json, tuner_json)
        key = self._selection_keys.get(memo_key)
        if key is None:
            key = selection_key(
                {
                    "operator": operator.canonical(),
                    "workload": tune_call.canonical(),
                    "backend": self.backend.backend_id,
                    "compiler": self.compiler.identity(self.backend),
                    "host": self.host.fingerprint,
                    "candidates": candidates_digest,
                    "tuner": self.tuner.cache_identity(),
                }
            )
            self._selection_keys[memo_key] = key
        cached = self._selection_cache.get(key, plans)
        if cached is not None:
            plan, records = cached
            return Selection(
                key,
                self._run_cached(operator, call, key, plan),
                "cache",
                records,
            )

        context = TuningContext(
            self.host,
            operator.canonical(),
            tune_call.canonical(),
            plans,
        )

        measured: dict[str, int] = {}

        def measure(plan: ExecutionPlan) -> int:
            _warm_host()
            # A discarded run absorbs one-off page-in before the timed run.
            self._run_plan(operator, tune_call, plan)
            if isinstance(call, ParallelCall) and call.kv_cache and plan.code.packing.value == "k_transposed":
                # Price at stream steady state or the full-pack tax hides the winner.
                _, _, edge_skv, _ = tune_call.k.shape
                timed = self._run_plan(
                    operator, tune_call, plan, packed_prefix=edge_skv
                )
            else:
                timed = self._run_plan(operator, tune_call, plan)
            measured[plan.identity] = timed.latency_ns
            return timed.latency_ns

        result = self.tuner.select(context, measure)
        winner_measurement = next(
            item
            for item in result.measurements
            if item.plan.identity == result.winner.identity
        )
        # The winner was measured at the tuning shape; run it on the real tensors.
        executed = self._run_cached(operator, call, key, result.winner)
        winner = TimedPlan(
            result.winner,
            executed.compiled,
            winner_measurement.latency_ns,
            executed.result,
        )
        records = tuple(
            PlanRecord(
                item.plan.identity,
                "selected"
                if item.plan.identity == result.winner.identity
                else "measured",
                item.latency_ns,
                item.samples_ns,
            )
            for item in result.measurements
        )
        selection = Selection(key, winner, "tune", records)
        self._selection_cache.publish(selection)
        return selection

    def _packed_prefix(
        self,
        call: ParallelCall,
        plan: ExecutionPlan,
    ) -> tuple[int, tuple[Any, ...], str | None]:
        """Rows of K already packed for this buffer; any mismatch returns 0.

        The reason names why a previously valid packed state was discarded.
        """
        if plan.code.packing.value != "k_transposed":
            return 0, (), None
        b, _, _, d = call.q.shape
        _, hkv, skv, _ = call.k.shape
        key = (plan.identity, call.k.ctypes.data, b, hkv, d)
        state = self._kv_packed
        if state is None:
            return 0, key, None
        if state.key != key:
            return 0, key, "buffer-changed"
        if state.seq != self._arena_seq:
            return 0, key, "arena-changed"
        # Strict growth: a same-length call may have edited rows in place.
        if state.skv >= skv:
            return 0, key, "non-growing"
        tile_k = plan.code.tile.k
        # The packed layout is strided by _packed_skv; when that stride
        # changes the retained rows are unreadable, so only equal strides
        # may reuse the prefix.
        if _packed_skv(state.skv, tile_k) != _packed_skv(skv, tile_k):
            return 0, key, "stride-changed"
        return state.skv, key, None

    def _run_cached(
        self,
        operator: Operator,
        call: ValidatedCall,
        key: str,
        plan: ExecutionPlan,
    ) -> TimedPlan:
        token = (key, plan.identity)
        kernel = self._winner_kernels.get(token)
        if kernel is None:
            compiled = self.compiler.compile(
                operator,
                call,
                plan.code,
                self.backend,
            )
            kernel = self.compiler.load(compiled)
            self.executor.prepare_launch(kernel, plan.launch)
            self._winner_kernels[token] = kernel
        packed_prefix, packed_key, fallback_reason = 0, (), None
        if isinstance(call, ParallelCall) and call.kv_cache:
            packed_prefix, packed_key, fallback_reason = self._packed_prefix(call, plan)
        if fallback_reason is not None:
            diagnostics.log_fallback(fallback_reason)
        timed = self.executor.run(kernel, plan, call, packed_prefix)
        self._arena_seq += 1
        if packed_key:
            self._kv_packed = _PackedKState(
                packed_key, call.k.shape[2], self._arena_seq, call.k
            )
        return timed

    def _run_plan(
        self,
        operator: Operator,
        call: ValidatedCall,
        plan: ExecutionPlan,
        packed_prefix: int = 0,
    ) -> TimedPlan:
        compiled = self.compiler.compile(
            operator,
            call,
            plan.code,
            self.backend,
        )
        kernel = self.compiler.load(compiled)
        self.executor.prepare_launch(kernel, plan.launch)
        timed = self.executor.run(kernel, plan, call, packed_prefix)
        self._arena_seq += 1
        return timed


__all__ = ["Runtime", "diagnostics"]
