from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from threading import Lock
from typing import Any, Mapping, NamedTuple

from .core.operator import Linear, Operator, Parallel
from .core.validate import (
    ValidatedCall,
    validate_linear_call,
    validate_parallel_call,
)
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
        self.last_selection: Selection | None = None
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
            plans, candidates_digest, workload_json = self._plans(operator, call)
            selection = self._select(
                operator,
                call,
                plans,
                candidates_digest,
                workload_json,
            )
            self.last_selection = selection
            event(
                "selection complete mode=%s plan=%s latency_ns=%d",
                selection.mode,
                selection.winner.plan.identity[:12],
                selection.winner.latency_ns,
            )
            return selection.winner.result

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

    def _plans(
        self,
        operator: Operator,
        call: ValidatedCall,
    ) -> _PlanEntry:
        workload = json.dumps(
            call.canonical(),
            sort_keys=True,
            separators=(",", ":"),
        )
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
            )
            self._plan_cache[key] = cached
        return cached

    def _select(
        self,
        operator: Operator,
        call: ValidatedCall,
        plans: tuple[ExecutionPlan, ...],
        candidates_digest: str,
        workload_json: str,
    ) -> Selection:
        tuner_json = json.dumps(
            self.tuner.cache_identity(),
            sort_keys=True,
            separators=(",", ":"),
        )
        memo_key = (operator.fingerprint, workload_json, tuner_json)
        key = self._selection_keys.get(memo_key)
        if key is None:
            key = selection_key(
                {
                    "operator": operator.canonical(),
                    "workload": call.canonical(),
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
            call.canonical(),
            plans,
        )

        measured: dict[str, TimedPlan] = {}

        def measure(plan: ExecutionPlan) -> int:
            timed = self._run_plan(operator, call, plan)
            measured[plan.identity] = timed
            return timed.latency_ns

        result = self.tuner.select(context, measure)
        winner_measurement = next(
            item
            for item in result.measurements
            if item.plan.identity == result.winner.identity
        )
        winner = replace(
            measured[result.winner.identity],
            latency_ns=winner_measurement.latency_ns,
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
        return self.executor.run(kernel, plan, call)

    def _run_plan(
        self,
        operator: Operator,
        call: ValidatedCall,
        plan: ExecutionPlan,
    ) -> TimedPlan:
        compiled = self.compiler.compile(
            operator,
            call,
            plan.code,
            self.backend,
        )
        kernel = self.compiler.load(compiled)
        self.executor.prepare_launch(kernel, plan.launch)
        return self.executor.run(kernel, plan, call)


__all__ = ["Runtime"]
