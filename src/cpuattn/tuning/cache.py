from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

from ..schedule.plan import ExecutionPlan
from .tuner import PlanRecord, Selection


CachedSelection = tuple[ExecutionPlan, tuple[PlanRecord, ...]]


class SelectionCache:
    """Persist exact tuning winners; Runtime owns and drives this cache."""

    def __init__(self, cache_dir: Path) -> None:
        self._directory = cache_dir / "selections"
        self._memory: dict[str, CachedSelection] = {}

    def get(
        self,
        key: str,
        candidates: tuple[ExecutionPlan, ...],
    ) -> CachedSelection | None:
        cached = self._memory.get(key)
        if cached is not None:
            return cached
        cached = self._read(key, candidates)
        if cached is not None:
            self._memory[key] = cached
        return cached

    def publish(self, selection: Selection) -> None:
        path = self._path(selection.key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "key": selection.key,
            "winner_plan_id": selection.winner.plan.identity,
            "records": [
                {
                    "plan_id": record.plan_id,
                    "status": record.status,
                    "latency_ns": record.latency_ns,
                    "samples_ns": list(record.samples_ns),
                }
                for record in selection.records
            ],
        }
        lock_path = path.with_suffix(".lock")
        with lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            temporary = path.with_name(
                f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
            )
            temporary.write_text(
                json.dumps(payload, sort_keys=True, indent=2),
                encoding="utf-8",
            )
            os.replace(temporary, path)
        self._memory[selection.key] = (selection.winner.plan, selection.records)

    def _read(
        self,
        key: str,
        candidates: tuple[ExecutionPlan, ...],
    ) -> CachedSelection | None:
        try:
            data = json.loads(self._path(key).read_text(encoding="utf-8"))
            if not isinstance(data, dict) or data.get("key") != key:
                return None
            by_id = {plan.identity: plan for plan in candidates}
            winner = by_id.get(data.get("winner_plan_id"))
            raw_records = data.get("records")
            if winner is None or not isinstance(raw_records, list):
                return None
            records = tuple(_record(value, by_id) for value in raw_records)
        except (OSError, TypeError, ValueError):
            return None
        if not records or any(record is None for record in records):
            return None
        typed_records = tuple(record for record in records if record is not None)
        if winner.identity not in {record.plan_id for record in typed_records}:
            return None
        return winner, typed_records

    def _path(self, key: str) -> Path:
        return self._directory / f"{key}.json"


def selection_key(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _record(
    value: object,
    candidates: dict[str, ExecutionPlan],
) -> PlanRecord | None:
    if not isinstance(value, dict):
        return None
    plan_id = value.get("plan_id")
    status = value.get("status")
    latency = value.get("latency_ns")
    samples = value.get("samples_ns")
    if (
        not isinstance(plan_id, str)
        or plan_id not in candidates
        or status not in {"measured", "selected"}
        or not isinstance(latency, int)
        or latency < 0
        or not isinstance(samples, list)
        or not samples
        or any(not isinstance(sample, int) or sample < 0 for sample in samples)
    ):
        return None
    return PlanRecord(plan_id, status, latency, tuple(samples))


__all__ = ["SelectionCache", "selection_key"]
