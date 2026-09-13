from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from .schedule.plan import ExecutionPlan, MemoryPlan

if TYPE_CHECKING:
    from .hardware.host import Host
    from .native.backends.base import Backend
    from .tuning.tuner import Selection, Tuner

SCHEMA = 1

_ENABLED: bool | None = None
_DIAGNOSTIC_LOGGER = logging.getLogger("cpuattn.debug")


@dataclass(frozen=True, slots=True)
class SelectionDiagnostics:
    """Structured, JSON-ready record of one selection decision.

    Assembled from data the runtime already holds; ``numeric`` stays empty
    unless the caller (a test or benchmark) attaches a reference comparison.
    """

    schema: int
    host: dict[str, object]
    backend: dict[str, object]
    workload: dict[str, object]
    plans: dict[str, object]
    tuner: dict[str, object]
    measurements: tuple[dict[str, object], ...]
    winner: dict[str, object]
    workspace: dict[str, object]
    compile_cache: dict[str, int]
    numeric: dict[str, object] | None = None

    def as_json(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "host": self.host,
            "backend": self.backend,
            "workload": self.workload,
            "plans": self.plans,
            "tuner": self.tuner,
            "measurements": list(self.measurements),
            "winner": self.winner,
            "workspace": self.workspace,
            "compile_cache": self.compile_cache,
            "numeric": self.numeric,
        }

    def with_numeric(self, numeric: Mapping[str, object]) -> SelectionDiagnostics:
        """Attach a reference comparison; the runtime itself has no reference."""
        return replace(self, numeric=dict(numeric))


def _diagnostic_logger() -> logging.Logger:
    """Dedicated stderr logger sharing the project's handler discipline."""
    if not _DIAGNOSTIC_LOGGER.handlers and debug_enabled():
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[cpuattn-debug] %(message)s"))
        _DIAGNOSTIC_LOGGER.addHandler(handler)
        _DIAGNOSTIC_LOGGER.setLevel(logging.INFO)
        _DIAGNOSTIC_LOGGER.propagate = False
    return _DIAGNOSTIC_LOGGER


def debug_enabled() -> bool:
    """Read CPUATTN_DEBUG once per process; any value but 0/false enables it."""
    global _ENABLED
    if _ENABLED is None:
        _ENABLED = os.environ.get("CPUATTN_DEBUG", "").lower() not in ("", "0", "false")
    return _ENABLED


def reset_debug_cache() -> None:
    """Full diagnostics reset: re-read the env and re-bind the debug stream."""
    global _ENABLED
    _ENABLED = None
    _DIAGNOSTIC_LOGGER.handlers.clear()


def plan_label(plan: ExecutionPlan) -> str:
    """One-line plan summary; the schedule shows only when it is not the default."""
    code = plan.code
    schedule = (
        "" if code.schedule.value == "static" else f"/{code.schedule.value}"
    )
    return (
        f"{code.lowering.value}/{code.packing.value}{schedule} "
        f"tile=({code.tile.q},{code.tile.k},{code.tile.d},{code.tile.dv}) "
        f"workers={plan.launch.workers}"
    )


def summarize_plans(plans: Sequence[ExecutionPlan]) -> dict[str, object]:
    """Count the legal plan space by each structural axis."""

    def counts(axis) -> dict[str, int]:
        tally: dict[str, int] = {}
        for plan in plans:
            name = axis(plan)
            tally[name] = tally.get(name, 0) + 1
        return dict(sorted(tally.items()))

    return {
        "execution_plans": len(plans),
        "by_lowering": counts(lambda plan: plan.code.lowering.value),
        "by_packing": counts(lambda plan: plan.code.packing.value),
        "by_decomposition": counts(lambda plan: plan.code.decomposition.value),
        "by_schedule": counts(lambda plan: plan.code.schedule.value),
        "worker_counts": sorted({plan.launch.workers for plan in plans}),
    }


def build_selection_diagnostics(
    *,
    selection: Selection,
    host: Host,
    backend: Backend,
    workload: Mapping[str, object],
    plans: Sequence[ExecutionPlan],
    tuner: Tuner,
    compiler_stats: Mapping[str, int],
    numeric: Mapping[str, object] | None = None,
) -> SelectionDiagnostics:
    """Assemble the record purely from objects the runtime already holds."""
    labels = {plan.identity: plan_label(plan) for plan in plans}
    winner_plan = selection.winner.plan
    measurements = tuple(
        {
            "plan_id": record.plan_id,
            "label": labels.get(record.plan_id, "?"),
            "status": record.status,
            "latency_ns": record.latency_ns,
            "samples_ns": list(record.samples_ns),
        }
        for record in selection.records
    )
    return SelectionDiagnostics(
        schema=SCHEMA,
        host=_host_record(host),
        backend=_backend_record(host, backend),
        workload={
            "pattern": workload.get("pattern"),
            "digest": _short(workload),
            "canonical": _plain(workload),
        },
        plans=summarize_plans(plans),
        tuner=tuner.cache_identity(),
        measurements=measurements,
        winner={
            "plan_id": winner_plan.identity,
            "label": plan_label(winner_plan),
            "mode": selection.mode,
            "latency_ns": selection.winner.latency_ns,
        },
        workspace=_workspace_record(winner_plan.memory, winner_plan.code.packing.value),
        compile_cache={key: int(value) for key, value in compiler_stats.items()},
        numeric=None if numeric is None else dict(numeric),
    )


def _host_record(host: Host) -> dict[str, object]:
    return {
        "fingerprint": host.fingerprint,
        "vendor": host.vendor,
        "model": host.model,
        "architecture": host.architecture,
        "physical_cores": host.physical_cores,
        "logical_cpus": len(host.cpus),
        "numa_nodes": len({cpu.numa_node for cpu in host.cpus}),
        "features": sorted(host.features),
        "sve_vector_bytes": host.sve_vector_bytes,
    }


def _backend_record(host: Host, backend: Backend) -> dict[str, object]:
    from .native.backends import BACKENDS

    return {
        "selected": backend.backend_id,
        "vector_bytes": backend.vector_bytes,
        "required_features": sorted(backend.required_features),
        "cflags": list(backend.cflags),
        "candidates": [
            {
                "id": candidate.backend_id,
                "compatible": candidate.supports(host),
                "missing_features": sorted(candidate.required_features - host.features),
                "sve_vector_bytes": candidate.sve_vector_bytes,
            }
            for candidate in BACKENDS
        ],
    }


def _workspace_record(memory: MemoryPlan, packing: str) -> dict[str, object]:
    regions = ", ".join(
        f"{region.name}={_bytes(region.size_bytes)}" for region in memory.regions
    )
    return {
        "total_bytes": memory.total_bytes,
        "packing": packing,
        "summary": f"total={_bytes(memory.total_bytes)} [{regions}]",
        "regions": [
            {
                "name": region.name,
                "size_bytes": region.size_bytes,
                "placement": region.placement.value,
                "owner_groups": list(region.owner_groups),
            }
            for region in memory.regions
        ],
    }


def render(record: SelectionDiagnostics) -> str:
    """Human-readable form of a record, written to stderr under CPUATTN_DEBUG."""
    host = record.host
    backend = record.backend
    workload = record.workload
    candidates = ", ".join(
        f"{item['id']}:{'ok' if item['compatible'] else 'missing=' + str(item['missing_features'])}"
        for item in backend["candidates"]
    )
    lines = [
        f"[cpuattn-debug] selection operator={workload['pattern']} workload={workload['digest']}",
        f"  host: {host['architecture']} {host['vendor']} {host['model']} "
        f"cores={host['physical_cores']} backend={backend['selected']}",
        f"  isa: features={host['features']} sve={host['sve_vector_bytes']} candidates=[{candidates}]",
        f"  legal plans: {record.plans['execution_plans']}",
        "  tuner measurements:",
        *[
            f"    {item['plan_id'][:12]} {item['label']} "
            f"{_ms(item['latency_ns'])} {item['status']}"
            for item in record.measurements
        ],
        f"  winner: {record.winner['label']} native={_ms(record.winner['latency_ns'])} "
        f"mode={record.winner['mode']}",
        f"  workspace: {record.workspace['summary']}",
        f"  compile cache: {record.compile_cache}",
    ]
    return "\n".join(lines)


def log_selection(record: SelectionDiagnostics) -> None:
    """Render one structured selection record to the debug logger."""
    if debug_enabled():
        _diagnostic_logger().info(render(record))


def log_fallback(reason: str) -> None:
    """One line per packed-K stream invalidation, only under CPUATTN_DEBUG."""
    if debug_enabled():
        _diagnostic_logger().info("packed-k full-pack fallback: %s", reason)


def _bytes(value: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{int(value)}B" if unit == "B" else f"{value:.1f}{unit}"
        value /= 1024
    return f"{value}B"


def _plain(value: object) -> object:
    """JSON-plain form for frozen mapping/tuple structures."""
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def _short(mapping: Mapping[str, object]) -> str:
    payload = json.dumps(_plain(mapping), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _ms(nanoseconds: int | None) -> str:
    return "?" if nanoseconds is None else f"{nanoseconds / 1e6:.3f}ms"
