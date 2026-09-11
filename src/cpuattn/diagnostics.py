from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Mapping
from typing import TYPE_CHECKING

from .schedule.plan import ExecutionPlan, MemoryPlan

if TYPE_CHECKING:
    from .hardware.host import Host
    from .native.backends.base import Backend
    from .tuning.tuner import Selection, TuningContext

_ENABLED: bool | None = None
_DIAGNOSTIC_LOGGER = logging.getLogger("cpuattn.debug")


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
    code = plan.code
    return (
        f"{code.lowering.value}/{code.packing.value} "
        f"tile=({code.tile.q},{code.tile.k},{code.tile.d},{code.tile.dv}) "
        f"workers={plan.launch.workers}"
    )


def _bytes(value: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{int(value)}B" if unit == "B" else f"{value:.1f}{unit}"
        value /= 1024
    return f"{value}B"


def workspace_summary(memory: MemoryPlan) -> str:
    regions = ", ".join(
        f"{region.name}={_bytes(region.size_bytes)}" for region in memory.regions
    )
    return f"total={_bytes(memory.total_bytes)} [{regions}]"


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


def log_selection(
    context: TuningContext,
    selection: Selection,
    host: Host,
    backend: Backend,
    compiler_stats: dict[str, int],
) -> None:
    """Print one structured block per tuning decision to stderr."""
    if not debug_enabled():
        return
    labels = {plan.identity: plan_label(plan) for plan in context.candidates}
    lines = [
        f"[cpuattn-debug] selection operator={context.pattern} workload={_short(context.workload)}",
        f"  host: {host.architecture} {host.vendor} {host.model} "
        f"cores={host.physical_cores} backend={backend.backend_id}",
        f"  legal plans: {len(context.candidates)}",
        "  tuner measurements:",
        *[
            f"    {record.plan_id[:12]} {labels.get(record.plan_id, '?')} "
            f"{_ms(record.latency_ns)} {record.status}"
            for record in selection.records
        ],
        f"  winner: {plan_label(selection.winner.plan)} "
        f"native={_ms(selection.winner.latency_ns)} mode={selection.mode}",
        f"  workspace: {workspace_summary(selection.winner.plan.memory)}",
        f"  compile cache: {compiler_stats}",
    ]
    _diagnostic_logger().info("\n".join(lines))


def log_fallback(reason: str) -> None:
    """One line per packed-K stream invalidation, only under CPUATTN_DEBUG."""
    if debug_enabled():
        _diagnostic_logger().info("packed-k full-pack fallback: %s", reason)


def _ms(nanoseconds: int | None) -> str:
    return "?" if nanoseconds is None else f"{nanoseconds / 1e6:.3f}ms"
