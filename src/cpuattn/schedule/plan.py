from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any


class LoweringKind(str, Enum):
    PARALLEL_BLOCKED = "parallel_blocked"
    PARALLEL_SPLIT_K = "parallel_split_k"
    PARALLEL_2D = "parallel_2d"
    LINEAR_SCAN = "linear_scan"
    LINEAR_CHUNKED = "linear_chunked"
    LINEAR_2D = "linear_2d"


class ScheduleKind(str, Enum):
    STATIC = "static"
    DYNAMIC = "dynamic"


class DecompositionKind(str, Enum):
    QUERY_BLOCKS = "query_blocks"
    SPLIT_KEY = "split_key"
    QUERY_GROUPS_KEY_PARTS = "query_groups_key_parts"
    STATE_HEADS = "state_heads"
    STATE_HEADS_DV_BLOCKS = "state_heads_dv_blocks"


class PackingKind(str, Enum):
    NONE = "none"
    K_TRANSPOSED = "k_transposed"


class MergeKind(str, Enum):
    NONE = "none"
    ROW_NORM = "row_norm"


class FusionKind(str, Enum):
    PATTERN_OWNED = "pattern_owned"


class PlacementKind(str, Enum):
    NUMA_LOCAL = "numa_local"
    WORKER_LOCAL = "worker_local"


@dataclass(frozen=True, slots=True)
class TileShape:
    q: int
    k: int
    d: int
    dv: int

    def __post_init__(self) -> None:
        if min(self.q, self.k, self.d, self.dv) <= 0:
            raise ValueError("tile dimensions must be positive")


@dataclass(frozen=True, slots=True)
class MicrokernelSpec:
    name: str
    qk_vectors: int = 1

    def __post_init__(self) -> None:
        if not self.name or self.qk_vectors <= 0:
            raise ValueError("microkernel name and qk_vectors must be valid")


@dataclass(frozen=True, slots=True)
class WorkspaceABI:
    region_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.region_names or any(not name for name in self.region_names):
            raise ValueError("workspace ABI must declare named regions")
        if len(self.region_names) != len(set(self.region_names)):
            raise ValueError("workspace ABI region names must be unique")


@dataclass(frozen=True, slots=True)
class CodePlan:
    backend_id: str
    lowering: LoweringKind
    microkernel: MicrokernelSpec
    tile: TileShape
    vector_bytes: int
    decomposition: DecompositionKind
    packing: PackingKind
    merge: MergeKind
    fusion: FusionKind
    workspace_abi: WorkspaceABI
    schedule: ScheduleKind = ScheduleKind.STATIC
    # canonical() memoizes into this slot and returns the shared dict:
    # callers must treat canonical() output as read-only.
    _canonical_memo: tuple[dict[str, Any], str] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not self.backend_id or self.vector_bytes <= 0:
            raise ValueError("code plan backend and vector width must be valid")
        parallel = self.lowering in {
            LoweringKind.PARALLEL_BLOCKED,
            LoweringKind.PARALLEL_SPLIT_K,
            LoweringKind.PARALLEL_2D,
        }
        if not parallel and self.packing is not PackingKind.NONE:
            raise ValueError("Linear code plans cannot pack K")
        if self.schedule is not ScheduleKind.STATIC and (
            self.lowering is not LoweringKind.PARALLEL_BLOCKED
        ):
            raise ValueError(
                "only parallel_blocked code plans carry a work-sharing schedule"
            )
        expects_packed = "packed_k" in self.workspace_abi.region_names
        if expects_packed != (self.packing is PackingKind.K_TRANSPOSED):
            raise ValueError("workspace ABI must match the code plan packing")
        if self.lowering in {LoweringKind.PARALLEL_SPLIT_K, LoweringKind.PARALLEL_2D}:
            if self.merge is not MergeKind.ROW_NORM:
                raise ValueError("split-key code plans require a RowNorm merge")
        elif self.merge is not MergeKind.NONE:
            raise ValueError("non-split code plans cannot declare a merge")

    def _memo(self) -> tuple[dict[str, Any], str]:
        if self._canonical_memo is None:
            canonical = _canonical(self)
            object.__setattr__(self, "_canonical_memo", (canonical, _digest(canonical)))
        return self._canonical_memo

    def canonical(self) -> dict[str, Any]:
        return self._memo()[0]

    @property
    def identity(self) -> str:
        return self._memo()[1]


@dataclass(frozen=True, slots=True)
class WorkerGroup:
    numa_node: int
    cpu_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.numa_node < 0 or not self.cpu_ids:
            raise ValueError("worker group must name a NUMA node and CPUs")
        if len(self.cpu_ids) != len(set(self.cpu_ids)):
            raise ValueError("worker group CPU IDs must be unique")


@dataclass(frozen=True, slots=True)
class LaunchPlan:
    cpu_ids: tuple[int, ...]
    groups: tuple[WorkerGroup, ...]

    def __post_init__(self) -> None:
        if not self.cpu_ids or len(self.cpu_ids) != len(set(self.cpu_ids)):
            raise ValueError("launch CPU IDs must be non-empty and unique")
        grouped = tuple(cpu for group in self.groups for cpu in group.cpu_ids)
        if len(grouped) != len(set(grouped)) or set(grouped) != set(self.cpu_ids):
            raise ValueError("every launch CPU must belong to exactly one worker group")

    @property
    def workers(self) -> int:
        return len(self.cpu_ids)

    @property
    def crosses_numa(self) -> bool:
        return len(self.groups) > 1

    def canonical(self) -> dict[str, Any]:
        return _canonical(self)


@dataclass(frozen=True, slots=True)
class MemoryRegion:
    name: str
    offset: int
    size_bytes: int
    alignment: int
    placement: PlacementKind
    owner_groups: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.name or self.offset < 0 or self.size_bytes <= 0:
            raise ValueError("memory region name, offset, and size must be valid")
        if self.alignment <= 0 or self.alignment & (self.alignment - 1):
            raise ValueError("memory region alignment must be a power of two")
        if self.offset % self.alignment:
            raise ValueError("memory region offset violates alignment")
        if not self.owner_groups:
            raise ValueError("memory regions must declare owner groups")
        if len(self.owner_groups) != len(set(self.owner_groups)):
            raise ValueError("memory region owners must be unique")
        if (
            self.placement is PlacementKind.NUMA_LOCAL
            and self.size_bytes % len(self.owner_groups)
        ):
            raise ValueError("NUMA-local region size must divide evenly across owners")

    @property
    def replica_count(self) -> int:
        if self.placement is not PlacementKind.NUMA_LOCAL:
            raise AttributeError("worker-local replicas are defined by the LaunchPlan")
        return len(self.owner_groups)

    @property
    def replica_bytes(self) -> int:
        return self.size_bytes // self.replica_count


@dataclass(frozen=True, slots=True)
class MemoryPlan:
    regions: tuple[MemoryRegion, ...]
    total_bytes: int

    def __post_init__(self) -> None:
        if not self.regions or self.total_bytes <= 0:
            raise ValueError("memory plan must contain storage")
        if len({region.name for region in self.regions}) != len(self.regions):
            raise ValueError("memory region names must be unique")
        ordered = sorted(self.regions, key=lambda region: region.offset)
        end = 0
        for region in ordered:
            if region.offset < end:
                raise ValueError("memory regions overlap")
            end = region.offset + region.size_bytes
        if end != self.total_bytes:
            raise ValueError("memory plan total must end at the final region")

    def canonical(self) -> dict[str, Any]:
        return _canonical(self)

    @property
    def identity(self) -> str:
        return _digest(self.canonical())


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    code: CodePlan
    launch: LaunchPlan
    memory: MemoryPlan
    _identity_memo: str | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def canonical(self) -> dict[str, Any]:
        return _canonical(self)

    @property
    def identity(self) -> str:
        if self._identity_memo is None:
            object.__setattr__(self, "_identity_memo", _digest(self.canonical()))
        return self._identity_memo


@dataclass(frozen=True, slots=True)
class CompiledPlan:
    code: CodePlan
    artifact_key: str
    library: Path
    source: Path
    manifest: Path


@dataclass(frozen=True, slots=True)
class TimedPlan:
    plan: ExecutionPlan
    compiled: CompiledPlan
    latency_ns: int
    result: object

    def __post_init__(self) -> None:
        if self.latency_ns < 0:
            raise ValueError("latency must be non-negative")


def _canonical(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _canonical(getattr(value, field.name))
            for field in fields(value)
            if not field.name.startswith("_")
        }
    if isinstance(value, tuple):
        return [_canonical(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in value.items()}
    return value


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "CodePlan",
    "CompiledPlan",
    "DecompositionKind",
    "ExecutionPlan",
    "FusionKind",
    "LaunchPlan",
    "LoweringKind",
    "MemoryPlan",
    "MemoryRegion",
    "MergeKind",
    "MicrokernelSpec",
    "PackingKind",
    "PlacementKind",
    "ScheduleKind",
    "TileShape",
    "TimedPlan",
    "WorkerGroup",
    "WorkspaceABI",
]
