from __future__ import annotations

from dataclasses import dataclass
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import re

from ..errors import UnsupportedError


@dataclass(frozen=True, slots=True)
class LogicalCpu:
    cpu_id: int
    core_id: int
    socket_id: int
    numa_node: int

    def canonical(self) -> dict[str, int]:
        return {
            "cpu_id": self.cpu_id,
            "core_id": self.core_id,
            "socket_id": self.socket_id,
            "numa_node": self.numa_node,
        }


@dataclass(frozen=True, slots=True)
class Cache:
    level: int
    kind: str
    size_bytes: int
    shared_cpus: tuple[int, ...]

    def canonical(self) -> dict[str, object]:
        return {
            "level": self.level,
            "kind": self.kind,
            "size_bytes": self.size_bytes,
            "shared_cpus": list(self.shared_cpus),
        }


@dataclass(frozen=True, slots=True)
class Host:
    architecture: str
    vendor: str
    model: str
    features: frozenset[str]
    cpus: tuple[LogicalCpu, ...]
    caches: tuple[Cache, ...] = ()
    sve_vector_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.architecture not in {"x86_64", "aarch64"}:
            raise UnsupportedError(f"unsupported CPU architecture {self.architecture!r}")
        ids = tuple(cpu.cpu_id for cpu in self.cpus)
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("Host must contain a non-empty unique allowed CPU set")
        if self.sve_vector_bytes is not None and self.sve_vector_bytes <= 0:
            raise ValueError("SVE vector length must be positive")

    @property
    def allowed_cpu_ids(self) -> tuple[int, ...]:
        return tuple(cpu.cpu_id for cpu in self.cpus)

    @property
    def physical_cores(self) -> int:
        return len({(cpu.socket_id, cpu.core_id) for cpu in self.cpus})

    def canonical(self) -> dict[str, object]:
        return {
            "architecture": self.architecture,
            "vendor": self.vendor,
            "model": self.model,
            "features": sorted(self.features),
            "cpus": [cpu.canonical() for cpu in self.cpus],
            "caches": [cache.canonical() for cache in self.caches],
            "sve_vector_bytes": self.sve_vector_bytes,
        }

    @property
    def fingerprint(self) -> str:
        data = json.dumps(self.canonical(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(data.encode()).hexdigest()


def detect_host() -> Host:
    architecture = _normalize_architecture(platform.machine())
    allowed = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(os.cpu_count() or 1))
    records = _cpuinfo_records(Path("/proc/cpuinfo"))
    selected = _allowed_cpuinfo_records(records, allowed)
    first = selected[0] if selected else {}
    vendor = first.get("vendor_id") or first.get("CPU implementer") or "unknown"
    model = first.get("model name") or first.get("Processor") or first.get("CPU part") or "unknown"
    feature_sets = [
        set((record.get("flags") or record.get("Features") or "").lower().split())
        for record in selected
    ]
    features = frozenset(set.intersection(*feature_sets)) if feature_sets else frozenset()
    cpus = tuple(_logical_cpu(cpu_id) for cpu_id in allowed)
    sve_vector_bytes = _sve_vector_bytes() if "sve" in features else None
    return Host(
        architecture,
        vendor,
        model,
        features,
        cpus,
        _read_caches(allowed),
        sve_vector_bytes,
    )


def _sve_vector_bytes() -> int | None:
    """Return the Linux thread's active SVE vector length in bytes."""
    pr_sve_get_vl = 51
    pr_sve_vl_len_mask = 0xFFFF
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        result = libc.prctl(pr_sve_get_vl, 0, 0, 0, 0)
    except (AttributeError, OSError):
        return None
    if result < 0:
        return None
    length = result & pr_sve_vl_len_mask
    return length or None


def _allowed_cpuinfo_records(
    records: list[dict[str, str]],
    allowed: list[int],
) -> list[dict[str, str]]:
    by_cpu: dict[int, dict[str, str]] = {}
    for record in records:
        try:
            by_cpu[int(record["processor"])] = record
        except (KeyError, ValueError):
            continue
    selected = [by_cpu[cpu_id] for cpu_id in allowed if cpu_id in by_cpu]
    if len(selected) == len(allowed):
        return selected
    # If cpuinfo cannot be mapped completely, intersecting every record is the
    # conservative option: it can disable an optimization but cannot enable an
    # instruction absent from an allowed CPU.
    return records


def _normalize_architecture(value: str) -> str:
    normalized = value.lower()
    if normalized in {"x86_64", "amd64"}:
        return "x86_64"
    if normalized in {"aarch64", "arm64"}:
        return "aarch64"
    raise UnsupportedError(f"unsupported CPU architecture {value!r}")


def _cpuinfo_records(path: Path) -> list[dict[str, str]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    records: list[dict[str, str]] = []
    for block in text.strip().split("\n\n"):
        record = {}
        for line in block.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                record[key.strip()] = value.strip()
        if record:
            records.append(record)
    return records


def _read_int(path: Path, fallback: int) -> int:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return fallback


def _logical_cpu(cpu_id: int) -> LogicalCpu:
    root = Path(f"/sys/devices/system/cpu/cpu{cpu_id}")
    core = _read_int(root / "topology/core_id", cpu_id)
    socket = _read_int(root / "topology/physical_package_id", 0)
    numa = 0
    try:
        nodes = sorted(root.glob("node[0-9]*"))
        if nodes:
            numa = int(nodes[0].name[4:])
    except (OSError, ValueError):
        pass
    return LogicalCpu(cpu_id, core, socket, numa)


def _read_caches(allowed: list[int]) -> tuple[Cache, ...]:
    if not allowed:
        return ()
    result: dict[tuple[int, str, int, tuple[int, ...]], Cache] = {}
    allowed_set = set(allowed)
    for cpu_id in allowed:
        root = Path(f"/sys/devices/system/cpu/cpu{cpu_id}/cache")
        for index in root.glob("index*"):
            try:
                level = int((index / "level").read_text().strip())
                kind = (index / "type").read_text().strip().lower()
                size = _parse_size((index / "size").read_text().strip())
                shared = tuple(sorted(_parse_cpu_list((index / "shared_cpu_list").read_text()) & allowed_set))
            except (OSError, ValueError):
                continue
            cache = Cache(level, kind, size, shared)
            result[(level, kind, size, shared)] = cache
    return tuple(sorted(result.values(), key=lambda item: (item.level, item.kind, item.shared_cpus)))


def _parse_size(text: str) -> int:
    match = re.fullmatch(r"(\d+)([KMG]?)", text.strip(), re.IGNORECASE)
    if match is None:
        raise ValueError(text)
    scale = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3}[match.group(2).upper()]
    return int(match.group(1)) * scale


def _parse_cpu_list(text: str) -> set[int]:
    result: set[int] = set()
    for part in text.strip().split(","):
        if not part:
            continue
        bounds = part.split("-", 1)
        start = int(bounds[0])
        end = int(bounds[1]) if len(bounds) == 2 else start
        result.update(range(start, end + 1))
    return result


__all__ = ["Cache", "Host", "LogicalCpu", "detect_host"]
