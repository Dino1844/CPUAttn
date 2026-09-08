import numpy as np
import pytest

import cpuattn.hardware.host as host_module
from cpuattn import Linear, Parallel, UnsupportedError, expr, rownorm, transition
from cpuattn.core.validate import validate_linear_call, validate_parallel_call
from cpuattn.hardware.host import Host, LogicalCpu, detect_host
from cpuattn.native.backends import select_backend
from cpuattn.schedule.plan import DecompositionKind, LoweringKind, PackingKind
from cpuattn.schedule.planning import PlanBuilder


def _host(
    arch: str,
    features: set[str],
    vendor: str = "fixture",
    *,
    sve_vector_bytes: int | None = None,
) -> Host:
    return Host(
        arch,
        vendor,
        "model",
        frozenset(features),
        tuple(LogicalCpu(i, i, 0, 0) for i in range(4)),
        sve_vector_bytes=sve_vector_bytes,
    )


@pytest.mark.parametrize(
    "fixture,expected",
    [
        (_host("x86_64", {"avx2", "fma"}, "GenuineIntel"), "x86_avx2_fma"),
        (
            _host("x86_64", {"avx2", "fma", "avx512f"}, "AuthenticAMD"),
            "x86_avx512",
        ),
        (_host("aarch64", {"asimd"}), "arm64_neon"),
        (
            _host(
                "aarch64", {"asimd", "sve"}, sve_vector_bytes=32
            ),
            "arm64_sve",
        ),
        (_host("aarch64", {"asimd", "sve"}), "arm64_neon"),
    ],
)
def test1(fixture: Host, expected: str) -> None:
    """Backend selection follows detected architecture and ISA capabilities."""
    assert select_backend(fixture).backend_id == expected


def test2() -> None:
    """Backend selection has no unsupported fallback."""
    with pytest.raises(UnsupportedError, match="no compiled backend"):
        select_backend(_host("x86_64", {"sse2"}))


def test3() -> None:
    """Parallel code plans are deterministic, complete, and structurally legal."""
    fixture = _host("x86_64", {"avx2", "fma"})
    backend = select_backend(fixture)
    operator = Parallel()
    call = validate_parallel_call(
        operator,
        q=np.zeros((1, 4, 17, 15), np.float32),
        k=np.zeros((1, 2, 65, 15), np.float32),
        v=np.zeros((1, 2, 65, 19), np.float32),
    )
    first = backend.enumerate_code_plans(operator, call, fixture)
    second = backend.enumerate_code_plans(operator, call, fixture)

    assert first == second
    assert len({code.identity for code in first}) == len(first)
    assert not hasattr(first[0], "cpu_ids")
    assert not hasattr(first[0], "workers")
    assert {code.decomposition for code in first} == {
        DecompositionKind.QUERY_BLOCKS,
        DecompositionKind.SPLIT_KEY,
        DecompositionKind.QUERY_GROUPS_KEY_PARTS,
    }
    blocked = [
        code for code in first if code.lowering is LoweringKind.PARALLEL_BLOCKED
    ]
    assert all(code.tile.q > 1 for code in blocked)
    assert {code.packing for code in blocked} == {
        PackingKind.NONE,
        PackingKind.K_TRANSPOSED,
    }
    assert {code.microkernel.qk_vectors for code in blocked} == {1, 2, 4}

    no_merge = rownorm.RowNorm(
        name="unmerged_identity",
        states=(),
        updates=(),
        rescale=expr.as_expr(1.0),
        weight=expr.var("score"),
        final_scale=expr.as_expr(1.0),
        reference=rownorm.Reference((), expr.var("score")),
    )
    arrays = {
        "q": np.zeros((1, 4, 2, 3), np.float32),
        "k": np.zeros((1, 1, 65, 3), np.float32),
        "v": np.zeros((1, 1, 65, 5), np.float32),
    }
    without = Parallel(row_norm=no_merge)
    codes = backend.enumerate_code_plans(
        without, validate_parallel_call(without, **arrays), fixture
    )
    assert all(code.lowering is LoweringKind.PARALLEL_BLOCKED for code in codes)


def test4() -> None:
    """Linear chunking follows transition structure and keeps the general scan."""
    fixture = _host("x86_64", {"avx2", "fma"})
    backend = select_backend(fixture)
    tensor = np.zeros((1, 1, 7, 9), np.float32)
    values = {"q": tensor, "k": tensor, "v": tensor}
    chunkable = Linear(
        transition=transition.program(
            transition.scale(0.97),
            transition.outer(expr.var("k"), expr.var("v")),
        )
    )
    general = Linear(
        transition=transition.program(
            transition.rank1(expr.var("k"), expr.var("k")),
            transition.outer(expr.var("k"), expr.var("v")),
        )
    )
    chunkable_ids = {
        code.lowering
        for code in backend.enumerate_code_plans(
            chunkable, validate_linear_call(chunkable, **values), fixture
        )
    }
    general_ids = {
        code.lowering
        for code in backend.enumerate_code_plans(
            general, validate_linear_call(general, **values), fixture
        )
    }
    assert chunkable_ids == {
        LoweringKind.LINEAR_SCAN,
        LoweringKind.LINEAR_CHUNKED,
        LoweringKind.LINEAR_2D,
    }
    assert general_ids == {LoweringKind.LINEAR_SCAN, LoweringKind.LINEAR_2D}


def test5() -> None:
    """PlanBuilder emits legal NUMA placements and page-aligned memory."""
    cpus = tuple(
        LogicalCpu(cpu_id, cpu_id % 4, cpu_id // 4, cpu_id // 4)
        for cpu_id in range(8)
    )
    fixture = Host(
        "x86_64", "fixture", "dual", frozenset({"avx2", "fma"}), cpus
    )
    backend = select_backend(fixture)
    operator = Parallel()
    call = validate_parallel_call(
        operator,
        q=np.zeros((1, 8, 8, 5), np.float32),
        k=np.zeros((1, 8, 65, 5), np.float32),
        v=np.zeros((1, 8, 65, 9), np.float32),
    )
    codes = backend.enumerate_code_plans(operator, call, fixture)
    plans = PlanBuilder(fixture, page_size=4096).build_all(codes, operator, call)
    placements = {
        (plan.launch.workers, plan.launch.crosses_numa, plan.launch.cpu_ids)
        for plan in plans
        if plan.code.lowering is LoweringKind.PARALLEL_BLOCKED
    }
    assert any(workers == 1 and not cross for workers, cross, _ in placements)
    assert any(workers == 4 and not cross for workers, cross, _ in placements)
    assert any(
        workers == 4 and cross and ids == (0, 4, 1, 5)
        for workers, cross, ids in placements
    )
    assert any(workers == 8 and cross for workers, cross, _ in placements)

    plan = plans[-1]
    assert tuple(region.name for region in plan.memory.regions) == (
        plan.code.workspace_abi.region_names
    )
    assert all(region.offset % region.alignment == 0 for region in plan.memory.regions)
    assert plan.memory.total_bytes == (
        plan.memory.regions[-1].offset + plan.memory.regions[-1].size_bytes
    )


def test6(monkeypatch: pytest.MonkeyPatch) -> None:
    """Host snapshots are stable and intersect features across allowed CPUs."""
    real = detect_host()
    assert real.allowed_cpu_ids
    assert real.physical_cores > 0
    assert real.fingerprint == detect_host().fingerprint

    records = [
        {
            "processor": "0",
            "vendor_id": "outside",
            "model name": "outside",
            "flags": "avx2 avx512f",
        },
        {
            "processor": "1",
            "vendor_id": "GenuineIntel",
            "model name": "allowed",
            "flags": "avx2 fma",
        },
        {
            "processor": "2",
            "vendor_id": "GenuineIntel",
            "model name": "allowed",
            "flags": "avx2",
        },
    ]
    monkeypatch.setattr(host_module.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(host_module, "_cpuinfo_records", lambda path: records)
    monkeypatch.setattr(host_module.os, "sched_getaffinity", lambda pid: {1, 2})
    monkeypatch.setattr(
        host_module,
        "_logical_cpu",
        lambda cpu_id: LogicalCpu(cpu_id, cpu_id, 0, 0),
    )
    monkeypatch.setattr(host_module, "_read_caches", lambda allowed: ())

    snapshot = detect_host()
    assert snapshot.allowed_cpu_ids == (1, 2)
    assert snapshot.vendor == "GenuineIntel"
    assert snapshot.model == "allowed"
    assert snapshot.features == frozenset({"avx2"})
