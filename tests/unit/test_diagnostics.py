from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from cpuattn import Parallel, Runtime, expr
from cpuattn.diagnostics import (
    debug_enabled,
    plan_label,
    reset_debug_cache,
)
from cpuattn.schedule.plan import (
    CodePlan,
    DecompositionKind,
    ExecutionPlan,
    FusionKind,
    LaunchPlan,
    LoweringKind,
    MemoryPlan,
    MemoryRegion,
    MicrokernelSpec,
    MergeKind,
    PackingKind,
    PlacementKind,
    ScheduleKind,
    TileShape,
    WorkerGroup,
    WorkspaceABI,
)


def test_debug_enabled_reads_the_environment_once(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_debug_cache()
    monkeypatch.setenv("CPUATTN_DEBUG", "1")
    assert debug_enabled() is True
    monkeypatch.setenv("CPUATTN_DEBUG", "0")
    assert debug_enabled() is True  # cached from the first read
    reset_debug_cache()
    assert debug_enabled() is False
    monkeypatch.setenv("CPUATTN_DEBUG", "true")
    reset_debug_cache()
    assert debug_enabled() is True
    reset_debug_cache()
    monkeypatch.delenv("CPUATTN_DEBUG", raising=False)
    reset_debug_cache()


def _region(name: str, offset: int, size: int) -> MemoryRegion:
    return MemoryRegion(
        name=name, offset=offset, size_bytes=size, alignment=64,
        placement=PlacementKind.NUMA_LOCAL, owner_groups=(0,),
    )


def _first_selection_plan(runtime: Runtime):
    rng = np.random.default_rng(3)
    operator = Parallel(score_mod=expr.identity() * 0.25, mask_mod=expr.causal())
    q = rng.normal(size=(1, 2, 4, 8)).astype("float32")
    k = rng.normal(size=(1, 1, 6, 8)).astype("float32")
    v = rng.normal(size=(1, 1, 6, 8)).astype("float32")
    runtime.run(operator, q=q, k=k, v=v)
    return runtime.last_selection.winner.plan


def test_plan_label_matches_rendered_fields(tmp_path) -> None:
    plan = _first_selection_plan(Runtime(cache_dir=tmp_path))
    label = plan_label(plan)
    assert plan.code.lowering.value in label
    assert f"workers={plan.launch.workers}" in label


def test_plan_label_surfaces_non_static_schedule() -> None:
    """The work-sharing schedule appears only when it is not the default."""
    cpus = (0,)
    code = CodePlan(
        backend_id="fixture",
        lowering=LoweringKind.PARALLEL_BLOCKED,
        microkernel=MicrokernelSpec("qk_m1n1_pv_m1", 1),
        tile=TileShape(1, 16, 8, 8),
        vector_bytes=32,
        decomposition=DecompositionKind.QUERY_BLOCKS,
        packing=PackingKind.NONE,
        merge=MergeKind.NONE,
        fusion=FusionKind.PATTERN_OWNED,
        workspace_abi=WorkspaceABI(("scratch",)),
        schedule=ScheduleKind.DYNAMIC,
    )
    plan = ExecutionPlan(
        code,
        LaunchPlan(cpus, (WorkerGroup(0, cpus),)),
        MemoryPlan((_region("scratch", 0, 128),), 128),
    )
    assert "/dynamic" in plan_label(plan)
    static = replace(plan, code=replace(code, schedule=ScheduleKind.STATIC))
    assert "/dynamic" not in plan_label(static)
    assert "/static" not in plan_label(static)


def test_runtime_explain_is_structured_and_serializable(tmp_path) -> None:
    runtime = Runtime(cache_dir=tmp_path)
    assert runtime.explain() is None

    rng = np.random.default_rng(9)
    operator = Parallel(score_mod=expr.identity() * 0.25, mask_mod=expr.causal())
    runtime.run(
        operator,
        q=rng.normal(size=(1, 2, 4, 8)).astype("float32"),
        k=rng.normal(size=(1, 1, 6, 8)).astype("float32"),
        v=rng.normal(size=(1, 1, 6, 8)).astype("float32"),
    )

    record = runtime.explain()
    assert record is not None
    payload = record.as_json()
    assert payload["host"]["fingerprint"] == runtime.host.fingerprint
    assert payload["backend"]["selected"] == runtime.backend.backend_id
    assert any(
        item["id"] == runtime.backend.backend_id and item["compatible"]
        for item in payload["backend"]["candidates"]
    )
    assert payload["plans"]["execution_plans"] > 0
    assert payload["measurements"] and payload["winner"]["plan_id"]
    assert payload["workspace"]["regions"]
    assert payload["workspace"]["summary"].startswith("total=")
    assert set(payload["compile_cache"]) == {"memory_hits", "disk_hits", "builds"}
    assert payload["numeric"] is None
    json.dumps(payload)  # must be JSON-serializable

    attached = record.with_numeric({"max_abs_err": 1e-6, "ok": True})
    assert attached.as_json()["numeric"] == {"max_abs_err": 1e-6, "ok": True}
    assert record.numeric is None


def test_selection_diagnostics_block_on_stderr(
    tmp_path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CPUATTN_DEBUG", "1")
    reset_debug_cache()
    runtime = Runtime(cache_dir=tmp_path)
    rng = np.random.default_rng(4)
    operator = Parallel(score_mod=expr.identity() * 0.25, mask_mod=expr.causal())
    q = rng.normal(size=(1, 2, 4, 8)).astype("float32")
    k = rng.normal(size=(1, 1, 6, 8)).astype("float32")
    v = rng.normal(size=(1, 1, 6, 8)).astype("float32")
    runtime.run(operator, q=q, k=k, v=v)
    err = capsys.readouterr().err
    assert "[cpuattn-debug] selection" in err
    assert "legal plans:" in err
    assert "tuner measurements:" in err
    assert "winner:" in err
    assert "workspace:" in err
    assert "compile cache:" in err
    reset_debug_cache()


def test_compile_stats_count_memory_hits(tmp_path) -> None:
    runtime = Runtime(cache_dir=tmp_path)
    rng = np.random.default_rng(5)
    operator = Parallel(score_mod=expr.identity() * 0.25, mask_mod=expr.causal())
    q = rng.normal(size=(1, 2, 4, 8)).astype("float32")
    k = rng.normal(size=(1, 1, 6, 8)).astype("float32")
    v = rng.normal(size=(1, 1, 6, 8)).astype("float32")
    runtime.run(operator, q=q, k=k, v=v)
    stats_first = dict(runtime.compiler.stats)
    # Tuner enumeration compiles candidates repeatedly; in-memory hits dominate.
    assert stats_first["memory_hits"] >= 1
    assert stats_first["builds"] + stats_first["disk_hits"] >= 1
    runtime.run(operator, q=q, k=k, v=v)
    # The replayed winner comes from the runtime kernel cache: zero compile calls.
    assert runtime.compiler.stats == stats_first


def test_log_fallback_prints_only_when_enabled(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from cpuattn.diagnostics import log_fallback

    reset_debug_cache()
    log_fallback("non-growing")
    assert capsys.readouterr().err == ""
    monkeypatch.setenv("CPUATTN_DEBUG", "1")
    reset_debug_cache()
    log_fallback("non-growing")
    assert "packed-k full-pack fallback: non-growing" in capsys.readouterr().err
    reset_debug_cache()


def test_default_info_visibility_survives_implicit_events(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from cpuattn import log as cpuattn_log

    cpuattn_log._LOGGER.handlers.clear()
    cpuattn_log._LOGGER.setLevel(cpuattn_log.logging.NOTSET)
    cpuattn_log.event("visibility probe %d", 7)
    assert "visibility probe 7" in capsys.readouterr().err
    cpuattn_log._LOGGER.handlers.clear()


def test_explicit_level_survives_implicit_events() -> None:
    from cpuattn import log as cpuattn_log

    cpuattn_log._LOGGER.handlers.clear()
    try:
        cpuattn_log.configure_logging(cpuattn_log.logging.ERROR)
        cpuattn_log.event("should stay hidden")
        assert cpuattn_log._LOGGER.level == cpuattn_log.logging.ERROR
        assert cpuattn_log._LOGGER.filters == []
    finally:
        cpuattn_log._LOGGER.setLevel(cpuattn_log.logging.INFO)
        cpuattn_log._LOGGER.handlers.clear()
