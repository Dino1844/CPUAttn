from dataclasses import replace
import os
from pathlib import Path

import numpy as np
import pytest

from cpuattn import Axis, Linear, Parallel, TensorArgSpec, expr, rownorm, transition
from cpuattn.native.backends import select_backend
from cpuattn.native.compiler import Compiler
from cpuattn.native.executor import Executor
from cpuattn.hardware.host import detect_host
from cpuattn.schedule.plan import LoweringKind
from cpuattn.schedule.planning import PlanBuilder
from cpuattn.core.validate import validate_linear_call, validate_parallel_call
from tests._reference import linear as reference_linear
from tests._reference import parallel as reference_parallel


@pytest.fixture(scope="module")
def native():
    host = detect_host()
    return host, select_backend(host)


def _plans(operator, call, host, backend):
    codes = backend.enumerate_code_plans(operator, call, host)
    return PlanBuilder(host).build_all(codes, operator, call)


def _execute(compiler, operator, call, plan, backend):
    compiled = compiler.compile(operator, call, plan.code, backend)
    return Executor().run(compiler.load(compiled), plan, call).result


def test1(tmp_path: Path, native) -> None:
    """Launch preparation restores the caller's CPU affinity."""
    host, backend = native
    operator = Parallel()
    tensor = np.ones((1, 1, 3, 5), np.float32)
    call = validate_parallel_call(operator, q=tensor, k=tensor, v=tensor)
    plan = min(
        _plans(operator, call, host, backend),
        key=lambda item: (
            abs(item.launch.workers - min(2, len(host.allowed_cpu_ids))),
            item.identity,
        ),
    )
    compiler = Compiler(tmp_path)
    kernel = compiler.load(compiler.compile(operator, call, plan.code, backend))
    affinity = os.sched_getaffinity(0)

    Executor().prepare_launch(kernel, plan.launch)

    assert os.sched_getaffinity(0) == affinity


def test2(tmp_path: Path, native) -> None:
    """Every Parallel plan handles Mods, GQA, tails, and 2D work."""
    host, backend = native
    rng = np.random.default_rng(31)
    scale = rng.uniform(0.02, 0.2, size=(4, 17)).astype(np.float32)
    operator = Parallel(
        score_mod=expr.identity() * expr.argument("scale"),
        mask_mod=expr.causal(),
        row_norm=rownorm.softmax(scale=0.125),
        arguments=(TensorArgSpec("scale", (Axis.QUERY_HEAD, Axis.QUERY)),),
    )
    call = validate_parallel_call(
        operator,
        q=rng.normal(size=(1, 4, 17, 15)).astype(np.float32),
        k=rng.normal(size=(1, 2, 65, 15)).astype(np.float32),
        v=rng.normal(size=(1, 2, 65, 7)).astype(np.float32),
        arguments={"scale": scale},
    )
    expected = reference_parallel(operator, call)
    compiler = Compiler(tmp_path)
    plans = _plans(operator, call, host, backend)
    assert {plan.code.lowering for plan in plans} == {
        LoweringKind.PARALLEL_BLOCKED,
        LoweringKind.PARALLEL_SPLIT_K,
        LoweringKind.PARALLEL_2D,
    }
    for plan in plans:
        actual = _execute(compiler, operator, call, plan, backend)
        np.testing.assert_allclose(
            actual, expected, rtol=2e-4, atol=2e-5, err_msg=plan.identity
        )


@pytest.mark.parametrize(
    "mask,expected",
    [
        (expr.causal(), 1),
        (expr.causal() & (expr.var("key_index") >= 2.0), 1),
        (expr.causal() | (expr.var("key_index") >= 2.0), 0),
        (expr.var("key_index") >= 2.0, 0),
    ],
)
def test3(
    tmp_path: Path, native, mask, expected: int
) -> None:
    """Generated loops prune only masks that structurally imply causality."""
    host, backend = native
    operator = Parallel(mask_mod=mask)
    q = np.ones((1, 1, 17, 15), np.float32)
    k = np.ones((1, 1, 33, 15), np.float32)
    v = np.ones((1, 1, 33, 19), np.float32)
    call = validate_parallel_call(operator, q=q, k=k, v=v)
    code = next(
        item
        for item in backend.enumerate_code_plans(operator, call, host)
        if item.lowering is LoweringKind.PARALLEL_BLOCKED
    )
    source = Compiler(tmp_path).source(operator, call, code)
    assert f"#define CAUSAL_BLOCK_LIMIT {expected}" in source


def test4(tmp_path: Path, native) -> None:
    """Parallel source selects SIMD and Softmax by structure."""
    host, backend = native
    tensor = np.ones((1, 1, 3, 5), np.float32)

    def source_for(operator: Parallel) -> str:
        call = validate_parallel_call(operator, q=tensor, k=tensor, v=tensor)
        code = next(
            item
            for item in backend.enumerate_code_plans(operator, call, host)
            if item.lowering is LoweringKind.PARALLEL_BLOCKED
        )
        return Compiler(tmp_path).source(operator, call, code)

    optimized = source_for(Parallel(row_norm=rownorm.softmax(scale=0.125)))
    generic = source_for(Parallel(row_norm=rownorm.sigmoid_norm(scale=0.125)))
    misleading = source_for(Parallel(row_norm=replace(rownorm.identity(), name="softmax")))
    assert '#include "simd/parallel.h"' in optimized
    assert "cpuattn_softmax_tile" in optimized
    assert "cpuattn_softmax_tile" not in generic
    assert "cpuattn_softmax_tile" not in misleading


def test5(tmp_path: Path, native) -> None:
    """Every Linear plan handles KDA updates and grouped broadcast."""
    host, backend = native
    rng = np.random.default_rng(32)
    operator = Linear(
        transition=transition.program(
            transition.rank1(expr.var("k") * -0.025, expr.var("k")),
            transition.outer(expr.var("k"), expr.var("v")),
        ),
        readout=transition.readout(timing=transition.ReadTiming.AFTER),
    )
    call = validate_linear_call(
        operator,
        q=rng.normal(size=(1, 1, 3, 5)).astype(np.float32),
        k=rng.normal(size=(1, 1, 3, 5)).astype(np.float32),
        v=rng.normal(size=(1, 8, 3, 17)).astype(np.float32),
    )
    expected = reference_linear(operator, call)
    compiler = Compiler(tmp_path)
    plans = _plans(operator, call, host, backend)
    assert LoweringKind.LINEAR_2D in {plan.code.lowering for plan in plans}
    for plan in plans:
        actual = _execute(compiler, operator, call, plan, backend)
        np.testing.assert_allclose(
            actual.output, expected.output, rtol=2e-4, atol=2e-5, err_msg=plan.identity
        )
        np.testing.assert_allclose(
            actual.state, expected.state, rtol=2e-4, atol=2e-5, err_msg=plan.identity
        )


@pytest.mark.parametrize("timing", ["before", "after"])
@pytest.mark.parametrize("scaled", [False, True])
def test6(
    tmp_path: Path, native, timing: str, scaled: bool
) -> None:
    """Every chunked Linear microkernel matches ordered scan semantics."""
    host, backend = native
    rng = np.random.default_rng(203 + scaled)
    steps = (() if not scaled else (transition.scale(0.97),)) + (
        transition.outer(expr.var("k"), expr.var("v")),
    )
    operator = Linear(
        transition=transition.program(*steps),
        readout=transition.readout(timing=timing),
        q_mod=expr.tanh(expr.identity()),
        k_mod=expr.sigmoid(expr.identity()),
        v_mod=expr.relu(expr.identity()),
    )
    call = validate_linear_call(
        operator,
        q=rng.normal(size=(1, 1, 7, 17)).astype(np.float32),
        k=rng.normal(size=(1, 1, 7, 17)).astype(np.float32),
        v=rng.normal(size=(1, 3, 7, 19)).astype(np.float32),
    )
    expected = reference_linear(operator, call)
    compiler = Compiler(tmp_path)
    plans = [
        plan
        for plan in _plans(operator, call, host, backend)
        if plan.code.lowering is LoweringKind.LINEAR_CHUNKED
        and plan.launch.workers == 1
    ]
    assert plans
    for plan in plans:
        actual = _execute(compiler, operator, call, plan, backend)
        np.testing.assert_allclose(actual.output, expected.output, rtol=2e-4, atol=2e-5)
        np.testing.assert_allclose(actual.state, expected.state, rtol=2e-4, atol=2e-5)


def test7(tmp_path: Path, native) -> None:
    """Complex elementwise Mods use the shared SIMD expression lowering."""
    host, backend = native
    rng = np.random.default_rng(207)
    x = expr.identity()
    operator = Parallel(
        score_mod=expr.where(
            x > 0.0,
            expr.log(expr.abs_(x) + 1.0) + expr.tanh(x),
            expr.sigmoid(x) - expr.exp(expr.minimum(x, 0.0)),
        ),
        mask_mod=expr.argument("gate") > 0.0,
        row_norm=rownorm.identity(),
        arguments=(TensorArgSpec("gate", (Axis.KEY,)),),
    )
    call = validate_parallel_call(
        operator,
        q=rng.normal(size=(1, 2, 5, 17)).astype(np.float32),
        k=rng.normal(size=(1, 1, 33, 17)).astype(np.float32),
        v=rng.normal(size=(1, 1, 33, 19)).astype(np.float32),
        arguments={"gate": (rng.random(33) > 0.2).astype(np.float32)},
    )
    plan = next(
        item
        for item in _plans(operator, call, host, backend)
        if item.code.lowering is LoweringKind.PARALLEL_BLOCKED
        and item.launch.workers == 1
    )
    compiler = Compiler(tmp_path)
    source = compiler.source(operator, call, plan.code)
    assert "cpuattn_simd_log" in source
    assert "cpuattn_simd_tanh" in source
    assert "cpuattn_simd_select" in source
    actual = _execute(compiler, operator, call, plan, backend)
    np.testing.assert_allclose(actual, reference_parallel(operator, call), rtol=2e-4, atol=2e-5)


def test8(tmp_path: Path, native) -> None:
    """Packing is compile-time selected and remains inside native timing."""
    host, backend = native
    operator = Parallel()
    tensor = np.ones((1, 1, 33, 17), np.float32)
    call = validate_parallel_call(operator, q=tensor[:, :, :1], k=tensor, v=tensor)
    codes = [
        item
        for item in backend.enumerate_code_plans(operator, call, host)
        if item.lowering is LoweringKind.PARALLEL_BLOCKED
    ]
    compiler = Compiler(tmp_path)
    direct = compiler.source(
        operator,
        call,
        next(item for item in codes if item.packing.value == "none"),
    )
    packed = compiler.source(
        operator,
        call,
        next(item for item in codes if item.packing.value == "k_transposed"),
    )

    assert "cpuattn_now_ns" in direct
    assert "cpuattn_now_ns" in packed
    assert "int pack_k" not in direct + packed
    assert "cpuattn_pack_k_for_owner(" not in direct
    assert "cpuattn_pack_k_for_owner(" in packed


@pytest.mark.parametrize(
    "sq,skv,d,dv",
    [(1, 255, 15, 17), (16, 257, 17, 32), (33, 17, 7, 15)],
)
def test9(
    tmp_path: Path, native, sq: int, skv: int, d: int, dv: int
) -> None:
    """Parallel kernels handle sequence and channel tail neighborhoods."""
    host, backend = native
    rng = np.random.default_rng(sq + skv + d + dv)
    operator = Parallel(row_norm=rownorm.sigmoid_norm(scale=0.125))
    call = validate_parallel_call(
        operator,
        q=rng.normal(size=(1, 2, sq, d)).astype(np.float32),
        k=rng.normal(size=(1, 1, skv, d)).astype(np.float32),
        v=rng.normal(size=(1, 1, skv, dv)).astype(np.float32),
    )
    plan = next(
        item
        for item in _plans(operator, call, host, backend)
        if item.code.lowering is LoweringKind.PARALLEL_BLOCKED
        and item.launch.workers == 1
    )
    actual = _execute(Compiler(tmp_path), operator, call, plan, backend)
    np.testing.assert_allclose(actual, reference_parallel(operator, call), rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("lowering", [LoweringKind.PARALLEL_SPLIT_K, LoweringKind.PARALLEL_2D])
def test10(
    tmp_path: Path, native, lowering: LoweringKind
) -> None:
    """Partitioned Parallel kernels merge generic stateful RowNorms."""
    host, backend = native
    operator = Parallel(mask_mod=expr.causal(), row_norm=rownorm.sigmoid_norm(0.125))
    rng = np.random.default_rng(94)
    call = validate_parallel_call(
        operator,
        q=rng.normal(size=(1, 4, 3, 7)).astype(np.float32),
        k=rng.normal(size=(1, 1, 65, 7)).astype(np.float32),
        v=rng.normal(size=(1, 1, 65, 9)).astype(np.float32),
    )
    plan = next(item for item in _plans(operator, call, host, backend) if item.code.lowering is lowering)
    actual = _execute(Compiler(tmp_path), operator, call, plan, backend)
    np.testing.assert_allclose(actual, reference_parallel(operator, call), rtol=2e-4, atol=2e-5)


def test11(tmp_path: Path, native) -> None:
    """Native Linear state continuation preserves axis-bound arguments."""
    host, backend = native
    rng = np.random.default_rng(92)
    operator = Linear(
        transition=transition.program(
            transition.scale(expr.argument("decay")),
            transition.outer(expr.var("k"), expr.var("v") * expr.argument("value_scale")),
        ),
        arguments=(
            TensorArgSpec("decay", (Axis.PARAMETER_GROUP, Axis.SEQUENCE, Axis.D)),
            TensorArgSpec("value_scale", (Axis.STATE_HEAD, Axis.SEQUENCE, Axis.DV)),
        ),
    )
    q = rng.normal(size=(1, 2, 7, 17)).astype(np.float32)
    k = rng.normal(size=(1, 2, 7, 17)).astype(np.float32)
    v = rng.normal(size=(1, 8, 7, 33)).astype(np.float32)
    decay = rng.uniform(0.8, 1.0, size=(2, 7, 17)).astype(np.float32)
    value_scale = rng.uniform(0.5, 1.5, size=(8, 7, 33)).astype(np.float32)
    arguments = {"decay": decay, "value_scale": value_scale}
    full_call = validate_linear_call(operator, q=q, k=k, v=v, arguments=arguments)
    code = next(
        item
        for item in backend.enumerate_code_plans(operator, full_call, host)
        if item.lowering is LoweringKind.LINEAR_SCAN
    )
    compiler = Compiler(tmp_path)

    first_call = validate_linear_call(
        operator,
        q=np.ascontiguousarray(q[:, :, :3]),
        k=np.ascontiguousarray(k[:, :, :3]),
        v=np.ascontiguousarray(v[:, :, :3]),
        arguments={
            "decay": np.ascontiguousarray(decay[:, :3]),
            "value_scale": np.ascontiguousarray(value_scale[:, :3]),
        },
    )
    first_plan = PlanBuilder(host).build_all((code,), operator, first_call)[0]
    first = _execute(compiler, operator, first_call, first_plan, backend)
    second_call = validate_linear_call(
        operator,
        q=np.ascontiguousarray(q[:, :, 3:]),
        k=np.ascontiguousarray(k[:, :, 3:]),
        v=np.ascontiguousarray(v[:, :, 3:]),
        state=first.state,
        arguments={
            "decay": np.ascontiguousarray(decay[:, 3:]),
            "value_scale": np.ascontiguousarray(value_scale[:, 3:]),
        },
    )
    second_plan = PlanBuilder(host).build_all((code,), operator, second_call)[0]
    second = _execute(compiler, operator, second_call, second_plan, backend)
    expected = reference_linear(operator, full_call)
    np.testing.assert_allclose(
        np.concatenate((first.output, second.output), axis=2),
        expected.output,
        rtol=2e-4,
        atol=2e-5,
    )
    np.testing.assert_allclose(second.state, expected.state, rtol=2e-4, atol=2e-5)


def test12(tmp_path: Path, native) -> None:
    """Every Parallel lowering keeps fully masked sigmoid rows finite and zero."""
    host, backend = native
    operator = Parallel(
        mask_mod=expr.as_expr(False), row_norm=rownorm.sigmoid_norm(0.125)
    )
    q = np.ones((1, 4, 3, 7), dtype=np.float32)
    k = np.ones((1, 1, 65, 7), dtype=np.float32)
    v = np.ones((1, 1, 65, 9), dtype=np.float32)
    call = validate_parallel_call(operator, q=q, k=k, v=v)
    plans = _plans(operator, call, host, backend)
    compiler = Compiler(tmp_path)

    for lowering in (
        LoweringKind.PARALLEL_BLOCKED,
        LoweringKind.PARALLEL_SPLIT_K,
        LoweringKind.PARALLEL_2D,
    ):
        plan = next(item for item in plans if item.code.lowering is lowering)
        actual = _execute(compiler, operator, call, plan, backend)
        assert np.isfinite(actual).all(), lowering
        np.testing.assert_array_equal(actual, 0.0, err_msg=lowering.value)


def test13(tmp_path: Path, native) -> None:
    """The chunked gated delta lowering matches the scalar reference."""
    host, backend = native
    rng = np.random.default_rng(41)
    operator = Linear(
        transition=transition.program(
            transition.scale(expr.argument("gate")),
            transition.rank1(-expr.argument("beta") * expr.var("k"), expr.var("k")),
            transition.outer(expr.var("k"), expr.argument("beta") * expr.var("v")),
        ),
        readout=transition.readout(timing=transition.ReadTiming.AFTER),
        arguments=(
            TensorArgSpec("gate", (Axis.BATCH, Axis.SEQUENCE)),
            TensorArgSpec("beta", (Axis.BATCH, Axis.SEQUENCE)),
        ),
    )
    call = validate_linear_call(
        operator,
        q=rng.normal(size=(1, 2, 7, 5)).astype(np.float32),
        k=rng.normal(size=(1, 2, 7, 5)).astype(np.float32),
        v=rng.normal(size=(1, 2, 7, 9)).astype(np.float32),
        arguments={
            "gate": rng.uniform(0.9, 1.0, size=(1, 7)).astype(np.float32),
            "beta": rng.uniform(0.05, 0.95, size=(1, 7)).astype(np.float32),
        },
    )
    expected = reference_linear(operator, call)
    plans = [
        plan
        for plan in _plans(operator, call, host, backend)
        if plan.code.lowering is LoweringKind.LINEAR_DELTA
    ]
    assert plans
    compiler = Compiler(tmp_path)
    for plan in plans:
        actual = _execute(compiler, operator, call, plan, backend)
        np.testing.assert_allclose(
            actual.output, expected.output, rtol=2e-4, atol=2e-5, err_msg=plan.identity
        )
        np.testing.assert_allclose(
            actual.state, expected.state, rtol=2e-4, atol=2e-5, err_msg=plan.identity
        )
