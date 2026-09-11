from __future__ import annotations

import numpy as np
import pytest

from cpuattn import Linear, Parallel, expr, transition
from cpuattn.core.tensor import Axis, TensorArgSpec
from cpuattn.core.validate import validate_linear_call, validate_parallel_call
from tests._reference import linear as reference_linear
from tests._reference import parallel as reference_parallel

from benchmarks.baselines import attention_numpy, attention_torch, kda_numpy, linear_numpy


def test_attention_numpy_matches_the_reference_implementation() -> None:
    rng = np.random.default_rng(11)
    operator = Parallel(score_mod=expr.identity() * 0.125, mask_mod=expr.causal())
    q = rng.normal(size=(1, 4, 5, 8)).astype(np.float32)
    k = rng.normal(size=(1, 2, 7, 8)).astype(np.float32)
    v = rng.normal(size=(1, 2, 7, 6)).astype(np.float32)
    call = validate_parallel_call(operator, q=q, k=k, v=v)
    expected = reference_parallel(operator, call)
    actual = attention_numpy(q, k, v, causal=True, scale=0.125)
    assert np.abs(actual - expected).max() < 2e-5


def test_attention_numpy_non_causal_matches_reference() -> None:
    rng = np.random.default_rng(12)
    operator = Parallel(score_mod=expr.identity() * 0.25, mask_mod=expr.as_expr(True))
    q = rng.normal(size=(1, 2, 3, 8)).astype(np.float32)
    k = rng.normal(size=(1, 2, 9, 8)).astype(np.float32)
    v = rng.normal(size=(1, 2, 9, 8)).astype(np.float32)
    call = validate_parallel_call(operator, q=q, k=k, v=v)
    expected = reference_parallel(operator, call)
    actual = attention_numpy(q, k, v, causal=False, scale=0.25)
    assert np.abs(actual - expected).max() < 2e-5


def test_linear_numpy_matches_the_reference_implementation() -> None:
    rng = np.random.default_rng(13)
    operator = Linear(
        transition=transition.program(
            transition.scale(0.95),
            transition.outer(expr.var("k"), expr.var("v")),
        ),
        readout=transition.readout(timing="after"),
    )
    q = rng.normal(size=(1, 2, 6, 8)).astype(np.float32)
    k = rng.normal(size=(1, 2, 6, 8)).astype(np.float32)
    v = rng.normal(size=(1, 2, 6, 4)).astype(np.float32)
    call = validate_linear_call(operator, q=q, k=k, v=v)
    expected = reference_linear(operator, call)
    output, state = linear_numpy(q, k, v, decay=0.95)
    assert np.abs(output - expected.output).max() < 2e-5
    assert np.abs(state - expected.state).max() < 2e-5


def test_linear_numpy_grouped_heads_match_reference() -> None:
    rng = np.random.default_rng(14)
    operator = Linear(
        transition=transition.program(
            transition.scale(0.9),
            transition.outer(expr.var("k"), expr.var("v")),
        ),
        readout=transition.readout(timing="after"),
    )
    q = rng.normal(size=(1, 2, 5, 8)).astype(np.float32)
    k = rng.normal(size=(1, 2, 5, 8)).astype(np.float32)
    v = rng.normal(size=(1, 6, 5, 4)).astype(np.float32)
    call = validate_linear_call(operator, q=q, k=k, v=v)
    expected = reference_linear(operator, call)
    output, state = linear_numpy(q, k, v, decay=0.9)
    assert np.abs(output - expected.output).max() < 2e-5
    assert np.abs(state - expected.state).max() < 2e-5


def test_linear_numpy_per_token_decay_matches_reference() -> None:
    rng = np.random.default_rng(15)
    operator = Linear(
        transition=transition.program(
            transition.scale(expr.argument("a")),
            transition.outer(expr.var("k"), expr.var("v")),
        ),
        readout=transition.readout(timing="after"),
        arguments=(TensorArgSpec("a", (Axis.BATCH, Axis.SEQUENCE)),),
    )
    q = rng.normal(size=(1, 2, 6, 8)).astype(np.float32)
    k = rng.normal(size=(1, 2, 6, 8)).astype(np.float32)
    v = rng.normal(size=(1, 2, 6, 4)).astype(np.float32)
    decay = rng.uniform(0.85, 0.99, size=(1, 6)).astype(np.float32)
    call = validate_linear_call(operator, q=q, k=k, v=v, arguments={"a": decay})
    expected = reference_linear(operator, call)
    output, state = linear_numpy(q, k, v, decay=decay)
    assert np.abs(output - expected.output).max() < 2e-5
    assert np.abs(state - expected.state).max() < 2e-5


def test_kda_numpy_matches_reference() -> None:
    rng = np.random.default_rng(16)
    operator = Linear(
        transition=transition.program(
            transition.scale(expr.argument("gate")),
            transition.rank1(
                -expr.argument("beta") * expr.var("k"), expr.var("k")
            ),
            transition.outer(
                expr.var("k"), expr.argument("beta") * expr.var("v")
            ),
        ),
        readout=transition.readout(timing="after"),
        arguments=(
            TensorArgSpec("gate", (Axis.BATCH, Axis.SEQUENCE)),
            TensorArgSpec("beta", (Axis.BATCH, Axis.SEQUENCE)),
        ),
    )
    q = rng.normal(size=(1, 2, 6, 8)).astype(np.float32)
    k = rng.normal(size=(1, 2, 6, 8)).astype(np.float32)
    k = k / np.linalg.norm(k, axis=-1, keepdims=True)
    v = rng.normal(size=(1, 2, 6, 4)).astype(np.float32)
    gate = rng.uniform(0.9, 1.0, size=(1, 6)).astype(np.float32)
    beta = rng.uniform(0.05, 0.95, size=(1, 6)).astype(np.float32)
    arguments = {"gate": gate, "beta": beta}
    call = validate_linear_call(operator, q=q.astype(np.float32), k=k.astype(np.float32), v=v, arguments=arguments)
    expected = reference_linear(operator, call)
    output, state = kda_numpy(q, k.astype(np.float32), v, gate=gate, beta=beta)
    assert np.abs(output - expected.output).max() < 2e-5
    assert np.abs(state - expected.state).max() < 2e-5


def test_attention_torch_suffix_decode_matches_reference() -> None:
    pytest.importorskip("torch")
    rng = np.random.default_rng(17)
    scale = 1.0 / 8**0.5  # SDPA's default, so the comparison is apples-to-apples
    operator = Parallel(score_mod=expr.identity() * scale, mask_mod=expr.causal())
    q = rng.normal(size=(1, 2, 1, 8)).astype(np.float32)
    k = rng.normal(size=(1, 2, 5, 8)).astype(np.float32)
    v = rng.normal(size=(1, 2, 5, 4)).astype(np.float32)
    call = validate_parallel_call(operator, q=q, k=k, v=v, kv_cache=True)
    expected = reference_parallel(operator, call)
    actual = attention_torch(q, k, v, causal=True, query_offset=4)
    assert np.abs(actual - expected).max() < 2e-5


def test_linear_numpy_batched_grouped_heads_match_reference() -> None:
    rng = np.random.default_rng(18)
    operator = Linear(
        transition=transition.program(
            transition.scale(0.9),
            transition.outer(expr.var("k"), expr.var("v")),
        ),
        readout=transition.readout(timing="after"),
    )
    q = rng.normal(size=(2, 2, 5, 8)).astype(np.float32)
    k = rng.normal(size=(2, 2, 5, 8)).astype(np.float32)
    v = rng.normal(size=(2, 6, 5, 4)).astype(np.float32)
    call = validate_linear_call(operator, q=q, k=k, v=v)
    expected = reference_linear(operator, call)
    output, state = linear_numpy(q, k, v, decay=0.9)
    assert np.abs(output - expected.output).max() < 2e-5
    assert np.abs(state - expected.state).max() < 2e-5


def test_linear_numpy_batched_per_token_decay_matches_reference() -> None:
    rng = np.random.default_rng(19)
    operator = Linear(
        transition=transition.program(
            transition.scale(expr.argument("a")),
            transition.outer(expr.var("k"), expr.var("v")),
        ),
        readout=transition.readout(timing="after"),
        arguments=(TensorArgSpec("a", (Axis.BATCH, Axis.SEQUENCE)),),
    )
    q = rng.normal(size=(2, 3, 6, 8)).astype(np.float32)
    k = rng.normal(size=(2, 3, 6, 8)).astype(np.float32)
    v = rng.normal(size=(2, 3, 6, 4)).astype(np.float32)
    decay = rng.uniform(0.85, 0.99, size=(2, 6)).astype(np.float32)
    call = validate_linear_call(operator, q=q, k=k, v=v, arguments={"a": decay})
    expected = reference_linear(operator, call)
    output, state = linear_numpy(q, k, v, decay=decay)
    assert np.abs(output - expected.output).max() < 2e-5
    assert np.abs(state - expected.state).max() < 2e-5


def test_attention_numpy_suffix_decode_matches_reference() -> None:
    rng = np.random.default_rng(20)
    operator = Parallel(score_mod=expr.identity() * 0.125, mask_mod=expr.causal())
    q = rng.normal(size=(1, 2, 1, 8)).astype(np.float32)
    k = rng.normal(size=(1, 2, 5, 8)).astype(np.float32)
    v = rng.normal(size=(1, 2, 5, 4)).astype(np.float32)
    call = validate_parallel_call(operator, q=q, k=k, v=v, kv_cache=True)
    expected = reference_parallel(operator, call)
    actual = attention_numpy(q, k, v, causal=True, scale=0.125, query_offset=4)
    assert np.abs(actual - expected).max() < 2e-5
