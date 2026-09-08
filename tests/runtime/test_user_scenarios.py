from pathlib import Path

import numpy as np

from cpuattn import Linear, Parallel, Runtime, expr, transition
from tests._reference import linear as reference_linear
from tests._reference import parallel as reference_parallel
from cpuattn.core.validate import validate_linear_call, validate_parallel_call


def test1(runtime_cache: Path, runtime_host) -> None:
    """A cache-backed MQA call works through the public Runtime API."""
    rng = np.random.default_rng(101)
    operator = Parallel(score_mod=expr.identity() * 0.125, mask_mod=expr.causal())
    q = rng.normal(size=(1, 8, 1, 17)).astype(np.float32)
    k = rng.normal(size=(1, 1, 33, 17)).astype(np.float32)
    v = rng.normal(size=(1, 1, 33, 15)).astype(np.float32)
    before_k, before_v = k.copy(), v.copy()
    runtime = Runtime(host=runtime_host, cache_dir=runtime_cache)
    actual = runtime.run(
        operator, q=q, k=k, v=v, kv_cache=True
    )
    expected = reference_parallel(
        operator,
        validate_parallel_call(operator, q=q, k=k, v=v, kv_cache=True),
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-5)
    np.testing.assert_array_equal(k, before_k)
    np.testing.assert_array_equal(v, before_v)

    k += 0.25
    updated = runtime.run(operator, q=q, k=k, v=v, kv_cache=True)
    expected_updated = reference_parallel(
        operator,
        validate_parallel_call(operator, q=q, k=k, v=v, kv_cache=True),
    )
    np.testing.assert_allclose(updated, expected_updated, rtol=2e-4, atol=2e-5)


def test2(
    runtime_cache: Path, runtime_host
) -> None:
    """A retention call returns state that a later call can reuse."""
    rng = np.random.default_rng(102)
    operator = Linear(
        transition=transition.program(
            transition.scale(0.95),
            transition.outer(expr.var("k"), expr.var("v")),
        ),
        readout=transition.readout(timing="after"),
    )
    q = rng.normal(size=(1, 1, 3, 7)).astype(np.float32)
    k = rng.normal(size=(1, 1, 3, 7)).astype(np.float32)
    v = rng.normal(size=(1, 8, 3, 9)).astype(np.float32)
    actual = Runtime(host=runtime_host, cache_dir=runtime_cache).run(
        operator, q=q, k=k, v=v
    )
    expected = reference_linear(operator, validate_linear_call(operator, q=q, k=k, v=v))
    np.testing.assert_allclose(actual.output, expected.output, rtol=2e-4, atol=2e-5)
    np.testing.assert_allclose(actual.state, expected.state, rtol=2e-4, atol=2e-5)

    q_next = rng.normal(size=(1, 1, 1, 7)).astype(np.float32)
    k_next = rng.normal(size=(1, 1, 1, 7)).astype(np.float32)
    v_next = rng.normal(size=(1, 8, 1, 9)).astype(np.float32)
    continued = Runtime(host=runtime_host, cache_dir=runtime_cache).run(
        operator, q=q_next, k=k_next, v=v_next, state=actual.state
    )
    expected_continued = reference_linear(
        operator,
        validate_linear_call(
            operator, q=q_next, k=k_next, v=v_next, state=actual.state
        ),
    )
    np.testing.assert_allclose(continued.output, expected_continued.output, rtol=2e-4, atol=2e-5)
    np.testing.assert_allclose(continued.state, expected_continued.state, rtol=2e-4, atol=2e-5)
