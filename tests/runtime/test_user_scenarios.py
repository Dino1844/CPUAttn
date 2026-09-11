from pathlib import Path

import numpy as np

from cpuattn import Linear, Parallel, Runtime, expr, transition
from cpuattn.runtime import _kv_bucket, _packed_skv
from cpuattn.tuning.tuner import Measurement, TuningResult, Tuner
from tests._reference import linear as reference_linear
from tests._reference import parallel as reference_parallel
from cpuattn.core.validate import validate_linear_call, validate_parallel_call


class _TransposedTuner(Tuner):
    """Deterministically elect the first k_transposed candidate as winner."""

    def cache_identity(self) -> dict[str, str]:
        return {"forced": "k_transposed"}

    def choose_next(self, context, measured):
        for plan in context.candidates:
            if plan.identity not in measured and plan.code.packing.value == "k_transposed":
                return plan
        return None

    def select(self, context, measure):
        measured = {}
        while (plan := self.choose_next(context, measured)) is not None:
            measured[plan.identity] = measure(plan)
        winner = next(
            plan for plan in context.candidates if plan.code.packing.value == "k_transposed"
        )
        latency = measured[winner.identity]
        return TuningResult(winner, (Measurement(winner, (latency,), latency),))


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


def test3(runtime_cache: Path, runtime_host) -> None:
    """A growing-KV decode stream tunes once per bucket, then replays."""
    rng = np.random.default_rng(103)
    operator = Parallel(score_mod=expr.identity() * 0.125, mask_mod=expr.causal())
    q = rng.normal(size=(1, 8, 1, 17)).astype(np.float32)
    k_full = rng.normal(size=(1, 1, 40, 17)).astype(np.float32)
    v_full = rng.normal(size=(1, 1, 40, 15)).astype(np.float32)
    runtime = Runtime(host=runtime_host, cache_dir=runtime_cache)
    modes = []
    for skv in (33, 34, 35):
        k = k_full[:, :, :skv].copy()
        v = v_full[:, :, :skv].copy()
        actual = runtime.run(operator, q=q, k=k, v=v, kv_cache=True)
        modes.append(runtime.last_selection.mode)
        expected = reference_parallel(
            operator,
            validate_parallel_call(operator, q=q, k=k, v=v, kv_cache=True),
        )
        np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-5)
    # The first step may hit a bucket published by an earlier test.
    assert modes[1] == "cache"
    assert modes[2] == "cache"


def test4(runtime_cache: Path, runtime_host) -> None:
    """A cache-backed decode stream re-packs only the grown K suffix."""
    rng = np.random.default_rng(104)
    operator = Parallel(score_mod=expr.identity() * 0.125, mask_mod=expr.causal())
    q = rng.normal(size=(1, 8, 1, 17)).astype(np.float32)
    runtime = Runtime(host=runtime_host, cache_dir=runtime_cache)
    k_cache = np.zeros((1, 1, 64, 17), dtype=np.float32)
    v_cache = np.zeros((1, 1, 64, 15), dtype=np.float32)
    k_cache[:, :, :33] = rng.normal(size=(33, 17))
    v_cache[:, :, :33] = rng.normal(size=(33, 15))
    for skv in (33, 34, 35, 36):
        actual = runtime.run(
            operator,
            q=q,
            k=k_cache[:, :, :skv],
            v=v_cache[:, :, :skv],
            kv_cache=True,
        )
        expected = reference_parallel(
            operator,
            validate_parallel_call(
                operator, q=q, k=k_cache[:, :, :skv], v=v_cache[:, :, :skv], kv_cache=True
            ),
        )
        np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-5)
    # A length that shrinks must not trust the stale packed prefix.
    actual = runtime.run(
        operator, q=q, k=k_cache[:, :, :34], v=v_cache[:, :, :34], kv_cache=True
    )
    expected = reference_parallel(
        operator,
        validate_parallel_call(
            operator, q=q, k=k_cache[:, :, :34], v=v_cache[:, :, :34], kv_cache=True
        ),
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-5)
    # A different buffer is a different stream automatically.
    k_other = rng.normal(size=(1, 1, 64, 17)).astype(np.float32)
    v_other = rng.normal(size=(1, 1, 64, 15)).astype(np.float32)
    k_other[:, :, :36] = rng.normal(size=(36, 17))
    v_other[:, :, :36] = rng.normal(size=(36, 15))
    actual = runtime.run(
        operator, q=q, k=k_other[:, :, :36], v=v_other[:, :, :36], kv_cache=True
    )
    expected = reference_parallel(
        operator,
        validate_parallel_call(
            operator, q=q, k=k_other[:, :, :36], v=v_other[:, :, :36], kv_cache=True
        ),
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-5)


def test5(runtime_cache: Path, runtime_host) -> None:
    """Growing across a packed-stride change inside one bucket stays correct."""
    rng = np.random.default_rng(105)
    operator = Parallel(score_mod=expr.identity() * 0.125, mask_mod=expr.causal())
    q = rng.normal(size=(1, 8, 1, 17)).astype(np.float32)
    cache_len = 520
    k_cache = np.zeros((1, 1, cache_len, 17), dtype=np.float32)
    v_cache = np.zeros((1, 1, cache_len, 15), dtype=np.float32)
    k_cache[:, :, :400] = rng.normal(size=(400, 17))
    v_cache[:, :, :400] = rng.normal(size=(400, 15))
    runtime = Runtime(host=runtime_host, cache_dir=runtime_cache, tuner=_TransposedTuner())
    runtime.run(operator, q=q, k=k_cache[:, :, :33], v=v_cache[:, :, :33], kv_cache=True)
    plan = runtime.last_selection.winner.plan
    assert plan.code.packing.value == "k_transposed"
    tile_k = plan.code.tile.k
    crossing = next(
        skv
        for skv in range(34, cache_len + 1)
        if _packed_skv(skv, tile_k) != _packed_skv(skv - 1, tile_k)
        and _kv_bucket(skv) == _kv_bucket(skv - 1)
    )
    for skv in (crossing - 1, crossing):
        actual = runtime.run(
            operator, q=q, k=k_cache[:, :, :skv], v=v_cache[:, :, :skv], kv_cache=True
        )
        expected = reference_parallel(
            operator,
            validate_parallel_call(
                operator, q=q, k=k_cache[:, :, :skv], v=v_cache[:, :, :skv], kv_cache=True
            ),
        )
        np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-5)


def test6(runtime_cache: Path, runtime_host) -> None:
    """Same-length re-runs after in-place edits repack instead of trusting the prefix."""
    rng = np.random.default_rng(106)
    operator = Parallel(score_mod=expr.identity() * 0.125, mask_mod=expr.causal())
    q = rng.normal(size=(1, 8, 1, 17)).astype(np.float32)
    k_cache = np.zeros((1, 1, 64, 17), dtype=np.float32)
    v_cache = np.zeros((1, 1, 64, 15), dtype=np.float32)
    k_cache[:, :, :33] = rng.normal(size=(33, 17))
    v_cache[:, :, :33] = rng.normal(size=(33, 15))
    runtime = Runtime(host=runtime_host, cache_dir=runtime_cache, tuner=_TransposedTuner())
    runtime.run(operator, q=q, k=k_cache[:, :, :33], v=v_cache[:, :, :33], kv_cache=True)
    # A per-row ramp makes stale K visible to softmax, unlike a uniform shift.
    k_cache[:, :, :33] += np.arange(33, dtype=np.float32).reshape(33, 1) * 0.25
    actual = runtime.run(
        operator, q=q, k=k_cache[:, :, :33], v=v_cache[:, :, :33], kv_cache=True
    )
    expected = reference_parallel(
        operator,
        validate_parallel_call(
            operator, q=q, k=k_cache[:, :, :33], v=v_cache[:, :, :33], kv_cache=True
        ),
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-5)


def test7(runtime_cache: Path, runtime_host) -> None:
    """Bucket tuning calls index K/V heads exactly like real calls."""
    rng = np.random.default_rng(107)
    operator = Parallel(score_mod=expr.identity() * 0.125, mask_mod=expr.causal())
    q = rng.normal(size=(1, 8, 1, 17)).astype(np.float32)
    k = rng.normal(size=(1, 1, 33, 17)).astype(np.float32)
    v = rng.normal(size=(1, 1, 33, 15)).astype(np.float32)
    runtime = Runtime(host=runtime_host, cache_dir=runtime_cache, tuner=_TransposedTuner())
    runtime.run(operator, q=q, k=k, v=v, kv_cache=True)
    assert runtime._bucket_calls
    for bucket_call in runtime._bucket_calls.values():
        k_bucket, v_bucket = bucket_call.k, bucket_call.v
        assert bucket_call.k_pitch == k_bucket.strides[1] // k_bucket.itemsize
        assert bucket_call.v_pitch == v_bucket.strides[1] // v_bucket.itemsize
