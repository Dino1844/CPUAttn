from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from cpuattn import Parallel, Runtime, expr
from cpuattn.core.validate import validate_parallel_call
from tests._reference import parallel as reference_parallel


def _decode_case(seed: int):
    """Random GQA geometry, scale, and buffers for one decode evolution chain."""
    rng = np.random.default_rng(seed)
    b = int(rng.integers(1, 3))
    hq = int(rng.integers(1, 5))
    hkv = int(rng.choice([div for div in range(1, hq + 1) if hq % div == 0]))
    d = int(rng.choice([8, 16]))
    dv = int(rng.choice([d, max(d // 2, 1), 3]))
    scale = float(rng.uniform(0.05, 0.3))
    operator = Parallel(score_mod=expr.identity() * scale, mask_mod=expr.causal())
    k_buffer = rng.normal(size=(b, hkv, 80, d)).astype(np.float32)
    v_buffer = rng.normal(size=(b, hkv, 80, dv)).astype(np.float32)
    return rng, operator, b, hq, hkv, d, dv, k_buffer, v_buffer


def _check(runtime, operator, q, k, v, **kwargs) -> None:
    call = validate_parallel_call(operator, q=q, k=k, v=v, **kwargs)
    expected = reference_parallel(operator, call)
    actual = runtime.run(operator, q=q, k=k, v=v, **kwargs)
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-5)


def test1(runtime_cache: Path, runtime_host) -> None:
    """Random decode chains grow a KV cache across a bucket edge and survive interference."""
    for seed in (11, 23, 47):
        rng, operator, b, hq, hkv, d, dv, k_buffer, v_buffer = _decode_case(seed)
        runtime = Runtime(host=runtime_host, cache_dir=runtime_cache)
        skv = int(rng.integers(2, 6))
        step = 0
        while skv < 70:  # grow until past the 64-key bucket edge, step count random
            skv += int(rng.integers(1, 5))
            step += 1
            q = rng.normal(size=(b, hq, 1, d)).astype(np.float32)
            _check(
                runtime,
                operator,
                q,
                k_buffer[:, :, :skv, :],
                v_buffer[:, :, :skv, :],
                kv_cache=True,
            )
            if step == 12:  # a prefill on fresh buffers must invalidate packed state
                prefill_sq = int(rng.integers(2, 5))
                _check(
                    runtime,
                    operator,
                    rng.normal(size=(b, hq, prefill_sq, d)).astype(np.float32),
                    rng.normal(size=(b, hkv, skv + prefill_sq, d)).astype(np.float32),
                    rng.normal(size=(b, hkv, skv + prefill_sq, dv)).astype(np.float32),
                )
        assert skv >= 70 > 64  # the chain really crossed the bucket edge


def test2(runtime_cache: Path, runtime_host) -> None:
    """Concurrent runs on random shapes all match their precomputed references."""
    rng = np.random.default_rng(97)
    operator = Parallel(score_mod=expr.identity() * 0.2, mask_mod=expr.causal())
    scenarios = []
    for _ in range(6):
        b = int(rng.integers(1, 3))
        hq = int(rng.integers(1, 4))
        hkv = int(rng.choice([div for div in range(1, hq + 1) if hq % div == 0]))
        sq = int(rng.integers(1, 5))
        skv = sq + int(rng.integers(0, 40))
        q = rng.normal(size=(b, hq, sq, 8)).astype(np.float32)
        k = rng.normal(size=(b, hkv, skv, 8)).astype(np.float32)
        v = rng.normal(size=(b, hkv, skv, 4)).astype(np.float32)
        scenarios.append((q, k, v, reference_parallel(
            operator, validate_parallel_call(operator, q=q, k=k, v=v)
        )))
    runtime = Runtime(host=runtime_host, cache_dir=runtime_cache)
    order = list(range(18))

    def worker(offset: int) -> None:
        for item in order[offset::3]:
            q, k, v, expected = scenarios[item % len(scenarios)]
            np.testing.assert_allclose(
                runtime.run(operator, q=q, k=k, v=v), expected, rtol=2e-4, atol=2e-5
            )

    with ThreadPoolExecutor(max_workers=3) as pool:
        for future in [pool.submit(worker, i) for i in range(3)]:
            future.result()
