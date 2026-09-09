from concurrent.futures import ThreadPoolExecutor
from collections.abc import Mapping
from pathlib import Path
import os
from threading import Lock
import time

import numpy as np

from cpuattn import Parallel, Runtime
import cpuattn.native.compiler as compiler_module
from cpuattn.tuning import Tuner
from tests._reference import parallel as reference_parallel
from cpuattn.core.validate import validate_parallel_call


def test1(
    runtime_cache: Path, runtime_host, monkeypatch
) -> None:
    """Runtime caches compiler work and replays the selected winner correctly."""
    affinity_before = os.sched_getaffinity(0)
    rng = np.random.default_rng(71)
    operator = Parallel()
    q = rng.normal(size=(1, 2, 2, 5)).astype(np.float32)
    k = rng.normal(size=(1, 1, 7, 5)).astype(np.float32)
    v = rng.normal(size=(1, 1, 7, 3)).astype(np.float32)
    version_calls = 0
    load_calls = 0
    original_version = compiler_module._compiler_version
    original_cdll = compiler_module.ctypes.CDLL

    def counted_version(compiler):
        nonlocal version_calls
        version_calls += 1
        return original_version(compiler)

    def counted_cdll(path):
        nonlocal load_calls
        load_calls += 1
        return original_cdll(path)

    monkeypatch.setattr(compiler_module, "_compiler_version", counted_version)
    monkeypatch.setattr(compiler_module.ctypes, "CDLL", counted_cdll)
    runtime = Runtime(host=runtime_host, cache_dir=runtime_cache)
    expected = reference_parallel(
        operator,
        validate_parallel_call(operator, q=q, k=k, v=v),
    )

    first = runtime.run(operator, q=q, k=k, v=v)
    first_load_calls = load_calls
    assert version_calls == 1
    assert runtime.last_selection is not None
    assert runtime.last_selection.mode == "tune"
    replay = runtime.run(operator, q=q, k=k, v=v)
    assert version_calls == 1
    assert load_calls == first_load_calls
    assert runtime.last_selection.mode == "cache"
    second_runtime = Runtime(host=runtime_host, cache_dir=runtime_cache)
    second = second_runtime.run(operator, q=q, k=k, v=v)
    assert version_calls == 2
    assert second_runtime.last_selection is not None

    cache_path = (
        runtime_cache
        / "selections"
        / f"{second_runtime.last_selection.key}.json"
    )
    cache_path.write_text("[]", encoding="utf-8")
    third_runtime = Runtime(host=runtime_host, cache_dir=runtime_cache)
    third = third_runtime.run(operator, q=q, k=k, v=v)

    np.testing.assert_allclose(first, expected, rtol=2e-4, atol=2e-5)
    np.testing.assert_array_equal(replay, first)
    np.testing.assert_array_equal(second, first)
    # Re-tuning after cache corruption may select a different plan; distinct
    # kernels may differ at the ulp level from fp contraction, so only
    # closeness is guaranteed for the third result.
    np.testing.assert_allclose(third, first, rtol=2e-4, atol=2e-5)
    assert second_runtime.last_selection.mode == "cache"
    assert third_runtime.last_selection is not None
    assert third_runtime.last_selection.mode == "tune"
    assert os.sched_getaffinity(0) == affinity_before


def test2(runtime_cache: Path, runtime_host) -> None:
    """One Runtime serializes access to its tuner and shared native workspace."""
    values = np.ones((1, 1, 3, 5), dtype=np.float32)
    operator = Parallel()
    runtime = Runtime(host=runtime_host, cache_dir=runtime_cache)
    runtime.run(operator, q=values, k=values, v=values)
    selection = runtime.last_selection
    assert selection is not None

    state_lock = Lock()
    active = 0
    maximum_active = 0
    choose_calls = 0

    class ProbeTuner(Tuner):
        def choose_next(self, context, history):
            nonlocal active, maximum_active, choose_calls
            if history:
                return None
            assert isinstance(context.workload, Mapping)
            assert "shape" in context.workload
            choose_calls += 1
            with state_lock:
                active += 1
                maximum_active = max(maximum_active, active)
            time.sleep(0.02)
            with state_lock:
                active -= 1
            return context.candidates[0]

    runtime.tuner = ProbeTuner()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(runtime.run, operator, q=values, k=values, v=values)
            for _ in range(2)
        ]
        for future in futures:
            future.result()

    assert maximum_active == 1
    assert choose_calls == 1
