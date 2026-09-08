#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON:-python3}"
export PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
export CPUATTN_CACHE_DIR="${CPUATTN_CACHE_DIR:-${project_root}/artifacts/cache}"
mkdir -p "${CPUATTN_CACHE_DIR}"

exec "${python_bin}" - <<'PY'
from __future__ import annotations

import logging
import math
import os
import platform
import statistics
import time

import numpy as np
import torch
import torch.nn.functional as F

from cpuattn import Linear, Parallel, Runtime, expr, transition


WARMUP = int(os.environ.get("CPUATTN_BENCH_WARMUP", "2"))
RUNS = int(os.environ.get("CPUATTN_BENCH_RUNS", "7"))
if WARMUP < 0 or RUNS < 1:
    raise SystemExit("CPUATTN_BENCH_WARMUP must be >= 0 and CPUATTN_BENCH_RUNS >= 1")


def measure(function):
    for _ in range(WARMUP):
        function()
    samples = []
    for _ in range(RUNS):
        started = time.perf_counter_ns()
        function()
        samples.append((time.perf_counter_ns() - started) / 1e6)
    return statistics.median(samples), min(samples)


def initialize(runtime, operator, **arguments):
    started = time.perf_counter_ns()
    runtime.run(operator, **arguments)
    return (time.perf_counter_ns() - started) / 1e6


def plan_name(runtime):
    selection = runtime.last_selection
    assert selection is not None
    plan = selection.winner.plan
    code = plan.code
    return (
        f"{code.lowering.value}/{code.microkernel.name}, "
        f"tile=({code.tile.q},{code.tile.k},{code.tile.d},{code.tile.dv}), "
        f"packing={code.packing.value}, workers={plan.launch.workers}"
    )


rng = np.random.default_rng(2026)
parallel_shape = (1, 8, 2, 256, 256, 64, 64)
b, hq, hkv, sq, skv, d, dv = parallel_shape
q = rng.normal(size=(b, hq, sq, d)).astype(np.float32)
k = rng.normal(size=(b, hkv, skv, d)).astype(np.float32)
v = rng.normal(size=(b, hkv, skv, dv)).astype(np.float32)
parallel = Parallel(
    score_mod=expr.identity() * (1.0 / math.sqrt(d)),
    mask_mod=expr.causal(),
)
parallel_runtime = Runtime()
parallel_first = initialize(parallel_runtime, parallel, q=q, k=k, v=v)
parallel_mode = parallel_runtime.last_selection.mode
logging.getLogger("cpuattn").disabled = True
parallel_median, parallel_min = measure(
    lambda: parallel_runtime.run(parallel, q=q, k=k, v=v)
)

# PyTorch SDPA has no grouped-query CPU path with identical inputs, so K/V are
# expanded before timing. The timed region measures only the attention call.
torch.set_num_threads(max(1, parallel_runtime.host.physical_cores))
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
tq = torch.from_numpy(q)
tk = torch.from_numpy(np.repeat(k, hq // hkv, axis=1))
tv = torch.from_numpy(np.repeat(v, hq // hkv, axis=1))
with torch.inference_mode():
    F.scaled_dot_product_attention(tq, tk, tv, is_causal=True)
    torch_median, torch_min = measure(
        lambda: F.scaled_dot_product_attention(tq, tk, tv, is_causal=True)
    )

linear_shape = (1, 1, 16, 256, 64, 64)
b, hk, hv, sequence, d, dv = linear_shape
lq = rng.normal(size=(b, hk, sequence, d)).astype(np.float32)
lk = rng.normal(size=(b, hk, sequence, d)).astype(np.float32)
lv = rng.normal(size=(b, hv, sequence, dv)).astype(np.float32)
linear = Linear(
    transition=transition.program(
        transition.scale(0.95),
        transition.outer(expr.var("k"), expr.var("v")),
    ),
    readout=transition.readout(timing="after"),
)
linear_runtime = Runtime()
linear_first = initialize(linear_runtime, linear, q=lq, k=lk, v=lv)
linear_mode = linear_runtime.last_selection.mode
linear_median, linear_min = measure(
    lambda: linear_runtime.run(linear, q=lq, k=lk, v=lv)
)

host = parallel_runtime.host
print("CPUAttn benchmark")
print(f"platform: {platform.platform()}")
print(f"python: {platform.python_version()}")
print(f"numpy: {np.__version__}")
print(f"torch: {torch.__version__}")
print(
    f"cpu: {host.vendor} {host.model}; arch={host.architecture}; "
    f"physical_cores={host.physical_cores}; allowed_logical_cpus={len(host.cpus)}"
)
print(f"samples: warmup={WARMUP}, measured={RUNS}")
print()
print("case                 first call ms   median ms      min ms")
print("-------------------  -------------  ----------  ----------")
print(
    f"Parallel CPUAttn      {parallel_first:13.3f}  "
    f"{parallel_median:10.3f}  {parallel_min:10.3f}"
)
print(
    f"Parallel torch SDPA  {'-':>13}  "
    f"{torch_median:10.3f}  {torch_min:10.3f}"
)
print(
    f"Linear CPUAttn       {linear_first:13.3f}  "
    f"{linear_median:10.3f}  {linear_min:10.3f}"
)
print()
print(f"Parallel shape: B={1}, HQ={8}, HKV={2}, SQ={256}, SKV={256}, D={64}, DV={64}")
print(f"Parallel first-call selection: {parallel_mode}")
print(f"Parallel plan: {plan_name(parallel_runtime)}")
print(f"Linear shape: B={1}, HK={1}, HV={16}, S={256}, D={64}, DV={64}")
print(f"Linear first-call selection: {linear_mode}")
print(f"Linear plan: {plan_name(linear_runtime)}")
PY
