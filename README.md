# CPUAttn

`CPUAttn` is a standalone, inference-only CPU attention framework. It
turns an attention definition into three explicit choices:

1. `CodePlan`: Pattern algorithm, decomposition, SIMD microkernel, and tiles;
2. `LaunchPlan`: physical cores and NUMA worker groups;
3. `MemoryPlan`: concrete workspace regions and first-touch ownership.

Release tests compare every generated kernel family with an independent NumPy
reference. The installed runtime trusts generated code and contains neither a
NumPy execution backend nor a correctness fallback.

Parallel lowerings choose between direct BHSD K and explicitly transposed K,
assign query/key work to physical cores, and run a fused
QK -> Mod/RowNorm -> PV tile pipeline.
Exact causal masks stop at the last K block reachable by each Q block; arbitrary
masks retain their elementwise semantics inside the same lowering.

The Linear backend always retains the general ordered state scan. Definitions
whose structure is exactly `Scale? -> Outer(k, v)` with a Q readout additionally
receive block sizes 2, 4, and 6. That lowering computes intra-block causal QK/PV
work and the end-of-block state update separately; it is never selected from an
operator name or used for `Rank1`/DPLR transitions.

Kernel code follows three layers. Jinja templates own the Pattern algorithm and
thread decomposition, `kernels/simd/{parallel,linear}.h` owns contiguous tiled
operations, and `kernels/simd/{x86,arm64}` owns ISA intrinsics. Elementwise Mod
expressions share one typed SIMD lowering for arithmetic, comparisons, boolean
logic, `where`, and the supported transcendental operations; only vector tails
use the scalar lowering. The compiler
recognizes the complete built-in Softmax protocol and uses its SIMD online
reduction; changing only a RowNorm name cannot select that path. Arbitrary
RowNorm definitions continue through the generated elementwise protocol.

Python code follows the same dependency direction: `core/` owns the public
attention language, `hardware/` owns immutable host facts, `schedule/` owns
legal execution plans, `native/` owns backend selection, compilation and
execution, and `tuning/` selects among complete runtime candidates.
`runtime.py` is the only outer orchestrator; the package root contains only
public exports, logging, and errors.

## Operator model

- `Parallel`: standard, causal, GQA, MQA, and cache-backed attention. Decode is
  represented by `kv_cache=True` and the actual Q/K/V dimensions, not a third
  Pattern.
- `Linear`: ordered fixed-size state transitions. Q/K parameter groups and
  V/output/state heads are separate axes, which covers grouped Mamba/SSD and
  KDA-style DPLR updates.
- `expr`: pure elementwise composition used by score, mask, Q/K/V, transition,
  and readout expressions.
- `rownorm`: fixed-size online row reductions with an independent full-row
  reference. Split-K is offered only when a structured merge is present.
- `transition`: ordered `Scale`, `Rank1`, and `Outer` state operations with
  readout before or after the update.

Definitions contain typed, canonical IR. They do not retain Python callables or
inject raw source code.

## Parallel example

```python
import numpy as np
from cpuattn import Parallel, Runtime, expr, rownorm

operator = Parallel(
    score_mod=expr.identity() * 0.125,
    mask_mod=expr.causal(),
    row_norm=rownorm.softmax(),
)

runtime = Runtime()
output = runtime.run(
    operator,
    q=np.empty((1, 8, 16, 64), dtype=np.float32),
    k=np.empty((1, 2, 256, 64), dtype=np.float32),
    v=np.empty((1, 2, 256, 64), dtype=np.float32),
    kv_cache=True,
)
```

K/V remain caller-owned and read-only. `kv_cache=True` describes short-query
cache-backed semantics; Q is interpreted as the suffix of K/V, so positional
expressions see query indices starting at `SKV - SQ`. Passing K/V as prefix
views of a preallocated cache (`buffer[:, :, :length]`) is supported and
zero-copy. Consecutive calls that grow the same K buffer are treated as one
decode stream: the transposed-K pack then re-packs only the grown suffix each
step instead of the whole cache. A changed buffer, a length that shrinks, a
same-length re-run, or an intervening run on another stream falls back to a
full pack. The stream contract is append-only: rows already covered by the
packed prefix must not be edited in place while the length later grows, or
the stale prefix is reused.

## Linear example

```python
import numpy as np
from cpuattn import Linear, Runtime, expr, transition

operator = Linear(
    transition=transition.program(
        transition.rank1(expr.var("k") * -0.02, expr.var("k")),
        transition.outer(expr.var("k"), expr.var("v")),
    ),
    readout=transition.readout(timing="after"),
)

runtime = Runtime()
result = runtime.run(
    operator,
    q=np.empty((1, 1, 32, 64), dtype=np.float32),
    k=np.empty((1, 1, 32, 64), dtype=np.float32),
    v=np.empty((1, 80, 32, 64), dtype=np.float32),
)

next_result = runtime.run(
    operator,
    q=np.empty((1, 1, 1, 64), dtype=np.float32),
    k=np.empty((1, 1, 1, 64), dtype=np.float32),
    v=np.empty((1, 80, 1, 64), dtype=np.float32),
    state=result.state,
)
```

## Custom tuning

`Runtime` selects one backend from the detected host. Its tuner then selects
among every legal `ExecutionPlan` for that backend. The base class owns
measurement, limits, repeated samples, median scoring, and winner selection; a
custom tuner implements only the next-plan decision.

```python
from dataclasses import dataclass

from cpuattn import Runtime, Tuner


@dataclass(frozen=True, slots=True)
class LargestFirstTuner(Tuner):
    arch = "x86_64"
    pattern = "parallel"
    version = 1

    def choose_next(self, context, history):
        measured = {item.plan.identity for item in history}
        ordered = sorted(
            context.candidates,
            key=lambda plan: (-plan.launch.workers, plan.identity),
        )
        return next(
            (plan for plan in ordered if plan.identity not in measured),
            None,
        )


runtime = Runtime(tuner=LargestFirstTuner(maxnum=6, repeat=1))
```

The context contains immutable host facts, canonical operator/workload
descriptions, and legal plans. It contains no input tensors, compiler, executor,
or cache object. Additional dataclass fields must be JSON-compatible because
they become part of the exact selection-cache identity. Increment `version`
when policy logic changes without a configuration change.

## Runtime ownership

At startup the runtime detects the allowed CPU set, topology, cache facts, ISA,
vendor, and model, then selects exactly one compatible backend. There is no
implicit AVX-512 or ARM default. The runtime does not mutate process-global
OpenMP settings.
ISA features are intersected across every CPU in the process affinity mask, so
a restricted or heterogeneous CPU set cannot accidentally select instructions
that one of its allowed workers does not support.
The fixed-width SVE backend is selected only when Linux reports an active
256-bit SVE vector length; other SVE lengths fall back to NEON.

The executor keeps one anonymous workspace arena keyed by memory layout and
NUMA group topology. A cached plan reuses it; changing placement discards it so
candidate first-touch histories cannot leak. Packing, when selected, remains
inside native timing. Output/state allocation remains outside native timing.
Workers pin themselves to their assigned CPU and keep that placement between
calls; the master thread, which returns to foreign code between calls,
restores its original affinity after every call, and any change of pin target
restores the previous original affinity before re-pinning. Calls through
one `Runtime` are serialized because its tuner and workspace are shared; use
separate Runtime instances when independent concurrent execution is required.

Compilation cache keys include operator IR, workload, backend, machine-code
kernel fields, compiler/version/flags, ABI, generated source, and shared
threading code. Worker count, CPU placement, NUMA policy, and workspace size are
runtime ABI data, so placements of the same microkernel share one artifact.
Selection keys include the exact workload, complete Host fingerprint, compiler
identity, selected backend, tuner configuration, and unordered legal candidate
identities. `PlanBuilder` exposes the full legal plan space without predicting
performance. The default `WorkerCountTuner` statically chooses one
representative for each topology-derived worker count and packing kind, and
measures at most six representatives once. Exact-workload cache hits bypass the
tuner and execute only the recorded winner. Within one Runtime, immutable
candidate plans, compiler identity, compiled artifacts, and loaded native
kernels are retained for later calls of the same workload.

`SingleCoreTuner` is available when the target is kernel selection on one
physical CPU rather than worker-count selection. It predicts a three-plan
shortlist with a compact roofline model, then measures those plans. The model
uses the SIMD width and FMA support reported by the backend and host, plus the
tile's padded work, memory traffic, K packing, cache footprint, and register
pressure. It sorts directly by the estimated cycle count; there are no
shape-specific rules or secondary candidate-selection policy. Estimates are
inspectable through `predictions(context)`. Setting `maxnum=None` measures every
legal plan on the selected CPU and is intended for offline validation of the
default Top-3 policy.

```python
from cpuattn import Runtime, SingleCoreTuner

runtime = Runtime(tuner=SingleCoreTuner(maxnum=3))
```

Artifacts and selections use `CPUATTN_CACHE_DIR` when set, otherwise the
platform-style `XDG_CACHE_HOME/cpuattn` user cache.

## Current support

- FP32 storage, accumulation, and output;
- contiguous BHSD public tensors;
- arbitrary positive sequence, D, and DV tails;
- native x86 AVX2+FMA and AVX-512;
- ARM64 NEON and SVE source/object cross-builds;
- forward inference only.

Native AMD and ARM correctness/performance measurements, non-FP32 lowerings,
and broader scheduler performance validation remain explicit work. They are not
silently reported as supported.

The package imports no implementation from the parent `try_attn/cpuattn`
tree.

## Reproducible environment

The supported submission environment is the DevContainer in
`.devcontainer/`. It uses Ubuntu 24.04, Python 3.12, GCC/OpenMP, an ARM64 cross
compiler, and pinned CPU-only PyTorch/NumPy test dependencies. Task B does not
use FUSE or overlayfs, so the container deliberately does not request
`--privileged`.

### VS Code DevContainer

Install the Dev Containers extension, open this directory, and run
`Dev Containers: Reopen in Container`. The project is installed in editable
mode when the container is created. Then run:

```bash
./scripts/codegen.sh
./scripts/correctness.sh
./scripts/benchmark.sh
```

### DevContainer CLI

With Docker and the Dev Container CLI installed, the same environment can be
started and tested without VS Code:

```bash
devcontainer up --workspace-folder .
devcontainer exec --workspace-folder . python -m pytest -q
devcontainer exec --workspace-folder . bash
```

`codegen.sh` exercises the complete native generation path for representative
Parallel and Linear operators. Each run creates a fresh ignored
`artifacts/codegen-*` directory containing the detected host and backend,
canonical operator/workload/plan JSON, generated C, compiler manifest, and
shared library for every legal code plan. `correctness.sh` runs the compact DSL,
reference, and end-to-end native Runtime checks. Run `python -m pytest -q` for
the complete test suite, including architecture and cross-compilation checks.

### Docker

The same image can also be built and used directly:

```bash
docker build -f .devcontainer/Dockerfile -t cpuattn .
docker run --rm -it \
  -v "$PWD:/workspace" -w /workspace \
  -e PYTHONPATH=/workspace/src \
  cpuattn python -m pytest -q
```

The benchmark reports the first call separately from steady-state latency and
states whether that call searched or reused a cached selection. It compares
causal GQA Parallel attention with PyTorch CPU SDPA and
also measures the optimized Linear recurrence. Override the sample counts with
`CPUATTN_BENCH_WARMUP` and `CPUATTN_BENCH_RUNS` when needed. Containerization
locks the software environment, but CPU ISA, topology, frequency, and system
load still determine performance results.
