#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON:-python3}"
artifact_root="${CPUATTN_ARTIFACT_DIR:-${project_root}/artifacts}"

mkdir -p "${artifact_root}"
output_dir="$(mktemp -d "${artifact_root}/codegen-XXXXXX")"
export CPUATTN_CODEGEN_OUTPUT="${output_dir}"
export PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

"${python_bin}" - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

import numpy as np

from cpuattn import Linear, Parallel, expr, transition
from cpuattn.core.validate import validate_linear_call, validate_parallel_call
from cpuattn.hardware import detect_host
from cpuattn.native.backends import select_backend
from cpuattn.native.compiler import Compiler


output = Path(os.environ["CPUATTN_CODEGEN_OUTPUT"])
host = detect_host()
backend = select_backend(host)
compiler = Compiler(output / ".cache")


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def emit(name, operator, call) -> int:
    destination = output / name
    destination.mkdir()
    write_json(destination / "operator.json", operator.canonical())
    write_json(destination / "workload.json", call.canonical())
    code_plans = backend.enumerate_code_plans(operator, call, host)
    for index, code in enumerate(code_plans):
        label = (
            f"{index:02d}-{code.lowering.value}-{code.packing.value}"
            f"-q{code.tile.q}-k{code.tile.k}-dv{code.tile.dv}"
        )
        candidate = destination / label
        candidate.mkdir()
        write_json(candidate / "code-plan.json", code.canonical())
        compiled = compiler.compile(operator, call, code, backend)
        compiler.load(compiled)
        shutil.copy2(compiled.source, candidate / "kernel.c")
        shutil.copy2(compiled.manifest, candidate / "manifest.json")
        shutil.copy2(compiled.library, candidate / "kernel.so")
    return len(code_plans)


parallel = Parallel(
    score_mod=expr.identity() * 0.125,
    mask_mod=expr.causal(),
)
parallel_call = validate_parallel_call(
    parallel,
    q=np.zeros((1, 2, 16, 17), dtype=np.float32),
    k=np.zeros((1, 1, 32, 17), dtype=np.float32),
    v=np.zeros((1, 1, 32, 15), dtype=np.float32),
    kv_cache=True,
)
linear = Linear(
    transition=transition.program(
        transition.scale(0.95),
        transition.outer(expr.var("k"), expr.var("v")),
    ),
    readout=transition.readout(timing="after"),
)
linear_call = validate_linear_call(
    linear,
    q=np.zeros((1, 1, 8, 7), dtype=np.float32),
    k=np.zeros((1, 1, 8, 7), dtype=np.float32),
    v=np.zeros((1, 4, 8, 9), dtype=np.float32),
)

counts = {
    "parallel": emit("parallel", parallel, parallel_call),
    "linear": emit("linear", linear, linear_call),
}
write_json(output / "host.json", host.canonical())
write_json(
    output / "backend.json",
    {
        "backend_id": backend.backend_id,
        "architecture": backend.architecture,
        "vector_bytes": backend.vector_bytes,
        "compiler": backend.compiler_path(),
        "cflags": backend.cflags,
    },
)
write_json(output / "summary.json", {"code_plans": counts})
shutil.rmtree(output / ".cache")
print(f"Generated {sum(counts.values())} kernels in {output}")
PY
