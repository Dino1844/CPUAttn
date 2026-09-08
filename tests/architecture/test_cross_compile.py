from pathlib import Path

import numpy as np
import pytest

from cpuattn import Linear, Parallel, expr, transition
from cpuattn.native.backends.arm64 import ARM_NEON, ARM_SVE
from cpuattn.native.backends.x86 import X86_AVX2, X86_AVX512
from cpuattn.native.compiler import Compiler
from cpuattn.hardware.host import Host, LogicalCpu
from cpuattn.core.validate import validate_linear_call, validate_parallel_call


TARGETS = (
    (X86_AVX2, "x86_64", {"avx2", "fma"}),
    (X86_AVX512, "x86_64", {"avx512f", "fma"}),
    (ARM_NEON, "aarch64", {"asimd"}),
    (ARM_SVE, "aarch64", {"asimd", "sve"}),
)

def fixture_host(architecture: str, features: set[str]) -> Host:
    return Host(
        architecture,
        "fixture",
        "fixture",
        frozenset(features),
        (LogicalCpu(0, 0, 0, 0), LogicalCpu(1, 1, 0, 0)),
        sve_vector_bytes=32 if "sve" in features else None,
    )


@pytest.mark.parametrize("backend,architecture,features", TARGETS, ids=lambda value: getattr(value, "backend_id", None) or str(value))
@pytest.mark.parametrize("pattern", ["parallel", "linear"])
def test1(
    tmp_path: Path, backend, architecture: str, features: set[str], pattern: str
) -> None:
    """Every advertised backend cross-compiles representative operator kernels."""
    host = fixture_host(architecture, features)
    assert backend.supports(host)
    if pattern == "parallel":
        x = expr.identity()
        operator = Parallel(score_mod=expr.tanh(x) + expr.sigmoid(x))
        tensor = np.ones((1, 1, 3, 5), np.float32)
        call = validate_parallel_call(operator, q=tensor, k=tensor, v=tensor)
    else:
        operator = Linear(
            transition=transition.program(
                transition.outer(expr.var("k"), expr.var("v"))
            ),
            q_mod=expr.tanh(expr.identity()),
            k_mod=expr.sigmoid(expr.identity()),
            v_mod=expr.relu(expr.identity()),
        )
        tensor = np.ones((1, 1, 3, 5), np.float32)
        call = validate_linear_call(operator, q=tensor, k=tensor, v=tensor)
    representatives = {}
    for code in backend.enumerate_code_plans(operator, call, host):
        representatives.setdefault(
            (code.lowering.value, code.microkernel.name, code.packing.value), code
        )
    for (lowering, microkernel, packing), code in representatives.items():
        destination = tmp_path / (
            f"{backend.backend_id}-{pattern}-{lowering}-{microkernel}-{packing}.o"
        )
        result = Compiler(tmp_path / "cache").cross_compile_object(
            operator, call, code, backend, destination
        )
        assert result.is_file()
        assert result.stat().st_size > 0
