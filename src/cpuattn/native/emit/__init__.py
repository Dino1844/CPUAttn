from __future__ import annotations

from jinja2 import Environment

from ...core.operator import Operator, Parallel
from ...core.validate import LinearCall, ParallelCall, ValidatedCall
from ...errors import UnsupportedError
from ...schedule.plan import CodePlan, LoweringKind
from .linear import render_linear
from .parallel import render_parallel


def render_source(
    templates: Environment,
    operator: Operator,
    call: ValidatedCall,
    code: CodePlan,
) -> str:
    expected = (
        {
            LoweringKind.PARALLEL_BLOCKED,
            LoweringKind.PARALLEL_SPLIT_K,
            LoweringKind.PARALLEL_2D,
        }
        if isinstance(operator, Parallel)
        else {
            LoweringKind.LINEAR_SCAN,
            LoweringKind.LINEAR_CHUNKED,
            LoweringKind.LINEAR_DELTA,
            LoweringKind.LINEAR_2D,
        }
    )
    if code.lowering not in expected:
        raise UnsupportedError(
            f"operation {operator.pattern} has no lowering "
            f"{code.lowering.value!r} for {code.backend_id}"
        )
    if isinstance(operator, Parallel):
        assert isinstance(call, ParallelCall)
        return render_parallel(templates, operator, call, code)
    assert isinstance(call, LinearCall)
    return render_linear(templates, operator, code)


__all__ = ["render_source"]
