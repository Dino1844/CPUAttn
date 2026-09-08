import numpy as np
import pytest

from cpuattn import Axis, Linear, Parallel, TensorArgSpec, expr, transition
from cpuattn.core.validate import validate_linear_call, validate_parallel_call


def f32(shape):
    return np.zeros(shape, dtype=np.float32)


@pytest.mark.parametrize("heads", [(1, 1), (4, 4), (8, 2), (8, 1)])
def test1(heads) -> None:
    """Parallel validation accepts MHA, GQA, and MQA head mappings."""
    hq, hkv = heads
    call = validate_parallel_call(
        Parallel(), q=f32((1, hq, 3, 5)), k=f32((1, hkv, 7, 5)), v=f32((1, hkv, 7, 9))
    )
    assert call.dimensions[Axis.QUERY_HEAD] == hq


def test2() -> None:
    """Parallel validation rejects invalid shapes, layouts, and dtypes."""
    operator = Parallel()
    with pytest.raises(ValueError, match="divisible"):
        validate_parallel_call(operator, q=f32((1, 6, 3, 5)), k=f32((1, 4, 7, 5)), v=f32((1, 4, 7, 9)))
    with pytest.raises(ValueError, match="contiguous BHSD"):
        validate_parallel_call(operator, q=f32((1, 2, 3, 5))[..., ::-1], k=f32((1, 1, 7, 5)), v=f32((1, 1, 7, 9)))
    with pytest.raises(ValueError, match="fp32"):
        validate_parallel_call(operator, q=np.zeros((1, 2, 3, 5), dtype=np.float64), k=f32((1, 1, 7, 5)), v=f32((1, 1, 7, 9)))
    with pytest.raises(ValueError, match="at least query_length"):
        validate_parallel_call(
            operator,
            q=f32((1, 2, 8, 5)),
            k=f32((1, 1, 7, 5)),
            v=f32((1, 1, 7, 9)),
            kv_cache=True,
        )

    cached = validate_parallel_call(
        operator,
        q=f32((1, 2, 3, 5)),
        k=f32((1, 1, 7, 5)),
        v=f32((1, 1, 7, 9)),
        kv_cache=True,
    )
    assert cached.query_offset == 4


@pytest.mark.parametrize("heads,groups", [(1, 1), (8, 8), (8, 2), (80, 1)])
def test3(heads: int, groups: int) -> None:
    """Linear validation keeps parameter groups separate from state heads."""
    operator = Linear(
        transition=transition.program(
            transition.outer(expr.var("k"), expr.var("v"))
        )
    )
    call = validate_linear_call(
        operator,
        q=f32((1, groups, 3, 5)),
        k=f32((1, groups, 3, 5)),
        v=f32((1, heads, 3, 7)),
    )
    assert call.dimensions[Axis.STATE_HEAD] == heads


def test4() -> None:
    """Auxiliary arguments must match their declared axes exactly."""
    scale = TensorArgSpec("scale", (Axis.QUERY_HEAD, Axis.QUERY))
    operator = Parallel(score_mod=expr.identity() * expr.argument("scale"), arguments=(scale,))
    args = {"scale": f32((4, 3))}
    validate_parallel_call(
        operator,
        q=f32((1, 4, 3, 5)),
        k=f32((1, 2, 7, 5)),
        v=f32((1, 2, 7, 9)),
        arguments=args,
    )
    with pytest.raises(ValueError, match="must have shape"):
        validate_parallel_call(
            operator,
            q=f32((1, 4, 3, 5)),
            k=f32((1, 2, 7, 5)),
            v=f32((1, 2, 7, 9)),
            arguments={"scale": f32((3, 4))},
        )
