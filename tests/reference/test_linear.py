import numpy as np

from cpuattn import Linear, expr, transition
from tests._reference import linear
from cpuattn.core.validate import validate_linear_call


def _call(operator: Linear, q, k, v, state=None):
    return linear(operator, validate_linear_call(operator, q=q, k=k, v=v, state=state))


def test1() -> None:
    """Outer-product transition matches a hand-computed state update."""
    operator = Linear(
        transition=transition.program(
            transition.outer(expr.var("k"), expr.var("v"))
        )
    )
    q = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
    k = np.array([[[[2.0, -1.0], [1.0, 1.0]]]], dtype=np.float32)
    v = np.array([[[[3.0], [2.0]]]], dtype=np.float32)
    result = _call(operator, q, k, v)
    np.testing.assert_allclose(result.output[0, 0, :, 0], [0.0, 20.0])
    np.testing.assert_allclose(result.state[0, 0, :, 0], [8.0, -1.0])


def test2() -> None:
    """Linear readout can observe the state after its update."""
    operator = Linear(
        transition=transition.program(
            transition.outer(expr.var("k"), expr.var("v"))
        ),
        readout=transition.readout(timing=transition.ReadTiming.AFTER),
    )
    q = np.ones((1, 1, 1, 2), dtype=np.float32)
    k = np.array([[[[2.0, 3.0]]]], dtype=np.float32)
    v = np.array([[[[4.0]]]], dtype=np.float32)
    np.testing.assert_allclose(_call(operator, q, k, v).output, 20.0)


def test3() -> None:
    """KDA rank-one updates and Mamba-style groups broadcast correctly."""
    rng = np.random.default_rng(8)
    transition_program = transition.program(
        transition.rank1(expr.var("k") * -0.05, expr.var("k")),
        transition.outer(expr.var("k"), expr.var("v")),
    )
    operator = Linear(transition=transition_program)
    q = rng.normal(size=(1, 1, 3, 5)).astype(np.float32)
    k = rng.normal(size=(1, 1, 3, 5)).astype(np.float32)
    v = rng.normal(size=(1, 80, 3, 7)).astype(np.float32)
    result = _call(operator, q, k, v)
    assert result.output.shape == (1, 80, 3, 7)
    assert result.state.shape == (1, 80, 5, 7)
    assert np.isfinite(result.output).all()


def test4() -> None:
    """Continued recurrent state equals one uninterrupted Linear call."""
    rng = np.random.default_rng(9)
    operator = Linear(
        transition=transition.program(
            transition.scale(0.9),
            transition.outer(expr.var("k"), expr.var("v")),
        ),
        readout=transition.readout(timing=transition.ReadTiming.AFTER),
    )
    q = rng.normal(size=(1, 2, 7, 3)).astype(np.float32)
    k = rng.normal(size=(1, 2, 7, 3)).astype(np.float32)
    v = rng.normal(size=(1, 8, 7, 5)).astype(np.float32)
    full = _call(operator, q, k, v)
    first = _call(operator, q[:, :, :3].copy(), k[:, :, :3].copy(), v[:, :, :3].copy())
    second = _call(
        operator,
        q[:, :, 3:].copy(),
        k[:, :, 3:].copy(),
        v[:, :, 3:].copy(),
        first.state,
    )
    np.testing.assert_allclose(np.concatenate((first.output, second.output), axis=2), full.output)
    np.testing.assert_allclose(second.state, full.state)
