import numpy as np
import pytest

from cpuattn import expr


def test1() -> None:
    """Composed expressions evaluate NumPy arrays elementwise."""
    x = expr.var("x")
    definition = expr.where(x > 0, expr.sigmoid(x) + 2 * x, expr.relu(x))
    values = np.array([-2.0, 0.0, 2.0], dtype=np.float32)

    actual = expr.evaluate(definition, {"x": values})

    expected = np.where(values > 0, 1 / (1 + np.exp(-values)) + 2 * values, 0)
    np.testing.assert_allclose(actual, expected)


def test2() -> None:
    """Expression IR has stable canonical form and value types."""
    left = expr.maximum(expr.var("x") + 1, 0)
    right = expr.maximum(expr.var("x") + 1.0, 0.0)
    assert left.canonical() == right.canonical()
    assert left.variables() == frozenset({"x"})

    with pytest.raises(TypeError, match="requires bool"):
        _ = expr.var("x") & expr.var("y")


def test3() -> None:
    """Symbolic expressions cannot silently enter Python control flow."""
    with pytest.raises(TypeError, match="cannot be converted"):
        bool(expr.var("x") > 0)
