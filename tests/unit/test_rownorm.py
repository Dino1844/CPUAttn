import numpy as np
import pytest

from cpuattn import expr, rownorm


@pytest.mark.parametrize("factory", [rownorm.softmax, rownorm.identity, rownorm.l1, rownorm.sigmoid_norm])
@pytest.mark.parametrize("length", [1, 7, 16, 17, 31, 32, 33])
def test1(factory, length: int) -> None:
    """Built-in online RowNorm protocols match full-row references."""
    scores = np.random.default_rng(length).normal(size=length).astype(np.float32)
    definition = factory()
    np.testing.assert_allclose(
        definition.online_weights(scores),
        definition.reference.weights(scores),
        rtol=2e-5,
        atol=2e-6,
    )


def test2() -> None:
    """Softmax defines fully masked rows as zero."""
    scores = np.full(9, -np.inf, dtype=np.float32)
    definition = rownorm.softmax()
    np.testing.assert_array_equal(definition.online_weights(scores), np.zeros(9))
    np.testing.assert_array_equal(definition.reference.weights(scores), np.zeros(9))


def test3() -> None:
    """Softmax merge reconstructs the state of a complete row."""
    definition = rownorm.softmax()
    scores = np.array([-1.5, 0.25, 3.0, -0.5], dtype=np.float32)

    def state(values: np.ndarray) -> dict[str, float]:
        maximum = np.max(values)
        return {"m": maximum, "l": np.sum(np.exp(values - maximum))}

    merged, left_scale, right_scale = definition.merge_states(state(scores[:2]), state(scores[2:]))
    expected = state(scores)
    np.testing.assert_allclose(merged["m"], expected["m"])
    np.testing.assert_allclose(merged["l"], expected["l"])
    np.testing.assert_allclose(
        left_scale * np.exp(scores[:2] - np.max(scores[:2])),
        np.exp(scores[:2] - expected["m"]),
    )
    np.testing.assert_allclose(
        right_scale * np.exp(scores[2:] - np.max(scores[2:])),
        np.exp(scores[2:] - expected["m"]),
    )


def test4() -> None:
    """Custom RowNorm definitions reject undeclared reference variables."""
    old_sum = expr.var("old.total")
    old_count = expr.var("old.count")
    score = expr.var("score")
    with pytest.raises(ValueError, match="unknown variables"):
        rownorm.RowNorm(
            name="mean",
            states=(rownorm.State("total", 0.0), rownorm.State("count", 0.0)),
            updates=(
                rownorm.Update("total", old_sum + score),
                rownorm.Update("count", old_count + 1),
            ),
            rescale=expr.as_expr(1.0),
            weight=expr.as_expr(1.0),
            final_scale=1 / expr.var("state.count"),
            reference=rownorm.Reference((), expr.as_expr(1.0) / expr.var("length")),
        )
