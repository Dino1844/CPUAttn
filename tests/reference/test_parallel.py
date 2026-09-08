import numpy as np

from cpuattn import Axis, Parallel, TensorArgSpec, expr, rownorm
from tests._reference import parallel
from cpuattn.core.validate import validate_parallel_call


def test1() -> None:
    """Parallel reference handles GQA, Mods, masks, and tail dimensions."""
    rng = np.random.default_rng(4)
    q = rng.normal(size=(1, 8, 17, 15)).astype(np.float32)
    k = rng.normal(size=(1, 2, 33, 15)).astype(np.float32)
    v = rng.normal(size=(1, 2, 33, 7)).astype(np.float32)
    scale = rng.uniform(0.01, 0.2, size=(8, 17)).astype(np.float32)
    spec = TensorArgSpec("scale", (Axis.QUERY_HEAD, Axis.QUERY))
    operator = Parallel(
        score_mod=expr.identity() * expr.argument("scale"),
        mask_mod=expr.causal(),
        row_norm=rownorm.softmax(),
        arguments=(spec,),
    )
    call = validate_parallel_call(operator, q=q, k=k, v=v, arguments={"scale": scale})
    output = parallel(operator, call)
    assert output.shape == (1, 8, 17, 7)
    assert np.isfinite(output).all()


def test2() -> None:
    """Softmax returns zero for fully masked MQA rows."""
    rng = np.random.default_rng(5)
    operator = Parallel(mask_mod=expr.as_expr(False))
    call = validate_parallel_call(
        operator,
        q=rng.normal(size=(1, 8, 3, 5)).astype(np.float32),
        k=rng.normal(size=(1, 1, 7, 5)).astype(np.float32),
        v=rng.normal(size=(1, 1, 7, 9)).astype(np.float32),
        kv_cache=True,
    )
    before_k = call.k.copy()
    before_v = call.v.copy()
    np.testing.assert_array_equal(parallel(operator, call), 0)
    np.testing.assert_array_equal(call.k, before_k)
    np.testing.assert_array_equal(call.v, before_v)


def test3() -> None:
    """Masks remove scores under the identity RowNorm."""
    operator = Parallel(
        mask_mod=expr.causal(),
        row_norm=rownorm.identity(),
    )
    q = np.ones((1, 1, 2, 1), dtype=np.float32)
    k = np.array([[[[2.0], [3.0]]]], dtype=np.float32)
    v = np.array([[[[5.0], [7.0]]]], dtype=np.float32)
    call = validate_parallel_call(operator, q=q, k=k, v=v)
    np.testing.assert_array_equal(parallel(operator, call)[0, 0, :, 0], [10.0, 31.0])


def test4() -> None:
    """Identity RowNorm also zeros a fully masked row."""
    operator = Parallel(mask_mod=expr.as_expr(False), row_norm=rownorm.identity())
    ones = np.ones((1, 1, 3, 2), dtype=np.float32)
    call = validate_parallel_call(operator, q=ones, k=ones, v=ones)
    np.testing.assert_array_equal(parallel(operator, call), 0.0)


def test5() -> None:
    """Cache-backed causal positions include the prefix and zero norms stay finite."""
    q = np.zeros((1, 1, 1, 2), dtype=np.float32)
    k = np.zeros((1, 1, 7, 2), dtype=np.float32)
    v = np.arange(1, 8, dtype=np.float32).reshape(1, 1, 7, 1)
    cached = validate_parallel_call(
        Parallel(mask_mod=expr.causal()), q=q, k=k, v=v, kv_cache=True
    )
    np.testing.assert_allclose(parallel(Parallel(mask_mod=expr.causal()), cached), 4.0)

    masked = validate_parallel_call(
        Parallel(mask_mod=expr.as_expr(False), row_norm=rownorm.sigmoid_norm()),
        q=q,
        k=k,
        v=v,
    )
    np.testing.assert_array_equal(
        parallel(
            Parallel(mask_mod=expr.as_expr(False), row_norm=rownorm.sigmoid_norm()),
            masked,
        ),
        0.0,
    )
