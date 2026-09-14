from cpuattn import Axis, Linear, TensorArgSpec, expr, transition
from cpuattn.native.lowering import delta_block_plan, linear_block_plan


def _operator(rank1_left, outer_right) -> Linear:
    beta = TensorArgSpec("beta", (Axis.BATCH, Axis.SEQUENCE))
    return Linear(
        transition=transition.program(
            transition.rank1(rank1_left, expr.var("k")),
            transition.outer(expr.var("k"), outer_right),
        ),
        readout=transition.readout(timing=transition.ReadTiming.AFTER),
        arguments=(beta,),
    )


def test_delta_recognition_normalizes_equivalent_forms() -> None:
    """Commuted/negated spellings of the gated delta rule all match."""
    beta = expr.argument("beta")
    k = expr.var("k")
    v = expr.var("v")
    forms = (
        (-beta * k, beta * v),
        (k * -beta, v * beta),
        (-(beta * k), beta * v),
        (beta * -k, beta * v),
    )
    for rank1_left, outer_right in forms:
        plan = delta_block_plan(_operator(rank1_left, outer_right))
        assert plan is not None
        assert plan.beta.canonical() == beta.canonical()
        assert linear_block_plan(_operator(rank1_left, outer_right)) is None


def test_delta_recognition_rejects_mismatched_beta() -> None:
    """A rank-one term whose beta does not scale the value term is not a delta."""
    operator = _operator(-expr.argument("beta") * expr.var("k"), expr.var("v"))
    assert delta_block_plan(operator) is None
