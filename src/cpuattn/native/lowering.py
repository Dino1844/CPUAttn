from __future__ import annotations

from dataclasses import dataclass

from ..core import transition
from ..core.expr import Expr, ValueType
from ..core.operator import Linear
from ..core.tensor import Axis


@dataclass(frozen=True, slots=True)
class LinearBlockPlan:
    scale: transition.Scale | None
    outer: transition.Outer


@dataclass(frozen=True, slots=True)
class DeltaBlockPlan:
    """A chunkable gated delta rule: Scale? -> Rank1 -> Outer on k/v."""

    scale: transition.Scale | None
    beta: Expr


def delta_block_plan(operator: Linear) -> DeltaBlockPlan | None:
    """Recognize the proven gated delta rule, never a name-based guess.

    The recurrence must be exactly ``Scale?; state -= beta k (k^T state);
    state += beta k (v)`` with a per-token beta that scales both the rank-1
    correction and the value term. Equivalent commuted/negated spellings are
    normalized by ``_coefficient_of``; anything else falls back to the scan.
    """
    steps = operator.transition.steps
    scale: transition.Scale | None = None
    if steps and isinstance(steps[0], transition.Scale):
        scale = steps[0]
        steps = steps[1:]
    if len(steps) != 2:
        return None
    rank1, outer = steps
    if not isinstance(rank1, transition.Rank1):
        return None
    if not isinstance(outer, transition.Outer):
        return None
    if not (_is_variable(rank1.right, "k") and _is_variable(outer.left, "k")):
        return None
    left = _coefficient_of(rank1.left, "k")
    if left is None or left.kind != "neg" or len(left.args) != 1:
        return None
    beta = left.args[0]
    right = _coefficient_of(outer.right, "v")
    if right is None or right.canonical() != beta.canonical():
        return None
    if not _is_variable(operator.readout.query, "q"):
        return None
    if operator.readout.timing is not transition.ReadTiming.AFTER:
        return None
    specs = {spec.name: spec for spec in operator.arguments}
    factors = set(beta.variables())
    if scale is not None:
        factors |= set(scale.factor.variables())
        if scale.factor.variables() & {"q", "k", "v"}:
            return None
    for name in factors:
        spec = specs.get(name)
        if spec is not None and ({Axis.D, Axis.DV} & set(spec.axes)):
            return None
    return DeltaBlockPlan(scale, beta)


def _coefficient_of(expression: Expr, name: str) -> Expr | None:
    """Return ``c`` for ``expression == c * var(name)``.

    Commuted multiplication and a negated variable or product are normalized:
    ``k*c``, ``c*k``, ``(-c)*k``, ``c*(-k)`` and ``-(c*k)`` all yield the same
    coefficient, so an algebraically identical definition is not missed. The
    gated delta rule is recognized through this, never by operator name.
    """
    if _is_variable(expression, name):
        return Expr("const", ValueType.FLOAT, value=1.0)
    if expression.kind == "neg" and len(expression.args) == 1:
        inner = _coefficient_of(expression.args[0], name)
        return None if inner is None else -inner
    if expression.kind == "mul" and len(expression.args) == 2:
        first, second = expression.args
        if _is_variable(second, name):
            return first
        if _is_variable(first, name):
            return second
        if second.kind == "neg" and _is_variable(second.args[0], name):
            return -first
        if first.kind == "neg" and _is_variable(first.args[0], name):
            return -second
    return None


def linear_block_plan(operator: Linear) -> LinearBlockPlan | None:
    """Return a proven chunkable recurrence, never a name-based guess."""
    steps = operator.transition.steps
    scale: transition.Scale | None
    outer: transition.Outer
    if len(steps) == 1 and isinstance(steps[0], transition.Outer):
        scale = None
        outer = steps[0]
    elif (
        len(steps) == 2
        and isinstance(steps[0], transition.Scale)
        and isinstance(steps[1], transition.Outer)
    ):
        scale = steps[0]
        outer = steps[1]
    else:
        return None

    if not (_is_variable(outer.left, "k") and _is_variable(outer.right, "v")):
        return None
    if not _is_variable(operator.readout.query, "q"):
        return None
    if scale is not None:
        if scale.factor.variables() & {"q", "k", "v"}:
            return None
        specs = {spec.name: spec for spec in operator.arguments}
        for name in scale.factor.variables():
            spec = specs.get(name)
            if spec is not None and ({Axis.D, Axis.DV} & set(spec.axes)):
                return None
    return LinearBlockPlan(scale, outer)


def _is_variable(value: Expr, name: str) -> bool:
    return value.kind == "var" and value.name == name


__all__ = ["DeltaBlockPlan", "LinearBlockPlan", "delta_block_plan", "linear_block_plan"]
