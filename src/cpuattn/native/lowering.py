from __future__ import annotations

from dataclasses import dataclass

from ..core import transition
from ..core.expr import Expr
from ..core.operator import Linear
from ..core.tensor import Axis


@dataclass(frozen=True, slots=True)
class LinearBlockPlan:
    scale: transition.Scale | None
    outer: transition.Outer


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


__all__ = ["LinearBlockPlan", "linear_block_plan"]
