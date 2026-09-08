from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import expr
from .expr import Expr, ValueType


@dataclass(frozen=True, slots=True)
class State:
    name: str
    initial: float

    def __post_init__(self) -> None:
        if not self.name.isidentifier():
            raise ValueError(f"invalid RowNorm state name {self.name!r}")

    def canonical(self) -> dict[str, object]:
        return {"name": self.name, "initial": self.initial}


@dataclass(frozen=True, slots=True)
class Update:
    state: str
    value: Expr

    def canonical(self) -> dict[str, object]:
        return {"state": self.state, "value": self.value.canonical()}


@dataclass(frozen=True, slots=True)
class Reduction:
    name: str
    operation: str
    value: Expr

    def __post_init__(self) -> None:
        if not self.name.isidentifier():
            raise ValueError(f"invalid reference reduction name {self.name!r}")
        if self.operation not in {"sum", "max"}:
            raise ValueError("reference reductions support only sum and max")

    def canonical(self) -> dict[str, object]:
        return {
            "name": self.name,
            "operation": self.operation,
            "value": self.value.canonical(),
        }


@dataclass(frozen=True, slots=True)
class Reference:
    reductions: tuple[Reduction, ...]
    weight: Expr

    def canonical(self) -> dict[str, object]:
        return {
            "reductions": [item.canonical() for item in self.reductions],
            "weight": self.weight.canonical(),
        }

    def weights(self, scores: np.ndarray) -> np.ndarray:
        env: dict[str, object] = {"score": scores}
        for reduction in self.reductions:
            values = np.asarray(expr.evaluate(reduction.value, env), dtype=np.float32)
            if reduction.operation == "sum":
                env[reduction.name] = np.sum(values, dtype=np.float32)
            else:
                env[reduction.name] = np.max(values)
        result = np.asarray(expr.evaluate(self.weight, env), dtype=np.float32)
        if result.shape == ():
            result = np.full(scores.shape, result, dtype=np.float32)
        if result.shape != scores.shape:
            raise ValueError(
                f"RowNorm reference produced {result.shape}, expected {scores.shape}"
            )
        return result


@dataclass(frozen=True, slots=True)
class Merge:
    updates: tuple[Update, ...]
    left_scale: Expr
    right_scale: Expr

    def canonical(self) -> dict[str, object]:
        return {
            "updates": [item.canonical() for item in self.updates],
            "left_scale": self.left_scale.canonical(),
            "right_scale": self.right_scale.canonical(),
        }


@dataclass(frozen=True, slots=True)
class RowNorm:
    name: str
    states: tuple[State, ...]
    updates: tuple[Update, ...]
    rescale: Expr
    weight: Expr
    final_scale: Expr
    reference: Reference
    merge: Merge | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("RowNorm name must not be empty")
        state_names = tuple(item.name for item in self.states)
        if len(set(state_names)) != len(state_names):
            raise ValueError("RowNorm state names must be unique")
        if tuple(item.state for item in self.updates) != state_names:
            raise ValueError("RowNorm updates must define every state once, in state order")
        for item in self.updates:
            _require_float(item.value, f"update {item.state}")
        for name, value in (
            ("rescale", self.rescale),
            ("weight", self.weight),
            ("final_scale", self.final_scale),
            ("reference weight", self.reference.weight),
        ):
            _require_float(value, name)
        self._validate_variables()

    @property
    def supports_split_k(self) -> bool:
        return self.merge is not None

    def canonical(self) -> dict[str, object]:
        return {
            "name": self.name,
            "states": [item.canonical() for item in self.states],
            "updates": [item.canonical() for item in self.updates],
            "rescale": self.rescale.canonical(),
            "weight": self.weight.canonical(),
            "final_scale": self.final_scale.canonical(),
            "reference": self.reference.canonical(),
            "merge": self.merge.canonical() if self.merge is not None else None,
        }

    def online_weights(self, scores: np.ndarray) -> np.ndarray:
        """Interpret the fixed-size online protocol independently of codegen."""
        states = {item.name: np.float32(item.initial) for item in self.states}
        weights = np.empty(scores.shape, dtype=np.float32)
        for index, score_value in enumerate(np.asarray(scores, dtype=np.float32)):
            env: dict[str, object] = {"score": score_value}
            env.update({f"old.{name}": value for name, value in states.items()})
            new_values: dict[str, np.float32] = {}
            for update in self.updates:
                value = np.float32(expr.evaluate(update.value, {**env, **new_values}))
                new_values[f"new.{update.state}"] = value
            step_env = {**env, **new_values}
            rescale = np.float32(expr.evaluate(self.rescale, step_env))
            weights[:index] *= rescale
            weights[index] = np.float32(expr.evaluate(self.weight, step_env))
            states = {
                item.name: new_values[f"new.{item.name}"] for item in self.states
            }
        final_env = {f"state.{name}": value for name, value in states.items()}
        weights *= np.float32(expr.evaluate(self.final_scale, final_env))
        return weights

    def merge_states(
        self,
        left: dict[str, float],
        right: dict[str, float],
    ) -> tuple[dict[str, np.float32], np.float32, np.float32]:
        if self.merge is None:
            raise ValueError(f"RowNorm {self.name!r} does not define merge")
        env: dict[str, object] = {
            **{f"left.{name}": np.float32(value) for name, value in left.items()},
            **{f"right.{name}": np.float32(value) for name, value in right.items()},
        }
        new_values: dict[str, np.float32] = {}
        for update in self.merge.updates:
            new_values[f"new.{update.state}"] = np.float32(
                expr.evaluate(update.value, {**env, **new_values})
            )
        scales_env = {**env, **new_values}
        merged = {
            item.name: new_values[f"new.{item.name}"] for item in self.states
        }
        return (
            merged,
            np.float32(expr.evaluate(self.merge.left_scale, scales_env)),
            np.float32(expr.evaluate(self.merge.right_scale, scales_env)),
        )

    def _validate_variables(self) -> None:
        state_names = [state.name for state in self.states]
        old = {f"old.{name}" for name in state_names}
        produced: set[str] = set()
        for update in self.updates:
            allowed = {"score"} | old | produced
            unknown = update.value.variables() - allowed
            if unknown:
                raise ValueError(
                    f"RowNorm update {update.state!r} has unavailable variables "
                    f"{sorted(unknown)}"
                )
            produced.add(f"new.{update.state}")
        online = {"score"} | old | {f"new.{name}" for name in state_names}
        for label, value in (("rescale", self.rescale), ("weight", self.weight)):
            unknown = value.variables() - online
            if unknown:
                raise ValueError(f"RowNorm {label} has unknown variables {sorted(unknown)}")
        final_allowed = {f"state.{name}" for name in state_names}
        unknown = self.final_scale.variables() - final_allowed
        if unknown:
            raise ValueError(f"RowNorm final_scale has unknown variables {sorted(unknown)}")

        reference_names: set[str] = set()
        for reduction in self.reference.reductions:
            unknown = reduction.value.variables() - ({"score"} | reference_names)
            if unknown:
                raise ValueError(
                    f"reference reduction {reduction.name!r} has unknown variables "
                    f"{sorted(unknown)}"
                )
            if reduction.name in reference_names:
                raise ValueError(f"duplicate reference reduction {reduction.name!r}")
            reference_names.add(reduction.name)
        unknown = self.reference.weight.variables() - ({"score"} | reference_names)
        if unknown:
            raise ValueError(f"reference weight has unknown variables {sorted(unknown)}")

        if self.merge is not None:
            if tuple(item.state for item in self.merge.updates) != tuple(state_names):
                raise ValueError("merge updates must define every state once, in state order")
            left_right = {
                f"left.{name}" for name in state_names
            } | {f"right.{name}" for name in state_names}
            merged: set[str] = set()
            for update in self.merge.updates:
                unknown = update.value.variables() - (left_right | merged)
                if unknown:
                    raise ValueError(
                        f"merge update {update.state!r} has unknown variables "
                        f"{sorted(unknown)}"
                    )
                merged.add(f"new.{update.state}")
            allowed = left_right | merged
            for label, value in (
                ("left_scale", self.merge.left_scale),
                ("right_scale", self.merge.right_scale),
            ):
                unknown = value.variables() - allowed
                if unknown:
                    raise ValueError(f"merge {label} has unknown variables {sorted(unknown)}")


def _require_float(value: Expr, label: str) -> None:
    if value.value_type is not ValueType.FLOAT:
        raise TypeError(f"RowNorm {label} must be a float expression")


def softmax(scale: float = 1.0) -> RowNorm:
    score = expr.var("score") * scale
    old_m = expr.var("old.m")
    old_l = expr.var("old.l")
    new_m = expr.maximum(old_m, score)
    old_scale = expr.where(old_l > 0.0, expr.exp(old_m - expr.var("new.m")), 0.0)
    new_weight = expr.where(
        score > -float("inf"), expr.exp(score - expr.var("new.m")), 0.0
    )
    new_l = old_l * old_scale + new_weight
    merge_m = expr.maximum(expr.var("left.m"), expr.var("right.m"))
    merge_left_scale = expr.where(
        expr.var("left.l") > 0.0,
        expr.exp(expr.var("left.m") - expr.var("new.m")),
        0.0,
    )
    merge_right_scale = expr.where(
        expr.var("right.l") > 0.0,
        expr.exp(expr.var("right.m") - expr.var("new.m")),
        0.0,
    )
    merge_l = expr.var("left.l") * merge_left_scale + expr.var(
        "right.l"
    ) * merge_right_scale
    ref_score = expr.var("score") * scale
    ref_numerator = expr.where(
        ref_score > -float("inf"), expr.exp(ref_score - expr.var("m")), 0.0
    )
    return RowNorm(
        name="softmax",
        states=(State("m", -float("inf")), State("l", 0.0)),
        updates=(Update("m", new_m), Update("l", new_l)),
        rescale=old_scale,
        weight=new_weight,
        final_scale=expr.where(
            expr.var("state.l") > 0.0, 1.0 / expr.var("state.l"), 0.0
        ),
        reference=Reference(
            reductions=(
                Reduction("m", "max", ref_score),
                Reduction("l", "sum", ref_numerator),
            ),
            weight=expr.where(
                expr.var("l") > 0.0, ref_numerator / expr.var("l"), 0.0
            ),
        ),
        merge=Merge(
            updates=(Update("m", merge_m), Update("l", merge_l)),
            left_scale=merge_left_scale,
            right_scale=merge_right_scale,
        ),
    )


def identity() -> RowNorm:
    score = expr.var("score")
    return RowNorm(
        name="identity",
        states=(),
        updates=(),
        rescale=expr.as_expr(1.0),
        weight=score,
        final_scale=expr.as_expr(1.0),
        reference=Reference((), score),
        merge=Merge((), expr.as_expr(1.0), expr.as_expr(1.0)),
    )


def l1(scale: float = 1.0) -> RowNorm:
    score = expr.var("score") * scale
    return RowNorm(
        name="l1",
        states=(State("r", 0.0),),
        updates=(Update("r", expr.var("old.r") + expr.abs_(score)),),
        rescale=expr.as_expr(1.0),
        weight=score,
        final_scale=1.0 / expr.maximum(expr.var("state.r"), 1.0),
        reference=Reference(
            (Reduction("r", "sum", expr.abs_(score)),),
            score / expr.maximum(expr.var("r"), 1.0),
        ),
        merge=Merge(
            (Update("r", expr.var("left.r") + expr.var("right.r")),),
            expr.as_expr(1.0),
            expr.as_expr(1.0),
        ),
    )


def sigmoid_norm(scale: float = 1.0) -> RowNorm:
    value = expr.sigmoid(expr.var("score") * scale)
    state_sum = expr.var("state.l")
    reference_sum = expr.var("l")
    return RowNorm(
        name="sigmoid_norm",
        states=(State("l", 0.0),),
        updates=(Update("l", expr.var("old.l") + value),),
        rescale=expr.as_expr(1.0),
        weight=value,
        final_scale=expr.where(state_sum > 0.0, 1.0 / state_sum, 0.0),
        reference=Reference(
            (Reduction("l", "sum", value),),
            expr.where(reference_sum > 0.0, value / reference_sum, 0.0),
        ),
        merge=Merge(
            (Update("l", expr.var("left.l") + expr.var("right.l")),),
            expr.as_expr(1.0),
            expr.as_expr(1.0),
        ),
    )


__all__ = [
    "Merge",
    "Reduction",
    "Reference",
    "RowNorm",
    "State",
    "Update",
    "identity",
    "l1",
    "sigmoid_norm",
    "softmax",
]
