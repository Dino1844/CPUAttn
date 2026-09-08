from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

import numpy as np


class ValueType(str, Enum):
    FLOAT = "float"
    BOOL = "bool"


@dataclass(frozen=True, slots=True)
class Expr:
    kind: str
    value_type: ValueType
    args: tuple["Expr", ...] = ()
    value: float | bool | None = None
    name: str | None = None

    def canonical(self) -> dict[str, object]:
        data: dict[str, object] = {
            "kind": self.kind,
            "type": self.value_type.value,
        }
        if self.args:
            data["args"] = [arg.canonical() for arg in self.args]
        if self.value is not None:
            data["value"] = self.value
        if self.name is not None:
            data["name"] = self.name
        return data

    def variables(self) -> frozenset[str]:
        names = {self.name} if self.kind == "var" and self.name is not None else set()
        for arg in self.args:
            names.update(arg.variables())
        return frozenset(names)

    def _float_binary(self, other: Any, kind: str) -> "Expr":
        rhs = as_expr(other)
        _require_type(self, ValueType.FLOAT, kind)
        _require_type(rhs, ValueType.FLOAT, kind)
        return Expr(kind, ValueType.FLOAT, (self, rhs))

    def __add__(self, other: Any) -> "Expr":
        return self._float_binary(other, "add")

    def __radd__(self, other: Any) -> "Expr":
        return as_expr(other)._float_binary(self, "add")

    def __sub__(self, other: Any) -> "Expr":
        return self._float_binary(other, "sub")

    def __rsub__(self, other: Any) -> "Expr":
        return as_expr(other)._float_binary(self, "sub")

    def __mul__(self, other: Any) -> "Expr":
        return self._float_binary(other, "mul")

    def __rmul__(self, other: Any) -> "Expr":
        return as_expr(other)._float_binary(self, "mul")

    def __truediv__(self, other: Any) -> "Expr":
        return self._float_binary(other, "div")

    def __rtruediv__(self, other: Any) -> "Expr":
        return as_expr(other)._float_binary(self, "div")

    def __neg__(self) -> "Expr":
        _require_type(self, ValueType.FLOAT, "neg")
        return Expr("neg", ValueType.FLOAT, (self,))

    def _compare(self, other: Any, kind: str) -> "Expr":
        rhs = as_expr(other)
        _require_type(self, ValueType.FLOAT, kind)
        _require_type(rhs, ValueType.FLOAT, kind)
        return Expr(kind, ValueType.BOOL, (self, rhs))

    def __lt__(self, other: Any) -> "Expr":
        return self._compare(other, "lt")

    def __le__(self, other: Any) -> "Expr":
        return self._compare(other, "le")

    def __gt__(self, other: Any) -> "Expr":
        return self._compare(other, "gt")

    def __ge__(self, other: Any) -> "Expr":
        return self._compare(other, "ge")

    def equal(self, other: Any) -> "Expr":
        return self._compare(other, "eq")

    def not_equal(self, other: Any) -> "Expr":
        return self._compare(other, "ne")

    def __and__(self, other: Any) -> "Expr":
        rhs = as_expr(other)
        _require_type(self, ValueType.BOOL, "and")
        _require_type(rhs, ValueType.BOOL, "and")
        return Expr("and", ValueType.BOOL, (self, rhs))

    def __or__(self, other: Any) -> "Expr":
        rhs = as_expr(other)
        _require_type(self, ValueType.BOOL, "or")
        _require_type(rhs, ValueType.BOOL, "or")
        return Expr("or", ValueType.BOOL, (self, rhs))

    def __invert__(self) -> "Expr":
        _require_type(self, ValueType.BOOL, "not")
        return Expr("not", ValueType.BOOL, (self,))

    def __bool__(self) -> bool:
        raise TypeError("symbolic expressions cannot be converted to bool")


def _require_type(expr: Expr, expected: ValueType, operation: str) -> None:
    if expr.value_type is not expected:
        raise TypeError(f"{operation} requires {expected.value} expressions")


def as_expr(value: Any) -> Expr:
    if isinstance(value, Expr):
        return value
    if isinstance(value, (bool, np.bool_)):
        return Expr("const", ValueType.BOOL, value=bool(value))
    if isinstance(value, (int, float, np.number)):
        return Expr("const", ValueType.FLOAT, value=float(value))
    raise TypeError(f"cannot use {type(value).__name__} as an expression")


def var(name: str, value_type: ValueType = ValueType.FLOAT) -> Expr:
    if not name:
        raise ValueError("variable name must not be empty")
    return Expr("var", value_type, name=name)


def argument(name: str, value_type: ValueType = ValueType.FLOAT) -> Expr:
    return var(name, value_type)


def identity() -> Expr:
    return var("x")


def causal() -> Expr:
    return var("key_index") <= var("query_index")


def _unary(kind: str, value: Any) -> Expr:
    expr = as_expr(value)
    _require_type(expr, ValueType.FLOAT, kind)
    return Expr(kind, ValueType.FLOAT, (expr,))


def exp(value: Any) -> Expr:
    return _unary("exp", value)


def log(value: Any) -> Expr:
    return _unary("log", value)


def tanh(value: Any) -> Expr:
    return _unary("tanh", value)


def sigmoid(value: Any) -> Expr:
    return _unary("sigmoid", value)


def relu(value: Any) -> Expr:
    return _unary("relu", value)


def abs_(value: Any) -> Expr:
    return _unary("abs", value)


def maximum(left: Any, right: Any) -> Expr:
    return as_expr(left)._float_binary(right, "max")


def minimum(left: Any, right: Any) -> Expr:
    return as_expr(left)._float_binary(right, "min")


def where(condition: Any, when_true: Any, when_false: Any) -> Expr:
    cond = as_expr(condition)
    lhs = as_expr(when_true)
    rhs = as_expr(when_false)
    _require_type(cond, ValueType.BOOL, "where")
    if lhs.value_type is not rhs.value_type:
        raise TypeError("where branches must have the same type")
    return Expr("where", lhs.value_type, (cond, lhs, rhs))


def evaluate(expr: Expr, env: Mapping[str, Any]) -> Any:
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        return _evaluate(expr, env)


def _evaluate(expr: Expr, env: Mapping[str, Any]) -> Any:
    if expr.kind == "const":
        return expr.value
    if expr.kind == "var":
        if expr.name not in env:
            raise ValueError(f"expression variable {expr.name!r} is not bound")
        return env[expr.name]
    values = tuple(_evaluate(arg, env) for arg in expr.args)
    if expr.kind == "add":
        return values[0] + values[1]
    if expr.kind == "sub":
        return values[0] - values[1]
    if expr.kind == "mul":
        return values[0] * values[1]
    if expr.kind == "div":
        return values[0] / values[1]
    if expr.kind == "neg":
        return -values[0]
    if expr.kind == "exp":
        return np.exp(values[0])
    if expr.kind == "log":
        return np.log(values[0])
    if expr.kind == "tanh":
        return np.tanh(values[0])
    if expr.kind == "sigmoid":
        return 1.0 / (1.0 + np.exp(-values[0]))
    if expr.kind == "relu":
        return np.maximum(values[0], 0.0)
    if expr.kind == "abs":
        return np.abs(values[0])
    if expr.kind == "max":
        return np.maximum(values[0], values[1])
    if expr.kind == "min":
        return np.minimum(values[0], values[1])
    if expr.kind == "lt":
        return values[0] < values[1]
    if expr.kind == "le":
        return values[0] <= values[1]
    if expr.kind == "gt":
        return values[0] > values[1]
    if expr.kind == "ge":
        return values[0] >= values[1]
    if expr.kind == "eq":
        return values[0] == values[1]
    if expr.kind == "ne":
        return values[0] != values[1]
    if expr.kind == "and":
        return np.logical_and(values[0], values[1])
    if expr.kind == "or":
        return np.logical_or(values[0], values[1])
    if expr.kind == "not":
        return np.logical_not(values[0])
    if expr.kind == "where":
        return np.where(values[0], values[1], values[2])
    raise ValueError(f"unsupported expression operation {expr.kind!r}")


__all__ = [
    "Expr",
    "ValueType",
    "abs_",
    "argument",
    "as_expr",
    "causal",
    "evaluate",
    "exp",
    "identity",
    "log",
    "maximum",
    "minimum",
    "relu",
    "sigmoid",
    "tanh",
    "var",
    "where",
]
