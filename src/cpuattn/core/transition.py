from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .expr import Expr, ValueType, as_expr, var


class ReadTiming(str, Enum):
    BEFORE = "before"
    AFTER = "after"


@dataclass(frozen=True, slots=True)
class Scale:
    factor: Expr

    def canonical(self) -> dict[str, object]:
        return {"operation": "scale", "factor": self.factor.canonical()}


@dataclass(frozen=True, slots=True)
class Rank1:
    left: Expr
    right: Expr

    def canonical(self) -> dict[str, object]:
        return {
            "operation": "rank1",
            "left": self.left.canonical(),
            "right": self.right.canonical(),
        }


@dataclass(frozen=True, slots=True)
class Outer:
    left: Expr
    right: Expr

    def canonical(self) -> dict[str, object]:
        return {
            "operation": "outer",
            "left": self.left.canonical(),
            "right": self.right.canonical(),
        }


TransitionStep = Scale | Rank1 | Outer


@dataclass(frozen=True, slots=True)
class Program:
    steps: tuple[TransitionStep, ...]

    def __post_init__(self) -> None:
        if not self.steps:
            raise ValueError("a Linear transition needs at least one operation")

    def canonical(self) -> dict[str, object]:
        return {"steps": [step.canonical() for step in self.steps]}


@dataclass(frozen=True, slots=True)
class Readout:
    query: Expr
    timing: ReadTiming = ReadTiming.AFTER

    def canonical(self) -> dict[str, object]:
        return {"query": self.query.canonical(), "timing": self.timing.value}


def _float(value: object, label: str) -> Expr:
    result = as_expr(value)
    if result.value_type is not ValueType.FLOAT:
        raise TypeError(f"{label} must be a float expression")
    return result


def scale(factor: object) -> Scale:
    return Scale(_float(factor, "scale factor"))


def rank1(left: object, right: object) -> Rank1:
    return Rank1(_float(left, "rank1 left"), _float(right, "rank1 right"))


def outer(left: object, right: object) -> Outer:
    return Outer(_float(left, "outer left"), _float(right, "outer right"))


def program(*steps: TransitionStep) -> Program:
    return Program(tuple(steps))


def readout(query: object | None = None, *, timing: str | ReadTiming = ReadTiming.AFTER) -> Readout:
    timing_value = timing if isinstance(timing, ReadTiming) else ReadTiming(timing)
    return Readout(_float(var("q") if query is None else query, "readout query"), timing_value)


__all__ = [
    "Outer",
    "Program",
    "Rank1",
    "ReadTiming",
    "Readout",
    "Scale",
    "outer",
    "program",
    "rank1",
    "readout",
    "scale",
]
