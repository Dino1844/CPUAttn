from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json

import numpy as np

from . import expr, rownorm, transition
from .expr import Expr
from .tensor import TensorArgSpec


def _arguments_canonical(arguments: tuple[TensorArgSpec, ...]) -> list[dict[str, object]]:
    return [item.canonical() for item in arguments]


def _validate_argument_names(arguments: tuple[TensorArgSpec, ...]) -> None:
    names = [item.name for item in arguments]
    if len(names) != len(set(names)):
        raise ValueError("operator argument names must be unique")


@dataclass(frozen=True, slots=True)
class Parallel:
    score_mod: Expr = field(default_factory=expr.identity)
    mask_mod: Expr = field(default_factory=lambda: expr.as_expr(True))
    row_norm: rownorm.RowNorm = field(default_factory=rownorm.softmax)
    arguments: tuple[TensorArgSpec, ...] = ()

    pattern: str = field(default="parallel", init=False)
    # canonical() memoizes into this slot and returns the shared dict:
    # callers must treat canonical() output as read-only.
    _canonical_memo: tuple[dict[str, object], str] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        _validate_argument_names(self.arguments)

    def _memo(self) -> tuple[dict[str, object], str]:
        if self._canonical_memo is None:
            canonical = {
                "pattern": self.pattern,
                "score_mod": self.score_mod.canonical(),
                "mask_mod": self.mask_mod.canonical(),
                "row_norm": self.row_norm.canonical(),
                "arguments": _arguments_canonical(self.arguments),
            }
            object.__setattr__(self, "_canonical_memo", (canonical, _fingerprint(canonical)))
        return self._canonical_memo

    def canonical(self) -> dict[str, object]:
        return self._memo()[0]

    @property
    def fingerprint(self) -> str:
        return self._memo()[1]


@dataclass(frozen=True, slots=True)
class Linear:
    transition: transition.Program
    readout: transition.Readout = field(default_factory=transition.readout)
    q_mod: Expr = field(default_factory=expr.identity)
    k_mod: Expr = field(default_factory=expr.identity)
    v_mod: Expr = field(default_factory=expr.identity)
    arguments: tuple[TensorArgSpec, ...] = ()

    pattern: str = field(default="linear", init=False)
    # canonical() memoizes into this slot and returns the shared dict:
    # callers must treat canonical() output as read-only.
    _canonical_memo: tuple[dict[str, object], str] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        _validate_argument_names(self.arguments)

    def _memo(self) -> tuple[dict[str, object], str]:
        if self._canonical_memo is None:
            canonical = {
                "pattern": self.pattern,
                "transition": self.transition.canonical(),
                "readout": self.readout.canonical(),
                "q_mod": self.q_mod.canonical(),
                "k_mod": self.k_mod.canonical(),
                "v_mod": self.v_mod.canonical(),
                "arguments": _arguments_canonical(self.arguments),
            }
            object.__setattr__(self, "_canonical_memo", (canonical, _fingerprint(canonical)))
        return self._canonical_memo

    def canonical(self) -> dict[str, object]:
        return self._memo()[0]

    @property
    def fingerprint(self) -> str:
        return self._memo()[1]


Operator = Parallel | Linear


@dataclass(frozen=True, slots=True)
class LinearResult:
    output: np.ndarray
    state: np.ndarray


def _fingerprint(value: dict[str, object]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = ["Linear", "LinearResult", "Operator", "Parallel"]
