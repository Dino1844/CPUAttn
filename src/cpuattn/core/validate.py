from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .expr import ValueType
from .operator import Linear, Operator, Parallel
from .tensor import Axis, TensorArgSpec, require_array, require_panel


_RESERVED_ARGUMENTS = {
    "x",
    "q",
    "k",
    "v",
    "score",
    "query_index",
    "key_index",
}


@dataclass(frozen=True, slots=True)
class ParallelCall:
    q: np.ndarray
    k: np.ndarray
    v: np.ndarray
    arguments: tuple[tuple[str, np.ndarray], ...]
    kv_cache: bool
    k_pitch: int = 0
    v_pitch: int = 0

    @property
    def argument_map(self) -> dict[str, np.ndarray]:
        return dict(self.arguments)

    @property
    def dimensions(self) -> dict[Axis, int]:
        b, hq, sq, d = self.q.shape
        _, hkv, skv, _ = self.k.shape
        dv = self.v.shape[-1]
        return {
            Axis.BATCH: b,
            Axis.QUERY_HEAD: hq,
            Axis.KV_HEAD: hkv,
            Axis.QUERY: sq,
            Axis.KEY: skv,
            Axis.D: d,
            Axis.DV: dv,
        }

    @property
    def query_offset(self) -> int:
        """Absolute position of the first query in a cache-backed sequence."""
        return self.k.shape[2] - self.q.shape[2] if self.kv_cache else 0

    def canonical(self) -> dict[str, object]:
        dims = self.dimensions
        return {
            "pattern": "parallel",
            "shape": {axis.value: dims[axis] for axis in dims},
            "kv_cache": self.kv_cache,
            "dtype": "fp32",
            "layout": "bhsd-contiguous",
        }


@dataclass(frozen=True, slots=True)
class LinearCall:
    q: np.ndarray
    k: np.ndarray
    v: np.ndarray
    state: np.ndarray | None
    arguments: tuple[tuple[str, np.ndarray], ...]

    @property
    def argument_map(self) -> dict[str, np.ndarray]:
        return dict(self.arguments)

    @property
    def dimensions(self) -> dict[Axis, int]:
        b, groups, sequence, d = self.q.shape
        heads = self.v.shape[1]
        dv = self.v.shape[-1]
        return {
            Axis.BATCH: b,
            Axis.PARAMETER_GROUP: groups,
            Axis.STATE_HEAD: heads,
            Axis.SEQUENCE: sequence,
            Axis.D: d,
            Axis.DV: dv,
        }

    def canonical(self) -> dict[str, object]:
        dims = self.dimensions
        return {
            "pattern": "linear",
            "shape": {axis.value: dims[axis] for axis in dims},
            "has_initial_state": self.state is not None,
            "dtype": "fp32",
            "layout": "bhsd-contiguous",
        }


ValidatedCall = ParallelCall | LinearCall

_DEFINITION_VERIFIED: set[str] = set()


def validate_definition(operator: Operator) -> None:
    # Operators are immutable and fingerprinted, so validating once is enough.
    if operator.fingerprint in _DEFINITION_VERIFIED:
        return
    argument_names = {item.name for item in operator.arguments}
    conflict = argument_names & _RESERVED_ARGUMENTS
    if conflict:
        raise ValueError(f"operator arguments use reserved names {sorted(conflict)}")
    if isinstance(operator, Parallel):
        if operator.score_mod.value_type is not ValueType.FLOAT:
            raise TypeError("Parallel score_mod must produce float")
        if operator.mask_mod.value_type is not ValueType.BOOL:
            raise TypeError("Parallel mask_mod must produce bool")
        _require_variables(operator.score_mod.variables(), {"x"} | argument_names, "score_mod")
        _require_variables(
            operator.mask_mod.variables(),
            {"query_index", "key_index"} | argument_names,
            "mask_mod",
        )
        allowed_axes = {
            Axis.BATCH,
            Axis.QUERY_HEAD,
            Axis.KV_HEAD,
            Axis.QUERY,
            Axis.KEY,
        }
    else:
        for label, value in (
            ("q_mod", operator.q_mod),
            ("k_mod", operator.k_mod),
            ("v_mod", operator.v_mod),
        ):
            if value.value_type is not ValueType.FLOAT:
                raise TypeError(f"Linear {label} must produce float")
            _require_variables(value.variables(), {"x"} | argument_names, label)
        transition_variables: set[str] = set(operator.readout.query.variables())
        for step in operator.transition.steps:
            if hasattr(step, "factor"):
                transition_variables.update(step.factor.variables())
            else:
                transition_variables.update(step.left.variables())
                transition_variables.update(step.right.variables())
        _require_variables(
            frozenset(transition_variables),
            {"q", "k", "v"} | argument_names,
            "Linear transition/readout",
        )
        allowed_axes = {
            Axis.BATCH,
            Axis.STATE_HEAD,
            Axis.PARAMETER_GROUP,
            Axis.SEQUENCE,
            Axis.D,
            Axis.DV,
        }
    for argument in operator.arguments:
        unsupported = set(argument.axes) - allowed_axes
        if unsupported:
            raise ValueError(
                f"argument {argument.name!r} uses axes unavailable to "
                f"{operator.pattern}: {sorted(axis.value for axis in unsupported)}"
            )
    _DEFINITION_VERIFIED.add(operator.fingerprint)


def validate_parallel_call(
    operator: Parallel,
    *,
    q: Any,
    k: Any,
    v: Any,
    arguments: Mapping[str, Any] | None = None,
    kv_cache: bool = False,
) -> ParallelCall:
    validate_definition(operator)
    q_array = require_array(q, name="q", rank=4)
    k_array, k_pitch = require_panel(k, name="k", rank=4)
    v_array, v_pitch = require_panel(v, name="v", rank=4)
    b, query_heads, _, d = q_array.shape
    kb, kv_heads, key_length, kd = k_array.shape
    vb, value_heads, value_length, _ = v_array.shape
    if kb != b or vb != b:
        raise ValueError("q, k, and v batch dimensions must match")
    if kd != d:
        raise ValueError("q and k channel dimensions must match")
    if kv_heads != value_heads:
        raise ValueError("k and v must use the same kv_heads")
    if key_length != value_length:
        raise ValueError("k and v sequence dimensions must match")
    if query_heads % kv_heads != 0:
        raise ValueError("query_heads must be divisible by kv_heads")
    if kv_cache and key_length < q_array.shape[2]:
        raise ValueError("kv_cache key_length must be at least query_length")
    dims = {
        Axis.BATCH: b,
        Axis.QUERY_HEAD: query_heads,
        Axis.KV_HEAD: kv_heads,
        Axis.QUERY: q_array.shape[2],
        Axis.KEY: key_length,
        Axis.D: d,
        Axis.DV: v_array.shape[3],
    }
    bound = _validate_arguments(operator.arguments, arguments, dims)
    return ParallelCall(
        q_array, k_array, v_array, bound, bool(kv_cache), k_pitch, v_pitch
    )


def validate_linear_call(
    operator: Linear,
    *,
    q: Any,
    k: Any,
    v: Any,
    state: Any | None = None,
    arguments: Mapping[str, Any] | None = None,
) -> LinearCall:
    validate_definition(operator)
    q_array = require_array(q, name="q", rank=4)
    k_array = require_array(k, name="k", rank=4)
    v_array = require_array(v, name="v", rank=4)
    b, groups, sequence, d = q_array.shape
    if k_array.shape != q_array.shape:
        raise ValueError("Linear q and k shapes must match exactly")
    vb, state_heads, value_sequence, dv = v_array.shape
    if vb != b or value_sequence != sequence:
        raise ValueError("Linear q/k/v batch and sequence dimensions must match")
    if state_heads % groups != 0:
        raise ValueError("state_heads must be divisible by parameter_groups")
    state_array = None
    if state is not None:
        state_array = require_array(state, name="state", rank=4)
        expected = (b, state_heads, d, dv)
        if state_array.shape != expected:
            raise ValueError(f"state must have shape {expected}; got {state_array.shape}")
    dims = {
        Axis.BATCH: b,
        Axis.PARAMETER_GROUP: groups,
        Axis.STATE_HEAD: state_heads,
        Axis.SEQUENCE: sequence,
        Axis.D: d,
        Axis.DV: dv,
    }
    bound = _validate_arguments(operator.arguments, arguments, dims)
    return LinearCall(q_array, k_array, v_array, state_array, bound)


def _require_variables(actual: frozenset[str], allowed: set[str], label: str) -> None:
    unknown = actual - allowed
    if unknown:
        raise ValueError(f"{label} uses undeclared variables {sorted(unknown)}")


def _validate_arguments(
    specs: tuple[TensorArgSpec, ...],
    values: Mapping[str, Any] | None,
    dimensions: Mapping[Axis, int],
) -> tuple[tuple[str, np.ndarray], ...]:
    supplied = dict(values or {})
    expected_names = {item.name for item in specs}
    if set(supplied) != expected_names:
        missing = sorted(expected_names - set(supplied))
        extra = sorted(set(supplied) - expected_names)
        raise ValueError(f"operator argument mismatch: missing={missing}, extra={extra}")
    result: list[tuple[str, np.ndarray]] = []
    for spec in specs:
        array = require_array(supplied[spec.name], name=spec.name, rank=len(spec.axes))
        expected_shape = tuple(dimensions[axis] for axis in spec.axes)
        if array.shape != expected_shape:
            raise ValueError(
                f"argument {spec.name!r} must have shape {expected_shape}; "
                f"got {array.shape}"
            )
        result.append((spec.name, array))
    return tuple(result)


def validate_call(operator: Operator, **kwargs: Any) -> ValidatedCall:
    if isinstance(operator, Parallel):
        return validate_parallel_call(operator, **kwargs)
    return validate_linear_call(operator, **kwargs)


__all__ = [
    "LinearCall",
    "ParallelCall",
    "ValidatedCall",
    "validate_call",
    "validate_definition",
    "validate_linear_call",
    "validate_parallel_call",
]
