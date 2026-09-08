from __future__ import annotations

import math
from typing import Mapping

from ...core.expr import Expr
from ...core.tensor import Axis, TensorArgSpec
from ...errors import UnsupportedError


def emit_expr(value: Expr, variables: Mapping[str, str]) -> str:
    if value.kind == "const":
        if value.value_type.value == "bool":
            return "1" if value.value else "0"
        return float_literal(float(value.value))
    if value.kind == "var":
        try:
            return variables[str(value.name)]
        except KeyError as exc:
            raise UnsupportedError(
                f"expression variable {value.name!r} has no lowering in this stage"
            ) from exc

    args = [emit_expr(arg, variables) for arg in value.args]
    binary = {
        "add": "+",
        "sub": "-",
        "mul": "*",
        "div": "/",
        "lt": "<",
        "le": "<=",
        "gt": ">",
        "ge": ">=",
        "eq": "==",
        "ne": "!=",
        "and": "&&",
        "or": "||",
    }
    if value.kind in binary:
        return f"(({args[0]}) {binary[value.kind]} ({args[1]}))"
    if value.kind == "neg":
        return f"(-({args[0]}))"
    if value.kind == "not":
        return f"(!({args[0]}))"
    if value.kind == "exp":
        return f"expf({args[0]})"
    if value.kind == "log":
        return f"logf({args[0]})"
    if value.kind == "tanh":
        return f"tanhf({args[0]})"
    if value.kind == "sigmoid":
        return f"(1.0f / (1.0f + expf(-({args[0]}))))"
    if value.kind == "relu":
        return f"fmaxf({args[0]}, 0.0f)"
    if value.kind == "abs":
        return f"fabsf({args[0]})"
    if value.kind == "max":
        return f"fmaxf({args[0]}, {args[1]})"
    if value.kind == "min":
        return f"fminf({args[0]}, {args[1]})"
    if value.kind == "where":
        return f"(({args[0]}) ? ({args[1]}) : ({args[2]}))"
    raise UnsupportedError(f"expression operation {value.kind!r} has no C lowering")


def emit_simd_expr(value: Expr, variables: Mapping[str, str]) -> str:
    if value.kind == "const":
        if value.value_type.value == "bool":
            return "cpuattn_mask_true()" if value.value else "cpuattn_mask_false()"
        return f"cpuattn_simd_set1({float_literal(float(value.value))})"
    if value.kind == "var":
        try:
            return variables[value.name or ""]
        except KeyError as exc:
            raise UnsupportedError(
                f"expression variable {value.name!r} has no SIMD lowering in this stage"
            ) from exc

    args = [emit_simd_expr(arg, variables) for arg in value.args]
    binary = {
        "add": "cpuattn_simd_add",
        "sub": "cpuattn_simd_sub",
        "mul": "cpuattn_simd_mul",
        "div": "cpuattn_simd_div",
        "max": "cpuattn_simd_max",
        "min": "cpuattn_simd_min",
        "lt": "cpuattn_simd_lt",
        "le": "cpuattn_simd_le",
        "gt": "cpuattn_simd_gt",
        "ge": "cpuattn_simd_ge",
        "eq": "cpuattn_simd_eq",
        "ne": "cpuattn_simd_ne",
        "and": "cpuattn_mask_and",
        "or": "cpuattn_mask_or",
    }
    unary = {
        "neg": "cpuattn_simd_neg",
        "exp": "cpuattn_simd_exp",
        "log": "cpuattn_simd_log",
        "tanh": "cpuattn_simd_tanh",
        "sigmoid": "cpuattn_simd_sigmoid",
        "relu": "cpuattn_simd_relu",
        "abs": "cpuattn_simd_abs",
        "not": "cpuattn_mask_not",
    }
    if value.kind in binary:
        return f"{binary[value.kind]}({args[0]}, {args[1]})"
    if value.kind in unary:
        return f"{unary[value.kind]}({args[0]})"
    if value.kind == "where":
        return f"cpuattn_simd_select({args[0]}, {args[1]}, {args[2]})"
    raise UnsupportedError(
        f"expression operation {value.kind!r} has no SIMD lowering"
    )


def float_literal(value: float) -> str:
    if math.isinf(value):
        return "INFINITY" if value > 0 else "(-INFINITY)"
    if math.isnan(value):
        return "NAN"
    literal = f"{value:.9g}"
    if "." not in literal and "e" not in literal.lower():
        literal += ".0"
    return literal + "f"


def parallel_argument_map(specs: tuple[TensorArgSpec, ...]) -> dict[str, str]:
    names = {
        Axis.BATCH: ("b", "B"),
        Axis.QUERY_HEAD: ("hq", "HQ"),
        Axis.KV_HEAD: ("hkv", "HKV"),
        Axis.QUERY: ("qi", "SQ"),
        Axis.KEY: ("ki", "SKV"),
    }
    return {spec.name: array_access(spec, names) for spec in specs}


def linear_argument_map(
    specs: tuple[TensorArgSpec, ...], vector_axis: str
) -> dict[str, str]:
    names = linear_names()
    result = {}
    for spec in specs:
        unavailable = Axis.DV if vector_axis == "d" else Axis.D
        if unavailable in spec.axes:
            continue
        result[spec.name] = array_access(spec, names)
    return result


def linear_names() -> dict[Axis, tuple[str, str]]:
    return {
        Axis.BATCH: ("b", "B"),
        Axis.STATE_HEAD: ("h", "H"),
        Axis.PARAMETER_GROUP: ("g", "G"),
        Axis.SEQUENCE: ("s", "S"),
        Axis.D: ("d", "D"),
        Axis.DV: ("dv", "DV"),
    }


def simd_argument_map(
    specs: tuple[TensorArgSpec, ...],
    names: Mapping[Axis, tuple[str, str]],
    vector_axis: Axis,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for spec in specs:
        unavailable = (
            Axis.DV
            if vector_axis is Axis.D
            else Axis.D
            if vector_axis is Axis.DV
            else None
        )
        if unavailable is not None and unavailable in spec.axes:
            continue
        scalar = array_access(spec, names)
        if vector_axis not in spec.axes:
            result[spec.name] = f"cpuattn_simd_set1({scalar})"
            continue
        stride = (
            " * ".join(
                names[axis][1]
                for axis in spec.axes[spec.axes.index(vector_axis) + 1 :]
            )
            or "1"
        )
        result[spec.name] = (
            f"cpuattn_simd_load_strided(arg_{spec.name} + "
            f"({array_offset(spec, names)}), {stride})"
        )
    return result


def array_offset(
    spec: TensorArgSpec, names: Mapping[Axis, tuple[str, str]]
) -> str:
    offset = "0"
    for axis in spec.axes:
        index, dimension = names[axis]
        offset = f"(({offset}) * {dimension} + {index})"
    return offset


def array_access(
    spec: TensorArgSpec, names: Mapping[Axis, tuple[str, str]]
) -> str:
    if not spec.axes:
        return f"arg_{spec.name}[0]"
    return f"arg_{spec.name}[{array_offset(spec, names)}]"


__all__ = [
    "emit_expr",
    "emit_simd_expr",
    "float_literal",
    "linear_argument_map",
    "linear_names",
    "parallel_argument_map",
    "simd_argument_map",
]
