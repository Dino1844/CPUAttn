from __future__ import annotations

from jinja2 import Environment

from ...core import rownorm
from ...core.expr import Expr
from ...core.operator import Parallel
from ...core.tensor import Axis
from ...core.validate import ParallelCall
from ...errors import UnsupportedError
from ...schedule.plan import CodePlan, LoweringKind
from .expression import (
    emit_expr,
    emit_simd_expr,
    float_literal,
    parallel_argument_map,
    simd_argument_map,
)


def render_parallel(
    templates: Environment,
    operator: Parallel,
    call: ParallelCall,
    code: CodePlan,
) -> str:
    argument_map = parallel_argument_map(operator.arguments)
    query_index = "(qi + query_offset)"
    base = {
        "x": "raw_score",
        "query_index": f"((float){query_index})",
        "key_index": "((float)ki)",
        **argument_map,
    }
    vector_base = {
        "x": "raw_score_v",
        "query_index": f"cpuattn_simd_set1((float){query_index})",
        "key_index": "cpuattn_simd_iota((float)ki)",
        **simd_argument_map(
            operator.arguments,
            {
                Axis.BATCH: ("b", "B"),
                Axis.QUERY_HEAD: ("hq", "HQ"),
                Axis.KV_HEAD: ("hkv", "HKV"),
                Axis.QUERY: ("qi", "SQ"),
                Axis.KEY: ("ki", "SKV"),
            },
            Axis.KEY,
        ),
    }
    old = {
        f"old.{state.name}": f"old_{state.name}"
        for state in operator.row_norm.states
    }
    new: dict[str, str] = {}
    updates = []
    for update in operator.row_norm.updates:
        rendered = emit_expr(update.value, {"score": "score", **old, **new})
        updates.append({"state": update.state, "value_c": rendered})
        new[f"new.{update.state}"] = f"new_{update.state}"
    row_norm: dict[str, object] = {
        "states": [
            {"name": state.name, "initial_c": float_literal(state.initial)}
            for state in operator.row_norm.states
        ],
        "updates": updates,
        "rescale_c": emit_expr(
            operator.row_norm.rescale, {"score": "score", **old, **new}
        ),
        "weight_c": emit_expr(
            operator.row_norm.weight, {"score": "score", **old, **new}
        ),
        "final_scale_c": emit_expr(
            operator.row_norm.final_scale,
            {
                f"state.{state.name}": f"state_{state.name}"
                for state in operator.row_norm.states
            },
        ),
    }
    softmax_scale = _structured_softmax_scale(operator.row_norm)
    row_norm["kind"] = "softmax" if softmax_scale is not None else "generic"
    row_norm["softmax_scale_c"] = float_literal(
        softmax_scale if softmax_scale is not None else 1.0
    )

    template = {
        LoweringKind.PARALLEL_BLOCKED: "parallel.c.j2",
        LoweringKind.PARALLEL_SPLIT_K: "parallel_split_k.c.j2",
        LoweringKind.PARALLEL_2D: "parallel_2d.c.j2",
    }[code.lowering]
    if code.lowering in {
        LoweringKind.PARALLEL_SPLIT_K,
        LoweringKind.PARALLEL_2D,
    }:
        merge = operator.row_norm.merge
        if merge is None:
            raise UnsupportedError(
                f"RowNorm {operator.row_norm.name!r} has no merge for parallel_split_k"
            )
        merge_variables = {
            **{
                f"left.{state.name}": f"left_{state.name}"
                for state in operator.row_norm.states
            },
            **{
                f"right.{state.name}": f"right_{state.name}"
                for state in operator.row_norm.states
            },
        }
        merge_updates = []
        for update in merge.updates:
            rendered = emit_expr(update.value, merge_variables)
            merge_updates.append({"state": update.state, "value_c": rendered})
            merge_variables[f"new.{update.state}"] = f"merged_{update.state}"
        row_norm.update(
            merge_updates=merge_updates,
            left_scale_c=emit_expr(merge.left_scale, merge_variables),
            right_scale_c=emit_expr(merge.right_scale, merge_variables),
        )

    return templates.get_template(template).render(
        code=code,
        arguments=operator.arguments,
        score_expr=emit_expr(operator.score_mod, base),
        mask_expr=emit_expr(operator.mask_mod, base),
        score_expr_simd=emit_simd_expr(operator.score_mod, vector_base),
        mask_expr_simd=emit_simd_expr(operator.mask_mod, vector_base),
        causal_block_limit=_implies_causality(operator.mask_mod),
        query_offset="query_offset",
        row_norm=row_norm,
    )


def _implies_causality(value: Expr) -> bool:
    """Return whether a mask structurally guarantees key_index <= query_index."""
    if value.kind == "and":
        return any(_implies_causality(arg) for arg in value.args)
    if value.kind not in {"le", "ge"} or len(value.args) != 2:
        return False
    left, right = value.args
    if value.kind == "ge":
        left, right = right, left
    return (
        left.kind == "var"
        and left.name == "key_index"
        and right.kind == "var"
        and right.name == "query_index"
    )


def _structured_softmax_scale(value: rownorm.RowNorm) -> float | None:
    """Recognize the exact built-in protocol without trusting its display name."""
    if len(value.updates) != 2 or value.updates[0].state != "m":
        return None
    maximum = value.updates[0].value
    if maximum.kind != "max" or len(maximum.args) != 2:
        return None
    old_maximum, scaled_score = maximum.args
    if old_maximum.kind != "var" or old_maximum.name != "old.m":
        return None
    if scaled_score.kind != "mul" or len(scaled_score.args) != 2:
        return None
    score, scale = scaled_score.args
    if score.kind != "var" or score.name != "score":
        return None
    if scale.kind != "const" or scale.value_type.value != "float":
        return None
    scale_value = float(scale.value)
    return scale_value if value == rownorm.softmax(scale_value) else None


__all__ = ["render_parallel"]
