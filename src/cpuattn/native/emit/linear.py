from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from jinja2 import Environment

from ...core import transition
from ...core.expr import Expr
from ...core.operator import Linear
from ...core.tensor import Axis, TensorArgSpec
from ...errors import UnsupportedError
from ...schedule.plan import CodePlan, LoweringKind
from ..lowering import LinearBlockPlan, linear_block_plan
from .expression import (
    emit_expr,
    emit_simd_expr,
    linear_argument_map,
    linear_names,
    simd_argument_map,
)


def render_linear(
    templates: Environment,
    operator: Linear,
    code: CodePlan,
) -> str:
    specs = {spec.name: spec for spec in operator.arguments}
    _check_linear_contexts(operator, specs)
    if code.lowering is LoweringKind.LINEAR_CHUNKED:
        plan = linear_block_plan(operator)
        if plan is None:
            raise UnsupportedError("Linear definition is not algebraically chunkable")
        return _render_chunked(templates, operator, code, plan)
    if code.lowering is LoweringKind.LINEAR_2D:
        return _render_2d(templates, operator, code)

    q_context = {"x": "x", **linear_argument_map(operator.arguments, "d")}
    v_context = {"x": "x", **linear_argument_map(operator.arguments, "dv")}
    q_vector_context = {
        "x": "x_v",
        **simd_argument_map(operator.arguments, linear_names(), Axis.D),
    }
    v_vector_context = {
        "x": "x_v",
        **simd_argument_map(operator.arguments, linear_names(), Axis.DV),
    }
    return templates.get_template("linear.c.j2").render(
        code=code,
        arguments=operator.arguments,
        q_mod=emit_expr(operator.q_mod, q_context),
        k_mod=emit_expr(operator.k_mod, q_context),
        v_mod=emit_expr(operator.v_mod, v_context),
        q_mod_simd=emit_simd_expr(operator.q_mod, q_vector_context),
        k_mod_simd=emit_simd_expr(operator.k_mod, q_vector_context),
        v_mod_simd=emit_simd_expr(operator.v_mod, v_vector_context),
        transition_code=_linear_transition_code(operator, specs, _DV_ALL),
        readout_code=_readout_code(operator, specs, _DV_ALL),
        read_before=operator.readout.timing is transition.ReadTiming.BEFORE,
    )


def _render_chunked(
    templates: Environment,
    operator: Linear,
    code: CodePlan,
    plan: LinearBlockPlan,
) -> str:
    q_context = {"x": "x", **linear_argument_map(operator.arguments, "d")}
    v_context = {"x": "x", **linear_argument_map(operator.arguments, "dv")}
    q_vector_context = {
        "x": "x_v",
        **simd_argument_map(operator.arguments, linear_names(), Axis.D),
    }
    v_vector_context = {
        "x": "x_v",
        **simd_argument_map(operator.arguments, linear_names(), Axis.DV),
    }
    factor_context = linear_argument_map(operator.arguments, "d")
    return templates.get_template("linear_chunked.c.j2").render(
        code=code,
        arguments=operator.arguments,
        q_mod=emit_expr(operator.q_mod, q_context),
        k_mod=emit_expr(operator.k_mod, q_context),
        v_mod=emit_expr(operator.v_mod, v_context),
        q_mod_simd=emit_simd_expr(operator.q_mod, q_vector_context),
        k_mod_simd=emit_simd_expr(operator.k_mod, q_vector_context),
        v_mod_simd=emit_simd_expr(operator.v_mod, v_vector_context),
        scale_expr=(
            emit_expr(plan.scale.factor, factor_context)
            if plan.scale is not None
            else "1.0f"
        ),
        read_before=operator.readout.timing is transition.ReadTiming.BEFORE,
    )


def _render_2d(
    templates: Environment,
    operator: Linear,
    code: CodePlan,
) -> str:
    specs = {spec.name: spec for spec in operator.arguments}
    q_context = {"x": "x", **linear_argument_map(operator.arguments, "d")}
    v_context = {"x": "x", **linear_argument_map(operator.arguments, "dv")}
    q_vector_context = {
        "x": "x_v",
        **simd_argument_map(operator.arguments, linear_names(), Axis.D),
    }
    v_vector_context = {
        "x": "x_v",
        **simd_argument_map(operator.arguments, linear_names(), Axis.DV),
    }
    return templates.get_template("linear_2d.c.j2").render(
        code=code,
        arguments=operator.arguments,
        q_mod=emit_expr(operator.q_mod, q_context),
        k_mod=emit_expr(operator.k_mod, q_context),
        v_mod=emit_expr(operator.v_mod, v_context),
        q_mod_simd=emit_simd_expr(operator.q_mod, q_vector_context),
        k_mod_simd=emit_simd_expr(operator.k_mod, q_vector_context),
        v_mod_simd=emit_simd_expr(operator.v_mod, v_vector_context),
        transition_code=_linear_transition_code(operator, specs, _DV_BLOCK),
        readout_code=_readout_code(operator, specs, _DV_BLOCK),
        read_before=operator.readout.timing is transition.ReadTiming.BEFORE,
    )


def _check_linear_contexts(
    operator: Linear, specs: Mapping[str, TensorArgSpec]
) -> None:
    contexts: list[tuple[str, Expr, set[str], Axis]] = [
        ("q_mod", operator.q_mod, set(), Axis.D),
        ("k_mod", operator.k_mod, set(), Axis.D),
        ("v_mod", operator.v_mod, set(), Axis.DV),
        ("readout", operator.readout.query, {"q", "k"}, Axis.D),
    ]
    for index, step in enumerate(operator.transition.steps):
        if isinstance(step, transition.Scale):
            contexts.append((f"Scale[{index}]", step.factor, {"q", "k"}, Axis.D))
        elif isinstance(step, transition.Rank1):
            contexts.extend(
                (
                    (f"Rank1[{index}].left", step.left, {"q", "k"}, Axis.D),
                    (f"Rank1[{index}].right", step.right, {"q", "k"}, Axis.D),
                )
            )
        else:
            contexts.extend(
                (
                    (f"Outer[{index}].left", step.left, {"q", "k"}, Axis.D),
                    (f"Outer[{index}].right", step.right, {"v"}, Axis.DV),
                )
            )
    for label, expression, vector_variables, vector_axis in contexts:
        unsupported_variables = (
            expression.variables() & {"q", "k", "v"}
        ) - vector_variables
        if unsupported_variables:
            raise UnsupportedError(
                f"{label} cannot use vector variables {sorted(unsupported_variables)}"
            )
        for name in expression.variables() & specs.keys():
            axes = set(specs[name].axes) & {Axis.D, Axis.DV}
            if axes and axes != {vector_axis}:
                raise UnsupportedError(
                    f"{label} cannot index argument {name!r} on vector axes "
                    f"{sorted(axis.value for axis in axes)}"
                )


def _transition_context(
    specs: Mapping[str, TensorArgSpec], axis: str
) -> dict[str, str]:
    context = {"q": "qv[d]", "k": "kv[d]", "v": "vv[dv]"}
    context.update(linear_argument_map(tuple(specs.values()), axis))
    return context


@dataclass(frozen=True)
class _DvSlice:
    """How generated code addresses the dv dimension of one lowering."""

    suffix: str
    span: str
    dv_loop: str


_DV_ALL = _DvSlice("", "DV", "for (int64_t dv = 0; dv < DV; ++dv)")
_DV_BLOCK = _DvSlice(
    " + dv_begin", "dv_count", "for (int64_t dv = dv_begin; dv < dv_end; ++dv)"
)


def _linear_transition_code(
    operator: Linear,
    specs: Mapping[str, TensorArgSpec],
    dv: _DvSlice,
) -> str:
    lines: list[str] = []
    for number, step in enumerate(operator.transition.steps):
        if isinstance(step, transition.Scale):
            factor = emit_expr(step.factor, _transition_context(specs, "d"))
            lines.extend(
                [
                    "for (int64_t d = 0; d < D; ++d) {",
                    f"    float factor_{number} = {factor};",
                    f"    cpuattn_scale_inplace(state + state_base + d * DV{dv.suffix},",
                    f"        {dv.span}, factor_{number});",
                    "}",
                ]
            )
        elif isinstance(step, transition.Rank1):
            left = emit_expr(step.left, _transition_context(specs, "d"))
            right = emit_expr(step.right, _transition_context(specs, "d"))
            lines.extend(
                [
                    f"cpuattn_zero(linear_tmp{dv.suffix}, {dv.span});",
                    "for (int64_t d = 0; d < D; ++d) {",
                    f"    float right_{number} = {right};",
                    f"    cpuattn_axpy_inplace(linear_tmp{dv.suffix},",
                    f"        state + state_base + d * DV{dv.suffix}, {dv.span},",
                    f"        right_{number});",
                    "}",
                    "for (int64_t d = 0; d < D; ++d) {",
                    f"    float left_{number} = {left};",
                    f"    cpuattn_axpy_inplace(state + state_base + d * DV{dv.suffix},",
                    f"        linear_tmp{dv.suffix}, {dv.span}, left_{number});",
                    "}",
                ]
            )
        else:
            left = emit_expr(step.left, _transition_context(specs, "d"))
            right = emit_expr(step.right, _transition_context(specs, "dv"))
            lines.extend(
                [
                    dv.dv_loop,
                    f"    linear_tmp[dv] = {right};",
                    "for (int64_t d = 0; d < D; ++d) {",
                    f"    float left_{number} = {left};",
                    f"    cpuattn_axpy_inplace(state + state_base + d * DV{dv.suffix},",
                    f"        linear_tmp{dv.suffix}, {dv.span}, left_{number});",
                    "}",
                ]
            )
    return "\n".join(lines)


def _readout_code(
    operator: Linear,
    specs: Mapping[str, TensorArgSpec],
    dv: _DvSlice,
) -> str:
    query = emit_expr(operator.readout.query, _transition_context(specs, "d"))
    return "\n".join(
        [
            f"cpuattn_zero(output + output_base{dv.suffix}, {dv.span});",
            "for (int64_t d = 0; d < D; ++d) {",
            f"    float query_value = {query};",
            f"    cpuattn_axpy_inplace(output + output_base{dv.suffix},",
            f"        state + state_base + d * DV{dv.suffix}, {dv.span}, query_value);",
            "}",
        ]
    )


__all__ = ["render_linear"]
