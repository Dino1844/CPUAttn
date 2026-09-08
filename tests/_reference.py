from __future__ import annotations

from typing import Mapping

import numpy as np

from cpuattn import expr, transition
from cpuattn.core.operator import Linear, LinearResult, Parallel
from cpuattn.core.tensor import Axis, TensorArgSpec
from cpuattn.core.validate import LinearCall, ParallelCall


def parallel(operator: Parallel, call: ParallelCall) -> np.ndarray:
    b_size, query_heads, query_length, _ = call.q.shape
    kv_heads = call.k.shape[1]
    key_length = call.k.shape[2]
    value_dim = call.v.shape[3]
    output = np.zeros((b_size, query_heads, query_length, value_dim), dtype=np.float32)
    arguments = call.argument_map
    specs = {item.name: item for item in operator.arguments}
    for b in range(b_size):
        for head in range(query_heads):
            kv_head = head * kv_heads // query_heads
            for query_index in range(query_length):
                absolute_query_index = call.query_offset + query_index
                scores: list[np.float32] = []
                active_keys: list[int] = []
                for key_index in range(key_length):
                    raw = np.dot(call.q[b, head, query_index], call.k[b, kv_head, key_index])
                    indices = {
                        Axis.BATCH: b,
                        Axis.QUERY_HEAD: head,
                        Axis.KV_HEAD: kv_head,
                        Axis.QUERY: query_index,
                        Axis.KEY: key_index,
                    }
                    env = _argument_env(specs, arguments, indices)
                    keep = bool(
                        expr.evaluate(
                            operator.mask_mod,
                            {
                                **env,
                                "query_index": float(absolute_query_index),
                                "key_index": float(key_index),
                            },
                        )
                    )
                    if not keep:
                        continue
                    active_keys.append(key_index)
                    scores.append(
                        np.float32(expr.evaluate(operator.score_mod, {**env, "x": raw}))
                    )
                if active_keys:
                    weights = operator.row_norm.reference.weights(
                        np.asarray(scores, dtype=np.float32)
                    )
                    output[b, head, query_index] = weights @ call.v[b, kv_head, active_keys]
    return output


def linear(operator: Linear, call: LinearCall) -> LinearResult:
    b_size, groups, sequence, d_size = call.q.shape
    state_heads = call.v.shape[1]
    dv_size = call.v.shape[3]
    state = (
        np.zeros((b_size, state_heads, d_size, dv_size), dtype=np.float32)
        if call.state is None
        else call.state.copy()
    )
    output = np.zeros((b_size, state_heads, sequence, dv_size), dtype=np.float32)
    arguments = call.argument_map
    specs = {item.name: item for item in operator.arguments}
    for b in range(b_size):
        for head in range(state_heads):
            group = head * groups // state_heads
            current = state[b, head]
            for token in range(sequence):
                indices: dict[Axis, int | slice] = {
                    Axis.BATCH: b,
                    Axis.STATE_HEAD: head,
                    Axis.PARAMETER_GROUP: group,
                    Axis.SEQUENCE: token,
                    Axis.D: slice(None),
                    Axis.DV: slice(None),
                }
                argument_env = _argument_env(specs, arguments, indices)
                q_vector = _vector(
                    expr.evaluate(operator.q_mod, {**argument_env, "x": call.q[b, group, token]}),
                    d_size,
                    "q_mod",
                )
                k_vector = _vector(
                    expr.evaluate(operator.k_mod, {**argument_env, "x": call.k[b, group, token]}),
                    d_size,
                    "k_mod",
                )
                v_vector = _vector(
                    expr.evaluate(operator.v_mod, {**argument_env, "x": call.v[b, head, token]}),
                    dv_size,
                    "v_mod",
                )
                env = {**argument_env, "q": q_vector, "k": k_vector, "v": v_vector}
                if operator.readout.timing is transition.ReadTiming.BEFORE:
                    output[b, head, token] = _readout(operator, env, current, d_size)
                for step in operator.transition.steps:
                    if isinstance(step, transition.Scale):
                        factor = np.asarray(expr.evaluate(step.factor, env), dtype=np.float32)
                        if factor.shape == ():
                            current *= factor
                        elif factor.shape == (d_size,):
                            current *= factor[:, None]
                        else:
                            raise ValueError(
                                f"Scale factor must be scalar or ({d_size},); got {factor.shape}"
                            )
                    elif isinstance(step, transition.Rank1):
                        left = _vector(expr.evaluate(step.left, env), d_size, "Rank1 left")
                        right = _vector(expr.evaluate(step.right, env), d_size, "Rank1 right")
                        current += np.outer(left, right @ current)
                    else:
                        left = _vector(expr.evaluate(step.left, env), d_size, "Outer left")
                        right = _vector(expr.evaluate(step.right, env), dv_size, "Outer right")
                        current += np.outer(left, right)
                if operator.readout.timing is transition.ReadTiming.AFTER:
                    output[b, head, token] = _readout(operator, env, current, d_size)
    return LinearResult(output, state)


def _readout(
    operator: Linear,
    env: Mapping[str, object],
    state: np.ndarray,
    d_size: int,
) -> np.ndarray:
    query = _vector(expr.evaluate(operator.readout.query, env), d_size, "readout query")
    return query @ state


def _vector(value: object, size: int, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (size,):
        raise ValueError(f"{label} must produce shape ({size},); got {array.shape}")
    return array


def _argument_env(
    specs: Mapping[str, TensorArgSpec],
    values: Mapping[str, np.ndarray],
    indices: Mapping[Axis, int | slice],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, spec in specs.items():
        selectors = tuple(indices[axis] for axis in spec.axes)
        value = values[name][selectors] if selectors else values[name][()]
        result[name] = value
    return result


__all__ = ["linear", "parallel"]
