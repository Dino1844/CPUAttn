from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class DType(str, Enum):
    FP32 = "fp32"


class Layout(str, Enum):
    BHSD = "bhsd"


class Axis(str, Enum):
    BATCH = "batch"
    QUERY_HEAD = "query_head"
    KV_HEAD = "kv_head"
    STATE_HEAD = "state_head"
    PARAMETER_GROUP = "parameter_group"
    QUERY = "query"
    KEY = "key"
    SEQUENCE = "sequence"
    D = "d"
    DV = "dv"


def _require_identifier_axes(kind: str, identifier: str, axes: tuple[Axis, ...]) -> None:
    if not identifier or not identifier.isidentifier():
        raise ValueError(f"invalid {kind} {identifier!r}")
    if len(set(axes)) != len(axes):
        raise ValueError(f"{kind} {identifier!r} repeats a logical axis")


@dataclass(frozen=True, slots=True)
class TensorSpec:
    role: str
    axes: tuple[Axis, ...]
    storage_dtype: DType = DType.FP32
    accumulation_dtype: DType = DType.FP32
    output_dtype: DType = DType.FP32
    layout: Layout | None = Layout.BHSD
    contiguous: bool = True

    def __post_init__(self) -> None:
        _require_identifier_axes("tensor role", self.role, self.axes)

    def canonical(self) -> dict[str, object]:
        return {
            "role": self.role,
            "axes": [axis.value for axis in self.axes],
            "storage_dtype": self.storage_dtype.value,
            "accumulation_dtype": self.accumulation_dtype.value,
            "output_dtype": self.output_dtype.value,
            "layout": self.layout.value if self.layout is not None else None,
            "contiguous": self.contiguous,
        }


@dataclass(frozen=True, slots=True)
class TensorArgSpec:
    name: str
    axes: tuple[Axis, ...] = ()
    dtype: DType = DType.FP32
    contiguous: bool = True

    def __post_init__(self) -> None:
        _require_identifier_axes("argument name", self.name, self.axes)

    def canonical(self) -> dict[str, object]:
        return {
            "name": self.name,
            "axes": [axis.value for axis in self.axes],
            "dtype": self.dtype.value,
            "contiguous": self.contiguous,
        }


def require_array(value: Any, *, name: str, rank: int) -> np.ndarray:
    """Return a zero-copy NumPy view and enforce the public tensor contract."""
    array = value
    if not isinstance(array, np.ndarray) and hasattr(array, "detach"):
        device = getattr(array, "device", None)
        if device is not None and getattr(device, "type", str(device)) != "cpu":
            raise ValueError(f"{name} must be a CPU tensor")
        array = array.detach().numpy()
    if not isinstance(array, np.ndarray):
        raise ValueError(f"{name} must be a NumPy array or contiguous CPU tensor")
    if array.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}; got shape {array.shape}")
    if array.dtype != np.float32:
        raise ValueError(f"{name} must use fp32 storage; got {array.dtype}")
    if not array.flags.c_contiguous:
        raise ValueError(f"{name} must be contiguous BHSD storage")
    if 0 in array.shape:
        raise ValueError(f"{name} dimensions must be positive; got {array.shape}")
    return array


__all__ = ["Axis", "DType", "Layout", "TensorArgSpec", "TensorSpec"]
