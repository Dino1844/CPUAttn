from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from enum import Enum
from types import MappingProxyType
from typing import ClassVar, final

from ..hardware.host import Host
from ..schedule.plan import ExecutionPlan, TimedPlan


Measure = Callable[[ExecutionPlan], int]

_CONFIRM_FINALISTS = 2
_CONFIRM_ROUNDS = 2


@dataclass(frozen=True, slots=True)
class TuningContext:
    host: Host
    operator: Mapping[str, object]
    workload: Mapping[str, object]
    candidates: tuple[ExecutionPlan, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "operator", _freeze(self.operator))
        object.__setattr__(self, "workload", _freeze(self.workload))
        object.__setattr__(self, "candidates", tuple(self.candidates))

    @property
    def pattern(self) -> str:
        pattern = self.operator.get("pattern")
        if not isinstance(pattern, str):
            raise ValueError("operator description must contain a pattern")
        return pattern


@dataclass(frozen=True, slots=True)
class Measurement:
    plan: ExecutionPlan
    samples_ns: tuple[int, ...]
    latency_ns: int

    def __post_init__(self) -> None:
        if not self.samples_ns or any(value < 0 for value in self.samples_ns):
            raise ValueError("measurement samples must be non-negative")
        if self.latency_ns < 0:
            raise ValueError("measurement latency must be non-negative")


@dataclass(frozen=True, slots=True)
class TuningResult:
    winner: ExecutionPlan
    measurements: tuple[Measurement, ...]


@dataclass(frozen=True, slots=True)
class PlanRecord:
    plan_id: str
    status: str
    latency_ns: int | None = None
    samples_ns: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class Selection:
    key: str
    winner: TimedPlan
    mode: str
    records: tuple[PlanRecord, ...]


@dataclass(frozen=True, slots=True)
class Tuner(ABC):
    """Own the tuning lifecycle while subclasses choose the next legal plan."""

    maxnum: int | None = None
    repeat: int = 3

    arch: ClassVar[str | None] = None
    pattern: ClassVar[str | None] = None
    version: ClassVar[int] = 1

    def __post_init__(self) -> None:
        if self.maxnum is not None and self.maxnum <= 0:
            raise ValueError("maxnum must be positive or None")
        if self.repeat <= 0:
            raise ValueError("repeat must be positive")
        if self.arch not in {None, "x86_64", "aarch64"}:
            raise ValueError(f"unsupported tuner architecture {self.arch!r}")
        if self.pattern not in {None, "parallel", "linear"}:
            raise ValueError(f"unsupported tuner pattern {self.pattern!r}")
        if self.version <= 0:
            raise ValueError("tuner version must be positive")

    @final
    def cache_identity(self) -> dict[str, object]:
        config = {
            field.name: _json_value(getattr(self, field.name))
            for field in fields(self)
        }
        return {
            "class": f"{type(self).__module__}.{type(self).__qualname__}",
            "arch": self.arch,
            "pattern": self.pattern,
            "version": self.version,
            "config": config,
        }

    @final
    def select(
        self,
        context: TuningContext,
        measure: Measure,
    ) -> TuningResult:
        self._validate_context(context)
        legal_by_id = {plan.identity: plan for plan in context.candidates}
        samples_by_id: dict[str, list[int]] = {}
        measured_order: list[str] = []

        while True:
            history = _history(measured_order, legal_by_id, samples_by_id)
            selected = self.choose_next(context, history)
            if selected is None:
                break
            legal = legal_by_id.get(selected.identity)
            if legal is None or legal != selected:
                raise ValueError("tuner selected a plan outside the legal candidates")
            if selected.identity not in samples_by_id:
                if self.maxnum is not None and len(samples_by_id) >= self.maxnum:
                    break
                samples_by_id[selected.identity] = []
                measured_order.append(selected.identity)
                # The first execution of a plan pays cold icache, first-touch
                # faults, and frequency ramp; run it once outside the samples.
                measure(legal)
            for _ in range(self.repeat):
                _sample(legal, measure, samples_by_id)

        self._confirm_finalists(legal_by_id, samples_by_id, measure)

        measurements = _history(measured_order, legal_by_id, samples_by_id)
        if not measurements:
            raise RuntimeError("tuner stopped before measuring a plan")
        winner = min(
            measurements,
            key=lambda item: (item.latency_ns, item.plan.identity),
        ).plan
        return TuningResult(winner, measurements)

    def _confirm_finalists(
        self,
        legal_by_id: dict[str, ExecutionPlan],
        samples_by_id: dict[str, list[int]],
        measure: Measure,
    ) -> None:
        """Re-measure the top plans interleaved so a transient burst cannot pick the winner."""
        if len(samples_by_id) < _CONFIRM_FINALISTS:
            return
        ranked = sorted(
            samples_by_id,
            key=lambda identity: (
                _median_sample(samples_by_id[identity]),
                identity,
            ),
        )[:_CONFIRM_FINALISTS]
        for _ in range(_CONFIRM_ROUNDS):
            for identity in ranked:
                _sample(legal_by_id[identity], measure, samples_by_id)

    def _validate_context(self, context: TuningContext) -> None:
        if not context.candidates:
            raise RuntimeError("planner produced no legal candidates")
        identities = [plan.identity for plan in context.candidates]
        if len(identities) != len(set(identities)):
            raise ValueError("tuning candidates must have unique identities")
        if self.arch is not None and context.host.architecture != self.arch:
            raise ValueError(
                f"tuner architecture {self.arch!r} does not support "
                f"{context.host.architecture!r}"
            )
        if self.pattern is not None and context.pattern != self.pattern:
            raise ValueError(
                f"tuner pattern {self.pattern!r} does not support "
                f"{context.pattern!r}"
            )
        if context.workload.get("pattern") != context.pattern:
            raise ValueError("operator and workload patterns do not match")

    @abstractmethod
    def choose_next(
        self,
        context: TuningContext,
        history: tuple[Measurement, ...],
    ) -> ExecutionPlan | None:
        """Return one supplied candidate, or None to finish tuning."""


def _history(
    order: list[str],
    legal_by_id: dict[str, ExecutionPlan],
    samples_by_id: dict[str, list[int]],
) -> tuple[Measurement, ...]:
    return tuple(
        Measurement(
            legal_by_id[identity],
            tuple(samples_by_id[identity]),
            _median_sample(samples_by_id[identity]),
        )
        for identity in order
    )


def _median_sample(samples: list[int]) -> int:
    """Median (lower middle for even counts); robust to one bad sample."""
    ordered = sorted(samples)
    return ordered[(len(ordered) - 1) // 2]


def _sample(
    plan: ExecutionPlan,
    measure: Measure,
    samples_by_id: dict[str, list[int]],
) -> None:
    latency = measure(plan)
    if not isinstance(latency, int) or latency < 0:
        raise ValueError("measure must return a non-negative integer")
    samples_by_id[plan.identity].append(latency)


def _shape(workload: Mapping[str, object]) -> dict[str, int]:
    value = workload.get("shape")
    if not isinstance(value, Mapping):
        raise ValueError("workload description must contain a shape")
    if not all(
        isinstance(key, str) and isinstance(item, int)
        for key, item in value.items()
    ):
        raise ValueError("workload shape must map axis names to integers")
    return dict(value)


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _json_value(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    raise TypeError(
        f"tuner configuration value {type(value).__name__} is not JSON-compatible"
    )


__all__ = [
    "Measurement",
    "PlanRecord",
    "Selection",
    "Tuner",
    "TuningContext",
    "TuningResult",
]
