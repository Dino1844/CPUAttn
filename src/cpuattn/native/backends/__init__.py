from __future__ import annotations

from ...errors import UnsupportedError
from ...hardware.host import Host
from .arm64 import ARM_NEON, ARM_SVE
from .base import Backend
from .x86 import X86_AVX2, X86_AVX512


BACKENDS = (X86_AVX512, X86_AVX2, ARM_SVE, ARM_NEON)


def compatible_backends(host: Host) -> tuple[Backend, ...]:
    return tuple(backend for backend in BACKENDS if backend.supports(host))


def select_backend(host: Host) -> Backend:
    compatible = compatible_backends(host)
    if not compatible:
        raise UnsupportedError(
            f"no compiled backend supports {host.architecture} features={sorted(host.features)}"
        )
    return compatible[0]


__all__ = ["Backend", "compatible_backends", "select_backend"]
