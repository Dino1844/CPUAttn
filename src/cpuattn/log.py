from __future__ import annotations

import logging
from threading import Lock

from .native.backends.base import Backend
from .hardware.host import Host


_LOGGER = logging.getLogger("cpuattn")
_LOCK = Lock()
_ANNOUNCED: set[tuple[str, str]] = set()


def configure_logging(level: int = logging.INFO) -> None:
    if not _LOGGER.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[cpuattn] %(levelname)s %(message)s"))
        _LOGGER.addHandler(handler)
    _LOGGER.setLevel(level)
    _LOGGER.propagate = False


def startup(host: Host, backend: Backend) -> None:
    key = (host.fingerprint, backend.backend_id)
    with _LOCK:
        if key in _ANNOUNCED:
            return
        _ANNOUNCED.add(key)
    configure_logging()
    _LOGGER.info(
        "host arch=%s vendor=%s model=%s logical_cpus=%d physical_cores=%d "
        "cpuset=%s sve_vector_bytes=%s; "
        "backend=%s build=available correctness=unverified performance=unverified",
        host.architecture,
        host.vendor,
        host.model,
        len(host.cpus),
        host.physical_cores,
        list(host.allowed_cpu_ids),
        host.sve_vector_bytes,
        backend.backend_id,
    )


def event(message: str, *args: object) -> None:
    configure_logging()
    _LOGGER.info(message, *args)


__all__ = ["configure_logging", "event", "startup"]
