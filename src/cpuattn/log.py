from __future__ import annotations

import logging
from threading import Lock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .native.backends.base import Backend
    from .hardware.host import Host


_LOGGER = logging.getLogger("cpuattn")
_LOCK = Lock()
_ANNOUNCED: set[tuple[str, str]] = set()


def _ensure_handler() -> None:
    """Install the default handler; INFO visibility unless a level was set."""
    if not _LOGGER.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[cpuattn] %(levelname)s %(message)s"))
        _LOGGER.addHandler(handler)
        if _LOGGER.level == logging.NOTSET:
            _LOGGER.setLevel(logging.INFO)
    _LOGGER.propagate = False


def configure_logging(level: int = logging.INFO) -> None:
    """Explicit user configuration; always takes effect."""
    _ensure_handler()
    _LOGGER.setLevel(level)


def startup(host: Host, backend: Backend) -> None:
    key = (host.fingerprint, backend.backend_id)
    with _LOCK:
        if key in _ANNOUNCED:
            return
        _ANNOUNCED.add(key)
    _ensure_handler()
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
    _ensure_handler()
    _LOGGER.info(message, *args)


__all__ = ["configure_logging", "event", "startup"]
