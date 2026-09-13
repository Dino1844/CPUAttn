from __future__ import annotations

import ctypes
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from ..core.operator import Operator
from ..core.validate import ValidatedCall
from ..log import event
from ..schedule.plan import CodePlan, CompiledPlan
from .backends.base import Backend
from .emit import render_source


ABI_VERSION = "cpuattn-abi-2"


@dataclass(frozen=True, slots=True)
class CompilerPolicy:
    optimization: str = "-O3"

    def canonical(self) -> dict[str, str]:
        return {"optimization": self.optimization, "abi": ABI_VERSION}


class Compiler:
    """Render, compile, cache, and load native kernels."""

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        policy: CompilerPolicy = CompilerPolicy(),
    ) -> None:
        self.cache_dir = (
            Path(cache_dir) if cache_dir is not None else _default_cache_dir()
        )
        self.policy = policy
        self._package_dir = Path(__file__).parents[1]
        self._kernel_dir = self._package_dir / "kernels"
        self._emit_dir = Path(__file__).with_name("emit")
        self._templates = Environment(
            loader=FileSystemLoader(self._kernel_dir),
            undefined=StrictUndefined,
            autoescape=False,
            keep_trailing_newline=True,
        )
        self._identities: dict[Backend, dict[str, object]] = {}
        self._compiled: dict[str, CompiledPlan] = {}
        self._loaded: dict[str, NativeKernel] = {}
        self.stats = {"memory_hits": 0, "disk_hits": 0, "builds": 0}

    def identity(self, backend: Backend) -> dict[str, object]:
        cached = self._identities.get(backend)
        if cached is not None:
            return cached
        compiler = backend.compiler_path()
        sources = [
            Path(__file__),
            *sorted(self._emit_dir.rglob("*.py")),
            *sorted(self._kernel_dir.rglob("*.j2")),
            *sorted(self._kernel_dir.rglob("*.h")),
        ]
        identity = {
            "compiler": compiler,
            "compiler_version": _compiler_version(compiler),
            "flags": self._flags(backend),
            "policy": self.policy.canonical(),
            "source_digests": {
                str(path.relative_to(self._package_dir)): _file_digest(path)
                for path in sources
            },
        }
        self._identities[backend] = identity
        return identity

    def source(
        self, operator: Operator, call: ValidatedCall, code: CodePlan
    ) -> str:
        return render_source(self._templates, operator, call, code)

    def compile(
        self,
        operator: Operator,
        call: ValidatedCall,
        code: CodePlan,
        backend: Backend,
    ) -> CompiledPlan:
        if code.backend_id != backend.backend_id:
            raise ValueError("code plan/backend identity mismatch")
        compiler_identity = self.identity(backend)
        request_key = _json_digest({
            "operator": operator.canonical(),
            "workload": call.canonical(),
            "code": code.canonical(),
            "backend": backend.backend_id,
            "compiler_identity": compiler_identity,
        })
        cached = self._compiled.get(request_key)
        if cached is not None:
            self.stats["memory_hits"] += 1
            return cached
        source = self.source(operator, call, code)
        compiler = str(compiler_identity["compiler"])
        flags = tuple(compiler_identity["flags"])
        identity = {
            "operator": operator.canonical(),
            "code": code.canonical(),
            "backend": backend.backend_id,
            "compiler_identity": compiler_identity,
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "threading_sha256": _file_digest(
                self._kernel_dir / "common/threading.h"
            ),
        }
        key = _json_digest(identity)
        directory = self.cache_dir / "artifacts" / key
        source_path = directory / "kernel.c"
        library_path = directory / "kernel.so"
        manifest_path = directory / "manifest.json"
        directory.mkdir(parents=True, exist_ok=True)
        lock_path = directory / ".lock"
        with lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if not library_path.is_file() or not manifest_path.is_file():
                self.stats["builds"] += 1
                _atomic_text(source_path, source)
                temporary = directory / f"kernel.{os.getpid()}.so"
                command = [compiler, *flags, str(source_path), "-o", str(temporary)]
                started = time.perf_counter()
                result = subprocess.run(command, text=True, capture_output=True)
                build_seconds = time.perf_counter() - started
                if result.returncode:
                    temporary.unlink(missing_ok=True)
                    raise RuntimeError(
                        f"code plan {code.identity[:12]} compilation failed: "
                        f"{result.stderr[-2000:]}"
                    )
                os.replace(temporary, library_path)
                event("compiled plan=%s in %.2fs", code.identity[:12], build_seconds)
                _atomic_text(
                    manifest_path,
                    json.dumps(identity, sort_keys=True, indent=2),
                )
            else:
                self.stats["disk_hits"] += 1
        compiled = CompiledPlan(code, key, library_path, source_path, manifest_path)
        self._compiled[request_key] = compiled
        return compiled

    def cross_compile_object(
        self,
        operator: Operator,
        call: ValidatedCall,
        code: CodePlan,
        backend: Backend,
        destination: str | Path,
    ) -> Path:
        source = self.source(operator, call, code)
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        source_path = destination.with_suffix(".c")
        _atomic_text(source_path, source)
        flags = [
            self.policy.optimization,
            "-std=c11",
            "-fopenmp",
            "-fPIC",
            *backend.cflags,
            "-I",
            str(self._kernel_dir),
            "-c",
        ]
        result = subprocess.run(
            [
                backend.compiler_path(),
                *flags,
                str(source_path),
                "-o",
                str(destination),
            ],
            text=True,
            capture_output=True,
        )
        if result.returncode:
            raise RuntimeError(
                f"cross compilation failed: {result.stderr[-2000:]}"
            )
        return destination

    def load(self, compiled: CompiledPlan) -> "NativeKernel":
        kernel = self._loaded.get(compiled.artifact_key)
        if kernel is None:
            kernel = NativeKernel(compiled)
            self._loaded[compiled.artifact_key] = kernel
        return kernel

    def _flags(self, backend: Backend) -> tuple[str, ...]:
        return (
            self.policy.optimization,
            "-std=c11",
            "-shared",
            "-fPIC",
            "-fopenmp",
            *backend.cflags,
            "-I",
            str(self._kernel_dir),
            "-lm",
        )


class NativeKernel:
    def __init__(self, compiled: CompiledPlan) -> None:
        self.compiled = compiled
        self._library = ctypes.CDLL(str(compiled.library))
        self._execute_packed = self._library.cpuattn_execute_packed
        self._execute_packed.restype = ctypes.c_int
        self._execute_packed.argtypes = (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint64),
        )
        self._prepare_launch = self._library.cpuattn_prepare_launch
        self._prepare_launch.argtypes = (ctypes.c_void_p, ctypes.c_int)
        self._prepare_launch.restype = ctypes.c_int


def _compiler_version(compiler: str) -> str:
    result = subprocess.run(
        [compiler, "--version"], text=True, capture_output=True
    )
    return (result.stdout or result.stderr).splitlines()[0]


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_digest(value: object) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode()).hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _default_cache_dir() -> Path:
    explicit = os.environ.get("CPUATTN_CACHE_DIR")
    if explicit:
        return Path(explicit).expanduser()
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return root / "cpuattn"


__all__ = ["ABI_VERSION", "Compiler", "CompilerPolicy", "NativeKernel"]
