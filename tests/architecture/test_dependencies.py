import ast
from pathlib import Path

import cpuattn


ROOT = Path(__file__).parents[2]
SOURCE = ROOT / "src" / "cpuattn"
CORE = SOURCE / "core"
HARDWARE = SOURCE / "hardware"
SCHEDULE = SOURCE / "schedule"
NATIVE = SOURCE / "native"
EMIT = NATIVE / "emit"
TUNING = SOURCE / "tuning"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    result = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            result.add(node.module)
    return result


def _depends_on(path: Path, layers: set[str]) -> bool:
    return any(
        name == layer or name.startswith(f"{layer}.")
        for name in _imports(path)
        for layer in layers
    )


def test1() -> None:
    """Source files remain grouped by their single architectural responsibility."""
    assert {path.name for path in SOURCE.glob("*.py")} >= {
        "__init__.py",
        "errors.py",
        "log.py",
        "runtime.py",
    }
    assert {path.name for path in CORE.glob("*.py")} >= {
        "expr.py",
        "operator.py",
        "rownorm.py",
        "tensor.py",
        "transition.py",
        "validate.py",
    }
    assert {path.name for path in EMIT.glob("*.py")} >= {
        "__init__.py",
        "expression.py",
        "linear.py",
        "parallel.py",
    }
    assert {path.name for path in TUNING.glob("*.py")} >= {
        "__init__.py",
        "cache.py",
        "single_core.py",
        "tuner.py",
        "worker_count.py",
    }


def test2() -> None:
    """Dependencies point inward and Runtime does not bypass execution layers."""
    upper_layers = {"hardware", "schedule", "native", "tuning", "runtime"}
    for path in CORE.glob("*.py"):
        assert not _depends_on(path, upper_layers), path.name
    assert not _depends_on(
        HARDWARE / "host.py", {"core", "schedule", "native", "tuning", "runtime"}
    )
    for path in SCHEDULE.glob("*.py"):
        assert not _depends_on(path, {"native", "tuning", "runtime"}), path.name
    for path in (NATIVE / "compiler.py", *EMIT.glob("*.py")):
        assert not _depends_on(path, {"tuning", "runtime"}), path.name
    assert "reference" not in _imports(SOURCE / "runtime.py")
    assert "kernels" not in _imports(SOURCE / "runtime.py")


def test3() -> None:
    """The public package stays small and independent of the parent project."""
    forbidden = {"Backend", "Compiler", "Selection"}
    assert forbidden.isdisjoint(cpuattn.__all__)
    assert {
        "SingleCoreTuner",
        "Tuner",
        "TuningContext",
        "WorkerCountTuner",
    } <= set(cpuattn.__all__)
    assert Path(cpuattn.__file__).resolve().is_relative_to(SOURCE.resolve())
    parent_package = str(ROOT.parent / "cpuattn")
    for path in SOURCE.rglob("*"):
        if path.is_file() and path.suffix in {".py", ".j2", ".h", ".c"}:
            assert parent_package not in path.read_text(encoding="utf-8"), path


def test4() -> None:
    """Every supported ISA ships the native SIMD assets used by codegen."""
    expected = (
        SOURCE / "kernels/simd/x86/avx2.h",
        SOURCE / "kernels/simd/x86/avx512.h",
        SOURCE / "kernels/simd/arm64/neon.h",
        SOURCE / "kernels/simd/arm64/sve.h",
        SOURCE / "kernels/simd/isa.h",
        SOURCE / "kernels/simd/linear.h",
        SOURCE / "kernels/simd/parallel.h",
        SOURCE / "kernels/common/packing.h",
    )
    assert all(path.is_file() and path.stat().st_size > 0 for path in expected)


def test5() -> None:
    """Tuner policy stays independent from execution and persistence mechanisms."""
    tuner_imports = _imports(TUNING / "tuner.py")
    assert not any(
        name.endswith(suffix)
        for name in tuner_imports
        for suffix in (
            "native.backends.base",
            "native.compiler",
            "native.executor",
            "tuning.cache",
        )
    )
    worker_tree = ast.parse(
        (TUNING / "worker_count.py").read_text(encoding="utf-8")
    )
    worker_methods = {
        node.name
        for node in ast.walk(worker_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "select" not in worker_methods
    planner_imports = _imports(SCHEDULE / "planning.py")
    assert not any(name.endswith("tuning") for name in planner_imports)
