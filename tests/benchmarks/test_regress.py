from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.regress import (
    compare,
    environment_differences,
    load_report,
    main,
    regressions,
    unavailable,
)


def _write(tmp_path: Path, name: str, cases, environment=None) -> Path:
    path = tmp_path / name
    path.write_text(
        json.dumps({"cases": cases, "environment": environment or {}}), "utf-8"
    )
    return path


def test_load_compare_and_gate(tmp_path: Path) -> None:
    baseline_path = _write(
        tmp_path,
        "base.json",
        [
            {"name": "a", "native": {"median_ms": 1.0}, "wall": {"median_ms": 2.0}},
            {"name": "b", "native": {"median_ms": 4.0}},
            {"name": "c"},  # no median: present but not comparable
            {"name": "d", "native": {"median_ms": 0.0}},  # zero baseline
        ],
        {"cpu": "X", "architecture": "x86_64", "git_commit": "aaa"},
    )
    candidate_path = _write(
        tmp_path,
        "cand.json",
        [
            {"name": "a", "native": {"median_ms": 1.25}},  # exactly the tolerance
            {"name": "b", "native": {"median_ms": 3.0}},
            {"name": "d", "native": {"median_ms": 1.0}},
        ],
        {"cpu": "Y", "architecture": "x86_64", "git_commit": "bbb"},
    )
    baseline, base_env = load_report(baseline_path)
    candidate, cand_env = load_report(candidate_path)

    assert set(baseline) == {"a", "b", "c", "d"}
    deltas = compare(baseline, candidate, "native")
    ratios = {delta.name: delta.ratio for delta in deltas}
    assert ratios == {"a": pytest.approx(1.25), "b": pytest.approx(0.75)}
    assert regressions(deltas, 1.25) == ()  # ratio == tolerance is not a regression
    assert [delta.name for delta in regressions(deltas, 1.24)] == ["a"]
    assert unavailable(baseline, candidate, "native") == ("d",)
    assert environment_differences(base_env, cand_env) == ("cpu", "git_commit")


def test_main_fails_on_crashed_or_missing_case(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    baseline = _write(
        tmp_path,
        "b.json",
        [
            {"name": "a", "native": {"median_ms": 1.0}},
            {"name": "b", "native": {"median_ms": 2.0}},
        ],
    )
    candidate = _write(
        tmp_path,
        "c.json",
        [
            {"name": "a", "native": {"median_ms": 1.0}},
            {"name": "b", "error": "RuntimeError: boom"},
        ],
    )
    assert main(["--baseline", str(baseline), "--candidate", str(candidate)]) == 1
    assert "unavailable in candidate: b" in capsys.readouterr().out


def test_main_passes_on_identity(tmp_path: Path) -> None:
    cases = [{"name": "a", "native": {"median_ms": 1.0}}]
    baseline = _write(tmp_path, "b.json", cases)
    candidate = _write(tmp_path, "c.json", cases)
    assert main(["--baseline", str(baseline), "--candidate", str(candidate)]) == 0


def test_load_report_rejects_bad_input(tmp_path: Path) -> None:
    foreign = tmp_path / "foreign.json"
    foreign.write_text("[1, 2, 3]", "utf-8")
    with pytest.raises(ValueError):
        load_report(foreign)
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", "utf-8")
    with pytest.raises(ValueError):
        load_report(broken)


def test_non_finite_and_bool_medians_are_ignored(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "n.json",
        [
            {"name": "a", "native": {"median_ms": True}},
            {"name": "b", "native": {"median_ms": float("nan")}},
            {"name": "c", "native": {"median_ms": 5.0}},
        ],
    )
    report, _ = load_report(path)
    assert report["a"] == {}
    assert report["b"] == {}
    assert report["c"] == {"native": 5.0}
