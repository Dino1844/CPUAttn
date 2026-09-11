from __future__ import annotations

import pytest

from benchmarks.harness import collect, collect_with_probe, summarize, timed_ns
from benchmarks.results import BaselineResult, CaseResult, Environment, ThreadLevel
from benchmarks.report import case_to_json, print_console, report_to_json
from benchmarks.workloads import (
    AttentionWorkload,
    LinearWorkload,
    ThreadScalingWorkload,
    full_matrix,
    select,
    smoke_matrix,
)


def test_timed_ns_measures_a_positive_duration() -> None:
    samples = [timed_ns(lambda: sum(range(1000))) for _ in range(3)]
    assert all(sample > 0 for sample in samples)


def test_collect_runs_warmup_then_requested_samples() -> None:
    counter = {"calls": 0}

    def function() -> int:
        counter["calls"] += 1
        return counter["calls"]

    samples = collect(function, warmup=2, runs=5)
    assert counter["calls"] == 7
    assert len(samples) == 5


def test_summarize_matches_hand_computed_statistics() -> None:
    stats = summarize([10, 20, 30, 40])
    assert stats.samples == 4
    assert stats.median == 25.0
    assert stats.minimum == 10.0
    assert stats.mean == 25.0
    assert stats.p25 == 17.5
    assert stats.p75 == 32.5


def test_summarize_odd_count_median_is_middle_value() -> None:
    stats = summarize([5, 1, 3])
    assert stats.median == 3.0


def test_summarize_rejects_empty_samples() -> None:
    with pytest.raises(ValueError):
        summarize([])


def test_full_matrix_contains_prefill_decode_linear_threads() -> None:
    kinds = {type(item) for item in full_matrix()}
    assert kinds == {AttentionWorkload, LinearWorkload, ThreadScalingWorkload}


def test_smoke_matrix_is_a_subset_of_the_full_matrix() -> None:
    full_names = {item.name for item in full_matrix()}
    assert {item.name for item in smoke_matrix()} <= full_names


def test_select_filters_by_regex() -> None:
    picked = select("full", r"^decode")
    assert picked
    assert all(item.name.startswith("decode") for item in picked)


def test_select_smoke_suite_returns_the_small_set() -> None:
    assert select("smoke", None) == smoke_matrix()


def _environment() -> Environment:
    return Environment(
        timestamp_utc="2026-09-11T00:00:00+00:00",
        platform="test-platform",
        cpu="Test CPU",
        architecture="x86_64",
        physical_cores=8,
        numpy_version="1.0",
        torch_version=None,
        git_commit="abcdef0",
        warmup=1,
        runs=3,
        stream_warmup=1,
        suite="smoke",
    )


def _case_result() -> CaseResult:
    return CaseResult(
        kind="attention_prefill",
        name="case_a",
        params={"sq": 8, "skv": 8},
        plan="parallel_blocked/none workers=1",
        first_call_ms=1.5,
        wall=summarize([100, 200, 300]),
        native=summarize([50, 60, 70]),
        baselines=(
            BaselineResult(
                backend="numpy_blas",
                wall=summarize([400, 500, 600]),
                max_abs_err=1e-6,
                speedup=2.5,
            ),
        ),
        max_abs_err=1e-6,
        numeric_ok=True,
    )


def test_case_to_json_round_trips_every_field() -> None:
    payload = case_to_json(_case_result())
    assert payload["kind"] == "attention_prefill"
    assert payload["name"] == "case_a"
    assert payload["plan"].startswith("parallel_blocked")
    assert payload["first_call_ms"] == 1.5
    assert payload["wall"]["median_ms"] == pytest.approx(0.0002)
    assert payload["baselines"][0]["speedup"] == 2.5
    assert payload["max_abs_err"] == 1e-6
    assert payload["numeric_ok"] is True


def test_error_case_serializes_only_the_error() -> None:
    result = CaseResult(kind="linear", name="broken", params={}, error="ValueError: no")
    payload = case_to_json(result)
    assert payload["error"] == "ValueError: no"
    assert "wall" not in payload


def test_report_to_json_embeds_environment_and_cases() -> None:
    payload = report_to_json(_environment(), [_case_result()])
    assert payload["environment"]["git_commit"] == "abcdef0"
    assert payload["environment"]["suite"] == "smoke"
    assert len(payload["cases"]) == 1


def test_print_console_lists_errors_and_thread_levels(capsys: pytest.CaptureFixture[str]) -> None:
    thread_case = CaseResult(
        kind="thread_scaling",
        name="threads_x",
        params={},
        thread_levels=(
            ThreadLevel(workers=1, native_median_ms=0.5, wall_median_ms=0.6, plan="a/b"),
        ),
    )
    print_console([thread_case, _case_result()], _environment())
    output = capsys.readouterr().out
    assert "threads_x" in output
    assert "workers=  1" in output
    assert "a/b" in output
    assert "case_a" in output


def test_collect_with_probe_pairs_each_wall_with_its_own_probe_value() -> None:
    counter = {"runs": 0}

    def function() -> None:
        counter["runs"] += 1

    walls, probes = collect_with_probe(
        function, lambda result: counter["runs"], warmup=2, runs=3
    )
    assert len(walls) == 3 and len(probes) == 3
    assert probes == [3, 4, 5]
    assert all(wall > 0 for wall in walls)
