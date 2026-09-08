#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON:-python3}"
export PYTHONPATH="${project_root}/src:${project_root}${PYTHONPATH:+:${PYTHONPATH}}"

exec "${python_bin}" -m pytest -q \
    "${project_root}/tests/unit" \
    "${project_root}/tests/reference" \
    "${project_root}/tests/runtime/test_user_scenarios.py"
