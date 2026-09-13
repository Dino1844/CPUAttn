#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON:-python3}"
export PYTHONPATH="${project_root}/src:${project_root}${PYTHONPATH:+:${PYTHONPATH}}"
export CPUATTN_CACHE_DIR="${CPUATTN_CACHE_DIR:-${project_root}/artifacts/cache}"
mkdir -p "${CPUATTN_CACHE_DIR}"

exec "${python_bin}" -m benchmarks.run --suite full
