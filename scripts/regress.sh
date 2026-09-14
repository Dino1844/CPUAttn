#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON:-python3}"
export PYTHONPATH="${project_root}/src:${project_root}${PYTHONPATH:+:${PYTHONPATH}}"
export CPUATTN_CACHE_DIR="${CPUATTN_CACHE_DIR:-${project_root}/artifacts/cache}"
mkdir -p "${CPUATTN_CACHE_DIR}"

arch="$("${python_bin}" -c 'from cpuattn.hardware.host import detect_host; print(detect_host().architecture)')"
baseline="${CPUATTN_BASELINE:-${project_root}/artifacts/benchmarks/quiet-baseline-${arch}.json}"
candidate="${CPUATTN_REGRESS_OUT:-${project_root}/artifacts/benchmarks/regress-latest-${arch}.json}"
tolerance="${CPUATTN_REGRESS_TOLERANCE:-1.25}"

if [[ ! -f "${baseline}" ]]; then
    echo "missing baseline report: ${baseline}" >&2
    echo "record one on this machine first:" >&2
    echo "  PYTHONPATH=${project_root}/src:${project_root} ${python_bin} -m benchmarks.run --suite full --no-torch --output ${baseline}" >&2
    exit 2
fi

# Remove any stale report so a hard-crashed run cannot be gated against old data.
candidate_second="${candidate%.json}-2.json"
rm -f "${candidate}" "${candidate_second}"

benchmark_status=0
"${python_bin}" -m benchmarks.run --suite full --no-torch --output "${candidate}" || benchmark_status=$?
"${python_bin}" -m benchmarks.run --suite full --no-torch --output "${candidate_second}" || benchmark_status=$?

regress_status=0
"${python_bin}" -m benchmarks.regress \
    --baseline "${baseline}" \
    --candidate "${candidate}" --candidate "${candidate_second}" \
    --tolerance "${tolerance}" || regress_status=$?

if [[ "${benchmark_status}" -ne 0 || "${regress_status}" -ne 0 ]]; then
    echo "regression gate failed (benchmark=${benchmark_status}, gate=${regress_status})" >&2
    exit 1
fi
