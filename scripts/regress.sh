#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'EOF'
usage: regress.sh [options]
  --baseline PATH    baseline report
                     (default artifacts/benchmarks/quiet-baseline-<arch>.json)
  --output PATH      candidate report
                     (default artifacts/benchmarks/regress-latest-<arch>.json)
  --tolerance X      regression threshold, candidate/baseline (default 1.25)
  --runs N           candidate runs, merged per case by minimum (default 2)
  --stat STAT        min or median (default min)
  --metric METRIC    native or wall (default native)
  --cache-dir PATH   CPUATTN_CACHE_DIR for the run (default artifacts/cache)
  --python BIN       python interpreter (default python3)
  -h, --help         show this help
EOF
}

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="python3"
baseline=""
output=""
tolerance="1.25"
runs="2"
stat="min"
metric="native"
cache_dir=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --baseline) baseline="$2"; shift 2 ;;
        --output) output="$2"; shift 2 ;;
        --tolerance) tolerance="$2"; shift 2 ;;
        --runs) runs="$2"; shift 2 ;;
        --stat) stat="$2"; shift 2 ;;
        --metric) metric="$2"; shift 2 ;;
        --cache-dir) cache_dir="$2"; shift 2 ;;
        --python) python_bin="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) usage; exit 2 ;;
    esac
done

export PYTHONPATH="${project_root}/src:${project_root}${PYTHONPATH:+:${PYTHONPATH}}"
export CPUATTN_CACHE_DIR="${cache_dir:-${project_root}/artifacts/cache}"
mkdir -p "${CPUATTN_CACHE_DIR}"

arch="$("${python_bin}" -c 'from cpuattn.hardware.host import detect_host; print(detect_host().architecture)')"
baseline="${baseline:-${project_root}/artifacts/benchmarks/quiet-baseline-${arch}.json}"
candidate="${output:-${project_root}/artifacts/benchmarks/regress-latest-${arch}.json}"

if [[ ! -f "${baseline}" ]]; then
    echo "missing baseline report: ${baseline}" >&2
    echo "record one on this machine first:" >&2
    echo "  PYTHONPATH=${project_root}/src:${project_root} ${python_bin} -m benchmarks.run --suite full --no-torch --output ${baseline}" >&2
    exit 2
fi

candidate_args=()
benchmark_status=0
for ((i = 0; i < runs; i++)); do
    path="${candidate}"
    if [[ "${i}" -gt 0 ]]; then
        path="${candidate%.json}-${i}.json"
    fi
    rm -f "${path}"
    candidate_args+=(--candidate "${path}")
    "${python_bin}" -m benchmarks.run --suite full --no-torch --output "${path}" \
        || benchmark_status=$?
done

regress_status=0
"${python_bin}" -m benchmarks.regress \
    --baseline "${baseline}" "${candidate_args[@]}" \
    --metric "${metric}" --stat "${stat}" --tolerance "${tolerance}" \
    || regress_status=$?

if [[ "${benchmark_status}" -ne 0 || "${regress_status}" -ne 0 ]]; then
    echo "regression gate failed (benchmark=${benchmark_status}, gate=${regress_status})" >&2
    exit 1
fi
