#!/usr/bin/env bash
#
# QuixiCore CPU bench entrypoint. Thin wrapper around the shared core
# (run_bench_core.sh, synced from the umbrella); this file is hand-written
# and backend-owned.
#
#   perf/harness/run_bench.sh --preset quick --kernel colibri_ops --label int8-ab
#   perf/harness/run_bench.sh --dry-run
#
# Wraps scripts/bench (the quixicore_cpu_bench binary; build it first with
# `cmake --preset perf && cmake --build --preset perf`). CPU runs are quiet:
# the noise limit is tightened to cv 0.15 (observed runs sit at 0.04-0.06).

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
QC_BACKEND="cpu"
QC_CV_LIMIT="${QC_CV_LIMIT:-0.15}"

qc_bench_cmd() {
    local args=()
    [ -n "${QC_PRESET:-}" ] && args+=(--preset "$QC_PRESET")
    [ -n "${QC_KERNELS:-}" ] && args+=(--kernel "$QC_KERNELS")
    qc_exec "$REPO_ROOT/scripts/bench" "${args[@]+"${args[@]}"}" \
        --out-dir "$OUT_DIR" \
        ${QC_PASSTHROUGH[@]+"${QC_PASSTHROUGH[@]}"}
}

qc_device_info() {
    if [ "$(uname -s)" = "Darwin" ]; then
        echo "cpu=$(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
    else
        echo "cpu=$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2)"
    fi
    echo "uname=$(uname -srm)"
}

source "$(dirname "${BASH_SOURCE[0]}")/run_bench_core.sh"
