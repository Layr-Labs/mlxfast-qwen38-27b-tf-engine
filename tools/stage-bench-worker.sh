#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
WORKER="${MLXFAST_ENGINE_BIN:-${TF_BENCH_WORKER:-${ROOT}/Vendor/TensorFold/zig-out/bin/bench-worker}}"
[[ -x "${WORKER}" ]] || {
  printf 'stage-bench-worker.sh: worker is missing: %s\n' "${WORKER}" >&2
  exit 1
}
if [[ "${MLXFAST_SKIP_WORKER_SELF_TEST:-0}" != "1" ]]; then
  result="$(MLXFAST_ENGINE_BIN="${WORKER}" "${ROOT}/tools/sandboxed-cli.sh" self-test)"
  [[ "${result}" == 'sandboxed-cli.sh: PASS: self-test' ]] || {
    printf 'stage-bench-worker.sh: self-test failed: %s\n' "${result}" >&2
    exit 1
  }
fi
printf 'stage-bench-worker.sh: worker ready at %s\n' "${WORKER}"
