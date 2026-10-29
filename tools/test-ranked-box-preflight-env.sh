#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
TARGET="${ROOT}/tools/ranked-box-preflight.sh"

for name in MLXFAST_QWEN38_GOLDEN_DIR MLXFAST_BASELINE_WORKSPACE MLXFAST_BASELINE_CALIBRATION MLXFAST_REFERENCE_DIR; do
  grep -F "${name}" "${TARGET}" >/dev/null
done
if grep -Eq '\bTF_(TARGET|GOLDEN|BASELINE|PRIVATE)' "${TARGET}"; then
  echo 'test-ranked-box-preflight-env.sh: stale TF_ control-plane variable found' >&2
  exit 1
fi

output="$(BENCHD=synthetic "${TARGET}" 2>&1 || true)"
grep -F 'BENCHD is set' <<<"${output}" >/dev/null
printf 'test-ranked-box-preflight-env.sh: PASS -- pinned MLXFAST names and override gate verified\n'
