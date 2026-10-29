#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
if [[ "$(uname -s)" == Darwin ]] && ! /usr/bin/sandbox-exec -p '(version 1) (allow default)' /usr/bin/true >/dev/null 2>&1; then
  printf 'test-bench-worker-staging.sh: SKIP -- sandbox-exec is blocked by the outer sandbox\n'
  exit 0
fi
TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/test-stage.XXXXXX")"
# Match the resolved path used by the profile's exact process-exec rule.
TEMP_DIR="$(cd -- "${TEMP_DIR}" && pwd -P)"
trap 'rm -rf "${TEMP_DIR}"' EXIT
MOCK="${TEMP_DIR}/bench-worker"
# The profile allows exec of the literal worker path only, so a #! script (which also execs /bin/sh) is denied.
# Build the mock as a native binary, as the real worker is.
printf '%s\n' '#include <stdio.h>' 'int main(void){puts("engine-protocol-v1 self-test: ok");return 0;}' > "${TEMP_DIR}/mock.c"
/usr/bin/cc -o "${MOCK}" "${TEMP_DIR}/mock.c"
TF_BENCH_WORKER="${MOCK}" "${ROOT}/tools/stage-bench-worker.sh" | grep -F 'worker ready' >/dev/null
printf 'test-bench-worker-staging.sh: PASS\n'
