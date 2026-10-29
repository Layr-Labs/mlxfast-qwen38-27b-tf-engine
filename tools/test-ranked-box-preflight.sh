#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
output="$(R2_ACCESS_KEY_ID=synthetic "${ROOT}/tools/ranked-box-preflight.sh" 2>&1 || true)"
grep -F 'R2_ACCESS_KEY_ID is set' <<<"${output}" >/dev/null
if grep -F 'macmon missing' <<<"${output}" >/dev/null; then
  echo 'test-ranked-box-preflight.sh: credential check did not run before thermal checks' >&2
  exit 1
fi
printf 'test-ranked-box-preflight.sh: PASS -- credential gate runs before thermal checks\n'
