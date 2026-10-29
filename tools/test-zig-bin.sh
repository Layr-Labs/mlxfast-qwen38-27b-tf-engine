#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
mkdir -p "${WORK}/empty"

cat > "${WORK}/zig-good" <<'MOCK'
#!/bin/bash
[[ "${1:-}" == version ]] || exit 2
printf '0.17.0\n'
MOCK
cat > "${WORK}/zig-wrong" <<'MOCK'
#!/bin/bash
[[ "${1:-}" == version ]] || exit 2
printf '0.16.0\n'
MOCK
chmod +x "${WORK}/zig-good" "${WORK}/zig-wrong"

resolved="$(ZIG="${WORK}/zig-good" "${ROOT}/tools/zig-bin.sh")"
[[ "${resolved}" == "${WORK}/zig-good" ]]

if output="$(ZIG="${WORK}/zig-wrong" "${ROOT}/tools/zig-bin.sh" 2>&1)"; then
  printf 'test-zig-bin.sh: wrong version was accepted\n' >&2
  exit 1
fi
[[ "${output}" == *"Zig 0.17.0 is required"* ]]

if output="$(env -u ZIG PATH="${WORK}/empty" /bin/bash "${ROOT}/tools/zig-bin.sh" 2>&1)"; then
  printf 'test-zig-bin.sh: missing Zig was accepted\n' >&2
  exit 1
fi
[[ "${output}" == *"set ZIG or add zig to PATH"* ]]

printf 'test-zig-bin.sh: PASS\n'
