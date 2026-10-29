#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
mkdir -p "${WORK}/repo/tools" "${WORK}/repo/editable"
cp "${ROOT}/tools/check-editable-surface-size.sh" "${WORK}/repo/tools/"
printf 'small\n' > "${WORK}/repo/editable/source.txt"
printf 'staged-delete-marker\n' > "${WORK}/repo/editable/staged-delete.txt"
printf '%s\n' '{"editablePaths":["editable"]}' > "${WORK}/repo/benchmark.json"
chmod +x "${WORK}/repo/tools/check-editable-surface-size.sh"
git -C "${WORK}/repo" init -q
git -C "${WORK}/repo" config user.email test@example.com
git -C "${WORK}/repo" config user.name test
git -C "${WORK}/repo" add .
git -C "${WORK}/repo" commit -qm small
(cd "${WORK}/repo" && tools/check-editable-surface-size.sh) | grep -q 'PASS'

mkdir -p "${WORK}/bin"
cat > "${WORK}/bin/gzip" <<'MOCK'
#!/bin/sh
if grep -aq 'staged-delete-marker'; then
  dd if=/dev/zero bs=1048576 count=20 2>/dev/null
  printf x
else
  printf ok
fi
MOCK
chmod +x "${WORK}/bin/gzip"
git -C "${WORK}/repo" rm -q editable/staged-delete.txt
(cd "${WORK}/repo" && PATH="${WORK}/bin:${PATH}" tools/check-editable-surface-size.sh) | grep -q 'PASS'

cat > "${WORK}/bin/gzip" <<'MOCK'
#!/bin/sh
cat >/dev/null
dd if=/dev/zero bs=1048576 count=20 2>/dev/null
printf x
MOCK
chmod +x "${WORK}/bin/gzip"
if (cd "${WORK}/repo" && PATH="${WORK}/bin:${PATH}" tools/check-editable-surface-size.sh >/dev/null 2>&1); then
  printf 'test-editable-surface-size.sh: FAIL: oversized archive passed\n' >&2
  exit 1
fi
printf 'test-editable-surface-size.sh: PASS\n'
