#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
mkdir -p "${WORK}/repo/tools" "${WORK}/repo/Vendor/TensorFold/zig"
cp "${ROOT}/tools/check-vendored-subset.sh" "${WORK}/repo/tools/"
printf 'source\n' > "${WORK}/repo/Vendor/TensorFold/zig/source.zig"
printf 'build\n' > "${WORK}/repo/Vendor/TensorFold/build.zig"
printf 'zon\n' > "${WORK}/repo/Vendor/TensorFold/build.zig.zon"
printf 'remove\n' > "${WORK}/repo/Vendor/TensorFold/unused.txt"
printf 'manifest\n' > "${WORK}/repo/mtp-head.manifest.json"
chmod +x "${WORK}/repo/tools/check-vendored-subset.sh"
git -C "${WORK}/repo" init -q
git -C "${WORK}/repo" config user.email test@example.com
git -C "${WORK}/repo" config user.name test
git -C "${WORK}/repo" add .
git -C "${WORK}/repo" commit -qm reference
reference="$(git -C "${WORK}/repo" rev-parse HEAD)"
git -C "${WORK}/repo" rm -q Vendor/TensorFold/unused.txt
git -C "${WORK}/repo" commit -qm subset
(cd "${WORK}/repo" && tools/check-vendored-subset.sh "${reference}") | grep -q 'PASS'

git -C "${WORK}/repo" rm -q Vendor/TensorFold/zig/source.zig
git -C "${WORK}/repo" commit -qm missing-core
if (cd "${WORK}/repo" && tools/check-vendored-subset.sh "${reference}" >/dev/null 2>&1); then
  printf 'test-vendored-subset.sh: FAIL: deleted core source passed\n' >&2
  exit 1
fi

mkdir -p "${WORK}/repo/Vendor/TensorFold/zig"
git -C "${WORK}/repo" show "${reference}:Vendor/TensorFold/zig/source.zig" > "${WORK}/repo/Vendor/TensorFold/zig/source.zig"
printf 'added\n' > "${WORK}/repo/Vendor/TensorFold/new.txt"
git -C "${WORK}/repo" add Vendor/TensorFold/zig/source.zig Vendor/TensorFold/new.txt
git -C "${WORK}/repo" commit -qm added
if (cd "${WORK}/repo" && tools/check-vendored-subset.sh "${reference}" >/dev/null 2>&1); then
  printf 'test-vendored-subset.sh: FAIL: added vendor source passed\n' >&2
  exit 1
fi
printf 'test-vendored-subset.sh: PASS\n'
