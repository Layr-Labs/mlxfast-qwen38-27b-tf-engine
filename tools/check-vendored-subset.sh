#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
REFERENCE="${1:-}"

fail() { printf 'check-vendored-subset.sh: FAIL: %s\n' "$*" >&2; exit 1; }
[[ -n "${REFERENCE}" && $# -eq 1 ]] || fail "usage: $0 <reference-commit>"
[[ "${REFERENCE}" =~ ^[0-9a-fA-F]{40}$ ]] || fail "reference commit must be a full 40-hex SHA"
git -C "${ROOT}" cat-file -e "${REFERENCE}^{commit}" 2>/dev/null || fail "reference commit does not exist: ${REFERENCE}"

changed="$(git -C "${ROOT}" diff --diff-filter=ACMRT --name-only "${REFERENCE}" HEAD -- Vendor/TensorFold)"
[[ -z "${changed}" ]] || fail "vendored subset has added or changed paths: ${changed}"
printf 'vendored subset: deletions only against %s\n' "${REFERENCE}"

for path in Vendor/TensorFold/zig Vendor/TensorFold/build.zig Vendor/TensorFold/build.zig.zon mtp-head.manifest.json; do
  git -C "${ROOT}" diff --quiet "${REFERENCE}" HEAD -- "${path}" \
    || fail "path differs from reference: ${path}"
  printf 'vendored subset: byte-equal %s\n' "${path}"
done

printf 'check-vendored-subset.sh: PASS\n'
