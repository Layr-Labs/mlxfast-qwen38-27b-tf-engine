#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/test-spec.XXXXXX")"
trap 'rm -rf "${TEMP_DIR}"' EXIT

printf '%s\n' '{"dflash_head":{"permitted_draft_depths":[1]}}' > "${TEMP_DIR}/contract.json"
printf '%s\n' '{"spec":{"enabled":false,"tokens":0}}' > "${TEMP_DIR}/manifest.json"
env SPEC_DECLARATION_MANIFEST="${TEMP_DIR}/manifest.json" SPEC_DECLARATION_CONTRACT="${TEMP_DIR}/contract.json" \
  "${ROOT}/tools/spec-declaration.sh" json | grep -Fx '{"mode":"serial"}' >/dev/null

printf '%s\n' '{"spec":{"enabled":true,"tokens":1}}' > "${TEMP_DIR}/manifest.json"
env SPEC_DECLARATION_MANIFEST="${TEMP_DIR}/manifest.json" SPEC_DECLARATION_CONTRACT="${TEMP_DIR}/contract.json" \
  "${ROOT}/tools/spec-declaration.sh" json | grep -Fx '{"mode":"dflash","dflash":{"depth":1}}' >/dev/null

printf '%s\n' '{"spec":{"enabled":true,"tokens":2}}' > "${TEMP_DIR}/manifest.json"
if env SPEC_DECLARATION_MANIFEST="${TEMP_DIR}/manifest.json" SPEC_DECLARATION_CONTRACT="${TEMP_DIR}/contract.json" \
  "${ROOT}/tools/spec-declaration.sh" validate >/dev/null 2>&1; then
  printf 'test-spec-declaration.sh: depth 2 was accepted\n' >&2
  exit 1
fi

for wrong in \
  '{"spec":{"enabled":"true","tokens":1}}' \
  '{"spec":{"enabled":true,"tokens":"1"}}'; do
  printf '%s\n' "${wrong}" > "${TEMP_DIR}/manifest.json"
  if env SPEC_DECLARATION_MANIFEST="${TEMP_DIR}/manifest.json" SPEC_DECLARATION_CONTRACT="${TEMP_DIR}/contract.json" \
    "${ROOT}/tools/spec-declaration.sh" validate >/dev/null 2>&1; then
    printf 'test-spec-declaration.sh: wrong JSON type was accepted: %s\n' "${wrong}" >&2
    exit 1
  fi
done
printf 'test-spec-declaration.sh: PASS\n'
