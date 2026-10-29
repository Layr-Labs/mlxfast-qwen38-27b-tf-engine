#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
MANIFEST="${SPEC_DECLARATION_MANIFEST:-${ROOT}/mtp-head.manifest.json}"
CONTRACT="${SPEC_DECLARATION_CONTRACT:-${ROOT}/fixtures/qwen3_8_27b_tf_mlx_v1_track.json}"

fail() { printf 'spec-declaration.sh: REFUSING -- %s\n' "$*" >&2; exit 1; }
command -v jq >/dev/null 2>&1 || fail "jq is required"

enabled=false
tokens=0
if [[ -f "${MANIFEST}" ]]; then
  jq -e . "${MANIFEST}" >/dev/null 2>&1 || fail "declaration is not valid JSON"
  [[ "$(jq -r '.spec // {} | type' "${MANIFEST}")" == object ]] || fail "spec must be an object"
  unknown="$(jq -r '.spec // {} | keys[] | select(. != "enabled" and . != "tokens")' "${MANIFEST}")"
  [[ -z "${unknown}" ]] || fail "unknown spec key: ${unknown}"
  [[ "$(jq -r 'if .spec | has("enabled") then .spec.enabled | type else "boolean" end' "${MANIFEST}")" == boolean ]] \
    || fail "spec.enabled must be a boolean"
  [[ "$(jq -r 'if .spec | has("tokens") then .spec.tokens | type else "number" end' "${MANIFEST}")" == number ]] \
    || fail "spec.tokens must be an integer"
  enabled="$(jq -r '.spec.enabled // false' "${MANIFEST}")"
  tokens="$(jq -r '.spec.tokens // 0' "${MANIFEST}")"
fi
[[ "${enabled}" == true || "${enabled}" == false ]] || fail "spec.enabled must be a boolean"
[[ "${tokens}" =~ ^[0-9]+$ ]] || fail "spec.tokens must be an integer"

depth=0
if [[ "${enabled}" == true ]]; then
  [[ "${tokens}" -gt 0 ]] || fail "enabled DFlash needs a positive depth"
  jq -e --argjson depth "${tokens}" '.dflash_head.permitted_draft_depths | index($depth) != null' "${CONTRACT}" >/dev/null \
    || fail "DFlash depth ${tokens} is not permitted; this track permits [1]"
  depth="${tokens}"
fi

case "${1:-}" in
  json)
    if [[ "${depth}" == 0 ]]; then printf '%s\n' '{"mode":"serial"}';
    else printf '{"mode":"dflash","dflash":{"depth":%s}}\n' "${depth}"; fi ;;
  speculative) [[ "${depth}" == 0 ]] && echo 0 || echo 1 ;;
  draft-len) echo "${depth}" ;;
  describe) [[ "${depth}" == 0 ]] && echo serial || echo "dflash${depth}" ;;
  validate) : ;;
  *) printf '%s\n' 'usage: tools/spec-declaration.sh {json|speculative|draft-len|describe|validate}' >&2; exit 2 ;;
esac
