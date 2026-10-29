#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
LIMIT=20971520

fail() { printf 'check-editable-surface-size.sh: FAIL: %s\n' "$*" >&2; exit 1; }
command -v jq >/dev/null 2>&1 || fail "jq is required"
command -v gzip >/dev/null 2>&1 || fail "gzip is required"

read_editable_paths() {
  local value
  while IFS= read -r value; do
    [[ -n "${value}" ]] && EDITABLE_PATHS+=("${value}")
  done
}

tree="$(git -C "${ROOT}" write-tree)"
manifest_json="$(git -C "${ROOT}" show "${tree}:benchmark.json")" || fail "benchmark.json is not in the archived tree"

EDITABLE_PATHS=()
editable_paths="$(printf '%s\n' "${manifest_json}" | jq -er '
  .editablePaths |
  if type == "array" and length > 0 and all(.[]; type == "string" and length > 0)
  then .[]
  else error("editablePaths must be a non-empty array of non-empty strings")
  end
')" || fail "benchmark.json has invalid editablePaths"
read_editable_paths <<<"${editable_paths}"
(( ${#EDITABLE_PATHS[@]} > 0 )) || fail "benchmark.json has no editablePaths"

for path in "${EDITABLE_PATHS[@]}"; do
  [[ "${path}" != /* && "${path}" != :* && "${path}" != *\\* ]] || fail "invalid editable path: ${path}"
  case "/${path}/" in
    *"/../"*|*"/./"*) fail "invalid editable path: ${path}" ;;
  esac
done

size="$(git -C "${ROOT}" archive "${tree}" -- "${EDITABLE_PATHS[@]}" | gzip -9 | wc -c | tr -d ' ')"
printf 'editable surface archive: %s bytes (limit %s)\n' "${size}" "${LIMIT}"
(( size <= LIMIT )) || fail "archive is ${size} bytes, above ${LIMIT}"
printf 'check-editable-surface-size.sh: PASS\n'
