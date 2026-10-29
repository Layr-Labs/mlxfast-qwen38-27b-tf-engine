#!/usr/bin/env bash
# Stage the fixture-pinned DFlash snapshot for an official candidate.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
FIXTURE="${TF_CONTRACT:-${ROOT}/fixtures/qwen3_8_27b_tf_mlx_v1_track.json}"
DECLARATION="${SPEC_DECLARATION_MANIFEST:-${ROOT}/mtp-head.manifest.json}"
DESTINATION="${DRAFTER_DEST:-${ROOT}/drafter_weights}"
fail() { printf 'stage-drafter.sh: FAIL: %s\n' "$1" >&2; exit 1; }

command -v jq >/dev/null 2>&1 || fail tool-missing
[[ -f "${FIXTURE}" ]] || fail manifest-missing
mode="$(SPEC_DECLARATION_MANIFEST="${DECLARATION}" SPEC_DECLARATION_CONTRACT="${FIXTURE}" \
  "${ROOT}/tools/spec-declaration.sh" describe 2>/dev/null)" || fail declaration
if [[ "${mode}" == "serial" ]]; then
  printf 'stage-drafter.sh: PASS: serial\n'
  exit 0
fi
[[ "${mode}" == "dflash1" ]] || fail declaration
command -v shasum >/dev/null 2>&1 || fail tool-missing

if [[ -n "${MLXFAST_DRAFTER_DIR:-}" ]]; then
  SOURCE="${MLXFAST_DRAFTER_DIR}"
else
  [[ -n "${MLXFAST_BASELINE_WORKSPACE:-}" ]] || fail source-missing
  SOURCE="${MLXFAST_BASELINE_WORKSPACE}/drafter_weights"
fi
SOURCE_MANIFEST="${MLXFAST_DRAFTER_MANIFEST:-${SOURCE}.manifest.json}"
[[ -d "${SOURCE}" && -f "${SOURCE_MANIFEST}" ]] || fail source-missing

want_model="$(jq -r '.target.drafter.upstream_model_id // ""' "${FIXTURE}")"
want_revision="$(jq -r '.target.drafter.upstream_revision // ""' "${FIXTURE}")"
[[ -n "${want_model}" && "${want_revision}" =~ ^[0-9a-f]{40}$ ]] || fail fixture-pin
[[ "$(jq -r '.dflash_head.upstream_model_id // ""' "${FIXTURE}")" == "${want_model}" \
   && "$(jq -r '.dflash_head.upstream_revision // ""' "${FIXTURE}")" == "${want_revision}" ]] \
  || fail fixture-pin

jq -e --arg model "${want_model}" --arg revision "${want_revision}" '
  .upstream_model_id == $model and .upstream_revision == $revision and
  (.files | type == "array" and length > 0) and
  ([.files[].path] | length == (unique | length)) and
  ([.files[].path] | index("config.json") != null and index("model.safetensors") != null) and
  all(.files[];
    (.path | type == "string" and length > 0 and startswith("/") | not) and
    (.path | test("^[A-Za-z0-9._/+ -]+$")) and
    (.path | test("(^|/)\\.\\.(/|$)") | not) and
    (.bytes | type == "number" and . > 0 and floor == .) and
    (.sha256 | type == "string" and test("^[0-9a-f]{64}$")))
' "${SOURCE_MANIFEST}" >/dev/null 2>&1 || fail source-manifest

verify_snapshot() {
  local root="$1" relative bytes digest file
  while IFS=$'\t' read -r relative bytes digest; do
    file="${root}/${relative}"
    [[ -f "${file}" ]] || return 1
    [[ "$(wc -c < "${file}" | tr -d '[:space:]')" == "${bytes}" ]] || return 1
    [[ "$(shasum -a 256 "${file}" | awk '{print $1}')" == "${digest}" ]] || return 1
  done < <(jq -r '.files[] | [.path, (.bytes | tostring), .sha256] | @tsv' "${SOURCE_MANIFEST}")
}

verify_snapshot "${SOURCE}" || fail source-content
[[ ! -e "${DESTINATION}" && ! -L "${DESTINATION}" ]] || fail destination-exists
mkdir -p "$(dirname -- "${DESTINATION}")" || fail copy
if ! cp -cRL "${SOURCE}" "${DESTINATION}" 2>/dev/null; then
  cp -RL "${SOURCE}" "${DESTINATION}" >/dev/null 2>&1 || fail copy
fi
if find "${DESTINATION}" -type l -print -quit | grep -q .; then
  fail staged-symlink
fi
[[ -f "${DESTINATION}/config.json" && -f "${DESTINATION}/model.safetensors" ]] \
  || fail staged-incomplete
verify_snapshot "${DESTINATION}" || fail staged-content
printf 'stage-drafter.sh: PASS: dflash\n'
