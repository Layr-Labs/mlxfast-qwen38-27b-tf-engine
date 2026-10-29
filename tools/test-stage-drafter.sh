#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
SOURCE="${WORK}/source"
FIXTURE="${WORK}/fixture.json"
DECLARATION="${WORK}/mtp-head.manifest.json"
DESTINATION="${WORK}/candidate/drafter_weights"
MANIFEST="${SOURCE}.manifest.json"
REVISION=50307d4c4cde6860d4eee73e2547cd786fe8e8a4
fail() { printf 'test-stage-drafter.sh: FAIL: %s\n' "$*" >&2; exit 1; }

mkdir -p "${SOURCE}"
printf '%s\n' '{"model_type":"qwen3"}' > "${SOURCE}/config.json"
printf '%s\n' 'mock weights' > "${SOURCE}/model.safetensors"
jq -n --arg revision "${REVISION}" '{
  target:{drafter:{upstream_model_id:"z-lab/Qwen3.8-27B-DFlash2",upstream_revision:$revision}},
  dflash_head:{
    upstream_model_id:"z-lab/Qwen3.8-27B-DFlash2",
    upstream_revision:$revision,
    permitted_draft_depths:[1]
  }
}' > "${FIXTURE}"
write_declaration() {
  jq -n --argjson enabled "$1" --argjson tokens "$2" \
    '{spec:{enabled:$enabled,tokens:$tokens}}' > "${DECLARATION}"
}
write_manifest() {
  jq -n --arg revision "$1" \
    --arg config_sha "$(shasum -a 256 "${SOURCE}/config.json" | awk '{print $1}')" \
    --argjson config_bytes "$(wc -c < "${SOURCE}/config.json" | tr -d '[:space:]')" \
    --arg model_sha "$(shasum -a 256 "${SOURCE}/model.safetensors" | awk '{print $1}')" \
    --argjson model_bytes "$(wc -c < "${SOURCE}/model.safetensors" | tr -d '[:space:]')" '{
      upstream_model_id:"z-lab/Qwen3.8-27B-DFlash2",
      upstream_revision:$revision,
      files:[
        {path:"config.json",bytes:$config_bytes,sha256:$config_sha},
        {path:"model.safetensors",bytes:$model_bytes,sha256:$model_sha}
      ]
    }' > "${MANIFEST}"
}
run_stage() {
  TF_CONTRACT="${FIXTURE}" SPEC_DECLARATION_MANIFEST="${DECLARATION}" \
    MLXFAST_DRAFTER_DIR="${SOURCE}" MLXFAST_DRAFTER_MANIFEST="${MANIFEST}" \
    DRAFTER_DEST="${DESTINATION}" "${ROOT}/tools/stage-drafter.sh" 2>&1
}

printf '%s\n' '{}' > "${DECLARATION}"
[[ "$(run_stage)" == 'stage-drafter.sh: PASS: serial' && ! -e "${DESTINATION}" ]] \
  || fail "an omitted spec block did not stay serial"

rm -f "${DECLARATION}"
[[ "$(run_stage)" == 'stage-drafter.sh: PASS: serial' && ! -e "${DESTINATION}" ]] \
  || fail "an absent declaration did not stay serial"

write_declaration false 0
[[ "$(run_stage)" == 'stage-drafter.sh: PASS: serial' && ! -e "${DESTINATION}" ]] \
  || fail "serial mode staged a drafter"

write_declaration true 1
write_manifest "${REVISION}"
[[ "$(run_stage)" == 'stage-drafter.sh: PASS: dflash' ]] || fail "pinned DFlash stage failed"
[[ -f "${DESTINATION}/config.json" && -f "${DESTINATION}/model.safetensors" ]] \
  || fail "DFlash stage is incomplete"
find "${DESTINATION}" -type l -print -quit | grep -q . && fail "DFlash stage kept a symlink"

find "${DESTINATION}" -type f -delete
rmdir "${DESTINATION}"
write_manifest 0000000000000000000000000000000000000000
out="$(run_stage || true)"
[[ "${out}" == 'stage-drafter.sh: FAIL: source-manifest' && ! -e "${DESTINATION}" ]] \
  || fail "wrong drafter revision was accepted"

write_manifest "${REVISION}"
printf 'changed\n' >> "${SOURCE}/model.safetensors"
out="$(run_stage || true)"
[[ "${out}" == 'stage-drafter.sh: FAIL: source-content' && ! -e "${DESTINATION}" ]] \
  || fail "wrong drafter content was accepted"

write_declaration true 2
out="$(run_stage || true)"
[[ "${out}" == 'stage-drafter.sh: FAIL: declaration' ]] \
  || fail "unsupported DFlash depth was accepted as serial"
printf 'test-stage-drafter.sh: PASS\n'
