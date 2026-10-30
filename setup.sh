#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
FIXTURE="${ROOT}/fixtures/qwen3_8_27b_tf_mlx_v1_track.json"
TARGET="${MLXFAST_REFERENCE_DIR:-${ROOT}/reference_weights/Qwen3.8-27B-MLX-4bit}"
ZIG_BIN="$("${ROOT}/tools/zig-bin.sh")"

fail() { printf 'setup.sh: %s\n' "$*" >&2; exit 1; }
[[ -d "${TARGET}" ]] || fail "MLXFAST_REFERENCE_DIR target snapshot is missing"
command -v jq >/dev/null 2>&1 || fail "jq is required"

while IFS=$'\t' read -r relative bytes digest; do
  file_name="${TARGET}/${relative}"
  [[ -f "${file_name}" ]] || fail "target file is missing: ${relative}"
  [[ "$(wc -c < "${file_name}" | tr -d ' ')" == "${bytes}" ]] || fail "target byte count differs: ${relative}"
  [[ "$(shasum -a 256 "${file_name}" | awk '{print $1}')" == "${digest}" ]] || fail "target hash differs: ${relative}"
done < <(jq -r '.target.files[] | [.path, (.bytes|tostring), .sha256] | @tsv' "${FIXTURE}")

export ZIG_GLOBAL_CACHE_DIR="${ROOT}/.zig-global-cache"
export ZIG_LOCAL_CACHE_DIR="${ROOT}/.zig-cache"
(
  cd "${ROOT}/Vendor/TensorFold"
  MLXFAST_BUILD_ZIG_BIN="${ZIG_BIN}" "${ROOT}/tools/sandboxed-build.sh" \
    --write "${ZIG_GLOBAL_CACHE_DIR}" \
    --write "${ZIG_LOCAL_CACHE_DIR}" \
    --write "${ROOT}/Vendor/TensorFold/.zig-cache" \
    --write "${ROOT}/Vendor/TensorFold/zig-out" \
    -- "${ZIG_BIN}" build bench-worker -Doptimize=ReleaseFast
)
"${ROOT}/tools/stage-bench-worker.sh"
printf 'setup.sh: target verified and bench-worker built\n'
