#!/usr/bin/env bash
set -euo pipefail

SOURCE=""
DEST=""
DRY_RUN=0
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --source) SOURCE="${2:-}"; shift 2 ;;
    --dest) DEST="${2:-}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) printf 'stage-baseline-workspace.sh: unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
fail() { printf 'stage-baseline-workspace.sh: %s\n' "$*" >&2; exit 1; }
[[ -n "${SOURCE}" && -d "${SOURCE}" ]] || fail "--source must name a worktree"
[[ -n "${DEST}" && ! -e "${DEST}" ]] || fail "--dest must not exist"
fixture="${TF_CONTRACT:-${ROOT}/fixtures/qwen3_8_27b_tf_mlx_v1_track.json}"
[[ -r "${fixture}" ]] || fail "track fixture is missing"
commit="$(jq -r '.baseline_reference_commit' "${fixture}")"
[[ "${commit}" =~ ^[0-9a-f]{40}$ ]] || fail "baseline commit is not pinned"
[[ "$(git -C "${SOURCE}" rev-parse HEAD 2>/dev/null)" == "${commit}" ]] || fail "source commit differs"
if [[ -x "${ROOT}/tools/zig-bin.sh" ]]; then
  zig_bin="$("${ROOT}/tools/zig-bin.sh")"
else
  zig_bin="${ZIG:-zig}"
fi
reference_dir="${MLXFAST_REFERENCE_DIR:-}"
if [[ "${DRY_RUN}" == 1 ]]; then
  printf 'git clone --no-local %q %q\n' "${SOURCE}" "${DEST}"
  printf 'cd %q && ZIG_GLOBAL_CACHE_DIR=%q ZIG_LOCAL_CACHE_DIR=%q %q build bench-worker -Doptimize=ReleaseFast\n' "${DEST}/Vendor/TensorFold" "${DEST}/.zig-global-cache" "${DEST}/.zig-cache" "${zig_bin}"
  printf 'clone-copy %q %q\n' "${reference_dir:-<MLXFAST_REFERENCE_DIR>}" "${DEST}/reference_weights/Qwen3.8-27B-MLX-4bit"
  printf 'mkdir-readonly %q\n' "${DEST}/drafter_weights"
  exit 0
fi
[[ -n "${reference_dir}" && -d "${reference_dir}" ]] || fail "MLXFAST_REFERENCE_DIR must name the verified target snapshot"
[[ -f "${reference_dir}/config.json" ]] || fail "MLXFAST_REFERENCE_DIR has no config.json"
git clone -q --no-local "${SOURCE}" "${DEST}"
(
  cd "${DEST}/Vendor/TensorFold"
  ZIG_GLOBAL_CACHE_DIR="${DEST}/.zig-global-cache" ZIG_LOCAL_CACHE_DIR="${DEST}/.zig-cache" "${zig_bin}" build bench-worker -Doptimize=ReleaseFast
)
[[ -x "${DEST}/Vendor/TensorFold/zig-out/bin/bench-worker" ]] || fail "staged worker is missing"
mkdir -p "${DEST}/reference_weights"
target_dir="${DEST}/reference_weights/Qwen3.8-27B-MLX-4bit"
if ! cp -cRL "${reference_dir}" "${target_dir}"; then
  cp -RL "${reference_dir}" "${target_dir}"
fi
if find "${target_dir}" -type l -print -quit | grep -q .; then
  fail "staged target snapshot contains a symlink"
fi
[[ -f "${target_dir}/config.json" ]] || fail "staged target snapshot is missing"
intake_shas="${ROOT}/fixtures/reference_qwen3_8_27b_4bit.sha256"
[[ -f "${intake_shas}" ]] || fail "target intake sha list is missing"
(cd "${target_dir}" && shasum -a 256 -c "${intake_shas}") >/dev/null \
  || fail "staged target snapshot differs from the intake shas"
# The pinned launcher checks this ignored path before it builds the serial argv.
# It adds --drafter only for DFlash, so the empty serial placeholder is unused.
mkdir "${DEST}/drafter_weights"
chmod 0555 "${DEST}/drafter_weights"
printf 'stage-baseline-workspace.sh: staged %s\n' "${DEST}"
