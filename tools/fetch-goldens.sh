#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
CONTRACT="${TF_CONTRACT:-${ROOT}/fixtures/qwen3_8_27b_tf_mlx_v1_track.json}"
R2_PATH=""
WANT_SHA=""
WANT_BYTES=""
OUT=""
FETCH_ALL=0
ALLOW_HIDDEN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --r2-path) R2_PATH="${2:-}"; shift 2 ;;
    --sha256) WANT_SHA="${2:-}"; shift 2 ;;
    --bytes) WANT_BYTES="${2:-}"; shift 2 ;;
    --out) OUT="${2:-}"; shift 2 ;;
    --all) FETCH_ALL=1; shift ;;
    --allow-hidden) ALLOW_HIDDEN=1; shift ;;
    *) printf 'fetch-goldens.sh: unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
die() { printf 'fetch-goldens.sh: %s\n' "$*" >&2; exit 1; }
[[ -r "${CONTRACT}" ]] || die "contract is missing"

if [[ "${FETCH_ALL}" == 1 ]]; then
  [[ -n "${OUT}" ]] || { printf 'fetch-goldens.sh: --all needs --out DIR\n' >&2; exit 2; }
  mkdir -p "${OUT}"
  while IFS=$'\t' read -r object digest bytes; do
    "${BASH_SOURCE[0]}" --allow-hidden --r2-path "${object}" --sha256 "${digest}" --bytes "${bytes}" --out "${OUT}/$(basename "${object}")"
  done < <(jq -r '[.timed_prompt_pool[], .warmup_golden, (.live_golden_speculative.dflash1[])][] | [.r2_path,.sha256,(.bytes|tostring)] | @tsv' "${CONTRACT}")
  exit 0
fi

[[ -n "${R2_PATH}" && -n "${WANT_SHA}" && -n "${WANT_BYTES}" && -n "${OUT}" ]] || { printf 'fetch-goldens.sh: --r2-path, --sha256, --bytes, and --out are required\n' >&2; exit 2; }
[[ "${R2_PATH}" =~ ^[A-Za-z0-9._/-]+$ && "${R2_PATH}" != /* && "${R2_PATH}" != *"/../"* ]] || { printf 'fetch-goldens.sh: invalid object path\n' >&2; exit 2; }
[[ "${WANT_SHA}" =~ ^[0-9a-f]{64}$ && "${WANT_BYTES}" =~ ^[1-9][0-9]*$ ]] || { printf 'fetch-goldens.sh: invalid pin\n' >&2; exit 2; }
if [[ "${ALLOW_HIDDEN}" != 1 ]] && jq -e --arg p "${R2_PATH}" --arg s "${WANT_SHA}" '[.timed_prompt_pool[], .warmup_golden, (.live_golden_speculative.dflash1[])][] | select(.r2_path == $p or .sha256 == $s)' "${CONTRACT}" >/dev/null; then
  die "requested pin is hidden organizer material"
fi
[[ -n "${R2_BUCKET_ENDPOINT:-}" ]] || die "R2_BUCKET_ENDPOINT is not set"
tmp="${OUT}.tmp.$$"
trap 'rm -f "${tmp}"' EXIT
mkdir -p "$(dirname "${OUT}")"
downloader="${TF_R2_DOWNLOADER:-${ROOT}/tools/download-r2-object.sh}"
[[ -x "${downloader}" ]] || die "R2 downloader is missing"
"${downloader}" "${R2_PATH}" "${tmp}"
[[ "$(wc -c < "${tmp}" | tr -d ' ')" == "${WANT_BYTES}" ]] || die "byte count differs"
[[ "$(shasum -a 256 "${tmp}" | awk '{print $1}')" == "${WANT_SHA}" ]] || die "digest differs"
mv "${tmp}" "${OUT}"
trap - EXIT
printf 'fetch-goldens.sh: verified %s\n' "${R2_PATH}"
