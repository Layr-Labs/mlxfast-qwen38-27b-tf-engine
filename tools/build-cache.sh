#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${ROOT}"
CACHE_ROOT="${MLXFAST_BUILD_CACHE_DIR:-${HOME}/.cache/mlxfast-engine-build}"
ARTIFACT="Vendor/TensorFold/zig-out/bin/bench-worker"
die() { printf 'build-cache.sh: %s\n' "$*" >&2; exit 2; }
sha_file() { shasum -a 256 -- "$1" | awk '{print $1}'; }
resolve_zig() {
  if [[ -x "${ROOT}/tools/zig-bin.sh" ]]; then
    "${ROOT}/tools/zig-bin.sh"
  else
    printf '%s\n' "${ZIG:-zig}"
  fi
}
ZIG_BIN="$(resolve_zig)"
content_key() {
  {
    git ls-files -s -z -- Vendor/TensorFold tools/stage-bench-worker.sh
    git diff --name-only -z -- Vendor/TensorFold tools/stage-bench-worker.sh | while IFS= read -r -d '' file; do
      [[ -f "${file}" ]] || die "tracked source is missing: ${file}"
      printf '%s\0%s\0' "${file}" "$(git hash-object -- "${file}")"
    done
    printf 'zig\0%s\0host\0%s\0root\0%s\0' "$("${ZIG_BIN}" version 2>/dev/null || echo missing)" "$(uname -m)" "${ROOT}"
  } | shasum -a 256 | awk '{print $1}'
}
case "${1:-}" in
  key) content_key ;;
  restore)
    key="$(content_key)"; dir="${CACHE_ROOT}/${key}"
    [[ -f "${dir}/MANIFEST" && -f "${dir}/${ARTIFACT}" ]] || { printf 'build-cache.sh: miss: %s\n' "${key:0:12}"; exit 1; }
    want="$(awk -v p="${ARTIFACT}" '$1==p {print $2}' "${dir}/MANIFEST")"
    [[ -n "${want}" && "$(sha_file "${dir}/${ARTIFACT}")" == "${want}" ]] || { printf 'build-cache.sh: miss: cached worker differs\n'; exit 1; }
    mkdir -p "$(dirname "${ARTIFACT}")"; cp -p "${dir}/${ARTIFACT}" "${ARTIFACT}"
    printf 'build-cache.sh: hit: %s\n' "${key:0:12}"
    ;;
  save)
    [[ -x "${ARTIFACT}" ]] || die "worker is missing: ${ARTIFACT}"
    key="$(content_key)"; dir="${CACHE_ROOT}/${key}"; tmp="${dir}.tmp.$$"
    rm -rf "${tmp}"; mkdir -p "${tmp}/$(dirname "${ARTIFACT}")"
    cp -p "${ARTIFACT}" "${tmp}/${ARTIFACT}"
    printf '%s %s\n' "${ARTIFACT}" "$(sha_file "${ARTIFACT}")" > "${tmp}/MANIFEST"
    rm -rf "${dir}"; mkdir -p "$(dirname "${dir}")"; mv "${tmp}" "${dir}"
    printf 'build-cache.sh: saved: %s\n' "${key:0:12}"
    ;;
  *) die "usage: tools/build-cache.sh key|restore|save" ;;
esac
