#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
work_dir=""
result_line=""
result_stderr=1
finish() {
  result_line="$1"
  result_stderr="$3"
  exit "$2"
}
refuse() {
  if [[ -n "${work_dir}" ]]; then
    finish "sandboxed-cli.sh: FAIL: $1" 2 1
  fi
  printf 'sandboxed-cli.sh: FAIL: %s\n' "$1" >&2
  exit 2
}

cleanup() {
  local original_rc=$? cleanup_rc=0 cleanup_output
  trap - EXIT
  # The cleanup log must live outside the tree it removes.
  cleanup_output="$(mktemp "${work_dir}.cleanup.XXXXXX")" || { printf 'sandboxed-cli.sh: FAIL: cleanup\n' >&2; exit 1; }
  python3 "${ROOT}/tools/bounded-run.py" --output "${cleanup_output}" --limit 65536 -- \
    /bin/bash -c '
      tree="$1"
      if command -v chflags >/dev/null 2>&1; then
        chflags -R nouchg,noschg -- "${tree}" || true
      fi
      chmod -R u+rwX -- "${tree}" || true
      rm -rf -- "${tree}"
      [[ ! -e "${tree}" ]]
    ' bash "${work_dir}" || cleanup_rc=$?
  if [[ ${cleanup_rc} -ne 0 || -e "${work_dir}" ]]; then
    printf 'sandboxed-cli.sh: FAIL: cleanup\n' >&2
    exit 1
  fi
  rm -f -- "${cleanup_output}"
  if [[ -n "${result_line}" ]]; then
    if [[ "${result_stderr}" == 1 ]]; then
      printf '%s\n' "${result_line}" >&2
    else
      printf '%s\n' "${result_line}"
    fi
  fi
  exit "${original_rc}"
}
[[ $# -ge 1 ]] || refuse bad-argument
case "$1" in
  self-test) [[ $# -eq 1 ]] || refuse bad-argument ;;
  manifest) [[ $# -eq 2 && "$2" == --digest ]] || refuse bad-argument ;;
  *) refuse bad-argument ;;
esac
verb="$1"
WORKER="${MLXFAST_ENGINE_BIN:-${ROOT}/Vendor/TensorFold/zig-out/bin/bench-worker}"
[[ -x "${WORKER}" ]] || refuse worker-missing
SANDBOX_EXEC=/usr/bin/sandbox-exec
[[ -x "${SANDBOX_EXEC}" ]] || refuse sandbox-unavailable
work_dir="$(mktemp -d "${RUNNER_TEMP:-/tmp}/tf-sandboxed-cli.XXXXXX")"  # not the long macOS $TMPDIR: the self-test socket path must stay under 104 bytes
trap cleanup EXIT
private_tmp="${work_dir}/tmp"
mkdir -m 700 "${private_tmp}"
profile="${work_dir}/profile.sb"
output="${work_dir}/output.log"
# The worker's stderr (diagnostics) stays private and apart; the result check reads stdout only.
errors="${work_dir}/stderr.log"
: > "${output}"
chmod 600 "${output}"
official=()
if [[ "${RUNNER_ENVIRONMENT:-}" == self-hosted || "${MLXFAST_OFFICIAL_BENCHMARK_RUN:-0}" == 1 ]]; then
  official=(--official)
fi
socket_args=()
if [[ "$1" == self-test ]]; then
  private_tmp="$(cd -- "${private_tmp}" && pwd -P)"
  self_test_dir="${private_tmp}/self-test"
  mkdir -m 700 "${self_test_dir}"
  self_test_dir="$(cd -- "${self_test_dir}" && pwd -P)"
  self_test_socket="${self_test_dir}/resident.sock"
  socket_bytes="$(printf '%s' "${self_test_socket}" | LC_ALL=C wc -c | tr -d '[:space:]')"
  (( socket_bytes < 104 )) || refuse socket-path
  socket_args=(--unix-socket "${self_test_socket}")
fi
python3 "${ROOT}/tools/seatbelt-profile.py" "${official[@]+"${official[@]}"}" \
  --exec "${WORKER}" --tree "${ROOT}" --write-subpath "${private_tmp}" \
  "${socket_args[@]+"${socket_args[@]}"}" > "${profile}" \
  2>/dev/null || refuse profile
rc=0
if [[ "$1" == self-test ]]; then
  BENCH_WORKER_SELF_TEST_DIR="${self_test_dir}" TMPDIR="${private_tmp}/" \
    python3 "${ROOT}/tools/bounded-run.py" --output "${output}" --stderr "${errors}" -- \
      "${SANDBOX_EXEC}" -f "${profile}" "${WORKER}" "$@" || rc=$?
else
  unset BENCH_WORKER_SELF_TEST_DIR
  TMPDIR="${private_tmp}/" python3 "${ROOT}/tools/bounded-run.py" --output "${output}" --stderr "${errors}" -- \
    "${SANDBOX_EXEC}" -f "${profile}" "${WORKER}" "$@" || rc=$?
fi
if [[ ${rc} -ne 0 ]]; then
  finish 'sandboxed-cli.sh: FAIL: command' "${rc}" 1
fi
python3 - "${output}" "${verb}" 2>/dev/null <<'PY' || refuse invalid-output
import re
import sys

data = open(sys.argv[1], "rb").read()
if sys.argv[2] == "self-test":
    valid = data == b"engine-protocol-v1 self-test: ok\n"
else:
    valid = re.fullmatch(b"[0-9a-f]{64}\\n", data) is not None
raise SystemExit(0 if valid else 1)
PY
finish "sandboxed-cli.sh: PASS: ${verb}" 0 0
