#!/usr/bin/env bash
#
# sandboxed-build.sh -- run one compile command of the build under Seatbelt.
#
# The build compiles Zig, C++, Metal, and CUDA files from editablePaths. A
# compiler reads every file that a source names (#include, #embed, .incbin).
# On an official run, this script writes a build profile
# (tools/seatbelt-profile.py --build) and starts the command under
# /usr/bin/sandbox-exec. The command and every process that it starts:
#   * cannot read or write the evaluator-only paths that the environment
#     names, the build cache root, or the runner registration files;
#   * can write only the --write trees, a private TMPDIR that this script
#     makes and removes, and the job account's cache directory
#     (DARWIN_USER_CACHE_DIR, shared between jobs of that account);
#   * cannot use the network.
# setup.sh can call this script for a compile command. Dependencies are already
# vendored. The build does not fetch them.
#
# Usage:
#   tools/sandboxed-build.sh --write DIR [--write DIR ...] -- COMMAND [ARG ...]
#
# Environment:
#   An official run (RUNNER_ENVIRONMENT=self-hosted or
#   MLXFAST_OFFICIAL_BENCHMARK_RUN=1) refuses when the runner environment does
#   not name the evaluator-only paths. On other runs, this script runs the
#   command without a sandbox.
#
# Output is captured in a private file and limited to 1 MiB. The job log gets
# one fixed result line. Exit codes: the command's exit code; 2 on a refusal;
# 125 when build processes survived cleanup; 128+N after cancellation by
# signal N (SIGTERM, SIGINT, SIGHUP). On cancellation the script forwards the
# signal and waits until bounded-run.py has stopped the build group.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
WORK_DIR=""
RESULT_LINE=""
RESULT_STDERR=1
finish() {
  RESULT_LINE="$1"
  RESULT_STDERR="$3"
  exit "$2"
}
refuse() {
  if [[ -n "${WORK_DIR}" ]]; then
    finish "sandboxed-build.sh: FAIL: $1" 2 1
  fi
  printf 'sandboxed-build.sh: FAIL: %s\n' "$1" >&2
  exit 2
}

# An extended ACL can give write access that the mode bits do not show.
# A path whose ACL cannot be read counts as having one (fail closed).
has_acl() {
  local out err rc=0
  err="$(mktemp)"
  if [[ "$(/usr/bin/uname -s)" == Darwin ]]; then
    out="$(/bin/ls -led "$1" 2>"${err}")" || rc=$?
  elif [[ -x /usr/bin/getfacl ]]; then
    out="$(/usr/bin/getfacl -p --omit-header "$1" 2>"${err}" | /usr/bin/grep -vE '^(user|group|other)::|^$')" || rc=$?
    [[ ${rc} -eq 1 && ! -s "${err}" ]] && rc=0
  else
    /bin/rm -f "${err}"
    return 0
  fi
  if [[ ${rc} -ne 0 || -s "${err}" ]]; then
    /bin/rm -f "${err}"
    return 0
  fi
  /bin/rm -f "${err}"
  if [[ "$(/usr/bin/uname -s)" == Darwin ]]; then
    [[ -n "$(printf '%s\n' "${out}" | /usr/bin/sed -n '2p')" ]]
  else
    [[ -n "${out}" ]]
  fi
}

xcrun_tool_path() {
  local kind="$1" found="$2" resolved resolved_dir stat_line stat_type stat_rest stat_uid stat_mode
  [[ "${OWNER_UID}" =~ ^[0-9]+$ ]] || return 1
  resolved="$(/bin/realpath "${found}" 2>/dev/null)" || return 1
  [[ -f "${resolved}" ]] || return 1
  stat_line="$(/usr/bin/stat -f '%HT %u %Lp' "${resolved}" 2>/dev/null)" || return 1
  stat_type="${stat_line% * *}"
  stat_rest="${stat_line% *}"
  stat_uid="${stat_rest##* }"
  stat_mode="${stat_line##* }"
  [[ "${stat_type}" == "Regular File" && "${stat_uid}" =~ ^[0-9]+$ && "${stat_mode}" =~ ^[0-7]+$ ]] \
    || return 1
  (( (8#${stat_mode} & 8#22) == 0 )) || return 1
  has_acl "${resolved}" && return 1
  case "${kind}" in
    python3)
      (( stat_uid == 0 || stat_uid == OWNER_UID )) || return 1
      case "${resolved}" in
        /Library/Developer/CommandLineTools/*) ;;
        *) [[ -n "${DEVELOPER_REAL}" && "${resolved}" == "${DEVELOPER_REAL}/"* ]] || return 1 ;;
      esac
      ;;
    metal|metallib)
      case "${resolved}" in
        /private/var/run/com.apple.security.cryptexd/mnt/com.apple.MobileAsset.MetalToolchain-*/*)
          (( stat_uid == 0 )) || return 1
          ;;
        "${DEVELOPER_REAL}/"*)
          [[ -n "${DEVELOPER_REAL}" ]] || return 1
          (( stat_uid == 0 || stat_uid == OWNER_UID )) || return 1
          resolved_dir="${resolved%/*}"
          if [[ "${resolved_dir}" != "${DEVELOPER_REAL}/Toolchains/XcodeDefault.xctoolchain/usr/bin" ]]; then
            toolchain_dir "${resolved_dir}" >/dev/null || return 1
          fi
          ;;
        *) return 1 ;;
      esac
      ;;
    *) return 1 ;;
  esac
  printf '%s\n' "${resolved}"
}

toolchain_dir() {
  local found="$1" resolved stat_line stat_type stat_rest stat_uid stat_mode
  resolved="$(/bin/realpath "${found}" 2>/dev/null)" || return 1
  stat_line="$(/usr/bin/stat -f '%HT %u %Lp' "${resolved}" 2>/dev/null)" || return 1
  stat_type="${stat_line% * *}"
  stat_rest="${stat_line% *}"
  stat_uid="${stat_rest##* }"
  stat_mode="${stat_line##* }"
  [[ "${stat_uid}" =~ ^[0-9]+$ && "${stat_mode}" =~ ^[0-7]+$ ]] || return 1
  [[ -d "${resolved}" && "${stat_type}" == "Directory" ]] || return 1
  (( (8#${stat_mode} & 8#22) == 0 )) || return 1
  has_acl "${resolved}" && return 1
  (( stat_uid == 0 || stat_uid == OWNER_UID )) || return 1
  printf '%s\n' "${resolved}"
}

xcrun_db_sha() {
  local digest hex
  digest="$(/usr/bin/openssl dgst -sha256 -r "${XCRUN_DB}" 2>/dev/null)" || return 1
  hex="${digest%% *}"
  [[ "${hex}" =~ ^[[:xdigit:]]{64}$ ]] || return 1
  printf '%.12s\n' "${hex}"
}

cleanup() {
  local original_rc=$? cleanup_rc=0 cleanup_output
  trap - EXIT
  # The cleanup log must live outside the tree it removes.
  cleanup_output="$(mktemp "${WORK_DIR}.cleanup.XXXXXX")" || { printf 'sandboxed-build.sh: FAIL: cleanup\n' >&2; exit 1; }
  python3 "${REPO_ROOT}/tools/bounded-run.py" --output "${cleanup_output}" --limit 65536 -- \
    /bin/bash -c '
      tree="$1"
      if command -v chflags >/dev/null 2>&1; then
        chflags -R nouchg,noschg -- "${tree}" || true
      fi
      chmod -R u+rwX -- "${tree}" || true
      rm -rf -- "${tree}"
      [[ ! -e "${tree}" ]]
    ' bash "${WORK_DIR}" || cleanup_rc=$?
  if [[ ${cleanup_rc} -ne 0 || -e "${WORK_DIR}" ]]; then
    printf 'sandboxed-build.sh: FAIL: cleanup\n' >&2
    exit 1
  fi
  rm -f -- "${cleanup_output}"
  if [[ -n "${RESULT_LINE}" ]]; then
    if [[ "${RESULT_STDERR}" == 1 ]]; then
      printf '%s\n' "${RESULT_LINE}" >&2
    else
      printf '%s\n' "${RESULT_LINE}"
    fi
  fi
  exit "${original_rc}"
}

# Cancellation: forward SIGTERM, SIGINT, and SIGHUP to bounded-run.py and wait
# until it has stopped and reaped the build group. The EXIT cleanup runs only
# after that.
BOUNDED_PID=""
CANCEL_SIGNAL=""
forward_signal() {
  [[ -n "${CANCEL_SIGNAL}" ]] || CANCEL_SIGNAL="$1"
  if [[ -n "${BOUNDED_PID}" ]]; then
    kill -s "$1" "${BOUNDED_PID}" 2>/dev/null || true
  fi
}
BUILD_SHELL=/bin/dash
XCRUN_ACTIVE=0
XCRUN_DB=""
XCRUN_WARM_SHA=""
XCRUN_WARM_METAL=""
XCRUN_WARM_METALLIB=""
DEVELOPER_REAL=""
OWNER_UID=""
run_bounded() {
  local rc=0 build_start=""
  [[ -z "${CANCEL_SIGNAL}" ]] || finish 'sandboxed-build.sh: FAIL: cancelled' 130 1
  if [[ "${RUNNER_ENVIRONMENT:-}" == "self-hosted" \
     || "${MLXFAST_OFFICIAL_BENCHMARK_RUN:-0}" == "1" ]]; then
    build_start="$(date +%s)"
  fi
  TMPDIR="${PRIVATE_TMP}/" SHELL="${BUILD_SHELL}" CONFIG_SHELL="${BUILD_SHELL}" \
    python3 "${REPO_ROOT}/tools/bounded-run.py" \
    --output "${OUTPUT}" --session -- "$@" 0<&0 &
  BOUNDED_PID=$!
  # A signal that came before BOUNDED_PID was set is recorded but not yet forwarded.
  [[ -z "${CANCEL_SIGNAL}" ]] || kill -s "${CANCEL_SIGNAL}" "${BOUNDED_PID}" 2>/dev/null || true
  while :; do
    rc=0
    wait "${BOUNDED_PID}" || rc=$?
    # A trapped signal interrupts wait. Wait again until bounded-run.py exits.
    kill -0 "${BOUNDED_PID}" 2>/dev/null || break
  done
  BOUNDED_PID=""
  if [[ -n "${build_start}" ]] \
    && ! "${REPO_ROOT}/tools/orphan-guard.sh" check --job-start "${build_start}" \
      >/dev/null 2>&1; then
    finish 'sandboxed-build.sh: FAIL: orphan-process' 125 1
  fi
  if [[ ${rc} -eq 0 && -z "${CANCEL_SIGNAL}" ]]; then
    if [[ "${XCRUN_ACTIVE}" == 1 ]]; then
      xcrun_after_sha="$(xcrun_db_sha)" \
        || finish 'sandboxed-build.sh: FAIL: xcrun-tamper' 2 1
      printf 'sandboxed-build.sh: info: xcrun_db sha256 %s -> %s\n' \
        "${XCRUN_WARM_SHA}" "${xcrun_after_sha}" >&2
      xcrun_after_metal="$(/usr/bin/xcrun -sdk macosx -f metal 2>/dev/null)" \
        || finish 'sandboxed-build.sh: FAIL: xcrun-tamper' 2 1
      xcrun_after_metallib="$(/usr/bin/xcrun -sdk macosx -f metallib 2>/dev/null)" \
        || finish 'sandboxed-build.sh: FAIL: xcrun-tamper' 2 1
      xcrun_after_metal="$(xcrun_tool_path metal "${xcrun_after_metal}")" \
        || finish 'sandboxed-build.sh: FAIL: xcrun-tamper' 2 1
      xcrun_after_metallib="$(xcrun_tool_path metallib "${xcrun_after_metallib}")" \
        || finish 'sandboxed-build.sh: FAIL: xcrun-tamper' 2 1
      [[ "${xcrun_after_metal}" == "${XCRUN_WARM_METAL}" \
        && "${xcrun_after_metallib}" == "${XCRUN_WARM_METALLIB}" ]] \
        || finish 'sandboxed-build.sh: FAIL: xcrun-tamper' 2 1
    fi
    finish 'sandboxed-build.sh: PASS' 0 0
  elif [[ ${rc} -eq 125 ]]; then
    finish 'sandboxed-build.sh: FAIL: build-left-processes' "${rc}" 1
  elif [[ -n "${CANCEL_SIGNAL}" ]]; then
    [[ ${rc} -ne 0 ]] || rc=130
    finish 'sandboxed-build.sh: FAIL: cancelled' "${rc}" 1
  else
    finish 'sandboxed-build.sh: FAIL: command' "${rc}" 1
  fi
}

writes=()
while [[ $# -gt 0 && "$1" != "--" ]]; do
  case "$1" in
    --write)
      [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || refuse bad-argument
      writes+=("$2")
      shift 2
      ;;
    *) refuse bad-argument ;;
  esac
done
[[ $# -ge 2 && "$1" == "--" ]] || refuse bad-argument
shift
[[ ${#writes[@]} -gt 0 ]] || refuse bad-argument

if [[ ( "${RUNNER_ENVIRONMENT:-}" == "self-hosted" \
    || "${MLXFAST_OFFICIAL_BENCHMARK_RUN:-0}" == "1" ) \
  && "$(uname -s)" == Darwin && -x /usr/bin/xcrun ]]; then
  [[ -x /bin/realpath ]] || refuse xcrun-warm
  xcrun_temp_dir="$(/usr/bin/getconf DARWIN_USER_TEMP_DIR 2>/dev/null)" \
    || refuse xcrun-warm
  [[ -n "${xcrun_temp_dir}" && -d "${xcrun_temp_dir}" ]] || refuse xcrun-warm
  xcrun_temp_dir="$(cd -P -- "${xcrun_temp_dir}" && pwd -P)" || refuse xcrun-warm
  XCRUN_DB="${xcrun_temp_dir%/}/xcrun_db"
  if [[ -n "${DEVELOPER_DIR:-}" ]]; then
    developer_dir="${DEVELOPER_DIR}"
  else
    developer_dir="$(/usr/bin/xcode-select -p 2>/dev/null)" || refuse xcrun-warm
  fi
  DEVELOPER_REAL="$(/bin/realpath "${developer_dir}" 2>/dev/null)" \
    || refuse toolchain-writable
  OWNER_UID="$(/usr/bin/stat -f %u "${DEVELOPER_REAL}" 2>/dev/null)" \
    || refuse toolchain-writable
  [[ "${OWNER_UID}" =~ ^[0-9]+$ ]] || refuse toolchain-writable
  case "${DEVELOPER_REAL}" in
    *.app/Contents/Developer)
      toolchain_dir "${DEVELOPER_REAL%/Contents/Developer}" >/dev/null \
        || refuse toolchain-writable
      toolchain_dir "${DEVELOPER_REAL%/Developer}" >/dev/null \
        || refuse toolchain-writable
      ;;
  esac
  toolchain_dir "${DEVELOPER_REAL}/Platforms" >/dev/null \
    || refuse toolchain-writable
  toolchain_dir "${DEVELOPER_REAL}/Toolchains" >/dev/null \
    || refuse toolchain-writable
  toolchain_dir \
    "${DEVELOPER_REAL}/Toolchains/XcodeDefault.xctoolchain" >/dev/null \
    || refuse toolchain-writable
  toolchain_dir \
    "${DEVELOPER_REAL}/Toolchains/XcodeDefault.xctoolchain/usr" >/dev/null \
    || refuse toolchain-writable
  toolchain_dir "${DEVELOPER_REAL}" >/dev/null \
    || refuse toolchain-writable
  toolchain_dir "${DEVELOPER_REAL}/usr/bin" >/dev/null \
    || refuse toolchain-writable
  toolchain_dir "${DEVELOPER_REAL}/Platforms/MacOSX.platform" >/dev/null \
    || refuse toolchain-writable
  xcrun_sdk="$(/usr/bin/xcrun -n --sdk macosx --show-sdk-path 2>/dev/null)" \
    || refuse toolchain-writable
  toolchain_dir "${xcrun_sdk}" >/dev/null \
    || refuse toolchain-writable
  toolchain_dir \
    "${DEVELOPER_REAL}/Toolchains/XcodeDefault.xctoolchain/usr/bin" >/dev/null \
    || refuse toolchain-writable
  xcrun_warm_metal="$(/usr/bin/xcrun -n -sdk macosx -f metal 2>/dev/null)" \
    || refuse xcrun-warm
  xcrun_warm_metallib="$(/usr/bin/xcrun -n -sdk macosx -f metallib 2>/dev/null)" \
    || refuse xcrun-warm
  xcrun_warm_python="$(/usr/bin/xcrun -n -f python3 2>/dev/null)" \
    || refuse xcrun-warm
  [[ -f "${XCRUN_DB}" && ! -L "${XCRUN_DB}" ]] || refuse xcrun-warm
  XCRUN_WARM_METAL="$(xcrun_tool_path metal "${xcrun_warm_metal}")" \
    || refuse xcrun-warm
  XCRUN_WARM_METALLIB="$(xcrun_tool_path metallib "${xcrun_warm_metallib}")" \
    || refuse xcrun-warm
  xcrun_tool_path python3 "${xcrun_warm_python}" >/dev/null \
    || refuse toolchain-writable
  XCRUN_WARM_SHA="$(xcrun_db_sha)" || refuse xcrun-warm
  XCRUN_ACTIVE=1
fi

temp_root="${RUNNER_TEMP:-${TMPDIR:-/tmp}}"
build_temp_root="${temp_root}"
if [[ "${RUNNER_ENVIRONMENT:-}" == "self-hosted" \
   || "${MLXFAST_OFFICIAL_BENCHMARK_RUN:-0}" == "1" ]]; then
  build_temp_root="${RUNNER_TEMP:-/private/tmp}/build.noindex"
  mkdir -p "${build_temp_root}" || refuse temp
  chmod 700 "${build_temp_root}" || refuse temp
fi
WORK_DIR="$(mktemp -d "${build_temp_root%/}/tensorfold-sandboxed-build.XXXXXX")"
trap cleanup EXIT
trap 'forward_signal TERM' TERM
trap 'forward_signal INT' INT
trap 'forward_signal HUP' HUP
chmod 700 "${WORK_DIR}"
OUTPUT="${WORK_DIR}/output.log"
: > "${OUTPUT}"
chmod 600 "${OUTPUT}"
runner=("$@")
[[ "${runner[0]}" != /bin/sh ]] || runner[0]="${BUILD_SHELL}"
PRIVATE_TMP="${WORK_DIR}/tmp"
mkdir -m 700 "${PRIVATE_TMP}"

if [[ "${RUNNER_ENVIRONMENT:-}" != "self-hosted" && "${MLXFAST_OFFICIAL_BENCHMARK_RUN:-0}" != "1" ]]; then
  run_bounded "${runner[@]}"
fi

PROFILE="${WORK_DIR}/profile.sb"
profile_args=()
deny_read_roots=()
resolved_zig=""
zig_toolchain_dir=""
add_deny_read_root() {
  local path="$1" known
  [[ -n "${path}" ]] || return 0
  for known in "${deny_read_roots[@]:-}"; do
    [[ "${known}" != "${path%/}" ]] || return 0
  done
  deny_read_roots+=("${path%/}")
}

real_path() {
  python3 -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "$1"
}

non_indexed_path() {
  python3 -c '
import os, sys
p = os.path.realpath(os.path.abspath(sys.argv[1]))
roots = ("/private/tmp", "/private/var/folders", "/var/folders")
raise SystemExit(0 if any(x.endswith(".noindex") for x in p.split(os.sep)) or any(p == r or p.startswith(r + "/") for r in roots) else 1)
' "$1"
}

# setup.sh passes the path that tools/zig-bin.sh already validated. A runner
# build accepts Zig only from one root-owned, non-writable toolchain directory.
zig_bin="${MLXFAST_BUILD_ZIG_BIN:-}"
if [[ -n "${zig_bin}" ]]; then
  [[ -x "${zig_bin}" ]] || refuse profile
  resolved_zig="$(real_path "${zig_bin}")"
  resolved_runner="$(real_path "${runner[0]}")"
  [[ "${resolved_runner}" == "${resolved_zig}" ]] || refuse profile
  zig_toolchain_dir="$("${REPO_ROOT}/tools/trusted-zig-dir.sh" "${resolved_zig}" 2>/dev/null)" \
    || refuse profile
  runner[0]="${resolved_zig}"
fi

# One account can host several track runners. A runner job denies the whole
# account home, then adds back only this runner and this build's paths. A local
# run with no runner variables keeps the narrower temporary-tree policy.
runner_roots=()
if [[ -n "${RUNNER_TEMP:-}" || -n "${RUNNER_WORKSPACE:-}" ]]; then
  [[ -n "${HOME:-}" ]] || refuse profile
  add_deny_read_root "${HOME}"
fi
for runner_path in "${RUNNER_TEMP:-}" "${RUNNER_WORKSPACE:-}"; do
  case "${runner_path}" in
    */_work/*) runner_roots+=("${runner_path%%/_work/*}") ;;
  esac
done
for runner_root in "${runner_roots[@]:-}"; do
  [[ -n "${runner_root}" ]] || continue
  add_deny_read_root "${runner_root}/_work"
  runner_parent="$(dirname -- "${runner_root}")"
  for runner_work in "${runner_parent}"/*/_work; do
    [[ -d "${runner_work}" ]] && add_deny_read_root "${runner_work}"
  done
done
if [[ -n "${HOME:-}" ]]; then
  for runner_work in "${HOME%/}"/*/_work; do
    [[ -d "${runner_work}" ]] && add_deny_read_root "${runner_work}"
  done
fi
add_deny_read_root "${temp_root}"
# A build tool that Zig starts gets the account cache directory, not TMPDIR.
for name in DARWIN_USER_TEMP_DIR DARWIN_USER_CACHE_DIR; do
  user_dir="$(/usr/bin/getconf "${name}" 2>/dev/null || true)"
  [[ -n "${user_dir}" && -d "${user_dir}" ]] || continue
  if [[ "${name}" == DARWIN_USER_TEMP_DIR ]]; then
    user_dir="$(cd -P -- "${user_dir}" && pwd -P)" || refuse profile
    add_deny_read_root "${user_dir}"
  else
    profile_args+=(--write-subpath "${user_dir%/}")
  fi
done
for deny_root in "${deny_read_roots[@]}"; do
  profile_args+=(--deny-read-subpath "${deny_root}")
done
profile_args+=(--read-subpath "${WORK_DIR}" --read-subpath "${REPO_ROOT}")
if [[ -n "${zig_toolchain_dir}" ]]; then
  zig_parent="${zig_toolchain_dir}"
  while [[ "${zig_parent}" != / ]]; do
    profile_args+=(--read-metadata-literal "${zig_parent}")
    zig_parent="$(dirname -- "${zig_parent}")"
  done
  profile_args+=(--read-subpath "${zig_toolchain_dir}")
fi
for dir in "${writes[@]}"; do
  if [[ "${RUNNER_ENVIRONMENT:-}" == "self-hosted" \
     || "${MLXFAST_OFFICIAL_BENCHMARK_RUN:-0}" == "1" ]]; then
    non_indexed_path "${dir}" || refuse profile
  fi
  mkdir -p "${dir}"
  profile_args+=(--read-subpath "${dir}" --write-subpath "${dir}")
done
if [[ "${XCRUN_ACTIVE}" == 1 ]]; then
  profile_args+=(--read-literal "${XCRUN_DB}")
fi
python3 "${REPO_ROOT}/tools/seatbelt-profile.py" --build --official --tree "${REPO_ROOT}" \
  --write-subpath "${PRIVATE_TMP}" "${profile_args[@]}" > "${PROFILE}" \
  2>/dev/null || refuse profile

SANDBOX_EXEC=/usr/bin/sandbox-exec
[[ -x "${SANDBOX_EXEC}" ]] || refuse sandbox-unavailable

run_bounded "${SANDBOX_EXEC}" -f "${PROFILE}" "${runner[@]}"
