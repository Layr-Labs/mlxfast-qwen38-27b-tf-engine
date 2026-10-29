#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
SANDBOXED_BUILD="${SANDBOXED_BUILD_UNDER_TEST:-${ROOT}/tools/sandboxed-build.sh}"
TRUSTED_ZIG_DIR="${ROOT}/tools/trusted-zig-dir.sh"
fail() { printf 'test-sandboxed-build.sh: FAIL: %s\n' "$*" >&2; exit 1; }
WORK="$(mktemp -d)"
SIBLING_PID=""
probe_tag="tf-f2-${$}-${WORK##*.}"
VAR_TMP_CANARY="/private/var/tmp/${probe_tag}.canary"
VAR_TMP_NEW="/private/var/tmp/${probe_tag}.new"
VAR_TMP_SH_THD="/private/var/tmp/sh-thd-${probe_tag}"
ACCOUNT_TMP_CANARY=""
XCRUN_DB_CANARY=""
XCRUN_DB_BACKUP=""
XCRUN_DB_TO_RESTORE=""
XCRUN_POLL_PID=""
cleanup_test() {
  local rc=$?
  trap - EXIT
  [[ -z "${SIBLING_PID}" ]] || kill "${SIBLING_PID}" 2>/dev/null || true
  rm -f -- "${VAR_TMP_CANARY}" "${VAR_TMP_NEW}" "${VAR_TMP_SH_THD}"
  [[ -z "${ACCOUNT_TMP_CANARY}" ]] || rm -f -- "${ACCOUNT_TMP_CANARY}"
  [[ -z "${XCRUN_DB_CANARY}" ]] || rm -f -- "${XCRUN_DB_CANARY}"
  if [[ -n "${XCRUN_POLL_PID}" ]]; then
    [[ -z "${xcrun_poll_stop:-}" ]] || : > "${xcrun_poll_stop}"
    wait "${XCRUN_POLL_PID}" 2>/dev/null || true
  fi
  if [[ -n "${XCRUN_DB_BACKUP}" && -n "${XCRUN_DB_TO_RESTORE}" ]]; then
    cp -p -- "${XCRUN_DB_BACKUP}" "${XCRUN_DB_TO_RESTORE}" || rc=1
    /usr/bin/xcrun -n -sdk macosx -f metal >/dev/null 2>&1 || true
  fi
  rm -rf "${WORK}"
  exit "${rc}"
}
trap cleanup_test EXIT
wait_for_file() {
  local file="$1" count=0
  # An official call first runs three uncached xcrun lookups (about 3 s).
  while [[ ! -s "${file}" && ${count} -lt 500 ]]; do
    sleep 0.02
    count=$((count + 1))
  done
  [[ -s "${file}" ]]
}
process_is_gone() {
  local pid="$1" count=0 state
  while [[ ${count} -lt 100 ]]; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      return 0
    fi
    state="$(ps -o stat= -p "${pid}" 2>/dev/null | tr -d '[:space:]' || true)"
    [[ "${state}" == Z* ]] && return 0
    sleep 0.02
    count=$((count + 1))
  done
  return 1
}
mkdir -p "${WORK}/goldens" "${WORK}/baseline" "${WORK}/benchd" "${WORK}/runner" "${WORK}/write"
TEST_DEVELOPER_DIR="${DEVELOPER_DIR:-}"
if [[ "$(uname -s)" == Darwin && -z "${TEST_DEVELOPER_DIR}" ]]; then
  TEST_DEVELOPER_DIR="$(/usr/bin/xcode-select -p 2>/dev/null || true)"
fi

# Use a virtual root and this test account as the expected owner. The helper
# still reads the real owner and mode with stat.
trust_root="${WORK}/trust-root"
users_toolchain="${trust_root}/Users/runner/zig"
bad_parent="${trust_root}/opt/group-writable"
bad_toolchain="${bad_parent}/zig"
good_toolchain="${trust_root}/usr/local/zig"
mkdir -p "${users_toolchain}" "${bad_toolchain}" "${good_toolchain}"
cp /usr/bin/true "${users_toolchain}/zig"
cp /usr/bin/true "${bad_toolchain}/zig"
cp /usr/bin/true "${good_toolchain}/zig"
chmod 0755 "${trust_root}" "${trust_root}/Users" "${trust_root}/Users/runner" \
  "${users_toolchain}" "${trust_root}/opt" "${bad_toolchain}" \
  "${trust_root}/usr" "${trust_root}/usr/local" "${good_toolchain}"
chmod 0775 "${bad_parent}"
if "${TRUSTED_ZIG_DIR}" "${users_toolchain}/zig" "${trust_root}" "$(id -u)" >/dev/null 2>&1; then
  fail "Zig below a fake /Users was trusted"
fi
if "${TRUSTED_ZIG_DIR}" "${bad_toolchain}/zig" "${trust_root}" "$(id -u)" >/dev/null 2>&1; then
  fail "Zig below a group-writable parent was trusted"
fi
acl_parent="${trust_root}/opt/acl-writable"
acl_toolchain="${acl_parent}/zig"
mkdir -p "${acl_toolchain}"
cp /usr/bin/true "${acl_toolchain}/zig"
chmod 0755 "${acl_parent}" "${acl_toolchain}"
/bin/chmod +a "everyone allow add_file,add_subdirectory,delete_child" "${acl_parent}"
if "${TRUSTED_ZIG_DIR}" "${acl_toolchain}/zig" "${trust_root}" "$(id -u)" >/dev/null 2>&1; then
  fail "Zig below a parent with a write ACL was trusted"
fi
unreadable_parent="${trust_root}/opt/acl-unreadable"
unreadable_toolchain="${unreadable_parent}/zig"
mkdir -p "${unreadable_toolchain}"
cp /usr/bin/true "${unreadable_toolchain}/zig"
chmod 0755 "${unreadable_parent}" "${unreadable_toolchain}"
/bin/chmod +a "everyone deny readsecurity" "${unreadable_parent}"
if "${TRUSTED_ZIG_DIR}" "${unreadable_toolchain}/zig" "${trust_root}" "$(id -u)" >/dev/null 2>&1; then
  /bin/chmod -N "${unreadable_parent}"
  fail "Zig below a parent whose ACL cannot be read was trusted"
fi
/bin/chmod -N "${unreadable_parent}"
trusted_dir="$("${TRUSTED_ZIG_DIR}" "${good_toolchain}/zig" "${trust_root}" "$(id -u)")"
good_toolchain_real="$(python3 -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "${good_toolchain}")"
[[ "${trusted_dir}" == "${good_toolchain_real}" ]] || fail "good root-owned Zig directory was refused"

profile="$(env HOME="${WORK}" MLXFAST_QWEN38_GOLDEN_DIR="${WORK}/goldens" MLXFAST_BASELINE_WORKSPACE="${WORK}/baseline" MLXFAST_BASELINE_CALIBRATION="${WORK}/cal.json" BENCHD_BIN_DIR="${WORK}/benchd" RUNNER_WORKSPACE="${WORK}/runner/work" python3 "${ROOT}/tools/seatbelt-profile.py" --build --official --tree "${ROOT}" --write-subpath "${WORK}/write")"
[[ "${profile}" == *'(deny network*)'* ]]
[[ "${profile}" == *"${WORK}/goldens"* ]]
[[ "${profile}" == *'(deny signal)'* && "${profile}" == *'(allow signal (target same-sandbox))'* ]] \
  || fail "build signal rules are missing"
[[ "${profile}" == *'(deny file-write-flags)'* && "${profile}" == *'(deny file-write-acl)'* ]] \
  || fail "build file flag or ACL denies are missing"
for rule in '(deny ipc-posix-shm*)' '(deny ipc-sysv*)' '(deny ipc-posix-sem*)' '(deny user-preference-write)'; do
  [[ "${profile}" == *"${rule}"* ]] || fail "build profile is missing ${rule}"
done
out="$(RUNNER_TEMP="${WORK}" "${SANDBOXED_BUILD}" --write "${WORK}/write" -- /bin/sh -c 'printf ok > "$1/result"' sh "${WORK}/write")"
[[ "${out}" == 'sandboxed-build.sh: PASS' ]] || fail "local result line changed"
[[ "$(<"${WORK}/write/result")" == ok ]]
out="$(env -u RUNNER_TEMP -u RUNNER_WORKSPACE -u RUNNER_ENVIRONMENT \
  -u MLXFAST_OFFICIAL_BENCHMARK_RUN "${SANDBOXED_BUILD}" \
  --write "${WORK}/write" -- /usr/bin/true)"
[[ "${out}" == 'sandboxed-build.sh: PASS' ]] || fail "local run without runner environment failed"
official_refusal='sandboxed-build.sh: FAIL: profile'
official_refusal_alt=''
if [[ "$(uname -s)" == Darwin ]] \
  && { ! env DEVELOPER_DIR="${TEST_DEVELOPER_DIR}" /usr/bin/xcrun -n -sdk macosx -f metal >/dev/null 2>&1 \
    || ! env DEVELOPER_DIR="${TEST_DEVELOPER_DIR}" /usr/bin/xcrun -n -sdk macosx -f metallib >/dev/null 2>&1 \
    || ! env DEVELOPER_DIR="${TEST_DEVELOPER_DIR}" /usr/bin/xcrun -n -f python3 >/dev/null 2>&1; }; then
  official_refusal='sandboxed-build.sh: FAIL: xcrun-warm'
  official_refusal_alt='sandboxed-build.sh: FAIL: toolchain-writable'
fi
out="$(DEVELOPER_DIR="${TEST_DEVELOPER_DIR}" MLXFAST_OFFICIAL_BENCHMARK_RUN=1 \
  "${SANDBOXED_BUILD}" --write "${WORK}/write" -- /usr/bin/true 2>&1 || true)"
[[ "${out}" == "${official_refusal}" \
  || ( -n "${official_refusal_alt}" && "${out}" == "${official_refusal_alt}" ) ]] \
  || fail "official refusal is not fixed: got '${out}', expected '${official_refusal}' or '${official_refusal_alt}'"

# Static order check: the official warm must precede trusted Python and helpers.
warm_line="$(grep -nF '/usr/bin/xcrun -n' "${SANDBOXED_BUILD}" | head -1 | cut -d: -f1)"
work_dir_line="$(grep -nF 'WORK_DIR="$(mktemp' "${SANDBOXED_BUILD}" | head -1 | cut -d: -f1)"
cleanup_trap_line="$(grep -nF 'trap cleanup EXIT' "${SANDBOXED_BUILD}" | head -1 | cut -d: -f1)"
first_trusted_line="$({
  grep -nF 'resolved_zig="$(real_path' "${SANDBOXED_BUILD}" || true
  grep -nF 'resolved_runner="$(real_path' "${SANDBOXED_BUILD}" || true
  grep -nF 'zig_toolchain_dir="$("${REPO_ROOT}/tools/' "${SANDBOXED_BUILD}" || true
  grep -nF 'python3 "${REPO_ROOT}/tools/seatbelt-profile.py"' "${SANDBOXED_BUILD}" || true
} | sort -n | head -1 | cut -d: -f1)"
[[ -n "${warm_line}" && -n "${work_dir_line}" && -n "${cleanup_trap_line}" \
  && -n "${first_trusted_line}" && ${warm_line} -lt ${work_dir_line} \
  && ${warm_line} -lt ${cleanup_trap_line} && ${warm_line} -lt ${first_trusted_line} ]] \
  || fail "xcrun warm order is unsafe"

out="$(RUNNER_TEMP="${WORK}" "${SANDBOXED_BUILD}" --write "${WORK}/write" -- /bin/sh -c 'printf BUILD-SECRET >&2; exit 9' 2>&1 || true)"
[[ "${out}" == 'sandboxed-build.sh: FAIL: command' ]] || fail "build output reached the public result"

out="$(RUNNER_TEMP="${WORK}" "${SANDBOXED_BUILD}" --write "${WORK}/write" -- \
  /bin/sh -c 'mkdir "${TMPDIR}/locked"; : > "${TMPDIR}/locked/file"; chmod 000 "${TMPDIR}/locked"')"
[[ "${out}" == 'sandboxed-build.sh: PASS' ]] || fail "restricted private temp cleanup failed"
[[ -z "$(find "${WORK}" -maxdepth 1 -name 'tensorfold-sandboxed-build.*' -print -quit)" ]] \
  || fail "private build temp tree remained"

descendant_file="${WORK}/write/descendant.pid"
out="$(RUNNER_TEMP="${WORK}" "${SANDBOXED_BUILD}" --write "${WORK}/write" -- /bin/sh -c '/bin/sleep 300 & printf "%s\n" "$!" > "$1"' sh "${descendant_file}")"
descendant_pid="$(cat "${descendant_file}")"
[[ "${out}" == 'sandboxed-build.sh: PASS' ]] && process_is_gone "${descendant_pid}" \
  || fail "build descendant survived cleanup"

normal_pid_file="${WORK}/normal.pid"
python3 "${ROOT}/tools/bounded-run.py" --output "${WORK}/normal.log" --session -- \
  /bin/sh -c '/bin/sleep 300 & printf "%s\n" "$!" > "$1"' sh "${normal_pid_file}"
normal_pid="$(cat "${normal_pid_file}")"
process_is_gone "${normal_pid}" || fail "normal exit left group child ${normal_pid}"

cancel_pid_file="${WORK}/cancel.pid"
python3 "${ROOT}/tools/bounded-run.py" --output "${WORK}/cancel.log" --session -- \
  /bin/sh -c '/bin/sleep 300 & printf "%s\n" "$!" > "$1"; wait' sh "${cancel_pid_file}" &
supervisor_pid=$!
wait_for_file "${cancel_pid_file}" || fail "SIGTERM probe did not start"
kill -TERM "${supervisor_pid}"
cancel_rc=0
wait "${supervisor_pid}" || cancel_rc=$?
[[ ${cancel_rc} -eq 143 ]] || fail "SIGTERM supervisor exit was ${cancel_rc}"
cancel_pid="$(cat "${cancel_pid_file}")"
process_is_gone "${cancel_pid}" || fail "SIGTERM left child ${cancel_pid}"

script_pid_file="${WORK}/write/script-cancel.pid"
cancel_out="${WORK}/script-cancel.out"
RUNNER_TEMP="${WORK}" "${SANDBOXED_BUILD}" --write "${WORK}/write" -- \
  /bin/sh -c '/bin/sleep 300 & printf "%s\n" "$!" > "$1"; wait' sh "${script_pid_file}" \
  > "${cancel_out}" 2>&1 &
script_pid=$!
wait_for_file "${script_pid_file}" || fail "script SIGTERM probe did not start"
kill -TERM "${script_pid}"
script_rc=0
wait "${script_pid}" || script_rc=$?
[[ ${script_rc} -eq 143 ]] || fail "script SIGTERM exit was ${script_rc}"
[[ "$(<"${cancel_out}")" == 'sandboxed-build.sh: FAIL: cancelled' ]] || fail "script SIGTERM result line changed"
script_child="$(cat "${script_pid_file}")"
process_is_gone "${script_child}" || fail "script SIGTERM left child ${script_child}"
[[ -z "$(find "${WORK}" -maxdepth 1 -name 'tensorfold-sandboxed-build.*' -print -quit)" ]] \
  || fail "script SIGTERM left the private build temp tree"

escaped_pid_file="${WORK}/escaped.pid"
if /bin/ps -A -o pid= >/dev/null 2>&1; then
  python3 "${ROOT}/tools/bounded-run.py" --output "${WORK}/escaped.log" --session -- \
    python3 -c 'import subprocess, sys, time; child = subprocess.Popen(["/bin/sleep", "300"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True); open(sys.argv[1], "w").write(str(child.pid)); time.sleep(0.3)' "${escaped_pid_file}"
  escaped_pid="$(cat "${escaped_pid_file}")"
  process_is_gone "${escaped_pid}" || fail "tracked session escape survived as ${escaped_pid}"
fi

# Tracker identity = (pid, kernel start time in microseconds): a reused PID is never signalled,
# a new descendant on a reused PID is still tracked after it is orphaned, and each signal re-checks.
python3 - "${ROOT}/tools/bounded-run.py" <<'PY' || fail "descendant tracker identity checks failed"
import importlib.util, os, signal, subprocess, sys
spec = importlib.util.spec_from_file_location("bounded_run", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Real processes: the kernel gives each one a distinct start identity.
me = mod.start_identity(os.getpid())
child = subprocess.Popen(["/bin/sleep", "5"])
try:
    other = mod.start_identity(child.pid)
    assert me is not None and other is not None and me != other, (me, other)
finally:
    child.kill(); child.wait()
assert mod.start_identity(child.pid) is None, "a reaped PID still has an identity"

UID = os.getuid()
state = {"table": {}, "ident": {}, "kppid": {}}
mod.process_table = lambda: dict(state["table"])
# Kernel view: identity from state["ident"]; kernel ppid from state["kppid"], else the ps ppid.
def fake_kernel(pid):
    if pid not in state["ident"]:
        return None
    ppid = state["kppid"].get(pid, state["table"].get(pid, (1, "S"))[0])
    return (state["ident"][pid], ppid, UID)
mod.kernel_info = fake_kernel
sent = []
mod.os = type("FakeOS", (), {"kill": staticmethod(lambda pid, sig: sent.append((pid, sig))), "getuid": staticmethod(lambda: UID)})()

# 1. PID 200 is a descendant, exits, and an unrelated process reuses it.
state["table"] = {100: (1, "S"), 200: (100, "S")}
state["ident"] = {100: (100, 1, 0), 200: (200, 1, 5)}
tracker = mod.DescendantTracker(100)
tracker.capture()
assert tracker.seen == {200: (200, 1, 5)}, tracker.seen
state["table"] = {100: (1, "S"), 200: (999, "S")}
state["ident"][200] = (200, 1, 6)
assert tracker.live() == {}, "reused PID is still live"
assert mod.stop_descendants(tracker) and sent == [], sent

# 2. A NEW descendant reuses PID 200, then is orphaned: it stays tracked and is signalled.
state["table"] = {100: (1, "S"), 200: (100, "S")}
state["ident"][200] = (200, 2, 0)
tracker.capture()
state["table"] = {100: (1, "S"), 200: (1, "S")}
assert tracker.live() == {200: (200, 2, 0)}, tracker.live()

# 4. ps still shows PID 300 under the build, but the PID was reused AFTER the ps snapshot
#    (start time in the future): it is never adopted, so it is never signalled.
state["table"] = {100: (1, "S"), 300: (100, "S")}
state["ident"][300] = (300, 10**10, 0)
tracker300 = mod.DescendantTracker(100)
tracker300.capture()
assert 300 not in tracker300.seen, tracker300.seen

# 5. A real descendant whose parent exits between ps and the kernel read (kernel ppid 1) is adopted.
state["table"] = {100: (1, "S"), 400: (100, "S")}
state["ident"][400] = (400, 4, 0)
state["kppid"][400] = 1
tracker400 = mod.DescendantTracker(100)
tracker400.capture()
assert tracker400.seen == {400: (400, 4, 0)}, tracker400.seen
state["table"] = {100: (1, "S"), 200: (1, "S")}

# 3. The identity changes between live() and the signal: no signal is sent.
real_live = tracker.live
def live_then_reuse():
    result = real_live()
    state["ident"][200] = (200, 3, 0)
    return result
tracker.live = live_then_reuse
sent.clear()
mod.stop_descendants(tracker)
assert sent == [], sent
PY

if [[ "$(uname -s)" == Darwin ]]; then
  if ! /usr/bin/sandbox-exec -p '(version 1) (allow default)' /usr/bin/true >/dev/null 2>&1; then
    if [[ "${MLXFAST_REQUIRE_METAL_TEST:-0}" == 1 ]]; then
      fail "outer Seatbelt guard unavailable"
    fi
    printf 'test-sandboxed-build.sh: SKIP: outer Seatbelt guard unavailable\n'
  else
    official_build() {
      exec env HOME="${WORK}" DEVELOPER_DIR="${TEST_DEVELOPER_DIR}" RUNNER_TEMP="${WORK}" \
        MLXFAST_QWEN38_GOLDEN_DIR="${WORK}/goldens" \
        MLXFAST_BASELINE_WORKSPACE="${WORK}/baseline" MLXFAST_BASELINE_CALIBRATION="${WORK}/cal.json" \
        BENCHD_BIN_DIR="${WORK}/benchd" RUNNER_WORKSPACE="${WORK}/runner/work" \
        MLXFAST_OFFICIAL_BENCHMARK_RUN=1 "${SANDBOXED_BUILD}" "$@"
    }

    if ! env DEVELOPER_DIR="${TEST_DEVELOPER_DIR}" \
      /usr/bin/xcrun -sdk macosx -f metal >/dev/null 2>&1; then
      if [[ "${MLXFAST_REQUIRE_METAL_TEST:-0}" == 1 ]]; then
        fail "metal toolchain absent"
      fi
      out="$(official_build --write "${WORK}/write" -- /usr/bin/true 2>&1 || true)"
      [[ "${out}" == 'sandboxed-build.sh: FAIL: xcrun-warm' \
        || "${out}" == 'sandboxed-build.sh: FAIL: toolchain-writable' ]] \
        || fail "missing Metal toolchain refusal changed: got '${out}', expected 'sandboxed-build.sh: FAIL: xcrun-warm' or 'sandboxed-build.sh: FAIL: toolchain-writable'"
      printf 'test-sandboxed-build.sh: SKIP: metal toolchain absent (official build section)\n'
    else

  if env -u DEVELOPER_DIR /usr/bin/xcode-select -p >/dev/null 2>&1 \
    && env -u DEVELOPER_DIR /usr/bin/xcrun -sdk macosx -f metal >/dev/null 2>&1; then
    out="$(env -u DEVELOPER_DIR HOME="${WORK}" RUNNER_TEMP="${WORK}" \
      MLXFAST_QWEN38_GOLDEN_DIR="${WORK}/goldens" \
      MLXFAST_BASELINE_WORKSPACE="${WORK}/baseline" MLXFAST_BASELINE_CALIBRATION="${WORK}/cal.json" \
      BENCHD_BIN_DIR="${WORK}/benchd" RUNNER_WORKSPACE="${WORK}/runner/work" \
      MLXFAST_OFFICIAL_BENCHMARK_RUN=1 "${SANDBOXED_BUILD}" \
      --write "${WORK}/write" -- /usr/bin/true)"
    [[ "${out}" == 'sandboxed-build.sh: PASS' ]] \
      || fail "official build with DEVELOPER_DIR unset failed"
  elif [[ "${MLXFAST_REQUIRE_METAL_TEST:-0}" == 1 ]]; then
    fail "xcode-select Metal toolchain absent"
  else
    printf 'test-sandboxed-build.sh: SKIP: xcode-select Metal toolchain absent\n'
  fi

  current_runner="${WORK}/actions-runner-current"
  other_runner="${WORK}/actions-runner-other"
  current_temp="${current_runner}/_work/_temp"
  current_workspace="${current_runner}/_work/repo/repo"
  current_runner_file="${current_runner}/runner-owned"
  sibling_temp_secret="${current_temp}/sibling-job/secret"
  sibling_work_secret="${current_runner}/_work/sibling-repo/sibling-repo/secret"
  other_temp_secret="${other_runner}/_work/_temp/other-secret"
  other_work_secret="${other_runner}/_work/other-repo/other-secret"
  custom_peer_secret="${WORK}/peer-with-custom-layout/private/secret"
  mkdir -p "${current_temp}" "${current_workspace}" \
    "$(dirname -- "${sibling_temp_secret}")" "$(dirname -- "${sibling_work_secret}")" \
    "$(dirname -- "${other_temp_secret}")" "$(dirname -- "${other_work_secret}")" \
    "$(dirname -- "${custom_peer_secret}")"
  printf 'runner-owned\n' > "${current_runner_file}"
  printf 'sibling-temp-secret\n' > "${sibling_temp_secret}"
  printf 'sibling-work-secret\n' > "${sibling_work_secret}"
  printf 'other-temp-secret\n' > "${other_temp_secret}"
  printf 'other-work-secret\n' > "${other_work_secret}"
  printf 'custom-peer-secret\n' > "${custom_peer_secret}"
  account_temp="$(/usr/bin/getconf DARWIN_USER_TEMP_DIR)"
  account_temp_real="$(python3 -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "${account_temp}")"
  xcrun_db="${account_temp_real%/}/xcrun_db"
  XCRUN_DB_TO_RESTORE="${xcrun_db}"
  XCRUN_DB_BACKUP="${WORK}/xcrun-db.backup"
  cp -p -- "${xcrun_db}" "${XCRUN_DB_BACKUP}" || fail "xcrun database backup failed"
  ACCOUNT_TMP_CANARY="${account_temp_real%/}/${probe_tag}.account-canary"
  XCRUN_DB_CANARY="${xcrun_db}.canary"
  printf 'account-temp-secret\n' > "${ACCOUNT_TMP_CANARY}"
  printf 'xcrun-canary-secret\n' > "${XCRUN_DB_CANARY}"
  planted_db="${WORK}/planted-xcrun-db"
  printf 'planted-xcrun-db\n' > "${planted_db}"
  cp "${planted_db}" "${xcrun_db}"
  xcrun_inode_before="$(/usr/bin/stat -f '%i' "${xcrun_db}")"
  xcrun_poll_ready="${WORK}/xcrun-poll.ready"
  xcrun_poll_stop="${WORK}/xcrun-poll.stop"
  xcrun_poll_missing="${WORK}/xcrun-poll.missing"
  (
    printf 'ready\n' > "${xcrun_poll_ready}"
    while [[ ! -e "${xcrun_poll_stop}" ]]; do
      [[ -e "${xcrun_db}" ]] || { : > "${xcrun_poll_missing}"; break; }
      /bin/sleep 0.01
    done
  ) &
  XCRUN_POLL_PID=$!
  wait_for_file "${xcrun_poll_ready}" || fail "xcrun database poll did not start"
  xcrun_profile="${WORK}/write/xcrun-profile.sb"
  xcrun_result="${WORK}/write/xcrun.result"
  out="$(official_build --write "${WORK}/write" -- /bin/sh -c '
    canary_rc=0; write_rc=0
    /bin/cat "$1" >/dev/null 2>&1 || canary_rc=$?
    printf changed > "$2" 2>/dev/null || write_rc=$?
    /bin/cp "${TMPDIR}/../profile.sb" "$3" || exit 7
    printf "%s %s\n" "${canary_rc}" "${write_rc}" > "$4"' \
    sh "${XCRUN_DB_CANARY}" "${xcrun_db}" "${xcrun_profile}" "${xcrun_result}")"
  : > "${xcrun_poll_stop}"
  wait "${XCRUN_POLL_PID}" || fail "xcrun database poll failed"
  XCRUN_POLL_PID=""
  [[ ! -e "${xcrun_poll_missing}" ]] || fail "xcrun database was absent during warm"
  xcrun_inode_after="$(/usr/bin/stat -f '%i' "${xcrun_db}")"
  [[ "${xcrun_inode_before}" != "${xcrun_inode_after}" ]] \
    || fail "xcrun database inode did not change during warm"
  [[ "${out}" == 'sandboxed-build.sh: PASS' ]] || fail "xcrun profile probe did not run"
  read -r xcrun_canary_rc xcrun_write_rc < "${xcrun_result}"
  [[ "${xcrun_canary_rc}" != 0 ]] || fail "xcrun canary was readable"
  [[ "${xcrun_write_rc}" != 0 ]] || fail "xcrun database was writable"
  xcrun_rule="(allow file-read* (literal \"${xcrun_db}\"))"
  [[ "$(grep -Fxc "${xcrun_rule}" "${xcrun_profile}")" == 1 ]] \
    || fail "xcrun database read rule count is wrong"
  if grep -F 'file-write' "${xcrun_profile}" | grep -Fq "${xcrun_db}"; then
    fail "xcrun database has a write rule"
  fi
  if grep -Fq "(allow file-write* (subpath \"${account_temp_real%/}\"))" "${xcrun_profile}"; then
    fail "account temp has a write rule"
  fi
  if grep -F 'file-write' "${xcrun_profile}" \
    | grep -Fq -e "\"${xcrun_db}\"" -e "(subpath \"${account_temp_real%/}\")"; then
    fail "xcrun database or account temp is named by a write rule"
  fi
  cmp -s "${planted_db}" "${xcrun_db}" && fail "planted xcrun database was kept"

  metal_dir="${WORK}/write/metal"
  metal_modules="${metal_dir}/modules"
  mkdir -p "${metal_modules}"
  printf '%s\n' '#include <metal_stdlib>' 'using namespace metal;' \
    'kernel void k(device float* a [[buffer(0)]]) {}' > "${metal_dir}/k.metal"
  out="$(official_build --write "${WORK}/write" -- /usr/bin/xcrun -sdk macosx metal \
    -c "${metal_dir}/k.metal" -fmodules-cache-path="${metal_modules}" \
    -o "${metal_dir}/k.air")"
  [[ "${out}" == 'sandboxed-build.sh: PASS' && -s "${metal_dir}/k.air" ]] \
    || fail "sandboxed Metal compile failed"

  runner_isolation_result="${WORK}/write/runner-isolation.rc"
  out="$(env HOME="${WORK}" DEVELOPER_DIR="${TEST_DEVELOPER_DIR}" \
    RUNNER_TEMP="${current_temp}" RUNNER_WORKSPACE="${current_workspace}" \
    MLXFAST_QWEN38_GOLDEN_DIR="${WORK}/goldens" MLXFAST_BASELINE_WORKSPACE="${WORK}/baseline" \
    MLXFAST_BASELINE_CALIBRATION="${WORK}/cal.json" BENCHD_BIN_DIR="${WORK}/benchd" \
    MLXFAST_OFFICIAL_BENCHMARK_RUN=1 "${SANDBOXED_BUILD}" --write "${WORK}/write" -- \
    /bin/sh -c '
      temp_rc=0; work_rc=0; account_rc=0; custom_rc=0; root_rc=0
      sibling_temp_rc=0; sibling_work_rc=0
      /bin/cat "$1" >/dev/null 2>&1 || temp_rc=$?
      /bin/cat "$2" >/dev/null 2>&1 || work_rc=$?
      /bin/cat "$3" >/dev/null 2>&1 || account_rc=$?
      /bin/cat "$4" >/dev/null 2>&1 || custom_rc=$?
      /bin/cat "$5" >/dev/null 2>&1 || root_rc=$?
      /bin/cat "$6" >/dev/null 2>&1 || sibling_temp_rc=$?
      /bin/cat "$7" >/dev/null 2>&1 || sibling_work_rc=$?
      /bin/cat "$8/README.md" >/dev/null || exit 7
      printf private > "${TMPDIR}/private" || exit 8
      /bin/cat "${TMPDIR}/private" >/dev/null || exit 9
      printf "%s %s %s %s %s %s %s\n" "${temp_rc}" "${work_rc}" "${account_rc}" \
        "${custom_rc}" "${root_rc}" "${sibling_temp_rc}" "${sibling_work_rc}" > "$9"' \
    sh "${other_temp_secret}" "${other_work_secret}" "${ACCOUNT_TMP_CANARY}" \
    "${custom_peer_secret}" "${current_runner_file}" "${sibling_temp_secret}" \
    "${sibling_work_secret}" "${ROOT}" "${runner_isolation_result}")"
  [[ "${out}" == 'sandboxed-build.sh: PASS' ]] || fail "runner isolation probe did not run"
  read -r other_temp_rc other_work_rc account_temp_rc custom_peer_rc runner_root_rc \
    sibling_temp_rc sibling_work_rc < "${runner_isolation_result}"
  [[ "${other_temp_rc}" != 0 && "${other_work_rc}" != 0 && "${account_temp_rc}" != 0 \
    && "${custom_peer_rc}" != 0 && "${runner_root_rc}" != 0 \
    && "${sibling_temp_rc}" != 0 && "${sibling_work_rc}" != 0 ]] \
    || fail "runner account read isolation is wrong"

  fake_home="${WORK}/zig-home"
  fake_toolchain="${fake_home}/tools/zig-good"
  peer_tree="${fake_home}/peer-tree"
  mkdir -p "${fake_toolchain}" "${peer_tree}"
  ln -s /usr/bin/python3 "${fake_toolchain}/zig"
  ln -s "${peer_tree}" "${fake_toolchain}/lib"
  printf 'peer-secret\n' > "${peer_tree}/secret"
  zig_profile="${WORK}/write/zig-profile.sb"
  zig_peer_result="${WORK}/write/zig-peer.rc"
  out="$(env HOME="${fake_home}" DEVELOPER_DIR="${TEST_DEVELOPER_DIR}" \
    ZIG="${fake_toolchain}/zig" RUNNER_TEMP="${current_temp}" \
    RUNNER_WORKSPACE="${current_workspace}" MLXFAST_QWEN38_GOLDEN_DIR="${WORK}/goldens" \
    MLXFAST_BASELINE_WORKSPACE="${WORK}/baseline" MLXFAST_BASELINE_CALIBRATION="${WORK}/cal.json" \
    BENCHD_BIN_DIR="${WORK}/benchd" MLXFAST_OFFICIAL_BENCHMARK_RUN=1 \
    MLXFAST_BUILD_ZIG_BIN="${fake_toolchain}/zig" \
    "${SANDBOXED_BUILD}" --write "${WORK}/write" -- "${fake_toolchain}/zig" -c '
import os, shutil, sys
shutil.copyfile(os.path.join(os.environ["TMPDIR"], "..", "profile.sb"), sys.argv[1])
try:
    open(sys.argv[2], "rb").read()
    rc = 0
except OSError:
    rc = 1
open(sys.argv[3], "w").write(str(rc))
' "${zig_profile}" "${peer_tree}/secret" "${zig_peer_result}")"
  [[ "${out}" == 'sandboxed-build.sh: PASS' ]] || fail "trusted Zig profile probe did not run"
  [[ "$(<"${zig_peer_result}")" != 0 ]] || fail "Zig lib symlink exposed its peer tree"
  grep -Fq '(allow file-read* (subpath "/usr/bin"))' "${zig_profile}" \
    || fail "trusted Zig directory was not readable"
  if grep -Fq '(allow file-read* (literal "/usr/bin/python3"))' "${zig_profile}" \
    || grep -Fq "${fake_toolchain}" "${zig_profile}" \
    || grep -Fq "${peer_tree}" "${zig_profile}"; then
    fail "Zig profile allowed more than its resolved toolchain directory"
  fi

  printf 'canary-original\n' > "${VAR_TMP_CANARY}"
  var_tmp_rc="${WORK}/write/var-tmp.rc"
  out="$(official_build --write "${WORK}/write" -- /bin/sh -c '
    first=0; second=0; third=0
    printf changed > "$1" 2>/dev/null || first=$?
    printf new > "$2" 2>/dev/null || second=$?
    printf sh-thd > "$3" 2>/dev/null || third=$?
    printf "%s %s %s\n" "$first" "$second" "$third" > "$4"' \
    sh "${VAR_TMP_CANARY}" "${VAR_TMP_NEW}" "${VAR_TMP_SH_THD}" "${var_tmp_rc}")"
  [[ "${out}" == 'sandboxed-build.sh: PASS' ]] || fail "var tmp canary probe did not run"
  read -r first_rc second_rc third_rc < "${var_tmp_rc}"
  [[ "${first_rc}" != 0 && "${second_rc}" != 0 && "${third_rc}" != 0 ]] \
    || fail "build wrote in /private/var/tmp"
  [[ "$(<"${VAR_TMP_CANARY}")" == canary-original ]] || fail "build changed the var tmp canary"
  [[ ! -e "${VAR_TMP_NEW}" && ! -e "${VAR_TMP_SH_THD}" ]] || fail "build created a var tmp file"

  heredoc_result="${WORK}/write/heredoc.result"
  out="$(official_build --write "${WORK}/write" -- /bin/sh -c 'cat > "$1" <<EOF
heredoc-ok
EOF' sh "${heredoc_result}")"
  [[ "${out}" == 'sandboxed-build.sh: PASS' && "$(<"${heredoc_result}")" == heredoc-ok ]] \
    || fail "sandboxed here document failed"

  peer_path_file="${WORK}/write/peer.path"
  peer_release="${WORK}/write/peer.release"
  peer_out="${WORK}/peer.out"
  official_build --write "${WORK}/write" -- /bin/sh -c '
    printf peer-secret > "${TMPDIR}/peer-secret"
    printf "%s\n" "${TMPDIR}/peer-secret" > "$1"
    while test ! -e "$2"; do /bin/sleep 0.02; done' sh "${peer_path_file}" "${peer_release}" \
    > "${peer_out}" 2>&1 &
  SIBLING_PID=$!
  wait_for_file "${peer_path_file}" || fail "peer temp probe did not start"
  peer_read="${WORK}/write/peer.read"
  out="$(official_build --write "${WORK}/write" -- /bin/sh -c '
    if /bin/cat "$(/bin/cat "$1")" > "$2" 2>/dev/null; then exit 9; fi' \
    sh "${peer_path_file}" "${peer_read}")"
  [[ "${out}" == 'sandboxed-build.sh: PASS' && ! -s "${peer_read}" ]] \
    || fail "sandboxed jobs saw each other's temp files"
  : > "${peer_release}"
  wait "${SIBLING_PID}" || fail "peer temp writer failed"
  SIBLING_PID=""
  [[ "$(tail -1 "${peer_out}")" == 'sandboxed-build.sh: PASS' ]] || fail "peer temp writer result changed"

  sandbox_cancel_info="${WORK}/write/sandbox-cancel.info"
  sandbox_cancel_out="${WORK}/sandbox-cancel.out"
  official_build --write "${WORK}/write" -- /bin/sh -c '
    /bin/sleep 300 & printf "%s\n%s\n" "$!" "$TMPDIR" > "$1"; wait' \
    sh "${sandbox_cancel_info}" > "${sandbox_cancel_out}" 2>&1 &
  SIBLING_PID=$!
  wait_for_file "${sandbox_cancel_info}" || fail "sandbox cancellation probe did not start"
  sandbox_private="$(sed -n '2p' "${sandbox_cancel_info}")"
  kill -TERM "${SIBLING_PID}"
  sandbox_cancel_rc=0
  wait "${SIBLING_PID}" || sandbox_cancel_rc=$?
  SIBLING_PID=""
  [[ ${sandbox_cancel_rc} -eq 143 ]] || fail "sandbox cancellation exit was ${sandbox_cancel_rc}"
  [[ "$(<"${sandbox_cancel_out}")" == 'sandboxed-build.sh: FAIL: cancelled' ]] \
    || fail "sandbox cancellation result changed"
  [[ ! -e "${sandbox_private}" ]] || fail "sandbox cancellation left its private temp tree"

  /bin/sleep 300 &
  SIBLING_PID=$!
  out="$(env HOME="${WORK}" DEVELOPER_DIR="${TEST_DEVELOPER_DIR}" RUNNER_TEMP="${WORK}" \
    MLXFAST_QWEN38_GOLDEN_DIR="${WORK}/goldens" \
    MLXFAST_BASELINE_WORKSPACE="${WORK}/baseline" MLXFAST_BASELINE_CALIBRATION="${WORK}/cal.json" \
    BENCHD_BIN_DIR="${WORK}/benchd" RUNNER_WORKSPACE="${WORK}/runner/work" MLXFAST_OFFICIAL_BENCHMARK_RUN=1 \
    "${SANDBOXED_BUILD}" --write "${WORK}/write" -- \
    /bin/sh -c 'kill -TERM "$1" 2>/dev/null; test "$?" -ne 0' sh "${SIBLING_PID}")"
  [[ "${out}" == 'sandboxed-build.sh: PASS' ]] || fail "sandboxed sibling-signal probe failed"
  kill -0 "${SIBLING_PID}" 2>/dev/null || fail "build signalled a sibling process"
  kill "${SIBLING_PID}" 2>/dev/null || true
  wait "${SIBLING_PID}" 2>/dev/null || true
  SIBLING_PID=""

  # Control: outside the sandbox this account can set uchg, so a refusal
  # below comes from the build profile.
  : > "${WORK}/flag-control"
  /usr/bin/chflags uchg "${WORK}/flag-control" || fail "flag control could not set uchg"
  /usr/bin/chflags nouchg "${WORK}/flag-control"
  flag_rc_file="${WORK}/write/flag.rc"
  out="$(env HOME="${WORK}" DEVELOPER_DIR="${TEST_DEVELOPER_DIR}" RUNNER_TEMP="${WORK}" \
    MLXFAST_QWEN38_GOLDEN_DIR="${WORK}/goldens" \
    MLXFAST_BASELINE_WORKSPACE="${WORK}/baseline" MLXFAST_BASELINE_CALIBRATION="${WORK}/cal.json" \
    BENCHD_BIN_DIR="${WORK}/benchd" RUNNER_WORKSPACE="${WORK}/runner/work" MLXFAST_OFFICIAL_BENCHMARK_RUN=1 \
    "${SANDBOXED_BUILD}" --write "${WORK}/write" -- \
    /bin/sh -c ': > "${TMPDIR}/flagged" || exit 3
      flag_rc=0; /usr/bin/chflags uchg "${TMPDIR}/flagged" 2>/dev/null || flag_rc=$?
      acl_rc=0; /bin/chmod +a "everyone deny delete" "${TMPDIR}/flagged" 2>/dev/null || acl_rc=$?
      printf "%s %s %s\n" "${flag_rc}" "${acl_rc}" "$(/usr/bin/stat -f %f "${TMPDIR}/flagged")" > "$1"' \
    sh "${flag_rc_file}")"
  [[ "${out}" == 'sandboxed-build.sh: PASS' ]] || fail "sandboxed file flag probe did not run"
  read -r flag_rc acl_rc flag_bits < "${flag_rc_file}"
  [[ "${flag_rc}" != 0 && $(( flag_bits & 2 )) -eq 0 ]] || fail "build set uchg on a file in its tmp"
  [[ "${acl_rc}" != 0 ]] || fail "build set an ACL on a file in its tmp"
  [[ -z "$(find "${WORK}" -maxdepth 1 -name 'tensorfold-sandboxed-build.*' -print -quit)" ]] \
    || fail "file flag probe left the private build temp tree"
    fi
  fi
fi
printf 'test-sandboxed-build.sh: PASS\n'
