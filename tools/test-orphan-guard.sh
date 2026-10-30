#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
GUARD="${ROOT}/tools/orphan-guard.sh"

make_runner() {
  local output="$1"
  mkdir -p "$(dirname -- "${output}")"
  /usr/bin/xcrun -sdk macosx clang -x c -o "${output}" - <<'C'
#include <sys/wait.h>
#include <unistd.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc < 2) return 2;
    pid_t child = fork();
    if (child < 0) return 2;
    if (child == 0) {
        char **args = calloc((size_t)argc + 1, sizeof(char *));
        if (!args) _exit(2);
        args[0] = "/bin/bash";
        for (int i = 1; i < argc; i++) args[i] = argv[i];
        execv(args[0], args);
        _exit(2);
    }
    int status = 0;
    while (waitpid(child, &status, 0) < 0) {}
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
    return 2;
}
C
}

MODE="${1:-}"
if [[ "${MODE}" != "--under-runner" && "${MODE}" != "--kill-case" ]]; then
  work="$(mktemp -d)"
  /bin/sleep 3600 &
  old_pid=$!
  printf '%s\n' '#include <fcntl.h>' '#include <unistd.h>' \
    'int main(int c, char **v) { if (c > 1) close(open(v[1], O_WRONLY | O_CREAT, 0600)); sleep(3600); return 0; }' \
    | /usr/bin/xcrun -sdk macosx clang -x c - -o "${work}/old-unlinked"
  "${work}/old-unlinked" "${work}/old-unlinked.ready" &
  old_unlinked_pid=$!
  while [[ ! -e "${work}/old-unlinked.ready" ]]; do sleep 0.01; done
  rm "${work}/old-unlinked"
  sleep 1.1
  make_runner "${work}/current-runner/bin/Runner.Worker"
  mkdir -p "${work}/runner-jobs/_work/repo"
  export ORPHAN_GUARD_TEST_WORK="${work}" ORPHAN_GUARD_OLD_PID="${old_pid}"
  export ORPHAN_GUARD_OLD_UNLINKED_PID="${old_unlinked_pid}"
  export RUNNER_WORKSPACE="${work}/runner-jobs/_work/repo"
  # A new process, not exec: exec keeps this shell's start time, which is
  # older than the "old" processes above.
  "${work}/current-runner/bin/Runner.Worker" "$0" --under-runner
  exit $?
fi

if [[ "${MODE}" == "--kill-case" ]]; then
  WORK="$2"
  OLD_PID=""
  REMOVE_WORK=0
else
  WORK="${ORPHAN_GUARD_TEST_WORK}"
  OLD_PID="${ORPHAN_GUARD_OLD_PID}"
  OLD_UNLINKED_PID="${ORPHAN_GUARD_OLD_UNLINKED_PID}"
  REMOVE_WORK=1
fi
PIDS=("${OLD_PID}" "${OLD_UNLINKED_PID:-}")
TMP_FILES=("")
LAUNCH_LABELS=("")
cleanup() {
  local rc=$? cleanup_rc=0 label pid
  trap - EXIT
  for label in "${LAUNCH_LABELS[@]}"; do
    [[ -n "${label}" ]] || continue
    /bin/launchctl remove "${label}" >/dev/null 2>&1 || true
    if /bin/launchctl list "${label}" >/dev/null 2>&1; then
      printf 'test-orphan-guard.sh: FAIL: could not remove launch job %s\n' \
        "${label}" >&2
      cleanup_rc=1
    fi
  done
  for pid in "${PIDS[@]}"; do
    [[ -n "${pid}" ]] || continue
    kill -KILL "${pid}" 2>/dev/null || true
  done
  for path in "${TMP_FILES[@]}"; do
    [[ -n "${path}" ]] || continue
    rm -f -- "${path}"
  done
  if [[ "${REMOVE_WORK}" -eq 1 ]]; then
    rm -rf "${WORK}"
  fi
  (( rc != 0 )) && exit "${rc}"
  exit "${cleanup_rc}"
}
trap cleanup EXIT

fail() { printf 'test-orphan-guard.sh: FAIL: %s\n' "$*" >&2; exit 1; }
"${GUARD}" recheck-self-test || fail "identity recheck function self-test failed"
if ! /bin/ps -A -o pid=,uid=,stat= >/dev/null 2>&1; then
  printf 'test-orphan-guard.sh: SKIP: process table is blocked for the fake Runner.Worker\n'
  exit 0
fi

process_is_gone() {
  local pid="$1" state tries=0
  while (( tries < 100 )); do
    if ! kill -0 "${pid}" 2>/dev/null; then return 0; fi
    state="$(ps -o stat= -p "${pid}" 2>/dev/null | tr -d '[:space:]' || true)"
    [[ "${state}" == Z* ]] && return 0
    sleep 0.02
    tries=$((tries + 1))
  done
  return 1
}
wait_for_ppid_one() {
  local pid="$1" tries=0
  while (( tries < 200 )); do
    [[ "$(ps -o ppid= -p "${pid}" 2>/dev/null | tr -d '[:space:]')" == 1 ]] && return 0
    sleep 0.02
    tries=$((tries + 1))
  done
  return 1
}
wait_for_exe() {
  local pid="$1" expected="$2" actual tries=0 expected_dir
  # The kernel reports resolved paths (/var -> /private/var).
  if expected_dir="$(cd -P -- "$(dirname -- "${expected}")" 2>/dev/null && pwd -P)"; then
    expected="${expected_dir}/$(basename -- "${expected}")"
  fi
  while (( tries < 200 )); do
    actual="$(kernel_path "${pid}" 2>/dev/null || true)"
    # OS binaries such as jsc run from the sealed OS cryptex; the kernel reports that path.
    [[ "${actual}" == "${expected}" \
      || "${actual}" == "/System/Volumes/Preboot/Cryptexes/OS${expected}" ]] && return 0
    sleep 0.02
    tries=$((tries + 1))
  done
  return 1
}
kernel_path() {
  /usr/bin/python3 - "$1" <<'PY'
import ctypes
import sys
libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
libproc.proc_pidpath.argtypes = (ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32)
libproc.proc_pidpath.restype = ctypes.c_int
buffer = ctypes.create_string_buffer(4096)
size = libproc.proc_pidpath(int(sys.argv[1]), buffer, len(buffer))
if size <= 0:
    raise SystemExit(1)
print(buffer.value.decode())
PY
}
launch_pid() {
  /bin/launchctl list | awk -v label="$1" '$3 == label { print $1; exit }'
}
wait_for_launch_pid() {
  local label="$1" pid tries=0
  while (( tries < 200 )); do
    pid="$(launch_pid "${label}")"
    if [[ "${pid}" =~ ^[0-9]+$ ]]; then
      printf '%s\n' "${pid}"
      return 0
    fi
    sleep 0.02
    tries=$((tries + 1))
  done
  return 1
}
remove_launch_job() {
  local label="$1" pid="$2"
  remove_launch_label "${label}"
  process_is_gone "${pid}" || fail "launch job process still exists: ${label}"
}
remove_launch_label() {
  local label="$1"
  /bin/launchctl remove "${label}" || fail "could not remove launch job ${label}"
  if /bin/launchctl list "${label}" >/dev/null 2>&1; then
    fail "launch job still exists: ${label}"
  fi
}
spawn_orphan() {
  local pid_file="$1"
  /usr/bin/python3 - "${pid_file}" <<'PY'
import os
import sys
import time

if os.fork():
    raise SystemExit(0)
os.setsid()
if os.fork():
    raise SystemExit(0)
with open(sys.argv[1], "w") as handle:
    handle.write(str(os.getpid()) + "\n")
time.sleep(300)
PY
  local tries=0
  while [[ ! -s "${pid_file}" && ${tries} -lt 200 ]]; do
    sleep 0.02
    tries=$((tries + 1))
  done
  [[ -s "${pid_file}" ]] || fail "orphan did not write its pid"
}
build_sleeper() {
  printf '%s\n' '#include <unistd.h>' 'int main(void) { sleep(300); return 0; }' \
    | /usr/bin/xcrun -sdk macosx clang -x c - -o "$1" || fail "could not build the test sleeper"
}
build_pid_sleeper() {
  printf '%s\n' '#include <fcntl.h>' '#include <stdio.h>' '#include <unistd.h>' \
    'int main(int c, char **v) { if (c != 2) return 2; FILE *f = fopen(v[1], "w"); if (!f) return 2; fprintf(f, "%d\n", getpid()); fclose(f); sleep(300); return 0; }' \
    | /usr/bin/xcrun -sdk macosx clang -x c - -o "$1" \
    || fail "could not build the launch test sleeper"
}

spawn_exec_orphan() {
  local pid_file="$1"
  shift
  /usr/bin/python3 - "${pid_file}" "$@" <<'PY'
import os
import sys

pid_file = sys.argv[1]
args = sys.argv[2:]
if os.fork():
    raise SystemExit(0)
os.setsid()
if os.fork():
    raise SystemExit(0)
with open(pid_file, "w") as handle:
    handle.write(str(os.getpid()) + "\n")
os.execv(args[0], args)
PY
  local tries=0
  while [[ ! -s "${pid_file}" && ${tries} -lt 200 ]]; do
    sleep 0.02
    tries=$((tries + 1))
  done
  [[ -s "${pid_file}" ]] || fail "exec orphan did not write its pid"
}

spawn_spoofed_exec_orphan() {
  local pid_file="$1" executable="$2" argv_zero="$3"
  /usr/bin/python3 - "${pid_file}" "${executable}" "${argv_zero}" <<'PY'
import os
import sys

if os.fork():
    raise SystemExit(0)
os.setsid()
if os.fork():
    raise SystemExit(0)
with open(sys.argv[1], "w") as handle:
    handle.write(str(os.getpid()) + "\n")
os.execv(sys.argv[2], [sys.argv[3]])
PY
  local tries=0
  while [[ ! -s "${pid_file}" && ${tries} -lt 200 ]]; do
    sleep 0.02
    tries=$((tries + 1))
  done
  [[ -s "${pid_file}" ]] || fail "spoofed exec orphan did not write its pid"
}

if [[ "${MODE}" == "--kill-case" ]]; then
  expected_root="$3"
  /bin/sleep 300 &
  helper_pid=$!
  PIDS+=("${helper_pid}")
  job_start="$(${GUARD} job-start)" || fail "updated runner job-start failed"
  set +e
  mismatch_out="$(${GUARD} kill --job-start "${job_start}" \
    --runner-root "${WORK}/wrong-runner" 2>&1)"
  mismatch_rc=$?
  set -e
  [[ ${mismatch_rc} -ne 0 && "${mismatch_out}" == *'runner root mismatch'* ]] \
    || fail "mismatching runner root was not clearly refused: ${mismatch_out}"
  kill -0 "${helper_pid}" 2>/dev/null \
    || fail "runner root mismatch killed the updated runner helper"
  "${GUARD}" kill --job-start "${job_start}" >/dev/null \
    || fail "updated runner kill mode reported survivors"
  process_is_gone "${helper_pid}" \
    || fail "updated runner kill mode left its own-tree helper"
  [[ ! -e "${expected_root}/_work" ]] \
    || fail "updated runner test put the work folder inside the installation"
  exit 0
fi

"${GUARD}" self-test || fail "kernel start or identity recheck self-test failed"
show_new_orphans() {
  # Diagnose on failure: this account's PPID-1 processes (the hosted runner has its own per-user agents).
  ps -axo pid=,ppid=,uid=,lstart=,command= | awk -v uid="$(id -u)" '$3 == uid && $2 == 1' >&2 || true
}
if [[ "${RUNNER_ENVIRONMENT:-}" == github-hosted && "${ORPHAN_GUARD_TEST_ON_HOSTED:-0}" != 1 ]]; then
  # A GitHub-hosted VM is a fresh login session: macOS per-user agents keep
  # starting as PPID-1 processes for minutes after boot, so the clean, kill
  # and measure cases cannot see a quiet account. The self-tests above ran;
  # the full test runs on the ranked box. Keep the list as a diagnostic.
  show_new_orphans
  printf 'test-orphan-guard.sh: SKIP: process cases need a quiet account (GitHub-hosted runner)\n'
  exit 0
fi
JOB_START="$(${GUARD} job-start)" || fail "job-start did not find Runner.Worker"
"${GUARD}" check --job-start "${JOB_START}" || { show_new_orphans; fail "clean check failed"; }
old_kill_out="$("${GUARD}" kill --job-start "${JOB_START}" 2>&1)" \
  || fail "old unlinked process broke kill mode: ${old_kill_out}"
kill -0 "${OLD_UNLINKED_PID}" 2>/dev/null \
  || fail "kill mode stopped the old unlinked process"

# A runner or official environment cannot use the local fallback when no
# Runner.Worker exists in its ancestry.
no_runner_label="com.gumbii.orphan-guard.no-runner.$$.$RANDOM"
no_runner_out="${WORK}/no-runner.out"
no_runner_rc="${WORK}/no-runner.rc"
LAUNCH_LABELS+=("${no_runner_label}")
/bin/launchctl submit -l "${no_runner_label}" -- /bin/bash -c \
  'GITHUB_ACTIONS=1 "$1" check --job-start 0 >"$2" 2>&1; printf "%s\n" "$?" >"$3"' \
  bash "${GUARD}" "${no_runner_out}" "${no_runner_rc}" \
  || fail "could not submit no-Runner.Worker test"
tries=0
while [[ ! -s "${no_runner_rc}" && ${tries} -lt 200 ]]; do
  sleep 0.02
  tries=$((tries + 1))
done
[[ -s "${no_runner_rc}" ]] || fail "no-Runner.Worker test did not finish"
[[ "$(<"${no_runner_rc}")" == 2 ]] \
  || fail "official no-Runner.Worker case did not return 2"
grep -Fq 'Runner.Worker ancestor not found' "${no_runner_out}" \
  || fail "official no-Runner.Worker case had no clear message"
remove_launch_label "${no_runner_label}"

# Use one long-running, non-interpreter system binary for both coalition cases.
if [[ -x /usr/sbin/iostat ]]; then
  SYSTEM_STANDIN=(/usr/sbin/iostat -w 60)
elif [[ -x /usr/sbin/netstat ]]; then
  SYSTEM_STANDIN=(/usr/sbin/netstat -w 60)
else
  fail "neither /usr/sbin/iostat nor /usr/sbin/netstat exists"
fi
SYSTEM_STANDIN_PATH="${SYSTEM_STANDIN[0]}"

# A system binary forked by the job keeps the job coalition. It is an orphan.
job_system_pid_file="${WORK}/job-system.pid"
spawn_exec_orphan "${job_system_pid_file}" "${SYSTEM_STANDIN[@]}"
JOB_SYSTEM_PID="$(<"${job_system_pid_file}")"
PIDS+=("${JOB_SYSTEM_PID}")
wait_for_ppid_one "${JOB_SYSTEM_PID}" || fail "job system binary was not reparented"
wait_for_exe "${JOB_SYSTEM_PID}" "${SYSTEM_STANDIN_PATH}" \
  || fail "job system binary did not exec"
out="$("${GUARD}" check --job-start "${JOB_START}" 2>&1 || true)"
[[ "${out}" == *'orphan-process'* ]] || fail "in-coalition system binary was not refused"
kill_out="$("${GUARD}" kill --job-start "${JOB_START}" 2>&1)" \
  || fail "kill mode did not clear in-coalition system binary: ${kill_out}"
process_is_gone "${JOB_SYSTEM_PID}" || fail "kill mode left in-coalition system binary"

# A runner-looking name cannot hide a helper in the job coalition.
runner_named_exe="${WORK}/build.noindex/.zig-cache/Runner.Worker"
runner_named_pid_file="${WORK}/runner-named.pid"
mkdir -p "$(dirname -- "${runner_named_exe}")"
build_sleeper "${runner_named_exe}"
spawn_exec_orphan "${runner_named_pid_file}" "${runner_named_exe}"
RUNNER_NAMED_PID="$(<"${runner_named_pid_file}")"
PIDS+=("${RUNNER_NAMED_PID}")
wait_for_ppid_one "${RUNNER_NAMED_PID}" \
  || fail "runner-named helper was not reparented"
wait_for_exe "${RUNNER_NAMED_PID}" "${runner_named_exe}" \
  || fail "runner-named helper did not exec"
out="$("${GUARD}" check --job-start "${JOB_START}" 2>&1 || true)"
[[ "${out}" == *'orphan-process'* ]] \
  || fail "runner-named helper was not refused"
kill_out="$("${GUARD}" kill --job-start "${JOB_START}" 2>&1)" \
  || fail "kill mode did not clear runner-named helper: ${kill_out}"
process_is_gone "${RUNNER_NAMED_PID}" \
  || fail "kill mode left runner-named helper"

# A launchd job has a new coalition. Its writable executable path still fails.
own_launch_bin="${WORK}/launch-sleeper"
own_launch_pid_file="${WORK}/launch-sleeper.pid"
build_pid_sleeper "${own_launch_bin}"
own_label="com.gumbii.orphan-guard.own.$$.$RANDOM"
LAUNCH_LABELS+=("${own_label}")
/bin/launchctl submit -l "${own_label}" -- "${own_launch_bin}" "${own_launch_pid_file}" \
  || fail "could not submit own launch job"
tries=0
while [[ ! -s "${own_launch_pid_file}" && ${tries} -lt 200 ]]; do
  sleep 0.02
  tries=$((tries + 1))
done
[[ -s "${own_launch_pid_file}" ]] || fail "own launch job did not write its pid"
OWN_LAUNCH_PID="$(<"${own_launch_pid_file}")"
PIDS+=("${OWN_LAUNCH_PID}")
wait_for_ppid_one "${OWN_LAUNCH_PID}" || fail "own launch job is not a PPID-1 process"
out="$("${GUARD}" check --job-start "${JOB_START}" 2>&1 || true)"
[[ "${out}" == *'orphan-process'* ]] || fail "own launch job was not refused"
remove_launch_job "${own_label}" "${OWN_LAUNCH_PID}"

# A launchd system binary is outside the job coalition. It passes and is logged.
system_label="com.gumbii.orphan-guard.system.$$.$RANDOM"
LAUNCH_LABELS+=("${system_label}")
/bin/launchctl submit -l "${system_label}" -- "${SYSTEM_STANDIN[@]}" \
  || fail "could not submit system stand-in"
SYSTEM_LAUNCH_PID="$(wait_for_launch_pid "${system_label}")" \
  || fail "system stand-in did not start"
PIDS+=("${SYSTEM_LAUNCH_PID}")
wait_for_ppid_one "${SYSTEM_LAUNCH_PID}" || fail "system stand-in is not a PPID-1 process"
wait_for_exe "${SYSTEM_LAUNCH_PID}" "${SYSTEM_STANDIN_PATH}" \
  || fail "system stand-in did not exec"
set +e
out="$("${GUARD}" check --job-start "${JOB_START}" 2>&1)"
rc=$?
set -e
[[ ${rc} -eq 0 && "${out}" == *"system agent passed: pid=${SYSTEM_LAUNCH_PID} "* ]] \
  || fail "system stand-in did not pass with a log line: ${out}"
kill_out="$("${GUARD}" kill --job-start "${JOB_START}" 2>&1)" \
  || fail "system stand-in broke kill mode: ${kill_out}"
kill -0 "${SYSTEM_LAUNCH_PID}" 2>/dev/null \
  || fail "kill mode signalled the system stand-in"
remove_launch_job "${system_label}" "${SYSTEM_LAUNCH_PID}"

# An interpreter stays forbidden even when launchd gives it a new coalition.
JSC="/System/Library/Frameworks/JavaScriptCore.framework/Versions/A/Helpers/jsc"
if [[ -x "${JSC}" ]]; then
  jsc_script="${WORK}/jsc-wait.js"
  printf '%s\n' 'var end = Date.now() + 300000; while (Date.now() < end) {}' > "${jsc_script}"
  jsc_label="com.gumbii.orphan-guard.jsc.$$.$RANDOM"
  LAUNCH_LABELS+=("${jsc_label}")
  /bin/launchctl submit -l "${jsc_label}" -- "${JSC}" "${jsc_script}" \
    || fail "could not submit jsc launch job"
  JSC_PID="$(wait_for_launch_pid "${jsc_label}")" || fail "jsc launch job did not start"
  PIDS+=("${JSC_PID}")
  wait_for_ppid_one "${JSC_PID}" || fail "jsc launch job is not a PPID-1 process"
  wait_for_exe "${JSC_PID}" "${JSC}" || fail "jsc launch job did not exec"
  out="$("${GUARD}" check --job-start "${JOB_START}" 2>&1 || true)"
  [[ "${out}" == *'orphan-process'* ]] || fail "jsc launch job was not refused"
  remove_launch_job "${jsc_label}" "${JSC_PID}"
else
  printf 'test-orphan-guard.sh: INFO: jsc absent; launch interpreter case skipped\n'
fi

orphan_pid_file="${WORK}/orphan.pid"
spawn_orphan "${orphan_pid_file}"
ORPHAN_PID="$(<"${orphan_pid_file}")"
PIDS+=("${ORPHAN_PID}")
wait_for_ppid_one "${ORPHAN_PID}" || fail "setsid child was not reparented to pid 1"

out="$("${GUARD}" check --job-start "${JOB_START}" 2>&1 || true)"
[[ "${out}" == *'orphan-process'* ]] || fail "real orphan was not refused"
if "${GUARD}" check --job-start "${JOB_START}" \
  --allow-pid "${ORPHAN_PID}:/not/the/executable" >/dev/null 2>&1; then
  fail "mismatched executable path was allowed"
fi
orphan_exe="$(kernel_path "${ORPHAN_PID}")"
"${GUARD}" check --job-start "${JOB_START}" \
  --allow-pid "${ORPHAN_PID}:${orphan_exe}" || fail "exact pid and executable were not allowed"
kill -KILL "${ORPHAN_PID}" 2>/dev/null || true
process_is_gone "${ORPHAN_PID}" || fail "test orphan did not stop"

# A live process with an unlinked executable has no kernel path. Its kernel
# identity and job coalition still prove that kill may signal it.
unlinked_exe="${WORK}/build.noindex/unlinked-sleeper"
unlinked_ready="${WORK}/unlinked-orphan.ready"
mkdir -p "$(dirname -- "${unlinked_exe}")"
# The sleeper writes a ready file from main(): deleting the binary while it
# is still starting up stops it, so wait until it runs before the rm.
printf '%s\n' '#include <fcntl.h>' '#include <unistd.h>' \
  'int main(int c, char **v) { if (c > 1) close(open(v[1], O_WRONLY | O_CREAT, 0600)); sleep(300); return 0; }' \
  | /usr/bin/xcrun -sdk macosx clang -x c - -o "${unlinked_exe}" || fail "could not build the unlinked sleeper"
unlinked_pid_file="${WORK}/unlinked-orphan.pid"
spawn_exec_orphan "${unlinked_pid_file}" "${unlinked_exe}" "${unlinked_ready}"
tries=0
while [[ ! -e "${unlinked_ready}" && ${tries} -lt 500 ]]; do sleep 0.01; tries=$((tries + 1)); done
[[ -e "${unlinked_ready}" ]] || fail "unlinked sleeper did not start"
sleep 0.5
UNLINKED_PID="$(<"${unlinked_pid_file}")"
PIDS+=("${UNLINKED_PID}")
wait_for_ppid_one "${UNLINKED_PID}" || fail "unlinked sleeper was not reparented"
wait_for_exe "${UNLINKED_PID}" "${unlinked_exe}" || fail "unlinked sleeper did not exec"
rm "${unlinked_exe}"
kill -0 "${UNLINKED_PID}" 2>/dev/null || fail "unlinked sleeper stopped when its binary was removed"
tries=0
while kernel_path "${UNLINKED_PID}" >/dev/null 2>&1 && (( tries < 200 )); do
  sleep 0.02
  tries=$((tries + 1))
done
(( tries < 200 )) || fail "unlinked sleeper kept a kernel path"
out="$("${GUARD}" check --job-start "${JOB_START}" 2>&1 || true)"
[[ "${out}" == *'orphan-process'* ]] || fail "unlinked orphan was not refused"
out="$("${GUARD}" kill --job-start "${JOB_START}" 2>&1)" \
  || fail "kill mode did not clear unlinked orphan: ${out}"
[[ "${out}" == *'remain=0'* ]] || fail "unlinked orphan remained: ${out}"
process_is_gone "${UNLINKED_PID}" || fail "kill mode left unlinked orphan"

# A system interpreter in the job coalition must still be refused.
SYSTEM_HELPER="/System/Library/Frameworks/JavaScriptCore.framework/Versions/A/Helpers/jsc"
system_pid_file="${WORK}/system-helper.pid"
if [[ -x "${SYSTEM_HELPER}" ]]; then
  spawn_exec_orphan "${system_pid_file}" "${SYSTEM_HELPER}" -e \
    'var t=Date.now(); while(Date.now()-t<20000){}'
else
  SYSTEM_HELPER="/usr/libexec/PlistBuddy"
  [[ -x "${SYSTEM_HELPER}" ]] || fail "no system helper is available"
  helper_fifo="${WORK}/plist.fifo"
  mkfifo "${helper_fifo}"
  printf 'test-orphan-guard.sh: INFO: jsc absent; using %s\n' "${SYSTEM_HELPER}"
  spawn_exec_orphan "${system_pid_file}" "${SYSTEM_HELPER}" -c Print "${helper_fifo}"
fi
SYSTEM_PID="$(<"${system_pid_file}")"
PIDS+=("${SYSTEM_PID}")
wait_for_ppid_one "${SYSTEM_PID}" || fail "system helper was not reparented to pid 1"
wait_for_exe "${SYSTEM_PID}" "${SYSTEM_HELPER}" || fail "system helper did not exec"
out="$("${GUARD}" check --job-start "${JOB_START}" 2>&1 || true)"
[[ "${out}" == *'orphan-process'* ]] || fail "unlisted system helper was not refused"
kill -KILL "${SYSTEM_PID}" 2>/dev/null || true
process_is_gone "${SYSTEM_PID}" || fail "system helper did not stop"

# A hardlink or copy is under a writable path, so it is not a system agent.
[[ -x /usr/sbin/cfprefsd ]] || fail "/usr/sbin/cfprefsd is missing"
mkdir -p "${WORK}/build.noindex"
CFPREFSD_COPY="${WORK}/build.noindex/cfprefsd"
if ln /usr/sbin/cfprefsd "${CFPREFSD_COPY}" 2>/dev/null; then
  hardlink_pid_file="${WORK}/cfprefsd-hardlink.pid"
  spawn_exec_orphan "${hardlink_pid_file}" "${CFPREFSD_COPY}"
  HARDLINK_PID="$(<"${hardlink_pid_file}")"
  PIDS+=("${HARDLINK_PID}")
  wait_for_ppid_one "${HARDLINK_PID}" || fail "cfprefsd hardlink was not reparented"
  wait_for_exe "${HARDLINK_PID}" "${CFPREFSD_COPY}" || fail "cfprefsd hardlink did not run"
  out="$("${GUARD}" check --job-start "${JOB_START}" 2>&1 || true)"
  [[ "${out}" == *'orphan-process'* ]] || fail "cfprefsd hardlink was incorrectly excluded"
  kill -KILL "${HARDLINK_PID}" 2>/dev/null || true
  process_is_gone "${HARDLINK_PID}" || fail "cfprefsd hardlink did not stop"
  rm -f "${CFPREFSD_COPY}"
else
  printf 'test-orphan-guard.sh: INFO: hardlink impossible: cross-device\n'
fi
# A copied Apple binary cannot run here. The compiled sleeper uses the exact
# writable path name and spoofs argv[0] as a system path.
build_sleeper "${CFPREFSD_COPY}"
chmod 0755 "${CFPREFSD_COPY}"
copy_pid_file="${WORK}/cfprefsd-copy.pid"
spawn_spoofed_exec_orphan "${copy_pid_file}" "${CFPREFSD_COPY}" /usr/sbin/cfprefsd
COPY_PID="$(<"${copy_pid_file}")"
PIDS+=("${COPY_PID}")
wait_for_ppid_one "${COPY_PID}" || fail "cfprefsd copy was not reparented to pid 1"
wait_for_exe "${COPY_PID}" "${CFPREFSD_COPY}" || fail "cfprefsd copy did not exec"
out="$("${GUARD}" check --job-start "${JOB_START}" 2>&1 || true)"
[[ "${out}" == *'orphan-process'* ]] || fail "spoofed cfprefsd copy was incorrectly excluded"
kill -KILL "${COPY_PID}" 2>/dev/null || true
process_is_gone "${COPY_PID}" || fail "cfprefsd copy did not stop"

# A /tmp copy of mdworker_shared is not under an allowed system path.
MDWORKER_SHARED="/System/Library/Frameworks/CoreServices.framework/Versions/A/Frameworks/Metadata.framework/Versions/A/Support/mdworker_shared"
[[ -x "${MDWORKER_SHARED}" ]] || fail "mdworker_shared is missing"
MDWORKER_COPY="$(mktemp /tmp/mdworker-shared-copy.XXXXXX)"
TMP_FILES+=("${MDWORKER_COPY}")
build_sleeper "${MDWORKER_COPY}"
chmod 0755 "${MDWORKER_COPY}"
mdworker_pid_file="${WORK}/mdworker-copy.pid"
spawn_exec_orphan "${mdworker_pid_file}" "${MDWORKER_COPY}"
MDWORKER_PID="$(<"${mdworker_pid_file}")"
PIDS+=("${MDWORKER_PID}")
wait_for_ppid_one "${MDWORKER_PID}" || fail "mdworker_shared copy was not reparented to pid 1"
wait_for_exe "${MDWORKER_PID}" "${MDWORKER_COPY}" || fail "mdworker_shared copy did not exec"
out="$("${GUARD}" check --job-start "${JOB_START}" 2>&1 || true)"
[[ "${out}" == *'orphan-process'* ]] || fail "mdworker_shared copy was incorrectly excluded"
kill -KILL "${MDWORKER_PID}" 2>/dev/null || true
process_is_gone "${MDWORKER_PID}" || fail "mdworker_shared copy did not stop"

# Measure-level post-timing gate. The mock replaces benchd only. It writes all
# three sealed files, starts a real PPID-1 helper during iterate, and returns 0.
TREE="${WORK}/measure-tree"
GOLDENS="${WORK}/goldens"
BASELINE="${WORK}/baseline"
mkdir -p "${TREE}/tools" "${TREE}/fixtures" \
  "${TREE}/Vendor/TensorFold/zig-out/bin" \
  "${TREE}/reference_weights/Qwen3.8-27B-MLX-4bit" \
  "${BASELINE}/Vendor/TensorFold/zig-out/bin" \
  "${BASELINE}/reference_weights/Qwen3.8-27B-MLX-4bit" "${GOLDENS}"
cp "${ROOT}/tools/qwen38-27b-tf-measure-and-score.sh" \
  "${ROOT}/tools/spec-declaration.sh" "${ROOT}/tools/orphan-guard.sh" "${TREE}/tools/"
chmod +x "${TREE}/tools/"*.sh
printf '#!/usr/bin/env bash\nexit 0\n' > "${TREE}/Vendor/TensorFold/zig-out/bin/bench-worker"
cp "${TREE}/Vendor/TensorFold/zig-out/bin/bench-worker" \
  "${BASELINE}/Vendor/TensorFold/zig-out/bin/bench-worker"
chmod +x "${TREE}/Vendor/TensorFold/zig-out/bin/bench-worker" \
  "${BASELINE}/Vendor/TensorFold/zig-out/bin/bench-worker"
printf '{}\n' > "${TREE}/mtp-head.manifest.json"
printf '{"trackId":"test-track"}\n' > "${TREE}/benchmark.json"
printf '{}\n' > "${GOLDENS}/one.golden.json"
golden_sha="$(shasum -a 256 "${GOLDENS}/one.golden.json" | awk '{print $1}')"
golden_bytes="$(wc -c < "${GOLDENS}/one.golden.json" | tr -d '[:space:]')"
jq -n --arg sha "${golden_sha}" --argjson bytes "${golden_bytes}" '{
  official_scoring_enabled: true,
  live_goldens: ["one"],
  timed_prompt_pool: [{r2_path:"pool/one.golden.json", sha256:$sha, bytes:$bytes}],
  dflash_head: {permitted_draft_depths:[1]}
}' > "${TREE}/fixtures/qwen3_8_27b_tf_mlx_v1_track.json"
printf '{}\n' > "${WORK}/calibration.json"

MOCK_BENCHD="${WORK}/benchd"
cat > "${MOCK_BENCHD}" <<'MOCK'
#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == iterate && "${2:-}" == --help ]]; then
  printf '%s\n' '--control-golden'
  exit 0
fi
[[ "${1:-}" == iterate ]] || exit 2
score=""
shift
while [[ $# -gt 0 ]]; do
  if [[ "$1" == --score-path ]]; then score="$2"; shift 2; else shift; fi
done
/usr/bin/python3 - "${TEST_ORPHAN_PID_FILE}" <<'PY'
import os
import sys
import time
if os.fork():
    raise SystemExit(0)
os.setsid()
if os.fork():
    raise SystemExit(0)
with open(sys.argv[1], "w") as handle:
    handle.write(str(os.getpid()) + "\n")
time.sleep(300)
PY
tries=0
while [[ ! -s "${TEST_ORPHAN_PID_FILE}" && ${tries} -lt 200 ]]; do sleep 0.02; tries=$((tries + 1)); done
pid="$(<"${TEST_ORPHAN_PID_FILE}")"
tries=0
while [[ "$(ps -o ppid= -p "${pid}" 2>/dev/null | tr -d '[:space:]')" != 1 && ${tries} -lt 200 ]]; do
  sleep 0.02
  tries=$((tries + 1))
done
printf '{"passed":true,"score":1}\n' > "${score}"
printf 'sealed\n' > "${score}.sha256"
printf '{}\n' > "$(dirname "${score}")/benchmark-integrity.json"
MOCK
chmod +x "${MOCK_BENCHD}"

measure_orphan_pid_file="${WORK}/measure-orphan.pid"
score_path="${TREE}/score.json"
set +e
measure_out="$(cd "${TREE}" && \
  HOME="${WORK}" BENCHD="${MOCK_BENCHD}" MLXFAST_GPU_WINDOW_HELD=1 \
  MLXFAST_OFFICIAL_BENCHMARK_RUN=1 MLXFAST_SCORE_PATH="${score_path}" \
  MLXFAST_QWEN38_GOLDEN_DIR="${GOLDENS}" MLXFAST_PRIVATE_DIR="${GOLDENS}" \
  MLXFAST_BASELINE_WORKSPACE="${BASELINE}" \
  MLXFAST_BASELINE_CALIBRATION="${WORK}/calibration.json" \
  MLXFAST_RESIDENT_LOG_ROOT="${WORK}/resident-logs" \
  TEST_ORPHAN_PID_FILE="${measure_orphan_pid_file}" \
  ./tools/qwen38-27b-tf-measure-and-score.sh 2>&1)"
measure_rc=$?
set -e
if [[ ! -s "${measure_orphan_pid_file}" ]]; then
  # The mock benchd never ran or never made its orphan: show what the
  # before-timing check saw (the hosted runner has its own per-user agents).
  show_new_orphans
  fail "measure refused before the mock benchd made its orphan (rc ${measure_rc}): ${measure_out: -600}"
fi
[[ ${measure_rc} -ne 0 && "${measure_out}" == *'orphan-process (after timing)'* ]] \
  || fail "measure after-timing orphan check did not refuse (rc ${measure_rc}): ${measure_out: -600}"
[[ ! -e "${score_path}" && ! -e "${score_path}.sha256" \
   && ! -e "${TREE}/benchmark-integrity.json" ]] \
  || fail "measure orphan refusal left a sealed file"
[[ "$(<"${score_path}.refusal-reason")" == orphan-process ]] \
  || fail "measure orphan refusal reason is missing"
MEASURE_ORPHAN_PID="$(<"${measure_orphan_pid_file}")"
PIDS+=("${MEASURE_ORPHAN_PID}")
kill -KILL "${MEASURE_ORPHAN_PID}" 2>/dev/null || true
process_is_gone "${MEASURE_ORPHAN_PID}" || fail "measure test orphan did not stop"

# Kill mode removes this runner's new process. It preserves an older process
# and processes owned by another runner installation.
/bin/sleep 300 &
CURRENT_PID=$!
PIDS+=("${CURRENT_PID}")
mkdir -p "${WORK}/other-runner/bin"
make_runner "${WORK}/other-runner/bin/Runner.Worker"
other_child_file="${WORK}/other-child.pid"
# Another runner's jobs live in that runner's own launchd coalition, not in
# this job's: start the fixture through launchd so the guard sees it there.
other_runner_label="com.gumbii.orphan-guard.otherrunner.$$.$RANDOM"
LAUNCH_LABELS+=("${other_runner_label}")
/bin/launchctl submit -l "${other_runner_label}" -- "${WORK}/other-runner/bin/Runner.Worker" -c \
  '/bin/sleep 300 & child=$!; printf "%s\n" "${child}" > "$1"; wait "${child}"' \
  Runner.Worker "${other_child_file}" || fail "could not submit the other runner fixture"
tries=0
while [[ ! -s "${other_child_file}" && ${tries} -lt 200 ]]; do sleep 0.02; tries=$((tries + 1)); done
[[ -s "${other_child_file}" ]] || fail "other runner child did not start"
OTHER_CHILD_PID="$(<"${other_child_file}")"
PIDS+=("${OTHER_CHILD_PID}")

other_install="${WORK}/actions-runner-other"
other_orphan_exe="${other_install}/_work/repo/build.noindex/helper"
mkdir -p "$(dirname "${other_orphan_exe}")"
build_pid_sleeper "${other_orphan_exe}"
other_orphan_file="${WORK}/other-runner-orphan.pid"
other_orphan_label="com.gumbii.orphan-guard.otherorphan.$$.$RANDOM"
LAUNCH_LABELS+=("${other_orphan_label}")
/bin/launchctl submit -l "${other_orphan_label}" -- "${other_orphan_exe}" "${other_orphan_file}" \
  || fail "could not submit the other runner orphan"
tries=0
while [[ ! -s "${other_orphan_file}" && ${tries} -lt 200 ]]; do sleep 0.02; tries=$((tries + 1)); done
[[ -s "${other_orphan_file}" ]] || fail "other runner orphan did not write its pid"
OTHER_ORPHAN_PID="$(<"${other_orphan_file}")"
PIDS+=("${OTHER_ORPHAN_PID}")
wait_for_ppid_one "${OTHER_ORPHAN_PID}" || fail "other runner orphan was not reparented"
wait_for_exe "${OTHER_ORPHAN_PID}" "${other_orphan_exe}" || fail "other runner orphan did not exec"
# Prove the separation: both protected fixtures are outside this job's coalition.
/usr/bin/python3 - "$$" "${OTHER_CHILD_PID}" "${OTHER_ORPHAN_PID}" <<'PY' || fail "a protected fixture shares the job coalition"
import ctypes, sys
lib = ctypes.CDLL("/usr/lib/libproc.dylib")
class CI(ctypes.Structure):
    _fields_ = (("ids", ctypes.c_uint64 * 2), ("reserved", ctypes.c_uint64 * 3))
def coalition(pid):
    info = CI()
    if lib.proc_pidinfo(int(pid), 20, 0, ctypes.byref(info), 40) != 40:
        raise SystemExit(1)
    return info.ids[0]
job = coalition(sys.argv[1])
raise SystemExit(0 if all(coalition(p) != job for p in sys.argv[2:]) else 1)
PY

set +e
mismatch_out="$("${GUARD}" kill --job-start "${JOB_START}" \
  --runner-root "${WORK}/runner-jobs/_work" 2>&1)"
mismatch_rc=$?
set -e
[[ ${mismatch_rc} -ne 0 && "${mismatch_out}" == *'runner root mismatch'* ]] \
  || fail "mismatching runner root was not clearly refused: ${mismatch_out}"
kill -0 "${CURRENT_PID}" 2>/dev/null \
  || fail "runner root mismatch killed this runner's process"

set +e
kill_out="$("${GUARD}" kill --job-start "${JOB_START}" 2>&1)"
kill_rc=$?
set -e
if [[ ${kill_rc} -ne 0 ]]; then
  # Diagnose on failure: list this account's processes with their parents.
  ps -axo pid=,ppid=,uid=,lstart=,comm= | awk -v uid="$(id -u)" '$3 == uid' >&2 || true
  fail "kill mode reported survivors: ${kill_out}"
fi
process_is_gone "${CURRENT_PID}" || fail "kill mode left this runner's new process"
kill -0 "${OLD_PID}" 2>/dev/null || fail "kill mode stopped an older process"
kill -0 "${OTHER_CHILD_PID}" 2>/dev/null || fail "kill mode stopped another runner's process"
kill -0 "${OTHER_ORPHAN_PID}" 2>/dev/null || fail "kill mode stopped another runner's PPID-1 process"
remove_launch_label "${other_runner_label}"
remove_launch_job "${other_orphan_label}" "${OTHER_ORPHAN_PID}"

updated_root="${WORK}/updated-runner"
updated_work="${WORK}/updated-runner-jobs/_work/repo"
updated_worker="${updated_root}/bin.2.338.0/Runner.Worker"
make_runner "${updated_worker}"
mkdir -p "${updated_work}"
RUNNER_WORKSPACE="${updated_work}" \
  "${updated_worker}" "$0" --kill-case "${WORK}" "${updated_root}" \
  || fail "bin.<version> runner root derivation failed"

printf 'test-orphan-guard.sh: PASS\n'
