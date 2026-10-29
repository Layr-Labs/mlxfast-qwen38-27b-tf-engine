#!/usr/bin/env bash
# Native tests for tools/stop-leaked-residents.sh.
# No model, GPU, network, sudo, or process-table shim is used.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
STOP="${ROOT_DIR}/tools/stop-leaked-residents.sh"

if ! ps -A -o pid=,args= >/dev/null 2>&1; then
  echo "test-stop-leaked-residents: SKIP -- process enumeration is blocked on this host"
  exit 0
fi

WORK="$(mktemp -d "${TMPDIR:-/tmp}/test-stop-leaked-residents.XXXXXX")"
JOB_ROOT="${WORK}/root"
LOCK="${WORK}/gpu.lock"
SOCKET_PATH="${WORK}/s.sock"
HELPERS=("")
LOCK_HOLDER_PID=""
fails=0
ALL_OUTPUT=""
mkdir -p "${JOB_ROOT}/.build/release" "${WORK}/outside" "${WORK}/lock-directory"
export RESIDENT_UP_LOCK_PATH="${LOCK}"

cleanup() {
  local args pid
  for pid in "${HELPERS[@]}"; do
    args="$(ps -ww -o args= -p "${pid}" 2>/dev/null || true)"
    [[ "${args}" != *"${WORK}"* ]] || kill -KILL "${pid}" 2>/dev/null || true
  done
  if [[ -n "${LOCK_HOLDER_PID}" ]]; then
    args="$(ps -ww -o args= -p "${LOCK_HOLDER_PID}" 2>/dev/null || true)"
    [[ "${args}" != *"${WORK}"* ]] || kill -KILL "${LOCK_HOLDER_PID}" 2>/dev/null || true
  fi
  for pid in "${HELPERS[@]}" "${LOCK_HOLDER_PID}"; do
    [[ -z "${pid}" ]] || wait "${pid}" 2>/dev/null || true
  done
  rm -rf "${WORK}"
}
trap cleanup EXIT

pass() { printf 'test-stop-leaked-residents: PASS %s\n' "$1"; }
fail() { printf 'test-stop-leaked-residents: FAIL %s\n' "$1" >&2; fails=$((fails + 1)); }
record() { ALL_OUTPUT="${ALL_OUTPUT}${1}"$'\n'; }

is_alive() {
  local state
  kill -0 "$1" 2>/dev/null || return 1
  state="$(ps -o stat= -p "$1" 2>/dev/null | tr -d '[:space:]' || true)"
  [[ "${state}" != Z* ]]
}

gone_within() {
  local pid="$1"
  local tries="${2:-30}"
  while is_alive "${pid}" && [[ ${tries} -gt 0 ]]; do
    sleep 0.1
    tries=$((tries - 1))
  done
  ! is_alive "${pid}"
}

HELPER_SOURCE="${WORK}/helper.c"
HELPER_BINARY="${WORK}/helper"
cat > "${HELPER_SOURCE}" <<'C'
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <wordexp.h>

static void write_pid(const char *path, pid_t pid) {
    FILE *handle = fopen(path, "w");
    if (handle == NULL) _exit(10);
    fprintf(handle, "%d\n", pid);
    fclose(handle);
}

static void run_title(const char *title) {
    wordexp_t words;
    char **args;
    size_t i;
    if (wordexp(title, &words, WRDE_NOCMD) != 0 || words.we_wordc < 2) _exit(11);
    args = calloc(words.we_wordc + 3, sizeof(char *));
    if (args == NULL) _exit(12);
    for (i = 0; i < words.we_wordc; i++) args[i] = words.we_wordv[i];
    args[words.we_wordc] = "--helper-mode";
    args[words.we_wordc + 1] = "child";
    execv(args[0], args);
    _exit(13);
}

int main(int argc, char **argv) {
    const char *mode = NULL;
    const char *child_file = "";
    const char *child_title = "";
    const char *sibling_file = "";
    const char *sibling_title = "";
    int i;
    pid_t child;

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--helper-mode") == 0 && i + 1 < argc) {
            mode = argv[i + 1];
            if (i + 2 < argc) child_file = argv[i + 2];
            if (i + 3 < argc) child_title = argv[i + 3];
            if (i + 4 < argc) sibling_file = argv[i + 4];
            if (i + 5 < argc) sibling_title = argv[i + 5];
            break;
        }
    }
    if (mode == NULL) return 2;
    if (strcmp(mode, "child") != 0 && setsid() < 0) return 2;
    if (strcmp(mode, "ignore") == 0) signal(SIGTERM, SIG_IGN);

    if (strcmp(mode, "group") == 0) {
        child = fork();
        if (child < 0) return 3;
        if (child == 0) for (;;) pause();
        write_pid(child_file, child);
    }
    if (strcmp(mode, "nonleader") == 0) {
        child = fork();
        if (child < 0) return 4;
        if (child == 0) run_title(child_title);
        write_pid(child_file, child);

        child = fork();
        if (child < 0) return 5;
        if (child == 0) run_title(sibling_title);
        write_pid(sibling_file, child);
    }
    for (;;) pause();
}
C
cc -O2 -o "${HELPER_BINARY}" "${HELPER_SOURCE}" || {
  echo "test-stop-leaked-residents: FAIL 0 -- could not build native process helper" >&2
  exit 1
}

check_title() {
  local pid="$1"
  local title="$2"
  local got=""
  local got_argv0=""
  local got_argv1=""
  local title_argv0=""
  local title_argv1=""
  local tries=50

  read -r title_argv0 title_argv1 _ <<< "${title}"
  while [[ ${tries} -gt 0 ]]; do
    got="$(ps -ww -o args= -p "${pid}" 2>/dev/null || true)"
    read -r got_argv0 got_argv1 _ <<< "${got}"
    [[ "${got_argv0}" == "${title_argv0}" && "${got_argv1}" == "${title_argv1}" && "${got}" == "${title}"* ]] && break
    sleep 0.1
    tries=$((tries - 1))
  done
  if [[ "${got_argv0}" != "${title_argv0}" || "${got_argv1}" != "${title_argv1}" || "${got}" != "${title}"* ]]; then
    echo "test-stop-leaked-residents: FAIL 0 -- helper process title does not match argv0, argv1, and full title prefix" >&2
    exit 1
  fi
}

start_helper() {
  local mode="$1"
  local title="$2"
  local child_file="${3:-}"
  local child_title="${4:-}"
  local sibling_file="${5:-}"
  local sibling_title="${6:-}"

  local -a title_parts=()
  local path
  read -r -a title_parts <<< "${title}"
  path="${title_parts[0]}"
  mkdir -p "$(dirname "${path}")"
  [[ -x "${path}" ]] || cp "${HELPER_BINARY}" "${path}"
  if [[ "${mode}" == "nonleader" ]]; then
    for path in "${child_title%% *}" "${sibling_title%% *}"; do
      mkdir -p "$(dirname "${path}")"
      [[ -x "${path}" ]] || cp "${HELPER_BINARY}" "${path}"
    done
  fi
  "${title_parts[@]}" --helper-mode "${mode}" "${child_file}" "${child_title}" "${sibling_file}" "${sibling_title}" &
  STARTED_PID=$!
  HELPERS+=("${STARTED_PID}")
  check_title "${STARTED_PID}" "${title}"
}

read_pid_file() {
  local path="$1"
  local tries=50
  while [[ ${tries} -gt 0 ]]; do
    [[ -s "${path}" ]] && break
    sleep 0.1
    tries=$((tries - 1))
  done
  sed -n '1p' "${path}" 2>/dev/null || true
}

run_stop() {
  "${STOP}" "$@" --root "${JOB_ROOT}" 2>&1
}

resident_title="${JOB_ROOT}/.build/release/bench-worker resident --socket ${SOCKET_PATH}"

# 1. No resident.
out="$(run_stop --mode preflight)"; rc=$?; record "${out}"
if [[ ${rc} -eq 0 && "${out}" == *"stop-leaked-residents: no leftover resident"* ]]; then
  pass "1 -- no resident"
else
  fail "1 -- no resident (rc=${rc})"
fi

# 2. One rooted resident is stopped and named by pid only.
start_helper normal "${resident_title}"
pid2="${STARTED_PID}"
out="$(run_stop --mode preflight)"; rc=$?; record "${out}"
if [[ ${rc} -eq 0 && "${out}" == *"::warning::stopping resident bench-worker left by an earlier job (pid ${pid2}, elapsed "* ]] && gone_within "${pid2}"; then
  pass "2 -- one rooted resident"
else
  fail "2 -- one rooted resident (rc=${rc}, pid=${pid2})"
fi

# 3. A rooted group leader and its child are both stopped.
child_file="${WORK}/group-child.pid"
child_title="${WORK}/group-child keep"
start_helper group "${resident_title}" "${child_file}" "${child_title}"
pid3="${STARTED_PID}"
child3="$(read_pid_file "${child_file}")"
[[ -z "${child3}" ]] || HELPERS+=("${child3}")
out="$(run_stop --mode preflight)"; rc=$?; record "${out}"
if [[ ${rc} -eq 0 && -n "${child3}" ]] && gone_within "${pid3}" && gone_within "${child3}"; then
  pass "3 -- rooted process group and child"
else
  fail "3 -- rooted process group and child (rc=${rc}, leader=${pid3}, child=${child3:-missing})"
fi

# 4. A rooted resident that ignores SIGTERM is killed on time.
start_helper ignore "${resident_title}"
pid4="${STARTED_PID}"
SECONDS=0
out="$(run_stop --mode preflight --stop-wait 1)"; rc=$?; elapsed_s=${SECONDS}; record "${out}"
if [[ ${rc} -eq 0 && ${elapsed_s} -le 6 ]] && gone_within "${pid4}"; then
  pass "4 -- SIGTERM ignore escalates to SIGKILL"
else
  fail "4 -- SIGTERM ignore escalates to SIGKILL (rc=${rc}, elapsed=${elapsed_s}s)"
fi

# 5. Three nonmatches survive and lookalikes produce one count-only warning.
baseline_out="$(run_stop --mode preflight)"; baseline_rc=$?; record "${baseline_out}"
baseline_count="$(printf '%s\n' "${baseline_out}" | awk '/::warning::found [0-9]+ other process\(es\) that look like a resident; not stopped/{count=$2} END{print count+0}')"
outside_title="${WORK}/outside/bench-worker resident --socket ${SOCKET_PATH}"
wrong_arg_title="${JOB_ROOT}/.build/release/bench-worker worker --socket ${SOCKET_PATH}"
later_words_title="${WORK}/outside/not-worker 300 bench-worker resident"
start_helper normal "${outside_title}"; pid5a="${STARTED_PID}"
start_helper normal "${wrong_arg_title}"; pid5b="${STARTED_PID}"
start_helper normal "${later_words_title}"; pid5c="${STARTED_PID}"
out="$(run_stop --mode preflight)"; rc=$?; record "${out}"
warning_count="$(printf '%s\n' "${out}" | awk '/::warning::found [0-9]+ other process\(es\) that look like a resident; not stopped/{count++} END{print count+0}')"
reported_count="$(printf '%s\n' "${out}" | awk '/::warning::found [0-9]+ other process\(es\) that look like a resident; not stopped/{count=$2} END{print count+0}')"
expected_count=$((baseline_count + 2))
if [[ ${baseline_rc} -eq 0 && ${rc} -eq 0 && ${warning_count} -eq 1 && ${reported_count} -eq ${expected_count} ]] && is_alive "${pid5a}" && is_alive "${pid5b}" && is_alive "${pid5c}"; then
  pass "5 -- three nonmatches survive"
else
  fail "5 -- three nonmatches survive (baseline_rc=${baseline_rc}, rc=${rc}, warnings=${warning_count}, count=${reported_count}, want=${expected_count})"
fi

# 6. A busy lock skips preflight and defers cleanup without killing.
ready="${WORK}/lock-ready"
python3 - "${LOCK}" "${ready}" <<'PY' &
import fcntl
import os
import sys
import time

fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o666)
fcntl.flock(fd, fcntl.LOCK_EX)
open(sys.argv[2], "w").close()
while True:
    time.sleep(1)
PY
LOCK_HOLDER_PID=$!
for _ in 1 2 3 4 5 6 7 8 9 10; do
  [[ -f "${ready}" ]] && break
  sleep 0.1
done
start_helper normal "${resident_title}"
pid6="${STARTED_PID}"
out_pre="$(run_stop --mode preflight)"; rc_pre=$?; record "${out_pre}"
out_clean="$(run_stop --mode cleanup)"; rc_clean=$?; record "${out_clean}"
want_pre="::warning::the GPU lock is busy (a measurement is running on this box); leftover check skipped"
want_clean="::warning::the GPU lock is busy; cleanup deferred to the next job's preflight"
if [[ ${rc_pre} -eq 0 && "${out_pre}" == "${want_pre}" && ${rc_clean} -eq 3 && "${out_clean}" == "${want_clean}" ]] && is_alive "${pid6}"; then
  pass "6 -- busy lock uses exact outcomes"
else
  fail "6 -- busy lock uses exact outcomes (preflight=${rc_pre}, cleanup=${rc_clean})"
fi
kill "${LOCK_HOLDER_PID}" 2>/dev/null || true
wait "${LOCK_HOLDER_PID}" 2>/dev/null || true
LOCK_HOLDER_PID=""
out="$(run_stop --mode preflight)"; rc=$?; record "${out}"

# 8. Unknown options, a bad mode, and removed --lock all exit 2.
out_unknown="$("${STOP}" "--unknown=${SOCKET_PATH}" 2>&1)"; rc_unknown=$?; record "${out_unknown}"
out_mode="$("${STOP}" --mode bad --root "${JOB_ROOT}" 2>&1)"; rc_mode=$?; record "${out_mode}"
out_lock="$("${STOP}" --mode preflight --root "${JOB_ROOT}" --lock "${LOCK}" 2>&1)"; rc_lock=$?; record "${out_lock}"
if [[ ${rc_unknown} -eq 2 && ${rc_mode} -eq 2 && ${rc_lock} -eq 2 ]]; then
  pass "8 -- bad arguments"
else
  fail "8 -- bad arguments (unknown=${rc_unknown}, mode=${rc_mode}, lock=${rc_lock})"
fi

# 9. Lock open errors refuse preflight and defer cleanup without a traceback.
out_pre="$(RESIDENT_UP_LOCK_PATH="${WORK}/lock-directory" run_stop --mode preflight)"; rc_pre=$?
out_clean="$(RESIDENT_UP_LOCK_PATH="${WORK}/lock-directory" run_stop --mode cleanup)"; rc_clean=$?
record "${out_pre}"
record "${out_clean}"
if [[ ${rc_pre} -eq 1 && "${out_pre}" == "::error::cannot open the GPU lock ("*"; refusing" && "${out_pre}" != *"Traceback"* && ${rc_clean} -eq 3 && "${out_clean}" == "::warning::cannot open the GPU lock ("*"; cleanup deferred to the next job's preflight" && "${out_clean}" != *"Traceback"* ]]; then
  pass "9 -- lock open errors are handled"
else
  fail "9 -- lock open errors are handled (preflight=${rc_pre}, cleanup=${rc_clean})"
fi

# 10. The configured Qwen runner is allowed. A peer track survives.
qwen_runner="${WORK}/home/actions-runner-qwen38"
qwen_root="${qwen_runner}/_work/repo"
peer_root="${WORK}/home/actions-runner-peer/_work/repo"
mkdir -p "${qwen_root}/.build/release" "${peer_root}/.build/release"
qwen_title="${qwen_root}/.build/release/bench-worker resident"
peer_title="${peer_root}/.build/release/bench-worker resident"
start_helper normal "${qwen_title}"; pid10a="${STARTED_PID}"
start_helper normal "${peer_title}"; pid10b="${STARTED_PID}"
out="$(HOME="${WORK}/home" QWEN_RUNNER_DIR="${qwen_runner}" \
  MLXFAST_BASELINE_WORKSPACE="" "${STOP}" --mode preflight 2>&1)"; rc=$?; record "${out}"
if [[ ${rc} -eq 0 ]] && gone_within "${pid10a}" && is_alive "${pid10b}"; then
  pass "10 -- configured Qwen runner stops; peer runner survives"
else
  fail "10 -- configured Qwen runner stops; peer runner survives (rc=${rc})"
fi

# 11. A non-leader resident stops without its group owner or sibling.
owner_title="${WORK}/group-owner keep"
target_file="${WORK}/nonleader-target.pid"
sibling_file="${WORK}/nonleader-sibling.pid"
sibling_title="${WORK}/nonleader-sibling keep"
start_helper nonleader "${owner_title}" "${target_file}" "${resident_title}" "${sibling_file}" "${sibling_title}"
owner11="${STARTED_PID}"
target11="$(read_pid_file "${target_file}")"
sibling11="$(read_pid_file "${sibling_file}")"
[[ -z "${target11}" ]] || HELPERS+=("${target11}")
[[ -z "${sibling11}" ]] || HELPERS+=("${sibling11}")
[[ -z "${target11}" ]] || check_title "${target11}" "${resident_title}"
[[ -z "${sibling11}" ]] || check_title "${sibling11}" "${sibling_title}"
out="$(run_stop --mode preflight)"; rc=$?; record "${out}"
if [[ ${rc} -eq 0 && -n "${target11}" && -n "${sibling11}" ]] && gone_within "${target11}" && is_alive "${owner11}" && is_alive "${sibling11}"; then
  pass "11 -- non-leader stops alone"
else
  fail "11 -- non-leader stops alone (rc=${rc}, target=${target11:-missing}, sibling=${sibling11:-missing})"
fi

# 12. An explicit configured work folder is scanned.
configured_root="${WORK}/configured-runner-jobs/_work"
mkdir -p "${configured_root}/repo/repo/.build/release"
configured_title="${configured_root}/repo/repo/.build/release/bench-worker resident"
start_helper normal "${configured_title}"; pid12="${STARTED_PID}"
out="$("${STOP}" --mode preflight --root "${configured_root}" 2>&1)"; rc=$?; record "${out}"
if [[ ${rc} -eq 0 ]] && gone_within "${pid12}"; then
  pass "12 -- configured job root"
else
  fail "12 -- configured job root (rc=${rc}, pid=${pid12})"
fi

# 13. An explicit --root does not authorize a peer runner under the same HOME.
other_root="${WORK}/home13/actions-runner-other/_work/y"
job_root13="${WORK}/job13-runner/_work"
mkdir -p "${other_root}/.build/release" "${job_root13}/repo/.build/release"
start_helper normal "${other_root}/.build/release/bench-worker resident"; pid13a="${STARTED_PID}"
start_helper normal "${job_root13}/repo/.build/release/bench-worker resident"; pid13b="${STARTED_PID}"
out="$(HOME="${WORK}/home13" MLXFAST_BASELINE_WORKSPACE="" "${STOP}" --mode preflight --root "${job_root13}" 2>&1)"; rc=$?; record "${out}"
if [[ ${rc} -eq 0 ]] && is_alive "${pid13a}" && gone_within "${pid13b}"; then
  pass "13 -- explicit root leaves peer runner alive"
else
  fail "13 -- explicit root leaves peer runner alive (rc=${rc})"
fi

# 7. Public output from all cases hides the socket, root, and lock paths.
if [[ "${ALL_OUTPUT}" != *"${SOCKET_PATH}"* && "${ALL_OUTPUT}" != *"${JOB_ROOT}"* && "${ALL_OUTPUT}" != *"${LOCK}"* ]]; then
  pass "7 -- output hides all private paths"
else
  fail "7 -- output hides all private paths"
fi

if [[ ${fails} -ne 0 ]]; then
  printf 'test-stop-leaked-residents: %d case(s) failed\n' "${fails}" >&2
  exit 1
fi
echo "test-stop-leaked-residents: all 13 cases passed"
