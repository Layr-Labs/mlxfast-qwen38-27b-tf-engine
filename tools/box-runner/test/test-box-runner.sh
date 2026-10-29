#!/usr/bin/env bash
# test-box-runner.sh -- dry-run test suite for tools/box-runner/.
#
# Exercises supervisor.sh's spawn/stop/status logic and lib/pidtree.sh's
# kill-tree primitive against MOCK processes only (mock-run.sh,
# mock-orphan-child.sh, mock-gh-app-mint.sh, all in this directory) -- no
# GitHub network access, no GitHub App credential, and no real actions-runner
# binary. The default path needs no root. An opt-in root section exercises the
# real privilege drop when BOX_RUNNER_TEST_USER names a non-root user.
#
# REQUIRED detached-child coverage: section 1b proves the kill-tree mechanism
# reaps a detached grandchild of the tracked job root. Section 1c is the
# negative control: it shows that
# killing only the tracked root pid does not
# reap the grandchild, so 1b is not a vacuous pass. Section 2 repeats the
# same proof end-to-end through supervisor.sh's actual `run`/`stop`/`status`
# entrypoints, including calling `stop` from a separate process invocation
# from the one running the job loop. A restarted daemon can therefore reap a
# tree that another process was tracking.
#
# Usage: tools/box-runner/test/test-box-runner.sh [-v]
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null && pwd -P)"
BOX_RUNNER_DIR="$(cd -- "${SCRIPT_DIR}/.." >/dev/null && pwd -P)"
VERBOSE=0
[[ "${1:-}" == "-v" ]] && VERBOSE=1

# shellcheck source=tools/box-runner/lib/pidtree.sh
. "${BOX_RUNNER_DIR}/lib/pidtree.sh"

# See tools/test-submission-security.sh for why this floor exists: it is the
# only thing that notices a section silently stopping (an early `exit`, a
# deleted block) under `set -uo pipefail` without `-e`, where a truncated
# suite otherwise reports "0 failed" and looks green.
EXPECTED_MIN_ASSERTIONS=80

WORK="$(mktemp -d "${TMPDIR:-/tmp}/box-runner-test.XXXXXX")"

# The test sandbox can deny process-table reads. Identity tests still need a
# stable ps contract. This mock covers the fields the implementation reads.
identity_mock_bin="${WORK}/identity-mock-bin"
mkdir -p "${identity_mock_bin}"
# shellcheck disable=SC2016  # generated mock expands its own arguments.
printf '%s\n' \
  '#!/usr/bin/env bash' \
  'set -u' \
  'field=""; pid=""; prev=""' \
  'for arg in "$@"; do' \
  '  [ "${prev}" = "-o" ] && field="${arg}"' \
  '  [ "${prev}" = "-p" ] && pid="${arg}"' \
  '  prev="${arg}"' \
  'done' \
  'case "${field}" in' \
  '  pid=) kill -0 "${pid}" 2>/dev/null && printf " %s\n" "${pid}" ;;' \
  '  lstart=) kill -0 "${pid}" 2>/dev/null && printf "TEST-START-%s\n" "${pid}" ;;' \
  '  ppid=) /bin/ps -o ppid= -p "${pid}" ;;' \
  '  command=) kill -0 "${pid}" 2>/dev/null || exit 1;' \
  '    if [ -n "${IDENTITY_COMMAND_FILE:-}" ] && [ -f "${IDENTITY_COMMAND_FILE}" ]; then cat "${IDENTITY_COMMAND_FILE}"; else printf "TEST-COMMAND-%s\n" "${pid}"; fi ;;' \
  '  *) exit 1 ;;' \
  'esac' \
  > "${identity_mock_bin}/ps"
chmod 0755 "${identity_mock_bin}/ps"
PATH="${identity_mock_bin}:${PATH}"
export PATH

state_file_pid() {
  local path="$1" first
  first="$(sed -n '1p' "${path}" 2>/dev/null)"
  case "${first}" in
    pid=*) printf '%s\n' "${first#pid=}" ;;
    *) printf '%s\n' "${first}" ;;
  esac
}

# shellcheck disable=SC2329  # invoked indirectly via `trap cleanup EXIT` below
cleanup() {
  # Belt and suspenders: every test below is expected to clean up its own
  # processes: this just makes sure nothing outlives the suite even if one
  # of them fails partway through.
  local pf
  for pf in "${WORK}"/*/job.pid "${WORK}"/*/mock.pid "${WORK}"/*/orphan.pid \
            "${WORK}"/*-state/job.pid "${WORK}"/*cat-pid; do
    [ -f "${pf}" ] || continue
    kill -KILL "$(state_file_pid "${pf}")" 2>/dev/null || true
  done
  for pf in "${WORK}"/*jit-path; do
    [ -f "${pf}" ] || continue
    case "$(cat "${pf}" 2>/dev/null)" in
      /tmp/qwen38-jitconfig.*|/private/tmp/qwen38-jitconfig.*)
        rm -f -- "$(cat "${pf}" 2>/dev/null)"
        ;;
    esac
  done
  chmod -R u+w "${WORK}" 2>/dev/null || true
  rm -rf "${WORK}"
}
trap cleanup EXIT

PASSED=0
FAILED=0
SKIPPED=0
FAILURES=()
PROCESS_ENUMERATION_AVAILABLE=1
pgrep -P 1 >/dev/null 2>&1
[[ $? -le 1 ]] || PROCESS_ENUMERATION_AVAILABLE=0

pass() { PASSED=$((PASSED + 1)); [ "${VERBOSE}" -eq 1 ] && printf 'PASS  %s\n' "$1"; }
fail_case() {
  if [ "${PROCESS_ENUMERATION_AVAILABLE}" -eq 0 ]; then
    case "$1" in
      *"pidtree_collect includes orphan"*|*"DETACHED ORPHAN pid dead"*|*"held mock cat process is dead"*)
        SKIPPED=$((SKIPPED + 1))
        printf 'SKIP  %s (process enumeration is blocked)\n' "$1"
        return
        ;;
    esac
  fi
  FAILED=$((FAILED + 1))
  FAILURES+=("$1")
  printf 'FAIL  %s\n' "$1"
}

assert_true() {
  local name="$1"; shift
  if "$@"; then pass "${name}"; else fail_case "${name} (expected success: $*)"; fi
}

assert_false() {
  local name="$1"; shift
  if "$@"; then fail_case "${name} (expected failure: $*)"; else pass "${name}"; fi
}

assert_eq() {
  local name="$1" want="$2" got="$3"
  if [ "${got}" = "${want}" ]; then
    pass "${name}"
  else
    fail_case "${name} (want '${want}', got '${got}')"
  fi
}

# assert_exit NAME WANT_STATUS WANT_SUBSTRING CMD...
assert_exit() {
  local name="$1" want="$2" pattern="$3"
  shift 3
  local out status
  out="$("$@" 2>&1)"
  status=$?
  [ "${VERBOSE}" -eq 1 ] && printf -- '--- %s (exit %d)\n%s\n' "${name}" "${status}" "${out}"
  if [ "${status}" != "${want}" ]; then
    fail_case "${name} (exit ${status}, wanted ${want}) :: ${out}"
    return
  fi
  if [ -n "${pattern}" ]; then
    case "${out}" in
      *"${pattern}"*) : ;;
      *) fail_case "${name} (missing '${pattern}' in output) :: ${out}"; return ;;
    esac
  fi
  pass "${name}"
}

# The four helpers below are only ever invoked indirectly, as the CMD... of
# assert_true/assert_false ("$@" further down each), so shellcheck cannot
# see the call site; SC2329 ("never invoked") is a false positive on all of
# them for that reason.
# shellcheck disable=SC2329
pid_alive() { kill -0 "$1" 2>/dev/null; }
# shellcheck disable=SC2329
pid_dead() { ! kill -0 "$1" 2>/dev/null; }
# shellcheck disable=SC2329
contains() { case "$1" in *"$2"*) return 0 ;; *) return 1 ;; esac; }
# list_has LIST NEEDLE -- LIST is newline-separated; true iff one line
# equals NEEDLE exactly (used for pid-list membership, where a substring
# match could false-positive on e.g. pid 12 matching pid 123).
# shellcheck disable=SC2329
list_has() { printf '%s\n' "$1" | grep -qx -- "$2"; }

wait_for_file() {
  local path="$1" timeout="${2:-5}" waited=0
  while [ ! -s "${path}" ]; do
    waited=$((waited + 1))
    [ "${waited}" -gt $((timeout * 10)) ] && return 1
    sleep 0.1
  done
  return 0
}

wait_for_pid_dead() {
  local pid="$1" timeout="${2:-5}" waited=0
  while kill -0 "${pid}" 2>/dev/null; do
    waited=$((waited + 1))
    [ "${waited}" -gt $((timeout * 10)) ] && return 1
    sleep 0.1
  done
  return 0
}

file_mode() {
  if [ "$(uname -s)" = "Darwin" ]; then
    stat -f '%Lp' "$1"
  else
    stat -c '%a' "$1"
  fi
}

file_uid() {
  if [ "$(uname -s)" = "Darwin" ]; then
    stat -f '%u' "$1"
  else
    stat -c '%u' "$1"
  fi
}

# =========================================================================
# Section 1: lib/pidtree.sh primitives, directly, against mock-run.sh
# =========================================================================

# start_mock_tree STATE_DIR -- starts mock-run.sh (which spawns a detached
# mock-orphan-child.sh grandchild) and blocks until both have recorded their
# own pids, printing mock-run.sh's pid on stdout.
start_mock_tree() {
  local state_dir="$1"
  mkdir -p "${state_dir}"
  "${SCRIPT_DIR}/mock-run.sh" "${state_dir}" \
    > "${state_dir}/mock-run.log" 2>&1 &
  wait_for_file "${state_dir}/mock.pid" 5 || { echo "mock.pid never appeared" >&2; return 1; }
  wait_for_file "${state_dir}/orphan.pid" 5 || { echo "orphan.pid never appeared" >&2; return 1; }
  cat "${state_dir}/mock.pid"
}

# --- 1a. pidtree_collect finds root + orphan while alive -------------------
t1_dir="${WORK}/t1"
root_pid="$(start_mock_tree "${t1_dir}")"
orphan_pid="$(cat "${t1_dir}/orphan.pid")"
collected="$(pidtree_collect "${root_pid}")"
assert_true "pidtree_collect includes root pid" list_has "${collected}" "${root_pid}"
assert_true "pidtree_collect includes orphan grandchild pid" list_has "${collected}" "${orphan_pid}"
pidtree_kill_tree "${root_pid}" 5 >/dev/null 2>&1 || true
nonexistent_collected="$(pidtree_collect 999999)"
assert_eq "pidtree_collect: nonexistent pid returns nothing" "" "${nonexistent_collected}"

# kill -0 returns EPERM for another uid. ps still proves the pid exists.
# shellcheck disable=SC2016  # nested shell expands its own function arguments.
assert_true "pidtree_alive: EPERM is alive when ps sees the pid" bash -c '
  set -u
  # shellcheck disable=SC1090
  . "$1"
  kill() { return 1; }
  ps() { printf " 4242\n"; }
  pidtree_alive 4242
' _ "${BOX_RUNNER_DIR}/lib/pidtree.sh"

# --- 1b. THE REQUIRED REGRESSION TEST: kill-tree reaps a detached orphan ---
t2_dir="${WORK}/t2"
root_pid="$(start_mock_tree "${t2_dir}")"
orphan_pid="$(cat "${t2_dir}/orphan.pid")"
assert_true "orphan-kill setup: root alive before stop" pid_alive "${root_pid}"
assert_true "orphan-kill setup: orphan alive before stop" pid_alive "${orphan_pid}"
assert_false "orphan-kill setup: orphan is NOT a direct child of the test shell (pgrep -P misses it from here)" \
  bash -c "pgrep -P \$\$ | grep -qx '${orphan_pid}'"

pidtree_kill_tree "${root_pid}" 5
kill_tree_status=$?
assert_eq "orphan-kill: pidtree_kill_tree reports success" "0" "${kill_tree_status}"
sleep 0.3
assert_true "orphan-kill: root pid dead after kill_tree" pid_dead "${root_pid}"
assert_true "orphan-kill: DETACHED ORPHAN pid dead after kill_tree" pid_dead "${orphan_pid}"

# A mock su exits on TERM. Its saved child ignores TERM and becomes orphaned.
# The stop primitive must retain that child's identity through SIGKILL.
# shellcheck disable=SC2016  # nested shell owns the mock process table.
assert_true "TERM exit: saved TERM-ignoring child is KILLed after mock su exits" bash -c '
  set -u
  . "$1"
  root_alive=1
  child_alive=1
  pidtree_alive() {
    case "$1" in 100) [ "${root_alive}" -eq 1 ] ;; 200) [ "${child_alive}" -eq 1 ] ;; esac
  }
  pidtree_start_time() { case "$1" in 100) printf ROOT ;; 200) printf CHILD ;; esac; }
  pidtree_command_identity() { printf mock-su; }
  pidtree_collect_identities() {
    [ "${root_alive}" -eq 1 ] && printf "100\tROOT\n200\tCHILD\n"
  }
  kill() {
    sig="$2"
    pid="$3"
    case "${sig}:${pid}" in
      TERM:100) root_alive=0 ;;
      TERM:200) : ;;
      KILL:200) child_alive=0 ;;
    esac
  }
  sleep() { :; }
  pidtree_kill_tree 100 0 ROOT mock-su || exit 1
  [ "${child_alive}" -eq 0 ]
' _ "${BOX_RUNNER_DIR}/lib/pidtree.sh"

# --- 1c. Negative control: naive single-pid kill leaves the orphan alive --
#     (proves 1b is not vacuous -- this is the detached-child case:
#     killing only the tracked pid, or relying on a process-group signal
#     that never reaches a re-sessioned descendant, leaves it running.)
t3_dir="${WORK}/t3"
root_pid="$(start_mock_tree "${t3_dir}")"
orphan_pid="$(cat "${t3_dir}/orphan.pid")"
kill -KILL "${root_pid}" 2>/dev/null
sleep 0.3
assert_true "negative control: naive kill removes only the root" pid_dead "${root_pid}"
assert_true "negative control: naive kill LEAVES the detached orphan alive (the bug)" pid_alive "${orphan_pid}"
kill -KILL "${orphan_pid}" 2>/dev/null || true # clean up what the negative control deliberately left running

# --- 1d. a reused descendant pid is not signalled -------------------------
# shellcheck disable=SC2016  # nested shell owns all mock function variables.
assert_true "pid reuse: descendant identity is checked before signal" bash -c '
  set -u
  . "$1"
  signal_log="$2"
  root_alive=1
  pidtree_alive() { [ "$1" = 200 ] || [ "${root_alive}" -eq 1 ]; }
  pidtree_start_time() { if [ "$1" = 100 ]; then printf ROOT; else printf NEW-CHILD; fi; }
  pidtree_command_identity() { printf root-command; }
  pidtree_collect() { [ "${root_alive}" -eq 1 ] && printf "100\n200\n"; }
  pidtree_collect_identities() { [ "${root_alive}" -eq 1 ] && printf "100\tROOT\n200\tOLD-CHILD\n"; }
  kill() {
    printf "%s\n" "$*" >> "${signal_log}"
    case "$*" in *" 100") root_alive=0 ;; esac
  }
  sleep() { :; }
  pidtree_kill_tree 100 0 ROOT root-command || exit 1
  ! grep -q " 200$" "${signal_log}"
' _ "${BOX_RUNNER_DIR}/lib/pidtree.sh" "${WORK}/reused-child-signals"

# The root can be reused after the grace-period identity check but before the
# refresh walk. The replacement tree must not be collected or signalled.
# shellcheck disable=SC2016  # nested shell owns the mock process table.
assert_true "pid reuse: grace refresh rejects a replacement root tree" bash -c '
  set -u
  . "$1"
  state_dir="$2"
  signal_log="$3"
  mkdir -p "${state_dir}"
  pidtree_alive() {
    case "$1" in
      100) [ ! -e "${state_dir}/root-dead" ] ;;
      200) [ -e "${state_dir}/reused" ] && [ ! -e "${state_dir}/child-dead" ] ;;
      *) return 1 ;;
    esac
  }
  pidtree_start_time() {
    case "$1" in
      100)
        if [ -e "${state_dir}/reused" ]; then
          printf PEER-ROOT
        elif [ -e "${state_dir}/term-sent" ]; then
          if [ -e "${state_dir}/grace-checked" ]; then
            : > "${state_dir}/reused"
            printf PEER-ROOT
          else
            : > "${state_dir}/grace-checked"
            printf ROOT
          fi
        else
          printf ROOT
        fi
        ;;
      200) printf PEER-CHILD ;;
    esac
  }
  pidtree_command_identity() { printf root-command; }
  pidtree_children() {
    [ "$1" = 100 ] && [ -e "${state_dir}/reused" ] && printf "200\n"
  }
  pidtree_parent_pid() { [ "$1" = 200 ] && printf 100; }
  kill() {
    printf "%s\n" "$*" >> "${signal_log}"
    case "$*" in
      "-s TERM 100") : > "${state_dir}/term-sent" ;;
      "-s KILL 100") : > "${state_dir}/root-dead" ;;
      "-s KILL 200") : > "${state_dir}/child-dead" ;;
    esac
  }
  sleep() { :; }
  pidtree_kill_tree 100 1 ROOT root-command
  stop_rc=$?
  # The saved root is gone (its pid now names a peer), so stop succeeds, and
  # the peer tree is never collected or signalled.
  [ "${stop_rc}" -eq 0 ] || exit 1
  [ ! -e "${state_dir}/child-dead" ] || exit 1
  ! grep -q " 200$" "${signal_log}"
' _ "${BOX_RUNNER_DIR}/lib/pidtree.sh" "${WORK}/reused-root-state" "${WORK}/reused-root-signals"

# The root can exit and be reused during a grace refresh while a saved child
# ignores TERM. The saved child must still be KILLed; the peer is untouched.
# shellcheck disable=SC2016  # nested shell owns the mock process table.
assert_true "pid reuse: saved TERM-ignoring child is KILLed after the root is reused" bash -c '
  set -u
  . "$1"
  state_dir="$2"
  signal_log="$3"
  mkdir -p "${state_dir}"
  pidtree_alive() {
    case "$1" in
      100) return 0 ;;
      300) [ ! -e "${state_dir}/saved-dead" ] ;;
      200) [ -e "${state_dir}/reused" ] ;;
      *) return 1 ;;
    esac
  }
  pidtree_start_time() {
    case "$1" in
      100)
        if [ -e "${state_dir}/reused" ]; then
          printf PEER-ROOT
        elif [ -e "${state_dir}/term-sent" ]; then
          if [ -e "${state_dir}/grace-checked" ]; then
            : > "${state_dir}/reused"
            printf PEER-ROOT
          else
            : > "${state_dir}/grace-checked"
            printf ROOT
          fi
        else
          printf ROOT
        fi
        ;;
      300) [ -e "${state_dir}/saved-dead" ] || printf SAVED-CHILD ;;
      200) printf PEER-CHILD ;;
    esac
  }
  pidtree_command_identity() { printf root-command; }
  pidtree_children() {
    if [ "$1" = 100 ]; then
      if [ -e "${state_dir}/reused" ]; then printf "200\n"; else printf "300\n"; fi
    fi
  }
  pidtree_parent_pid() { case "$1" in 300|200) printf 100 ;; esac; }
  kill() {
    printf "%s\n" "$*" >> "${signal_log}"
    case "$*" in
      "-s TERM 100") : > "${state_dir}/term-sent" ;;
      "-s KILL 300") : > "${state_dir}/saved-dead" ;;
    esac
  }
  sleep() { :; }
  pidtree_kill_tree 100 1 ROOT root-command
  stop_rc=$?
  [ "${stop_rc}" -eq 0 ] || exit 1
  [ -e "${state_dir}/saved-dead" ] || exit 1
  ! grep -q " 200$" "${signal_log}"
' _ "${BOX_RUNNER_DIR}/lib/pidtree.sh" "${WORK}/reused-root-saved-state" "${WORK}/reused-root-saved-signals"

# A new child that the grace refresh collected while the root was still valid
# stays tracked after the root pid is reused, and is KILLed. Reads 1-4 of the
# root start time after TERM (the refresh check and the collector's checks)
# see the saved root; later reads see a reused pid.
# shellcheck disable=SC2016  # nested shell owns the mock process table.
assert_true "pid reuse: fresh child collected before the root is reused is KILLed" bash -c '
  set -u
  . "$1"
  state_dir="$2"
  signal_log="$3"
  mkdir -p "${state_dir}"
  : > "${state_dir}/reads"
  pidtree_alive() {
    case "$1" in
      100) return 0 ;;
      400) [ ! -e "${state_dir}/fresh-dead" ] ;;
      *) return 1 ;;
    esac
  }
  pidtree_start_time() {
    case "$1" in
      100)
        if [ -e "${state_dir}/term-sent" ]; then
          printf x >> "${state_dir}/reads"
          if [ "$(wc -c < "${state_dir}/reads" | tr -d " ")" -le 4 ]; then printf ROOT; else printf PEER-ROOT; fi
        else
          printf ROOT
        fi
        ;;
      400) [ -e "${state_dir}/fresh-dead" ] || printf FRESH-CHILD ;;
    esac
  }
  pidtree_command_identity() { printf root-command; }
  pidtree_children() {
    [ "$1" = 100 ] && [ -e "${state_dir}/term-sent" ] && printf "400\n"
  }
  pidtree_parent_pid() { [ "$1" = 400 ] && printf 100; }
  kill() {
    printf "%s\n" "$*" >> "${signal_log}"
    case "$*" in
      "-s TERM 100") : > "${state_dir}/term-sent" ;;
      "-s KILL 400") : > "${state_dir}/fresh-dead" ;;
    esac
  }
  sleep() { :; }
  pidtree_kill_tree 100 1 ROOT root-command
  stop_rc=$?
  [ "${stop_rc}" -eq 0 ] || exit 1
  [ -e "${state_dir}/fresh-dead" ]
' _ "${BOX_RUNNER_DIR}/lib/pidtree.sh" "${WORK}/reused-root-fresh-state" "${WORK}/reused-root-fresh-signals"

# A parent can be reused after pgrep returns its children. Its new children
# are peer processes and must not enter the tracked tree.
# shellcheck disable=SC2016  # nested shell owns the mock process table.
assert_true "pid reuse: changed parent and its child are not signalled" bash -c '
  set -u
  . "$1"
  reuse_marker="$2"
  signal_log="$3"
  pidtree_alive() { return 0; }
  pidtree_start_time() {
    case "$1" in
      100) printf ROOT ;;
      200) [ -e "${reuse_marker}" ] && printf NEW-PARENT || printf OLD-PARENT ;;
      300) printf PEER-CHILD ;;
    esac
  }
  pidtree_parent_pid() {
    case "$1" in 200) printf 100 ;; 300) printf 200 ;; esac
  }
  pidtree_children() {
    case "$1" in
      100) printf "200\n" ;;
      200) : > "${reuse_marker}"; printf "300\n" ;;
    esac
  }
  kill() { printf "%s\n" "$*" >> "${signal_log}"; }
  entries=()
  while IFS= read -r entry; do [ -n "${entry}" ] && entries+=("${entry}"); done \
    < <(pidtree_collect_identities 100)
  pidtree_signal_identities TERM "${entries[@]}"
  ! grep -Eq " (200|300)$" "${signal_log}"
' _ "${BOX_RUNNER_DIR}/lib/pidtree.sh" "${WORK}/reused-parent" "${WORK}/reused-parent-signals"

# lstart reports only whole seconds. The kernel identity must reject a PID
# that is reused within that same second but has a different microsecond.
# shellcheck disable=SC2016  # nested shell owns the mock kernel identity.
assert_true "pid reuse: same-second microsecond change does not match" bash -c '
  set -u
  . "$1"
  marker="$2"
  pidtree_alive() { return 0; }
  ps() {
    case "$*" in *"lstart="*) printf "Wed Oct 28 17:31:03 2026\n" ;; esac
  }
  pidtree_kernel_start_time() {
    [ -e "${marker}" ] && printf "1793223063.900002\n" || printf "1793223063.900001\n"
  }
  saved="$(pidtree_start_time 200)"
  : > "${marker}"
  ! pidtree_start_identity_matches 200 "${saved}"
' _ "${BOX_RUNNER_DIR}/lib/pidtree.sh" "${WORK}/same-second-reuse"

# shellcheck disable=SC2016  # nested shell owns the mock kernel identity.
assert_true "pid identity: unknown kernel start time does not match" bash -c '
  set -u
  . "$1"
  pidtree_alive() { return 0; }
  pidtree_kernel_start_time() { return 1; }
  ! pidtree_start_identity_matches 200 "saved"
' _ "${BOX_RUNNER_DIR}/lib/pidtree.sh"

# =========================================================================
# Section 2: supervisor.sh end-to-end (spawn / status / stop), test mode
# =========================================================================

# --- 2a. relative log/state paths are anchored at supervisor.sh ------------
relative_stage="${WORK}/relative-stage"
mkdir -p "${relative_stage}/lib"
cp "${BOX_RUNNER_DIR}/supervisor.sh" "${relative_stage}/supervisor.sh"
cp "${BOX_RUNNER_DIR}/lib/pidtree.sh" "${relative_stage}/lib/pidtree.sh"
(
  cd "${WORK}" || exit 1
  env LOG_DIR=./log STATE_DIR=./state BOX_RUNNER_CONFIG=/nonexistent \
    "${relative_stage}/supervisor.sh" status >/dev/null
)
assert_true "relative LOG_DIR resolves under SCRIPT_DIR" test -d "${relative_stage}/log"
assert_true "relative STATE_DIR resolves under SCRIPT_DIR" test -d "${relative_stage}/state"

run_dir="${WORK}/runner-dir"
mkdir -p "${run_dir}"
# Real copies, not symlinks: mock-run.sh locates mock-orphan-child.sh next
# to itself (mirroring how a real actions-runner install keeps its helper
# scripts beside run.sh), so both must live in RUNNER_DIR together.
cp "${SCRIPT_DIR}/mock-run.sh" "${run_dir}/run.sh"
cp "${SCRIPT_DIR}/mock-orphan-child.sh" "${run_dir}/mock-orphan-child.sh"
chmod +x "${run_dir}/run.sh" "${run_dir}/mock-orphan-child.sh"

# Record the path passed to cat when the runner reads its JIT config. The
# wrapper then delegates to the system cat unchanged.
mock_bin="${WORK}/mock-bin"
mock_jit_path_file="${WORK}/mock-jit-path"
mkdir -p "${mock_bin}"
# shellcheck disable=SC2016  # deferred: mock cat expands these at run time
printf '%s\n' \
  '#!/usr/bin/env bash' \
  'case "${1:-}" in' \
  '  *jitconfig*)' \
  '    printf "%s\n" "$1" > "${MOCK_JIT_PATH_FILE}"' \
  '    if [ "${MOCK_JIT_HOLD:-0}" = "1" ]; then' \
  '      printf "%s\n" "$$" > "${MOCK_JIT_CAT_PID_FILE}"' \
  '      waited=0' \
  '      while [ ! -e "${MOCK_JIT_RELEASE_FILE}" ] && [ "${waited}" -lt 300 ]; do' \
  '        sleep 0.1' \
  '        waited=$((waited + 1))' \
  '      done' \
  '    fi' \
  '    ;;' \
  'esac' \
  'exec /bin/cat "$@"' \
  > "${mock_bin}/cat"
chmod +x "${mock_bin}/cat"

e2e_state="${WORK}/e2e-state"
sup_log_dir="${WORK}/sup-log"
sup_state_dir="${WORK}/sup-state"
mkdir -p "${e2e_state}"

run_supervisor() {
  env \
    RUNNER_SU_ENABLED=0 \
    RUNNER_DIR="${run_dir}" \
    RUNNER_NAME="test-box-1" \
    RUNNER_LABELS="qwen38-27b-tf-box" \
    GH_APP_MINT="${SCRIPT_DIR}/mock-gh-app-mint.sh" \
    MOCK_MINT_LOG="${WORK}/mock-mint.log" \
    PATH="${mock_bin}:${PATH}" \
    MOCK_JIT_PATH_FILE="${mock_jit_path_file}" \
    LOG_DIR="${sup_log_dir}" \
    STATE_DIR="${sup_state_dir}" \
    MOCK_STATE_DIR="${e2e_state}" \
    PAUSE_BETWEEN_JOBS=1 \
    BOX_RUNNER_MAX_ITERATIONS=1 \
    BOX_RUNNER_CONFIG=/nonexistent \
    "${BOX_RUNNER_DIR}/supervisor.sh" "$@"
}

# --- 2b. status is idle before anything has run -----------------------------
out="$(run_supervisor status)"
assert_eq "status: idle before any job" "idle" "${out}"

# --- 2c. spawn a job (background, single-iteration loop) --------------------
run_supervisor run > "${WORK}/supervisor-run.log" 2>&1 &
sup_pid=$!
wait_for_file "${sup_state_dir}/job.pid" 5
assert_true "supervisor: job.pid appears" test -s "${sup_state_dir}/job.pid"
wait_for_file "${e2e_state}/mock.pid" 5
wait_for_file "${e2e_state}/orphan.pid" 5
wait_for_file "${mock_jit_path_file}" 5
wait_for_file "${WORK}/mock-mint.log" 5
job_pid="$(state_file_pid "${sup_state_dir}/job.pid")"
mock_pid="$(cat "${e2e_state}/mock.pid")"
orphan_pid="$(cat "${e2e_state}/orphan.pid")"
jit_file="$(/bin/cat "${mock_jit_path_file}" 2>/dev/null)"
assert_eq "supervisor: minted name equals stable RUNNER_NAME" \
  "jitconfig test-box-1 qwen38-27b-tf-box" \
  "$(sed -n '1p' "${WORK}/mock-mint.log")"
assert_true "supervisor: mock runner reads a temporary JIT config" test -n "${jit_file}"
# shellcheck disable=SC2016  # deferred: nested bash expands its arguments
assert_false "supervisor: JIT config path is outside STATE_DIR" \
  bash -c 'case "$1" in "$2"/*) exit 0;; *) exit 1;; esac' _ "${jit_file}" "${sup_state_dir}"
# The tracked root pid is whatever `exec` ultimately replaced spawn_job's
# background subshell with -- mock-run.sh itself when no timeout(1)/
# gtimeout(1) is on PATH, or a `timeout` WRAPPER process (a distinct pid)
# ahead of it when one is (as it is on this dev machine, via Homebrew
# coreutils). Both are correct: what matters is that the tracked pid's tree
# includes mock-run.sh, which is what the process-tree check depends on.
job_tree="$(pidtree_collect "${job_pid}")"
assert_true "supervisor: tracked root's tree includes mock-run.sh's pid" list_has "${job_tree}" "${mock_pid}"
assert_true "supervisor: mock-run.sh alive while job in progress" pid_alive "${mock_pid}"
assert_true "supervisor: detached orphan alive while job in progress" pid_alive "${orphan_pid}"

# --- 2d. status while running lists the tree --------------------------------
status_out="$(run_supervisor status)"
# shellcheck disable=SC2016  # deliberate: $0 must expand inside the nested
# bash -c (bound there to status_out, passed as its NAME argument), not here.
assert_true "status while running: reports 'running'" bash -c 'case "$0" in running:*) exit 0;; *) exit 1;; esac' "${status_out}"
assert_true "status while running: lists the mock-run.sh pid" contains "${status_out}" "${mock_pid}"

# --- 2e. stop, called as a SEPARATE process invocation (the standalone
#     entrypoint -- the recovery case: a separate command or a restarted
#     `run`, reaping a tree a DIFFERENT process instance is tracking) -------
stop_out="$(run_supervisor stop)"
assert_true "stop: reports stopped" contains "${stop_out}" "stopped"
sleep 0.3
assert_true "stop: mock-run.sh pid dead" pid_dead "${mock_pid}"
assert_true "stop: DETACHED ORPHAN pid dead (kill-tree via standalone stop)" pid_dead "${orphan_pid}"
assert_false "stop: job.pid file removed" test -f "${sup_state_dir}/job.pid"
assert_false "stop: temporary JIT config is removed after the job ends" test -e "${jit_file}"

# --- 2f. status is idle again after stop ------------------------------------
final_status="$(run_supervisor status)"
assert_eq "status: idle again after stop" "idle" "${final_status}"

# Reap the backgrounded `run` loop itself. Its single job was already killed
# by `stop` above; BOX_RUNNER_MAX_ITERATIONS=1 makes the loop exit on its
# own within PAUSE_BETWEEN_JOBS=1s, so this is a bounded wait, not a hang.
kill "${sup_pid}" 2>/dev/null || true
wait "${sup_pid}" 2>/dev/null || true

# --- 2g. TERM removes a JIT file before the job can read or unlink it ------
held_state="${WORK}/held-state"
held_log_dir="${WORK}/held-log"
held_sup_state_dir="${WORK}/held-sup-state"
held_jit_path_file="${WORK}/held-jit-path"
held_cat_pid_file="${WORK}/held-cat-pid"
held_release_file="${WORK}/held-release"
mkdir -p "${held_state}"

env \
  RUNNER_SU_ENABLED=0 \
  RUNNER_DIR="${run_dir}" \
  RUNNER_NAME="test-box-1" \
  RUNNER_LABELS="qwen38-27b-tf-box" \
  GH_APP_MINT="${SCRIPT_DIR}/mock-gh-app-mint.sh" \
  PATH="${mock_bin}:${PATH}" \
  MOCK_JIT_PATH_FILE="${held_jit_path_file}" \
  MOCK_JIT_HOLD=1 \
  MOCK_JIT_CAT_PID_FILE="${held_cat_pid_file}" \
  MOCK_JIT_RELEASE_FILE="${held_release_file}" \
  LOG_DIR="${held_log_dir}" \
  STATE_DIR="${held_sup_state_dir}" \
  MOCK_STATE_DIR="${held_state}" \
  PAUSE_BETWEEN_JOBS=0 \
  STOP_TERM_TIMEOUT_SECONDS=1 \
  BOX_RUNNER_MAX_ITERATIONS=1 \
  BOX_RUNNER_CONFIG=/nonexistent \
  "${BOX_RUNNER_DIR}/supervisor.sh" run > "${WORK}/held-supervisor.log" 2>&1 &
held_sup_pid=$!
wait_for_file "${held_jit_path_file}" 5
wait_for_file "${held_cat_pid_file}" 5
wait_for_file "${held_sup_state_dir}/job.pid" 5
held_jit_file="$(/bin/cat "${held_jit_path_file}" 2>/dev/null)"
held_cat_pid="$(/bin/cat "${held_cat_pid_file}" 2>/dev/null)"
held_job_pid="$(state_file_pid "${held_sup_state_dir}/job.pid")"
assert_true "TERM cleanup: held JIT config path is recorded" test -n "${held_jit_file}"
assert_true "TERM cleanup: held JIT config exists before TERM" test -f "${held_jit_file}"
assert_eq "TERM cleanup: held JIT config mode is 600" "600" "$(file_mode "${held_jit_file}")"

kill -TERM "${held_sup_pid}"
held_term_bounded=1
if ! wait_for_pid_dead "${held_sup_pid}" 5; then
  held_term_bounded=0
  kill -KILL "${held_sup_pid}" 2>/dev/null || true
fi
wait "${held_sup_pid}" 2>/dev/null
held_term_status=$?
assert_eq "TERM cleanup: supervisor exits with TERM status" "143" "${held_term_status}"
assert_eq "TERM cleanup: supervisor exits within five seconds" "1" "${held_term_bounded}"
assert_false "TERM cleanup: EXIT trap removes held JIT config" test -e "${held_jit_file}"
held_cat_dead=1
wait_for_pid_dead "${held_cat_pid}" 2 || held_cat_dead=0
held_job_dead=1
wait_for_pid_dead "${held_job_pid}" 2 || held_job_dead=0
assert_eq "TERM cleanup: held mock cat process is dead" "1" "${held_cat_dead}"
assert_eq "TERM cleanup: tracked mock job process is dead" "1" "${held_job_dead}"

# --- 2h. stale identity never redirects a signal to a reused pid -----------
stale_log_dir="${WORK}/stale-log"
stale_state_dir="${WORK}/stale-state"
mkdir -p "${stale_log_dir}" "${stale_state_dir}"
sleep 300 &
stale_pid=$!
stale_command="$(pidtree_command_identity "${stale_pid}")"

run_stale_stop() {
  env LOG_DIR="${stale_log_dir}" STATE_DIR="${stale_state_dir}" \
    BOX_RUNNER_CONFIG=/nonexistent "${BOX_RUNNER_DIR}/supervisor.sh" stop
}

printf 'pid=%s\nstart=%s\ncommand=%s\n' \
  "${stale_pid}" "wrong start" "${stale_command}" > "${stale_state_dir}/job.pid"
stale_out="$(run_stale_stop)"
assert_true "stale state: start-time mismatch signals nothing" pid_alive "${stale_pid}"
assert_true "stale state: start-time mismatch is reported" contains "${stale_out}" "start time differs"

exec_log_dir="${WORK}/exec-log"
exec_state_dir="${WORK}/exec-state"
exec_command_file="${WORK}/exec-command"
exec_release="${WORK}/exec-release"
exec_ready="${WORK}/exec-ready"
mkdir -p "${exec_log_dir}" "${exec_state_dir}"
(
  : > "${exec_ready}"
  while [ ! -e "${exec_release}" ]; do sleep 0.02; done
  exec /bin/sleep 300
) &
exec_pid=$!
wait_for_file "${exec_ready}" 5
exec_start="$(pidtree_start_time "${exec_pid}")"
printf 'before-exec\n' > "${exec_command_file}"
printf 'pid=%s\nstart=%s\ncommand=%s\n' \
  "${exec_pid}" "${exec_start}" "before-exec" > "${exec_state_dir}/job.pid"
: > "${exec_release}"
printf 'after-exec\n' > "${exec_command_file}"
sleep 0.1
exec_out="$(env LOG_DIR="${exec_log_dir}" STATE_DIR="${exec_state_dir}" \
  IDENTITY_COMMAND_FILE="${exec_command_file}" BOX_RUNNER_CONFIG=/nonexistent \
  "${BOX_RUNNER_DIR}/supervisor.sh" stop)"
exec_stopped=1
if kill -0 "${exec_pid}" 2>/dev/null; then
  exec_stopped=0
  kill -KILL "${exec_pid}" 2>/dev/null || true
fi
wait "${exec_pid}" 2>/dev/null || true
assert_true "exec identity: stop reports stopped after command change" contains "${exec_out}" "stopped"
assert_eq "exec identity: stop kills the same process after exec" "1" "${exec_stopped}"

# This is the legacy numeric format. A live PID in it must not be signaled.
printf '%s\n' "${stale_pid}" > "${stale_state_dir}/job.pid"
stale_out="$(run_stale_stop)"
assert_true "stale state: legacy pid-only file signals nothing" pid_alive "${stale_pid}"
assert_true "stale state: legacy pid-only file is reported invalid" contains "${stale_out}" "invalid state"
kill -KILL "${stale_pid}" 2>/dev/null || true
wait "${stale_pid}" 2>/dev/null || true

# =========================================================================
# Section 3: opt-in production privilege path
# =========================================================================

priv_user="${BOX_RUNNER_TEST_USER:-}"
if [ "$(id -u)" -ne 0 ] || [ -z "${priv_user}" ] || \
   ! id "${priv_user}" >/dev/null 2>&1 || [ "$(id -u "${priv_user}" 2>/dev/null)" = "0" ]; then
  printf 'SKIP: privileged JIT checks (need root and BOX_RUNNER_TEST_USER)\n'
else
  priv_uid="$(id -u "${priv_user}")"
  priv_run_dir="${WORK}/priv-runner-dir"
  priv_log_dir="${WORK}/priv-log"
  priv_state_dir="${WORK}/priv-state"
  priv_mock_bin="${WORK}/priv-mock-bin"
  priv_jit_path_file="${WORK}/priv-jit-path"
  priv_chown_release="${WORK}/priv-chown-release"
  priv_job_marker="${priv_run_dir}/job-ran"

  chmod 755 "${WORK}"
  mkdir -p "${priv_run_dir}" "${priv_log_dir}" "${priv_state_dir}" "${priv_mock_bin}"
  cp "${SCRIPT_DIR}/mock-privileged-run.sh" "${priv_run_dir}/run.sh"
  cp "${SCRIPT_DIR}/mock-chown.sh" "${priv_mock_bin}/chown"
  chmod 755 "${priv_run_dir}" "${priv_run_dir}/run.sh" "${priv_mock_bin}/chown"
  chown "${priv_user}" "${priv_run_dir}"
  chown root "${priv_state_dir}"
  chmod 700 "${priv_state_dir}"

  env \
    RUNNER_SU_ENABLED=1 \
    RUNNER_USER="${priv_user}" \
    RUNNER_DIR="${priv_run_dir}" \
    RUNNER_NAME="test-box-1" \
    RUNNER_LABELS="qwen38-27b-tf-box" \
    GH_APP_MINT="${SCRIPT_DIR}/mock-gh-app-mint.sh" \
    PATH="${priv_mock_bin}:${PATH}" \
    REAL_CHOWN="$(command -v chown)" \
    MOCK_JIT_PATH_FILE="${priv_jit_path_file}" \
    MOCK_CHOWN_RELEASE_FILE="${priv_chown_release}" \
    LOG_DIR="${priv_log_dir}" \
    STATE_DIR="${priv_state_dir}" \
    PAUSE_BETWEEN_JOBS=0 \
    BOX_RUNNER_MAX_ITERATIONS=1 \
    BOX_RUNNER_CONFIG=/nonexistent \
    "${BOX_RUNNER_DIR}/supervisor.sh" run > "${WORK}/priv-supervisor.log" 2>&1 &
  priv_sup_pid=$!
  wait_for_file "${priv_jit_path_file}" 5
  priv_jit_file="$(/bin/cat "${priv_jit_path_file}" 2>/dev/null)"
  assert_eq "privileged JIT: owner is test user" "${priv_uid}" "$(file_uid "${priv_jit_file}")"
  assert_eq "privileged JIT: mode is 600" "600" "$(file_mode "${priv_jit_file}")"
  assert_false "privileged JIT: test user cannot write STATE_DIR" \
    su - "${priv_user}" -c "touch '${priv_state_dir}/forbidden'"

  : > "${priv_chown_release}"
  priv_run_bounded=1
  if ! wait_for_pid_dead "${priv_sup_pid}" 5; then
    priv_run_bounded=0
    kill -KILL "${priv_sup_pid}" 2>/dev/null || true
  fi
  wait "${priv_sup_pid}" 2>/dev/null
  priv_run_status=$?
  assert_eq "privileged JIT: supervisor run exits 0" "0" "${priv_run_status}"
  assert_eq "privileged JIT: run finishes within five seconds" "1" "${priv_run_bounded}"
  assert_true "privileged JIT: user job runs after unlink" test -f "${priv_job_marker}"
  assert_false "privileged JIT: user job unlinks config" test -e "${priv_jit_file}"
fi

# =========================================================================
# Section 4: fail-closed configuration (supervisor.sh)
# =========================================================================

assert_exit "supervisor: refuses with no RUNNER_DIR" 78 "RUNNER_DIR is required" \
  env RUNNER_SU_ENABLED=0 RUNNER_LABELS=x GH_APP_MINT="${SCRIPT_DIR}/mock-gh-app-mint.sh" \
      LOG_DIR="${WORK}/x1-log" STATE_DIR="${WORK}/x1-state" BOX_RUNNER_MAX_ITERATIONS=1 \
      BOX_RUNNER_CONFIG=/nonexistent "${BOX_RUNNER_DIR}/supervisor.sh" run

assert_exit "supervisor: refuses with missing run.sh" 78 "run.sh" \
  env RUNNER_SU_ENABLED=0 RUNNER_DIR="${WORK}/x2-empty" RUNNER_LABELS=x \
      GH_APP_MINT="${SCRIPT_DIR}/mock-gh-app-mint.sh" LOG_DIR="${WORK}/x2-log" \
      STATE_DIR="${WORK}/x2-state" BOX_RUNNER_MAX_ITERATIONS=1 BOX_RUNNER_CONFIG=/nonexistent \
      "${BOX_RUNNER_DIR}/supervisor.sh" run

assert_exit "supervisor: refuses with no RUNNER_LABELS" 78 "RUNNER_LABELS is required" \
  env RUNNER_SU_ENABLED=0 RUNNER_DIR="${run_dir}" \
      GH_APP_MINT="${SCRIPT_DIR}/mock-gh-app-mint.sh" LOG_DIR="${WORK}/x3-log" \
      STATE_DIR="${WORK}/x3-state" BOX_RUNNER_MAX_ITERATIONS=1 BOX_RUNNER_CONFIG=/nonexistent \
      "${BOX_RUNNER_DIR}/supervisor.sh" run

assert_exit "supervisor: refuses with missing gh-app-mint helper" 78 "gh-app mint helper missing" \
  env RUNNER_SU_ENABLED=0 RUNNER_DIR="${run_dir}" RUNNER_LABELS=x \
      GH_APP_MINT="${WORK}/does-not-exist.sh" LOG_DIR="${WORK}/x4-log" \
      STATE_DIR="${WORK}/x4-state" BOX_RUNNER_MAX_ITERATIONS=1 BOX_RUNNER_CONFIG=/nonexistent \
      "${BOX_RUNNER_DIR}/supervisor.sh" run

assert_exit "supervisor: refuses with no RUNNER_NAME" 78 "RUNNER_NAME is required" \
  env -u RUNNER_NAME RUNNER_SU_ENABLED=0 RUNNER_DIR="${run_dir}" RUNNER_LABELS=x \
      GH_APP_MINT="${SCRIPT_DIR}/mock-gh-app-mint.sh" LOG_DIR="${WORK}/x5-log" \
      STATE_DIR="${WORK}/x5-state" BOX_RUNNER_MAX_ITERATIONS=1 BOX_RUNNER_CONFIG=/nonexistent \
      "${BOX_RUNNER_DIR}/supervisor.sh" run

assert_exit "supervisor: refuses an unsafe RUNNER_NAME" 78 "RUNNER_NAME must use" \
  env RUNNER_SU_ENABLED=0 RUNNER_DIR="${run_dir}" RUNNER_NAME="bad name" RUNNER_LABELS=x \
      GH_APP_MINT="${SCRIPT_DIR}/mock-gh-app-mint.sh" LOG_DIR="${WORK}/x6-log" \
      STATE_DIR="${WORK}/x6-state" BOX_RUNNER_MAX_ITERATIONS=1 BOX_RUNNER_CONFIG=/nonexistent \
      "${BOX_RUNNER_DIR}/supervisor.sh" run

# =========================================================================
# Section 4: fail-closed configuration (gh-app-mint.sh)
# =========================================================================

assert_exit "gh-app-mint: refuses with no GH_APP_ID" 78 "GH_APP_ID is required" \
  env GH_APP_KEY="${WORK}/fake-key.pem" GH_RUNNER_REPO="owner/repo" \
      "${BOX_RUNNER_DIR}/gh-app-mint.sh" installation-id

echo "fake" > "${WORK}/fake-key.pem"
assert_exit "gh-app-mint: refuses with no GH_RUNNER_REPO" 78 "GH_RUNNER_REPO is required" \
  env GH_APP_ID=1 GH_APP_KEY="${WORK}/fake-key.pem" \
      "${BOX_RUNNER_DIR}/gh-app-mint.sh" installation-id

assert_exit "gh-app-mint: refuses with missing GH_APP_KEY file" 78 "GH_APP_KEY does not exist" \
  env GH_APP_ID=1 GH_APP_KEY="${WORK}/no-such-key.pem" GH_RUNNER_REPO="owner/repo" \
      "${BOX_RUNNER_DIR}/gh-app-mint.sh" installation-id

assert_exit "gh-app-mint: usage error on unknown subcommand" 64 "usage" \
  env GH_APP_ID=1 GH_APP_KEY="${WORK}/fake-key.pem" GH_RUNNER_REPO="owner/repo" \
      "${BOX_RUNNER_DIR}/gh-app-mint.sh" bogus-subcommand

gh_mock_bin="${WORK}/gh-mock-bin"
mkdir -p "${gh_mock_bin}"
cat > "${gh_mock_bin}/openssl" <<'EOF'
#!/usr/bin/env bash
case "${1:-}" in
  base64) /usr/bin/base64 ;;
  dgst) /bin/cat >/dev/null; printf signature ;;
  *) exit 2 ;;
esac
EOF
cat > "${gh_mock_bin}/curl" <<'EOF'
#!/usr/bin/env bash
set -u
/bin/cat >/dev/null
printf '%s\n' "$*" >> "${MOCK_GH_CURL_LOG}"
case "$*" in
  *app/installations/1/access_tokens*) printf '{"token":"TEST-TOKEN"}\n' ;;
  *actions/runners\?name=*)
    case "${MOCK_GH_RUNNER_STATE}" in
      offline) printf '{"runners":[{"id":42,"name":"test-box-1","status":"offline"}]}\n' ;;
      online) printf '{"runners":[{"id":42,"name":"test-box-1","status":"online"}]}\n' ;;
      none) printf '{"runners":[]}\n' ;;
      *) exit 22 ;;
    esac
    ;;
  *actions/runners/42*) : ;;
  *actions/runners/generate-jitconfig*) printf '{"encoded_jit_config":"ENCODED-JIT"}\n' ;;
  *installation/token*) : ;;
  *) exit 22 ;;
esac
EOF
chmod 0755 "${gh_mock_bin}/openssl" "${gh_mock_bin}/curl"

stale_curl_log="${WORK}/stale-curl.log"
stale_out="$(env PATH="${gh_mock_bin}:${PATH}" GH_APP_ID=1 \
  GH_APP_KEY="${WORK}/fake-key.pem" GH_RUNNER_REPO=owner/repo GH_INSTALLATION_ID=1 \
  MOCK_GH_RUNNER_STATE=offline MOCK_GH_CURL_LOG="${stale_curl_log}" \
  "${BOX_RUNNER_DIR}/gh-app-mint.sh" jitconfig test-box-1 track-label 2>&1)"
stale_status=$?
assert_eq "gh-app-mint: stale offline same-name runner refuses" "78" "${stale_status}"
assert_true "gh-app-mint: stale offline refusal is clear" \
  contains "${stale_out}" "is still registered offline; remove it before retrying"
assert_false "gh-app-mint: stale offline same-name runner is never deleted" \
  grep -q 'actions/runners/42' "${stale_curl_log}"

online_curl_log="${WORK}/online-curl.log"
online_out="$(env PATH="${gh_mock_bin}:${PATH}" GH_APP_ID=1 \
  GH_APP_KEY="${WORK}/fake-key.pem" GH_RUNNER_REPO=owner/repo GH_INSTALLATION_ID=1 \
  MOCK_GH_RUNNER_STATE=online MOCK_GH_CURL_LOG="${online_curl_log}" \
  "${BOX_RUNNER_DIR}/gh-app-mint.sh" jitconfig test-box-1 track-label 2>&1)"
online_status=$?
assert_eq "gh-app-mint: online same-name runner refuses" "78" "${online_status}"
assert_true "gh-app-mint: online same-name refusal is clear" \
  contains "${online_out}" "is online; refusing to replace it"
assert_false "gh-app-mint: online same-name runner is never deleted" \
  grep -q 'actions/runners/42' "${online_curl_log}"

# =========================================================================
# Section 5: install.sh pure helpers (isolated subshell: install.sh's own
# `set -euo pipefail` must never leak into this suite's shell, so each call
# below runs install.sh in a fresh `bash -c`, not via `source` here).
# =========================================================================

install_call() {
  bash -c '
    set -uo pipefail
    BOX_RUNNER_INSTALL_SOURCE_ONLY=1
    # shellcheck disable=SC1091
    . "$1/install.sh"
    shift
    "$@"
  ' _ "${BOX_RUNNER_DIR}" "$@"
}

assert_eq "install.sh: pinned sha256 for osx-arm64 is 64 hex chars" "64" \
  "$(install_call pinned_sha256 osx-arm64 | wc -c | tr -d ' ')"
assert_eq "install.sh: pinned sha256 for osx-x64 is 64 hex chars" "64" \
  "$(install_call pinned_sha256 osx-x64 | wc -c | tr -d ' ')"
assert_false "install.sh: pinned_sha256 refuses an unknown platform tag" \
  install_call pinned_sha256 linux-arm64

echo -n "box-runner" > "${WORK}/checksum-fixture.txt"
known_sha="$(shasum -a 256 "${WORK}/checksum-fixture.txt" 2>/dev/null | awk '{print $1}')"
assert_eq "install.sh: compute_sha256 matches shasum" "${known_sha}" \
  "$(install_call compute_sha256 "${WORK}/checksum-fixture.txt")"
# shellcheck disable=SC2016  # install_call's shell expands LOG_DIR.
assert_eq "install.sh: macOS log default avoids the /var symlink" \
  "/private/var/log/qwen38tf-box-runner" "$(install_call eval 'printf %s "${LOG_DIR}"')"
# shellcheck disable=SC2016  # install_call's shell expands STATE_DIR.
assert_eq "install.sh: macOS state default avoids the /var symlink" \
  "/private/var/db/qwen38tf-box-runner" "$(install_call eval 'printf %s "${STATE_DIR}"')"

# =========================================================================
# Section 6: trusted install paths and user-only runner extraction
# =========================================================================

# shellcheck disable=SC2329  # invoked indirectly by assert_exit.
trusted_path_call() {
  local root="$1" target="$2" fake_owner="${3:-}"
  # shellcheck disable=SC2016  # nested shell expands installer variables.
  env BOX_RUNNER_INSTALL_ROOT="${root}" BOX_RUNNER_INSTALL_OWNER_UID="$(id -u)" \
    bash -c '
      set -uo pipefail
      BOX_RUNNER_INSTALL_SOURCE_ONLY=1
      # shellcheck disable=SC1091
      . "$1/install.sh"
      fake_uid="$3"
      if [ -n "${fake_uid}" ]; then
        path_uid() {
          case "$1" in
            *bench-owned*) printf "%s\n" "${fake_uid}" ;;
            *)
              if [ "$(uname -s)" = "Darwin" ]; then stat -f "%u" "$1"; else stat -c "%u" "$1"; fi
              ;;
          esac
        }
      fi
      assert_trusted_path "$2"
    ' _ "${BOX_RUNNER_DIR}" "${target}" "${fake_owner}"
}

path_test_root="${WORK}/path-root"
mkdir -p "${path_test_root}/safe/group-writable" "${path_test_root}/safe/bench-owned"
chmod 0755 "${path_test_root}" "${path_test_root}/safe" "${path_test_root}/safe/bench-owned"
chmod 0775 "${path_test_root}/safe/group-writable"
ln -s "${path_test_root}/safe" "${path_test_root}/linked"

assert_exit "install.sh: refuses a group-writable ancestor" 78 "group/other writable" \
  trusted_path_call "${path_test_root}" "${path_test_root}/safe/group-writable/target"
assert_exit "install.sh: refuses a bench-owned ancestor" 78 "owner" \
  trusted_path_call "${path_test_root}" "${path_test_root}/safe/bench-owned/target" "$(( $(id -u) + 1 ))"
assert_exit "install.sh: refuses a symlink ancestor" 78 "symlink" \
  trusted_path_call "${path_test_root}" "${path_test_root}/linked/target"

install_stage="${WORK}/install-stage"
mkdir -p "${install_stage}/lib" "${WORK}/install-mock-bin" "${WORK}/install-tmp"
cp "${BOX_RUNNER_DIR}/install.sh" \
   "${BOX_RUNNER_DIR}/supervisor.sh" \
   "${BOX_RUNNER_DIR}/gh-app-mint.sh" \
   "${BOX_RUNNER_DIR}/box-runner.env.example" \
   "${BOX_RUNNER_DIR}/com.qwen38tf.box-runner.plist.template" \
   "${install_stage}/"
cp "${BOX_RUNNER_DIR}/lib/pidtree.sh" "${install_stage}/lib/pidtree.sh"

install_root="${WORK}/install-root"
install_runner_parent="${WORK}/runner-parent"
install_runner_dir="${install_runner_parent}/qwen-only"
install_action_log="${WORK}/install-actions.log"
mkdir -p "${install_root}" "${install_runner_parent}"
chmod 0755 "${install_root}" "${install_runner_parent}"

# shellcheck disable=SC2016  # generated mocks expand their own arguments.
printf '%s\n' \
  '#!/usr/bin/env bash' \
  'set -u' \
  'out=""; prev=""' \
  'for arg in "$@"; do [ "${prev}" = "-o" ] && out="${arg}"; prev="${arg}"; done' \
  '[ -n "${out}" ] || exit 1' \
  'printf stub-tarball > "${out}"' \
  > "${WORK}/install-mock-bin/curl"
# shellcheck disable=SC2016
printf '%s\n' \
  '#!/usr/bin/env bash' \
  'set -u' \
  'target=""; prev=""' \
  'for arg in "$@"; do [ "${prev}" = "-C" ] && target="${arg}"; prev="${arg}"; done' \
  '[ -n "${target}" ] || exit 1' \
  'printf "#!/bin/sh\n" > "${target}/run.sh"' \
  'chmod 0755 "${target}/run.sh"' \
  > "${WORK}/install-mock-bin/tar"
# shellcheck disable=SC2016
printf '%s\n' \
  '#!/usr/bin/env bash' \
  'set -u' \
  'for arg in "$@"; do' \
  '  case "${arg}" in' \
  '    "${MOCK_RUNNER_DIR}"|"${MOCK_RUNNER_DIR}"/*)' \
  '      [ "${BOX_RUNNER_ACTOR:-root}" = runner ] || printf "root-write %s\n" "${arg}" >> "${MOCK_ACTION_LOG}"' \
  '      ;;' \
  '  esac' \
  'done' \
  'exec /bin/mkdir "$@"' \
  > "${WORK}/install-mock-bin/mkdir"
# shellcheck disable=SC2016
printf '%s\n' \
  '#!/usr/bin/env bash' \
  'set -u' \
  'user="$1"; shift' \
  'printf "run-as %s %s\n" "${user}" "$*" >> "${MOCK_ACTION_LOG}"' \
  'BOX_RUNNER_ACTOR=runner "$@"' \
  > "${WORK}/install-run-as-user"
# shellcheck disable=SC2016
printf '%s\n' \
  '#!/usr/bin/env bash' \
  'case "$*" in' \
  '  *actions-runner*.tar.gz*) printf "%s  tarball\n" "8e8839c49b7060b6b2154f4931f815df330c27f167d53ef2239ee3dfce28b079" ;;' \
  '  *) exec /usr/bin/shasum "$@" ;;' \
  'esac' \
  > "${WORK}/install-mock-bin/shasum"
chmod 0755 "${WORK}/install-mock-bin/"* "${WORK}/install-run-as-user"

# shellcheck disable=SC2329  # invoked indirectly by assert_exit.
run_install_main() {
  env \
    BOX_RUNNER_INSTALL_ROOT="${install_root}" \
    BOX_RUNNER_INSTALL_OWNER_UID="$(id -u)" \
    BOX_RUNNER_RUN_AS_USER="${WORK}/install-run-as-user" \
    RUNNER_USER=bench \
    RUNNER_DIR="${install_runner_dir}" \
    MOCK_RUNNER_DIR="${install_runner_dir}" \
    MOCK_ACTION_LOG="${install_action_log}" \
    TMPDIR="${WORK}/install-tmp" \
    PATH="${WORK}/install-mock-bin:${PATH}" \
    "${install_stage}/install.sh" | tee "${WORK}/install-output.log"
}

assert_exit "install.sh: staged first install exits 0" 0 "done." run_install_main
assert_true "install.sh: runner extraction used the bench executor" \
  grep -q '^run-as bench ' "${install_action_log}"
assert_false "install.sh: root never creates a path under RUNNER_DIR" \
  grep -q '^root-write ' "${install_action_log}"
assert_false "install.sh: source has no recursive chown" \
  grep -Eq 'chown[[:space:]]+-R' "${BOX_RUNNER_DIR}/install.sh"
assert_true "install.sh: runner payload exists" test -x "${install_runner_dir}/run.sh"
assert_eq "install.sh: runner version marker is user-created" "2.336.0" \
  "$(cat "${install_runner_dir}/.box-runner-installed-version")"

installed_code="${install_root}/usr/local/libexec/qwen38tf-box-runner"
installed_config="${install_root}/usr/local/etc/qwen38tf-box-runner/box-runner.env"
installed_plist="${install_root}/Library/LaunchDaemons/com.qwen38tf.box-runner.plist"
assert_eq "install.sh: supervisor mode is 555" "555" "$(file_mode "${installed_code}/supervisor.sh")"
assert_eq "install.sh: pidtree mode is 555" "555" "$(file_mode "${installed_code}/lib/pidtree.sh")"
assert_eq "install.sh: mint helper mode is 555" "555" "$(file_mode "${installed_code}/gh-app-mint.sh")"
assert_eq "install.sh: config mode is 644" "644" "$(file_mode "${installed_config}")"
assert_eq "install.sh: plist mode is 644" "644" "$(file_mode "${installed_plist}")"
assert_true "install.sh: plist uses installed supervisor" \
  grep -q "${installed_code}/supervisor.sh" "${installed_plist}"
assert_true "install.sh: plist uses installed config" \
  grep -q "${installed_config}" "${installed_plist}"
assert_false "install.sh: plist does not name checkout supervisor" \
  grep -q "${BOX_RUNNER_DIR}/supervisor.sh" "${installed_plist}"
assert_true "install.sh: output prints installed sha256 values" \
  grep -q 'sha256=' "${WORK}/install-output.log"

installed_default_state="${WORK}/installed-default-state"
installed_default_log="${WORK}/installed-default-log"
mkdir -p "${installed_default_state}"
printf '\nSTATE_DIR=%s\nLOG_DIR=%s\n' \
  "${installed_default_state}" "${installed_default_log}" >> "${installed_config}"
printf 'invalid-state\n' > "${installed_default_state}/job.pid"
installed_status="$(env -u BOX_RUNNER_CONFIG "${installed_code}/supervisor.sh" status)"
assert_true "installed supervisor: default config is under installed etc" \
  contains "${installed_status}" "${installed_default_state}/job.pid"

override_state="${WORK}/installed-override-state"
override_log="${WORK}/installed-override-log"
override_config="${WORK}/installed-override.env"
mkdir -p "${override_state}"
printf 'STATE_DIR=%s\nLOG_DIR=%s\n' "${override_state}" "${override_log}" > "${override_config}"
printf 'invalid-state\n' > "${override_state}/job.pid"
override_status="$(BOX_RUNNER_CONFIG="${override_config}" "${installed_code}/supervisor.sh" status)"
assert_true "installed supervisor: BOX_RUNNER_CONFIG overrides installed default" \
  contains "${override_status}" "${override_state}/job.pid"

existing_root="${WORK}/existing-root"
existing_runner="${WORK}/existing-runner"
mkdir -p "${existing_root}" "${existing_runner}"
chmod 0755 "${existing_root}" "${existing_runner}"
# shellcheck disable=SC2016  # nested shell expands its own source path.
assert_exit "install.sh: refuses an existing RUNNER_DIR" 78 "RUNNER_DIR already exists" \
  env BOX_RUNNER_INSTALL_ROOT="${existing_root}" BOX_RUNNER_INSTALL_OWNER_UID="$(id -u)" \
      BOX_RUNNER_RUN_AS_USER="${WORK}/install-run-as-user" RUNNER_USER=bench \
      RUNNER_DIR="${existing_runner}" MOCK_RUNNER_DIR="${existing_runner}" \
      MOCK_ACTION_LOG="${install_action_log}" PATH="${WORK}/install-mock-bin:${PATH}" \
      bash -c 'BOX_RUNNER_INSTALL_SOURCE_ONLY=1; . "$1/install.sh"; install_actions_runner' \
      _ "${install_stage}"

leftover_tmp="$(find "${WORK}/install-tmp" -maxdepth 1 -name 'box-runner-install.*' 2>/dev/null | wc -l | tr -d ' ')"
assert_eq "install.sh: first install leaves no temp download dir" "0" "${leftover_tmp}"

# =========================================================================
printf '\n%d passed, %d failed, %d unverified skips (floor: %d assertions)\n' "${PASSED}" "${FAILED}" "${SKIPPED}" "${EXPECTED_MIN_ASSERTIONS}"
total=$((PASSED + FAILED + SKIPPED))
if [ "${total}" -lt "${EXPECTED_MIN_ASSERTIONS}" ]; then
  printf 'FAIL  suite ran only %d assertions, floor is %d (a section stopped running silently)\n' "${total}" "${EXPECTED_MIN_ASSERTIONS}"
  exit 1
fi
if [ "${FAILED}" -gt 0 ]; then
  printf '\nFailures:\n'
  printf '  - %s\n' "${FAILURES[@]}"
  exit 1
fi
exit 0
