#!/usr/bin/env bash
# lib/pidtree.sh -- process-tree tracking and kill-tree primitives.
#
# PROCESS-TREE SUPPORT. The supervisor starts the ephemeral actions-runner
# with:
#
#   su - "${RUNNER_USER}" -c '... timeout ./run.sh'
#
# `su -` puts the runner and its descendants in a session and process group
# that differ from the supervisor's. Process-group signals therefore do not
# cover the runner tree. This library walks the live parent-PID chain from the
# tracked job root and signals each descendant directly. The caller invokes
# it between jobs while the root is still alive, before the kernel can reparent
# descendants to PID 1.
#
# The caller persists the root PID. A separate `supervisor.sh stop` invocation
# can read it and perform the same live parent-PID walk when the supervisor
# loop is not running.
#
# Every function in this file is read-only about WHERE state lives; it only
# ever takes a pid on its command line and returns/acts on live process
# state. No paths, no env vars, no secrets.
set -u

# pidtree_children PID
#   Print the pid of every LIVE direct child of PID, one per line. Silent
#   (empty output, exit 0) if PID has no children or does not exist.
pidtree_children() {
  pgrep -P "$1" 2>/dev/null || true
}

# pidtree_alive PID
#   True if PID exists. kill -0 can fail with EPERM for another uid, so ps is
#   the fallback. A caller without signal access must not report that PID dead.
pidtree_alive() {
  kill -0 "$1" 2>/dev/null && return 0
  ps -o pid= -p "$1" 2>/dev/null | grep -Eq '^[[:space:]]*[0-9]+'
}

# pidtree_signal_access PID
#   True only when this uid can signal PID. status uses this to mark a result
#   that is live but cannot be fully checked by the caller.
pidtree_signal_access() {
  kill -0 "$1" 2>/dev/null
}

pidtree_kernel_start_time() {
  /usr/bin/python3 - "$1" <<'PY' 2>/dev/null
import ctypes
import sys


class ProcBsdInfo(ctypes.Structure):
    _fields_ = [
        ("pbi_flags", ctypes.c_uint32), ("pbi_status", ctypes.c_uint32),
        ("pbi_xstatus", ctypes.c_uint32), ("pbi_pid", ctypes.c_uint32),
        ("pbi_ppid", ctypes.c_uint32), ("pbi_uid", ctypes.c_uint32),
        ("pbi_gid", ctypes.c_uint32), ("pbi_ruid", ctypes.c_uint32),
        ("pbi_rgid", ctypes.c_uint32), ("pbi_svuid", ctypes.c_uint32),
        ("pbi_svgid", ctypes.c_uint32), ("rfu_1", ctypes.c_uint32),
        ("pbi_comm", ctypes.c_char * 16), ("pbi_name", ctypes.c_char * 32),
        ("pbi_nfiles", ctypes.c_uint32), ("pbi_pgid", ctypes.c_uint32),
        ("pbi_pjobc", ctypes.c_uint32), ("e_tdev", ctypes.c_uint32),
        ("e_tpgid", ctypes.c_uint32), ("pbi_nice", ctypes.c_int32),
        ("pbi_start_tvsec", ctypes.c_uint64),
        ("pbi_start_tvusec", ctypes.c_uint64),
    ]


try:
    pid = int(sys.argv[1])
    libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    libproc.proc_pidinfo.argtypes = [ctypes.c_int, ctypes.c_int,
                                     ctypes.c_uint64, ctypes.c_void_p,
                                     ctypes.c_int]
    libproc.proc_pidinfo.restype = ctypes.c_int
    info = ProcBsdInfo()
    size = ctypes.sizeof(info)
    read = libproc.proc_pidinfo(pid, 3, 0, ctypes.byref(info), size)
    if read != size or info.pbi_pid != pid:
        raise RuntimeError("unknown process identity")
    print(f"{info.pbi_start_tvsec}.{info.pbi_start_tvusec:06d}")
except (OSError, RuntimeError, TypeError, ValueError):
    sys.exit(1)
PY
}

pidtree_start_time() {
  pidtree_kernel_start_time "$1"
}

pidtree_parent_pid() {
  ps -o ppid= -p "$1" 2>/dev/null | tr -d '[:space:]'
}

pidtree_command_identity() {
  ps -o command= -p "$1" 2>/dev/null | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'
}

# pidtree_identity_matches PID START COMMAND
#   Match the kernel start time immediately before a signal. COMMAND is a
#   saved log hint only because a valid process changes command across exec.
#   PID plus start time blocks reuse across tracks without rejecting exec.
pidtree_identity_matches() {
  local pid="$1" expected_start="$2" expected_command="$3"
  local actual_start
  : "${expected_command}"
  pidtree_alive "${pid}" || return 1
  actual_start="$(pidtree_start_time "${pid}")"
  [ -n "${actual_start}" ] || return 1
  [ "${actual_start}" = "${expected_start}" ]
}

pidtree_start_identity_matches() {
  local pid="$1" expected_start="$2" actual_start
  pidtree_alive "${pid}" || return 1
  actual_start="$(pidtree_start_time "${pid}")"
  [ -n "${actual_start}" ] && [ "${actual_start}" = "${expected_start}" ]
}

# pidtree_collect ROOT_PID
#   Breadth-first walk of the live ppid tree rooted at ROOT_PID. Prints one
#   pid per line: ROOT_PID first (if alive), then every live descendant,
#   deduplicated. Prints nothing if ROOT_PID is not alive. This is the
#   session/process-group-independent walk described above -- it finds a
#   `su -`-detached or otherwise re-sessioned descendant exactly as reliably
#   as an ordinary child, because it never consults session id or pgid.
pidtree_collect() {
  local root="$1"
  pidtree_alive "${root}" || return 0

  # Plain indexed arrays only (bash 3.2 compatible -- macOS ships 3.2 as
  # /bin/bash and this library must run there unmodified).
  local -a queue=("${root}")
  local -a seen=()
  local -a out=()
  local pid child already idx

  while [ "${#queue[@]}" -gt 0 ]; do
    pid="${queue[0]}"
    queue=("${queue[@]:1}")

    already=0
    for idx in "${seen[@]:-}"; do
      if [ "${idx}" = "${pid}" ]; then
        already=1
        break
      fi
    done
    [ "${already}" -eq 1 ] && continue
    seen+=("${pid}")

    pidtree_alive "${pid}" || continue
    out+=("${pid}")

    while IFS= read -r child; do
      [ -n "${child}" ] && queue+=("${child}")
    done < <(pidtree_children "${pid}")
  done

  if [ "${#out[@]}" -gt 0 ]; then
    printf '%s\n' "${out[@]}"
  fi
}

# pidtree_collect_identities ROOT_PID [EXPECTED_ROOT_START]
#   Print PID<TAB>START for each process in the live tree. The identity is
#   captured with the PID and must be checked again immediately before signal.
#   When EXPECTED_ROOT_START is set, reject a reused root instead of adopting
#   its new identity. The one-argument form captures the current root identity.
pidtree_collect_identities() {
  local root="$1" expected_root_start="${2:-}"
  local root_start pid start child child_start child_parent key known idx
  local -a queue_pid=()
  local -a queue_start=()
  local -a seen=()
  local -a child_pid=()
  local -a child_identity=()

  if [ -n "${expected_root_start}" ]; then
    root_start="${expected_root_start}"
  else
    root_start="$(pidtree_start_time "${root}")"
  fi
  [ -n "${root_start}" ] || return 0
  pidtree_start_identity_matches "${root}" "${root_start}" || return 0
  queue_pid+=("${root}")
  queue_start+=("${root_start}")

  while [ "${#queue_pid[@]}" -gt 0 ]; do
    pid="${queue_pid[0]}"
    start="${queue_start[0]}"
    queue_pid=("${queue_pid[@]:1}")
    queue_start=("${queue_start[@]:1}")
    key="${pid}"$'\t'"${start}"

    known=0
    for child in "${seen[@]:-}"; do
      [ "${child}" != "${key}" ] || known=1
    done
    [ "${known}" -eq 0 ] || continue
    seen+=("${key}")
    pidtree_start_identity_matches "${pid}" "${start}" || continue
    printf '%s\t%s\n' "${pid}" "${start}"

    child_pid=()
    child_identity=()
    while IFS= read -r child; do
      [ -n "${child}" ] || continue
      child_start="$(pidtree_start_time "${child}")"
      child_parent="$(pidtree_parent_pid "${child}")"
      if [ -n "${child_start}" ] && [ "${child_parent}" = "${pid}" ]; then
        child_pid+=("${child}")
        child_identity+=("${child_start}")
      fi
    done < <(pidtree_children "${pid}")
    # A reused parent can enumerate a peer process's children. Admit the
    # snapshot only while the parent still has the identity checked above.
    pidtree_start_identity_matches "${pid}" "${start}" || continue
    for ((idx = 0; idx < ${#child_pid[@]}; idx++)); do
      queue_pid+=("${child_pid[idx]}")
      queue_start+=("${child_identity[idx]}")
    done
  done
}

# pidtree_describe ROOT_PID
#   Human-readable listing of every live pid in ROOT_PID's tree, for the
#   `status` entrypoint. One `ps` line per process; prints nothing (exit 0)
#   if the tree is empty (the between-jobs quiescent state).
pidtree_describe() {
  local root="$1" pid
  local -a pids=()
  while IFS= read -r pid; do
    [ -n "${pid}" ] && pids+=("${pid}")
  done < <(pidtree_collect "${root}")
  [ "${#pids[@]}" -eq 0 ] && return 0
  # -o field set is BSD/macOS-and-Linux-compatible; one `ps -p` call per pid
  # keeps this portable (GNU ps and BSD ps disagree on comma-vs-space -p
  # lists in some versions).
  for pid in "${pids[@]}"; do
    ps -o pid,ppid,pgid,stat,command -p "${pid}" 2>/dev/null | tail -n +2
  done
}

# pidtree_signal SIGNAL PID...
#   Best-effort: send SIGNAL to every listed pid. Never fails the caller --
#   a pid that already exited between collection and signaling is not an
#   error.
pidtree_signal() {
  local sig="$1"; shift
  local pid
  for pid in "$@"; do
    kill -s "${sig}" "${pid}" 2>/dev/null || true
  done
}

pidtree_signal_identities() {
  local sig="$1" entry pid start
  shift
  for entry in "$@"; do
    pid="${entry%%$'\t'*}"
    start="${entry#*$'\t'}"
    [ "${pid}" != "${start}" ] || continue
    pidtree_start_identity_matches "${pid}" "${start}" || continue
    kill -s "${sig}" "${pid}" 2>/dev/null || true
  done
}

# pidtree_kill_tree ROOT_PID [TERM_TIMEOUT_SECONDS]
#   The `stop` primitive. Collects ROOT_PID's live tree, sends TERM to all of
#   it, and keeps every captured PID and start time even if the root exits.
#   It adds descendants found during the grace period. It then sends KILL to
#   each saved identity that remains and verifies that all are gone. Prints
#   nothing; returns 0 if the tree is fully gone afterward, 1 if something
#   survived KILL, or 2 if the root identity changed before TERM.
pidtree_kill_tree() {
  local root="$1" timeout="${2:-10}"
  local expected_start="${3:-}" expected_command="${4:-}"
  local -a tree=()
  local -a found=()
  local -a live=()
  local entry candidate pid waited known

  if [ -z "${expected_start}" ] || [ -z "${expected_command}" ]; then
    expected_start="$(pidtree_start_time "${root}")"
    expected_command="$(pidtree_command_identity "${root}")"
  fi
  pidtree_identity_matches "${root}" "${expected_start}" "${expected_command}" \
    || return 2

  while IFS= read -r pid; do
    [ -n "${pid}" ] && tree+=("${pid}")
  done < <(pidtree_collect_identities "${root}" "${expected_start}")

  pidtree_identity_matches "${root}" "${expected_start}" "${expected_command}" \
    || return 2
  [ "${#tree[@]}" -eq 0 ] && return 0
  pidtree_signal_identities TERM "${tree[@]}"

  waited=0
  while [ "${waited}" -lt "$((timeout * 5))" ]; do
    # Keep every saved identity after the root exits. Add descendants that
    # appear during the grace period while the root is still identifiable.
    found=()
    if pidtree_start_identity_matches "${root}" "${expected_start}"; then
      while IFS= read -r entry; do
        [ -n "${entry}" ] && found+=("${entry}")
      done < <(pidtree_collect_identities "${root}" "${expected_start}")
      # The collector checks the saved root start time and each parent again
      # after it lists that parent's children, so a pid reused during this
      # refresh adds no peer process. Saved identities are always kept.
    fi
    for candidate in "${found[@]:-}"; do
      [ -n "${candidate}" ] || continue
      known=0
      for entry in "${tree[@]}"; do
        [ "${entry}" != "${candidate}" ] || known=1
      done
      [ "${known}" -eq 1 ] || tree+=("${candidate}")
    done
    live=()
    for entry in "${tree[@]}"; do
      pid="${entry%%$'\t'*}"
      candidate="${entry#*$'\t'}"
      [ "${pid}" != "${candidate}" ] || continue
      pidtree_start_identity_matches "${pid}" "${candidate}" \
        && live+=("${entry}")
    done
    [ "${#live[@]}" -eq 0 ] && return 0
    sleep 0.2
    waited=$((waited + 1))
  done

  live=()
  for entry in "${tree[@]}"; do
    pid="${entry%%$'\t'*}"
    candidate="${entry#*$'\t'}"
    [ "${pid}" != "${candidate}" ] || continue
    pidtree_start_identity_matches "${pid}" "${candidate}" \
      && live+=("${entry}")
  done
  [ "${#live[@]}" -eq 0 ] && return 0
  pidtree_signal_identities KILL "${live[@]}"

  waited=0
  while [ "${waited}" -lt 25 ]; do
    live=()
    for entry in "${tree[@]}"; do
      pid="${entry%%$'\t'*}"
      candidate="${entry#*$'\t'}"
      [ "${pid}" != "${candidate}" ] || continue
      pidtree_start_identity_matches "${pid}" "${candidate}" \
        && live+=("${entry}")
    done
    [ "${#live[@]}" -eq 0 ] && return 0
    sleep 0.2
    waited=$((waited + 1))
  done
  return 1
}
