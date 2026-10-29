#!/usr/bin/env bash
# Document whether same-uid code can read a runner listener's initial environment.
set -uo pipefail

STRICT=0
SELF_TEST=0
case "${1:-}" in
  "") ;;
  --strict) STRICT=1 ;;
  --self-test) SELF_TEST=1 ;;
  *) echo "usage: $0 [--strict|--self-test]" >&2; exit 2 ;;
esac

WORK=""
VICTIM_PID=""
ARGV_VICTIM_PID=""

# shellcheck disable=SC2329  # Invoked indirectly by EXIT and signal traps.
cleanup() {
  local pid
  for pid in "${VICTIM_PID}" "${ARGV_VICTIM_PID}"; do
    if [ -n "${pid}" ]; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
  if [ -n "${WORK}" ]; then
    rm -rf -- "${WORK}"
    WORK=""
  fi
}

# shellcheck disable=SC2329  # Invoked indirectly by HUP/INT/TERM traps.
handle_signal() {
  local status="$1"
  cleanup
  trap - EXIT
  exit "${status}"
}

# Install cleanup before mktemp or any other resource creation. cleanup is
# deliberately safe while WORK and both tracked pids are still empty.
trap cleanup EXIT
trap 'handle_signal 129' HUP
trap 'handle_signal 130' INT
trap 'handle_signal 143' TERM

# A signal between a background launch and recording its $! would run cleanup
# without the child's pid. Defer HUP/INT/TERM across each launch, then honor it.
PENDING_SIGNAL=""
defer_signals() {
  trap 'PENDING_SIGNAL=129' HUP
  trap 'PENDING_SIGNAL=130' INT
  trap 'PENDING_SIGNAL=143' TERM
}
restore_signals() {
  trap 'handle_signal 129' HUP
  trap 'handle_signal 130' INT
  trap 'handle_signal 143' TERM
  if [ -n "${PENDING_SIGNAL}" ]; then handle_signal "${PENDING_SIGNAL}"; fi
}

ERROR_ANY=0
FOUND_ANY=0
READ_KIND=""
READ_DISPLAY=""
NEEDLE="RUNNER_CANARY=self-test"

read_target() {
  local target_pid="$1"
  shift
  local output status

  if ! kill -0 "${target_pid}" 2>/dev/null; then
    READ_KIND="error"
    READ_DISPLAY="error"
    return
  fi

  output="$("$@" "${target_pid}" "${NEEDLE}" 2>/dev/null)"
  status=$?
  if [ "${status}" -ne 0 ]; then
    READ_KIND="error"
    READ_DISPLAY="error"
  elif [ "${output}" = "found" ]; then
    READ_KIND="found"
    READ_DISPLAY="found"
  elif [ "${output}" = "absent" ]; then
    READ_KIND="absent"
    READ_DISPLAY="absent"
  elif [ "${output}" = "denied errno=1" ] ||
       [ "${output}" = "denied errno=13" ]; then
    READ_KIND="denied"
    READ_DISPLAY="${output}"
  else
    READ_KIND="error"
    READ_DISPLAY="error"
  fi
}

record_mode() {
  local mode="$1"
  case "${READ_KIND}" in
    found) FOUND_ANY=1 ;;
    absent|denied) ;;
    error) ERROR_ANY=1 ;;
    *)
      READ_KIND="error"
      READ_DISPLAY="error"
      ERROR_ANY=1
      ;;
  esac
  printf 'mode=%s result=%s\n' "${mode}" "${READ_DISPLAY}"
}

run_mode() {
  local mode="$1"
  local target_pid="$2"
  shift 2
  read_target "${target_pid}" "$@"
  record_mode "${mode}"
}

run_negative_mode() {
  local mode="$1"
  local target_pid="$2"
  shift 2
  read_target "${target_pid}" "$@"
  if [ "${READ_KIND}" != "absent" ]; then
    READ_KIND="error"
    READ_DISPLAY="error"
  fi
  record_mode "${mode}"
}

finish_result() {
  local strict="$1"
  if [ "${ERROR_ANY}" -ne 0 ]; then
    echo "INCONCLUSIVE"
    return 2
  fi
  if [ "${FOUND_ANY}" -ne 0 ]; then
    echo "EXPOSED"
    if [ "${strict}" -ne 0 ]; then
      return 1
    fi
    return 0
  fi
  echo "NOT EXPOSED"
  return 0
}

setup_error() {
  echo "mode=setup result=error"
  printf 'error: %s\n' "$1" >&2
  echo "INCONCLUSIVE"
  exit 2
}

make_work_dir() {
  WORK="$(mktemp -d "${TMPDIR:-/tmp}/jit-config-peer-read.XXXXXX")" || return 1
  WORK="$(cd -- "${WORK}" >/dev/null && pwd -P)" || return 1
}

self_test_case() {
  local name="$1"
  local target_pid="$2"
  local reader="$3"
  local output status

  output="$({
    ERROR_ANY=0
    FOUND_ANY=0
    run_mode test "${target_pid}" "${reader}"
    finish_result 0
  } 2>&1)"
  status=$?
  if [ "${status}" -eq 2 ] &&
     [ "${output}" = $'mode=test result=error\nINCONCLUSIVE' ]; then
    return 0
  fi
  printf 'self-test failure (%s): status=%s output=%q\n' \
    "${name}" "${status}" "${output}" >&2
  return 1
}

run_self_test() {
  local passed=0
  local failed=0
  local dead_pid

  if ! make_work_dir; then
    echo "self-test: 0 passed, 3 failed"
    return 1
  fi

  cat > "${WORK}/reader-exit-3" <<'EOF'
#!/bin/sh
echo "error errno=5"
exit 3
EOF
  cat > "${WORK}/reader-garbage" <<'EOF'
#!/bin/sh
echo "garbage"
exit 0
EOF
  cat > "${WORK}/reader-absent" <<'EOF'
#!/bin/sh
echo "absent"
exit 0
EOF
  chmod +x "${WORK}/reader-exit-3" "${WORK}/reader-garbage" \
    "${WORK}/reader-absent" || {
      echo "self-test: 0 passed, 3 failed"
      return 1
    }

  defer_signals
  /bin/sleep 600 &
  VICTIM_PID=$!
  restore_signals

  if self_test_case "reader exit 3" "${VICTIM_PID}" \
      "${WORK}/reader-exit-3"; then
    passed=$((passed + 1))
  else
    failed=$((failed + 1))
  fi
  if self_test_case "reader garbage" "${VICTIM_PID}" \
      "${WORK}/reader-garbage"; then
    passed=$((passed + 1))
  else
    failed=$((failed + 1))
  fi

  /bin/sh -c 'exit 0' &
  dead_pid=$!
  wait "${dead_pid}" 2>/dev/null || true
  if self_test_case "dead victim" "${dead_pid}" \
      "${WORK}/reader-absent"; then
    passed=$((passed + 1))
  else
    failed=$((failed + 1))
  fi

  printf 'self-test: %d passed, %d failed\n' "${passed}" "${failed}"
  [ "${failed}" -eq 0 ]
}

if [ "${SELF_TEST}" -eq 1 ]; then
  run_self_test
  exit $?
fi

if [ "$(uname -s)" != "Darwin" ]; then
  echo "SKIP: not macOS"
  exit 0
fi

for required_command in cc codesign sandbox-exec python3; do
  if ! command -v "${required_command}" >/dev/null 2>&1; then
    setup_error "required command not found: ${required_command}"
  fi
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null && pwd -P)" || \
  setup_error "could not resolve script directory"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." >/dev/null && pwd -P)" || \
  setup_error "could not resolve repository root"
make_work_dir || setup_error "could not create work directory"

cat > "${WORK}/reader.c" <<'EOF'
#include <sys/sysctl.h>
#include <sys/types.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int report_error(int error_number) {
  printf("error errno=%d\n", error_number);
  return 3;
}

static int report_sysctl_failure(int error_number) {
  if (error_number == EPERM || error_number == EACCES) {
    printf("denied errno=%d\n", error_number);
    return 0;
  }
  return report_error(error_number);
}

int main(int argc, char **argv) {
  char *pid_end = NULL;
  long parsed_pid;
  int mib[3];
  size_t size = 0;
  unsigned char *buf;
  int process_argc;
  size_t cursor;
  size_t needle_size;

  if (argc != 3 || argv[2][0] == '\0') return report_error(EINVAL);

  errno = 0;
  parsed_pid = strtol(argv[1], &pid_end, 10);
  if (errno != 0) return report_error(errno);
  if (pid_end == argv[1] || *pid_end != '\0' || parsed_pid <= 0 ||
      parsed_pid > INT_MAX) {
    return report_error(EINVAL);
  }

  mib[0] = CTL_KERN;
  mib[1] = KERN_PROCARGS2;
  mib[2] = (int)parsed_pid;
  if (sysctl(mib, 3, NULL, &size, NULL, 0) == -1) {
    return report_sysctl_failure(errno == 0 ? EIO : errno);
  }
  if (size < sizeof(process_argc)) return report_error(EINVAL);

  buf = malloc(size);
  if (buf == NULL) return report_error(errno == 0 ? ENOMEM : errno);
  if (sysctl(mib, 3, buf, &size, NULL, 0) == -1) {
    int saved_errno = errno == 0 ? EIO : errno;
    free(buf);
    return report_sysctl_failure(saved_errno);
  }
  if (size < sizeof(process_argc)) {
    free(buf);
    return report_error(EINVAL);
  }

  memcpy(&process_argc, buf, sizeof(process_argc));
  if (process_argc < 0) {
    free(buf);
    return report_error(EINVAL);
  }

  /* Skip argc, the exec path, its NUL padding, and exactly argc argv
     strings. Only the remaining environment strings may match. */
  cursor = sizeof(process_argc);
  while (cursor < size && buf[cursor] != '\0') cursor++;
  if (cursor == size) {
    free(buf);
    return report_error(EINVAL);
  }
  while (cursor < size && buf[cursor] == '\0') cursor++;

  for (int i = 0; i < process_argc; ++i) {
    if (cursor >= size) {
      free(buf);
      return report_error(EINVAL);
    }
    while (cursor < size && buf[cursor] != '\0') cursor++;
    if (cursor == size) {
      free(buf);
      return report_error(EINVAL);
    }
    cursor++;
  }

  needle_size = strlen(argv[2]);
  while (cursor < size && buf[cursor] != '\0') {
    size_t entry_start = cursor;
    while (cursor < size && buf[cursor] != '\0') cursor++;
    if (cursor == size) {
      free(buf);
      return report_error(EINVAL);
    }
    if (cursor - entry_start == needle_size &&
        memcmp(buf + entry_start, argv[2], needle_size) == 0) {
      free(buf);
      puts("found");
      return 0;
    }
    cursor++;
  }

  free(buf);
  puts("absent");
  return 0;
}
EOF

cat > "${WORK}/argv-victim.c" <<'EOF'
#include <unistd.h>

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  for (;;) pause();
}
EOF

cc -Wall -Wextra -O2 "${WORK}/reader.c" -o "${WORK}/reader" || \
  setup_error "reader compilation failed"
cc -Wall -Wextra -O2 "${WORK}/argv-victim.c" -o "${WORK}/argv-victim" || \
  setup_error "argv victim compilation failed"
cp /bin/sleep "${WORK}/sleep" || setup_error "could not copy /bin/sleep"
codesign -s - -f "${WORK}/sleep" >/dev/null 2>&1 || \
  setup_error "could not codesign sleep copy"

python3 "${REPO_ROOT}/tools/seatbelt-profile.py" \
  --exec "${WORK}/reader" > "${WORK}/exec.sb" || \
  setup_error "exec profile generation failed"
python3 "${REPO_ROOT}/tools/seatbelt-profile.py" \
  --build > "${WORK}/build.sb" || \
  setup_error "build profile generation failed"

CANARY="$(python3 -c 'import secrets; print(secrets.token_hex(16))')" || \
  setup_error "canary generation failed"
[ -n "${CANARY}" ] || setup_error "canary generation returned an empty value"
NEEDLE="RUNNER_CANARY=${CANARY}"

wait_for_exec() {
  local target_pid="$1"
  local expected_comm="$2"
  local comm
  local attempts=0

  while [ "${attempts}" -lt 50 ]; do
    if ! kill -0 "${target_pid}" 2>/dev/null; then
      return 1
    fi
    comm="$(ps -p "${target_pid}" -o comm= 2>/dev/null)" || comm=""
    if [ "${comm}" = "${expected_comm}" ]; then
      return 0
    fi
    /bin/sleep 0.1
    attempts=$((attempts + 1))
  done
  return 1
}

defer_signals
/usr/bin/env -i "${NEEDLE}" PATH=/usr/bin:/bin "${WORK}/sleep" 600 &
VICTIM_PID=$!
restore_signals
wait_for_exec "${VICTIM_PID}" "${WORK}/sleep" || \
  setup_error "environment victim did not exec the sleep copy within 5 seconds"

defer_signals
/usr/bin/env -i PATH=/usr/bin:/bin "${WORK}/argv-victim" "${NEEDLE}" &
ARGV_VICTIM_PID=$!
restore_signals
wait_for_exec "${ARGV_VICTIM_PID}" "${WORK}/argv-victim" || \
  setup_error "argv-only victim did not exec within 5 seconds"

run_mode unsandboxed "${VICTIM_PID}" "${WORK}/reader"
run_mode exec "${VICTIM_PID}" sandbox-exec -f "${WORK}/exec.sb" \
  "${WORK}/reader"
run_mode build "${VICTIM_PID}" sandbox-exec -f "${WORK}/build.sb" \
  "${WORK}/reader"
run_negative_mode argv-negative "${ARGV_VICTIM_PID}" "${WORK}/reader"

finish_result "${STRICT}"
exit $?
