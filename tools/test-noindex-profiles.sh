#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
WORK="$(mktemp -d)"
WORK="$(cd -- "${WORK}" && pwd -P)"
LAUNCH_LABELS=()
cleanup() {
  local rc=$? cleanup_rc=0 label
  trap - EXIT
  for label in ${LAUNCH_LABELS[@]+"${LAUNCH_LABELS[@]}"}; do
    /bin/launchctl remove "${label}" >/dev/null 2>&1 || true
    if /bin/launchctl list "${label}" >/dev/null 2>&1; then
      printf 'test-noindex-profiles.sh: FAIL: could not remove launch job %s\n' \
        "${label}" >&2
      cleanup_rc=1
    fi
  done
  rm -rf "${WORK}"
  (( rc != 0 )) && exit "${rc}"
  exit "${cleanup_rc}"
}
trap cleanup EXIT
fail() { printf 'test-noindex-profiles.sh: FAIL: %s\n' "$*" >&2; exit 1; }

mkdir -p "${WORK}/build.noindex" "${WORK}/runtime.noindex" \
  "${WORK}/goldens" "${WORK}/baseline" "${WORK}/benchd" "${WORK}/runner/work"
common_env=(
  HOME="${WORK}"
  MLXFAST_QWEN38_GOLDEN_DIR="${WORK}/goldens"
  MLXFAST_BASELINE_WORKSPACE="${WORK}/baseline"
  MLXFAST_BASELINE_CALIBRATION="${WORK}/cal.json"
  BENCHD_BIN_DIR="${WORK}/benchd"
  RUNNER_WORKSPACE="${WORK}/runner/work"
)
build_profile="${WORK}/build.sb"
runtime_profile="${WORK}/runtime.sb"
env "${common_env[@]}" python3 "${ROOT}/tools/seatbelt-profile.py" \
  --build --official --tree "${ROOT}" \
  --write-subpath "${WORK}/build.noindex" > "${build_profile}"
env "${common_env[@]}" python3 "${ROOT}/tools/seatbelt-profile.py" \
  --official --tree "${ROOT}" \
  --exec /usr/bin/mdimport --exec /usr/bin/mdutil --exec /usr/bin/mdfind \
  --exec /bin/launchctl \
  --write-subpath "${WORK}/runtime.noindex" \
  --unix-socket "${WORK}/runtime.noindex/resident.sock" > "${runtime_profile}"

python3 - "${build_profile}" "${runtime_profile}" <<'PY' \
  || fail "an official profile has an indexed write grant"
import ast
import os
import re
import sys

devices = {"/dev/null", "/dev/zero", "/dev/dtracehelper"}
safe_roots = ("/private/tmp", "/private/var/folders", "/var/folders")

def resolve(path):
    path = os.path.abspath(path)
    rest = []
    while not os.path.lexists(path) and path != "/":
        path, tail = os.path.split(path)
        rest.insert(0, tail)
    return os.path.join(os.path.realpath(path), *rest)

def safe(path):
    if path in devices:
        return True
    path = resolve(path)
    if any(path == root or path.startswith(root + "/") for root in safe_roots):
        return True
    return any(part.endswith(".noindex") for part in path.split(os.sep) if part)

for profile_name in sys.argv[1:]:
    with open(profile_name, encoding="utf-8") as profile:
        for raw in profile:
            rule = raw.strip()
            if not rule.startswith("(allow file-write*"):
                continue
            matches = re.findall(r'\((literal|subpath) ("(?:[^"\\]|\\.)*")\)', rule)
            if not matches or "(regex " in rule:
                print(rule, file=sys.stderr)
                raise SystemExit(1)
            for _, quoted in matches:
                if not safe(ast.literal_eval(quoted)):
                    print(rule, file=sys.stderr)
                    raise SystemExit(1)
PY

for rule in \
  '(deny mach-lookup (global-name-prefix "com.apple.metadata.mds"))' \
  '(deny process-exec (literal "/usr/bin/mdimport") (literal "/usr/bin/mdutil"))'; do
  grep -Fq "${rule}" "${build_profile}" || fail "build profile is missing ${rule}"
done
grep -Fq '(deny mach-lookup (global-name-prefix "com.apple.metadata.mds"))' \
  "${runtime_profile}" || fail "runtime profile is missing the Spotlight mach deny"
for profile in "${build_profile}" "${runtime_profile}"; do
  grep -Fqx '(deny job-creation)' "${profile}" \
    || fail "${profile##*/} is missing (deny job-creation)"
  for log_root in .mlxfast-resident-logs .mlxfast-resident-logs.noindex; do
    grep -Fq "(deny file-read* file-write* (subpath \"${WORK}/${log_root}\"))" \
      "${profile}" || fail "${profile##*/} does not deny ${log_root}"
  done
done

if [[ "$(uname -s)" == Darwin && -x /usr/bin/sandbox-exec ]]; then
  # mdfind exits 0 with no result when it cannot reach Spotlight, so a direct
  # bootstrap lookup of the Spotlight service is the real probe.
  probe="${WORK}/build.noindex/mds-probe"
  printf '%s\n' '#include <servers/bootstrap.h>' \
    'int main(int c, char **v) { mach_port_t p; (void)c;' \
    '  return bootstrap_look_up(bootstrap_port, v[1], &p) == KERN_SUCCESS ? 0 : 3; }' \
    | /usr/bin/xcrun -sdk macosx clang -x c - -o "${probe}" || fail "could not build the lookup probe"
  "${probe}" com.apple.metadata.mds || fail "Spotlight lookup control failed"
  /usr/bin/mdimport -L >/dev/null 2>&1 || fail "mdimport control failed"
  # The real runtime profile allows only the engine executable.
  probe_runtime_profile="${WORK}/runtime-probe.sb"
  env "${common_env[@]}" python3 "${ROOT}/tools/seatbelt-profile.py" \
    --official --tree "${ROOT}" --exec "${probe}" \
    --write-subpath "${WORK}/runtime.noindex" > "${probe_runtime_profile}"
  for profile in "${build_profile}" "${probe_runtime_profile}"; do
    if /usr/bin/sandbox-exec -f "${profile}" "${probe}" com.apple.metadata.mds; then
      fail "Spotlight mach lookup escaped ${profile##*/}"
    fi
    if /usr/bin/sandbox-exec -f "${profile}" /usr/bin/mdimport -L >/dev/null 2>&1; then
      fail "mdimport escaped ${profile##*/}"
    fi
    if /usr/bin/sandbox-exec -f "${profile}" /usr/bin/mdutil -s / >/dev/null 2>&1; then
      fail "mdutil escaped ${profile##*/}"
    fi
  done
  for profile in "${build_profile}" "${runtime_profile}"; do
    kind="${profile##*/}"
    label="com.gumbii.seatbelt-job-deny.$$.$RANDOM"
    LAUNCH_LABELS+=("${label}")
    if /usr/bin/sandbox-exec -f "${profile}" \
      /bin/launchctl submit -l "${label}" -- /bin/sleep 30; then
      fail "job creation escaped ${kind}"
    fi
    if /bin/launchctl list "${label}" >/dev/null 2>&1; then
      fail "failed job creation left launch job ${label} from ${kind}"
    fi
  done
else
  printf 'test-noindex-profiles.sh: SKIP: native Seatbelt probes need macOS sandbox-exec\n'
fi

printf 'test-noindex-profiles.sh: PASS\n'
