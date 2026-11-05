#!/usr/bin/env bash
# Sourced by the tools/test-*.sh scripts that drive the scripts with an
# official ranked path (sandboxed-build.sh, sandboxed-cli.sh). Not run alone.
#
# Those scripts take the official path when RUNNER_ENVIRONMENT is self-hosted
# or MLXFAST_OFFICIAL_BENCHMARK_RUN is 1. The path needs the ranked-box
# variables, which only a ranked box sets. On any other self-hosted runner the
# official path refuses, as designed. A test on such a host asserts that
# refusal once, then runs its local cases with both variables removed. The
# official cases that set the ranked-box variables themselves still run.

OFFICIAL_RUN_TEST_ENV_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

# The ranked-box variables: the ones seatbelt-profile.py requires for an
# official run, read from that script.
official_ranked_variables() {
  python3 -I - "${OFFICIAL_RUN_TEST_ENV_DIR}/../seatbelt-profile.py" <<'PY'
import importlib.util
import sys
spec = importlib.util.spec_from_file_location("seatbelt_profile", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print("\n".join(module.OFFICIAL_REQUIRED))
PY
}

# Succeeds when the caller's environment takes the official path and at least
# one ranked-box variable is unset.
official_path_without_ranked_box() {
  [[ "${RUNNER_ENVIRONMENT:-}" == self-hosted \
    || "${MLXFAST_OFFICIAL_BENCHMARK_RUN:-0}" == 1 ]] || return 1
  local names name
  names="$(official_ranked_variables)" || return 0
  [[ -n "${names}" ]] || return 0
  while IFS= read -r name; do
    [[ -n "${!name:-}" ]] || return 0
  done <<<"${names}"
  return 1
}

# official_refusal_then_local REFUSAL... -- COMMAND...
# Where official_path_without_ranked_box succeeds: run COMMAND in the caller's
# environment, require its combined output to equal one of the REFUSAL lines,
# then unset RUNNER_ENVIRONMENT and MLXFAST_OFFICIAL_BENCHMARK_RUN in the
# calling shell. Elsewhere it does nothing. It returns 1, and prints the
# output, when the refusal differs.
official_refusal_then_local() {
  local refusals=() out refusal
  while [[ $# -gt 0 && "$1" != -- ]]; do
    refusals+=("$1")
    shift
  done
  [[ "${1:-}" == -- && ${#refusals[@]} -gt 0 && $# -ge 2 ]] || return 2
  shift
  official_path_without_ranked_box || return 0
  out="$("$@" 2>&1)" || true
  for refusal in "${refusals[@]}"; do
    if [[ "${out}" == "${refusal}" ]]; then
      unset RUNNER_ENVIRONMENT MLXFAST_OFFICIAL_BENCHMARK_RUN
      return 0
    fi
  done
  printf 'official refusal differs: %s\n' "${out}" >&2
  return 1
}
