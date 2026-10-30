#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/test-public-log-paths.XXXXXX")"
trap 'rm -rf "${WORK}"' EXIT

fail() {
  printf 'test-public-log-paths.sh: FAIL: %s\n' "$*" >&2
  exit 1
}

public_shell=(
  "${ROOT}/.github/workflows/benchmark.yml"
  "${ROOT}/setup.sh"
  "${ROOT}/tools/build-cache.sh"
  "${ROOT}/tools/fetch-benchd.sh"
  "${ROOT}/tools/qwen38-27b-tf-measure-and-score.sh"
  "${ROOT}/tools/ranked-box-preflight.sh"
  "${ROOT}/tools/sandboxed-cli.sh"
  "${ROOT}/tools/stage-bench-worker.sh"
  "${ROOT}/tools/stage-drafter.sh"
  "${ROOT}/tools/stop-leaked-residents.sh"
  "${ROOT}/tools/zig-bin.sh"
)
risky='(^|[[:space:];])(echo|printf|ok|fail)[[:space:]].*\$\{(MLXFAST_[A-Z0-9_]*(DIR|PATH|WORKSPACE|CALIBRATION)|BENCHD_BIN_DIR|HOME|RUNNER_TEMP|RUNNER_WORKSPACE|RUNNER_NAME)(:|}|/)'
hits="$(grep -En "${risky}" "${public_shell[@]}" || true)"
[[ -z "${hits}" ]] || fail "public output expands a private path or box variable: ${hits}"
alias_risky='(^|[[:space:];])(echo|printf|ok|fail)[[:space:]].*\$\{(ZIG|ZIG_BIN|zig_bin|reason)(:|}|/)'
hits="$(grep -En "${alias_risky}" "${ROOT}/.github/workflows/benchmark.yml" || true)"
[[ -z "${hits}" ]] || fail "public output expands a private variable alias: ${hits}"
hits="$(grep -En 'ZIG=.*GITHUB_ENV|GITHUB_ENV.*ZIG=' "${ROOT}/.github/workflows/benchmark.yml" || true)"
[[ -z "${hits}" ]] || fail "the Zig path is exported to later steps: ${hits}"
# Every displayed refusal reason must come from the filter, not a plain copy.
hits="$(grep -En 'displayed_reason=' "${ROOT}/.github/workflows/benchmark.yml" | grep -v 'public-log-filter.py' || true)"
[[ -z "${hits}" ]] || fail "a displayed refusal reason skips the filter: ${hits}"
[[ "$(grep -c 'displayed_reason=.*public-log-filter.py' "${ROOT}/.github/workflows/benchmark.yml")" == "2" ]] \
  || fail "the refusal step does not filter both displayed reasons"
# A plain cp in the ranked job prints staged paths on failure.
hits="$(grep -En '(^|[[:space:]]|\$\()cp -' "${ROOT}/.github/workflows/benchmark.yml" | grep -v '2>' || true)"
[[ -z "${hits}" ]] || fail "a cp in the workflow sends its errors to the public log: ${hits}"
short_out="$(printf '/various /tmpfile /usr/bin\n' | env HOME=/var RUNNER_TEMP=/tmp/ python3 -u "${ROOT}/tools/public-log-filter.py")"
[[ "${short_out}" == "/various /tmpfile /usr/bin" ]] || fail "a system root was rewritten: ${short_out}"
short_out="$(printf '/tmp/private-run-27/file /tmp\n' | env RUNNER_TEMP=/tmp python3 -u "${ROOT}/tools/public-log-filter.py")"
[[ "${short_out}" == "<runner-work>/private-run-27/file <runner-work>" ]] || fail "a short configured root was not hidden: ${short_out}"
esc_out="$(printf '\033[31m/home/op-x/f\033[0m {"p":"\\/home\\/op-x\\/f"} \033[1mbox-x\033[0m\n' \
  | env HOME=/home/op-x RUNNER_NAME=box-x python3 -u "${ROOT}/tools/public-log-filter.py")"
[[ "${esc_out}" != *op-x* && "${esc_out}" != *box-x* ]] || fail "a coloured or JSON-escaped value leaked: ${esc_out}"

private_zig="${WORK}/private/toolchain/zig"
if output="$(ZIG="${private_zig}" bash "${ROOT}/tools/zig-bin.sh" 2>&1)"; then
  fail "a missing configured Zig was accepted"
fi
[[ "${output}" != *"${private_zig}"* ]] || fail "the Zig error leaked its path"
mkdir -p "$(dirname -- "${private_zig}")"
printf '#!/usr/bin/env bash\nexit 1\n' > "${private_zig}"
chmod +x "${private_zig}"
if output="$(ZIG="${private_zig}" bash "${ROOT}/tools/zig-bin.sh" 2>&1)"; then
  fail "a Zig with no readable version was accepted"
fi
[[ "${output}" != *"${private_zig}"* ]] || fail "the Zig version error leaked its path"
printf '#!/usr/bin/env bash\nprintf "0.16.0\\n"\n' > "${private_zig}"
if output="$(ZIG="${private_zig}" bash "${ROOT}/tools/zig-bin.sh" 2>&1)"; then
  fail "a Zig with the wrong version was accepted"
fi
[[ "${output}" != *"${private_zig}"* ]] || fail "the wrong Zig version leaked its path"

mkdir -p "${WORK}/stage-tree/tools" \
  "${WORK}/stage-tree/Vendor/TensorFold/zig-out/bin"
cp "${ROOT}/tools/stage-bench-worker.sh" "${WORK}/stage-tree/tools/"
printf '#!/usr/bin/env bash\nexit 0\n' \
  > "${WORK}/stage-tree/Vendor/TensorFold/zig-out/bin/bench-worker"
chmod +x "${WORK}/stage-tree/Vendor/TensorFold/zig-out/bin/bench-worker"
stage_out="$(MLXFAST_SKIP_WORKER_SELF_TEST=1 \
  bash "${WORK}/stage-tree/tools/stage-bench-worker.sh")"
[[ "${stage_out}" == \
  'stage-bench-worker.sh: worker ready at Vendor/TensorFold/zig-out/bin/bench-worker' ]] \
  || fail "worker message is not repo-relative: ${stage_out}"
[[ "${stage_out}" != *"${WORK}"* ]] || fail "worker message contains an absolute path"

staging="${WORK}/organizer"
staging_real="${WORK}/organizer-real"
runner="${WORK}/runner/actions"
mkdir -p "${staging_real}/baseline" "${staging_real}/reference" "${staging_real}/benchd" \
  "${staging_real}/goldens" "${runner}/_work/repo/repo" "${WORK}/runner/tmp" \
  "${WORK}/home"
staging_real="$(cd -- "${staging_real}" && pwd -P)"
ln -s "${staging_real}" "${staging}"
calibration="${staging}/baseline-calibration.json"
printf '{"box":"calibration-box"}\n' > "${calibration}"

filter_env=(
  "MLXFAST_BASELINE_WORKSPACE=${staging}/baseline"
  "MLXFAST_REFERENCE_DIR=${staging}/reference"
  "BENCHD_BIN_DIR=${staging}/benchd"
  "MLXFAST_QWEN38_GOLDEN_DIR=${staging}/goldens"
  "MLXFAST_BASELINE_CALIBRATION=${calibration}"
  "MLXFAST_PRIVATE_DIR=/var/mlxfast/private"
  "GITHUB_WORKSPACE=${runner}/_work/repo/repo"
  "RUNNER_WORKSPACE=${runner}/_work/repo"
  "RUNNER_TEMP=${WORK}/runner/tmp"
  "HOME=${WORK}/home"
  "RUNNER_NAME=runner-box"
  "RESIDENT_UP_LOCK_PATH=/private/tmp/mtplx-test.lock"
)

printf '%s \377\n' "${staging}/reference" \
  | env "${filter_env[@]}" python3 -u "${ROOT}/tools/public-log-filter.py" \
  > "${WORK}/non-utf8.out"
printf '<staging> \377\n' > "${WORK}/non-utf8.expected"
cmp "${WORK}/non-utf8.expected" "${WORK}/non-utf8.out" >/dev/null \
  || fail "the filter did not preserve a non-UTF-8 byte"

filtered="$({
  printf 'benchd iterate: control loads %s/reference_weights/model\n' "${staging}/baseline"
  printf 'paired official run on box "%s" using calibration box "%s"\n' \
    runner-box calibration-box
  printf 'resident is up from %s/tree.noindex and temp %s/file\n' \
    "${runner}/_work/repo/repo" "${WORK}/runner/tmp"
  printf 'install %s/bin home %s/cache goldens %s/tape\n' \
    "${runner}" "${WORK}/home" "${staging}/goldens"
  printf 'resolved stage %s/model private aliases %s/file and %s\n' \
    "${staging_real}/reference" /private/var/mlxfast/private /tmp/mtplx-test.lock
} | env "${filter_env[@]}" python3 -u "${ROOT}/tools/public-log-filter.py")"

for secret in "${staging}" "${staging_real}" "${runner}" "${WORK}/runner/tmp" "${WORK}/home" \
  runner-box calibration-box; do
  [[ "${filtered}" != *"${secret}"* ]] || fail "filter leaked ${secret}"
done
[[ "${filtered}" == *'<staging>/reference_weights/model'* ]] \
  || fail "filter did not name staged paths"
[[ "${filtered}" == *'<runner-work>/tree.noindex'* ]] \
  || fail "filter did not name runner work paths"
[[ "${filtered}" == *'box "<box>" using calibration box "<box>"'* ]] \
  || fail "filter did not hide both box names"
[[ "${filtered}" == *'<home>/cache'* ]] || fail "filter did not hide HOME"
[[ "${filtered}" == *'resolved stage <staging>/model private aliases <staging>/file and <gpu-lock>'* ]] \
  || fail "filter did not hide resolved and private path aliases"

sealed_error="cannot read ${staging_real}/goldens/tape.json on box runner-box"
displayed_error="$(printf '%s\n' "${sealed_error}" \
  | env "${filter_env[@]}" python3 -u "${ROOT}/tools/public-log-filter.py")"
[[ "${displayed_error}" == 'cannot read <staging>/tape.json on box <box>' ]] \
  || fail "refusal text was not mapped to public role names: ${displayed_error}"

set +e
{
  printf 'benchd failed under %s on %s\n' "${staging}/reference" runner-box
  exit 37
} 2>&1 | env "${filter_env[@]}" python3 -u \
  "${ROOT}/tools/public-log-filter.py" > "${WORK}/failure.log"
pipe_status=("${PIPESTATUS[@]}")
producer_rc="${pipe_status[0]}"
filter_rc="${pipe_status[1]}"
set -e
[[ "${producer_rc}" == 37 && "${filter_rc}" == 0 ]] \
  || fail "pipeline changed exit codes (producer=${producer_rc}, filter=${filter_rc})"
grep -F 'benchd failed under <staging> on <box>' "${WORK}/failure.log" >/dev/null \
  || fail "failure output was not filtered"
grep -F 'PIPESTATUS[0]' "${ROOT}/.github/workflows/benchmark.yml" >/dev/null \
  || fail "ranked workflow does not preserve the benchd exit code"

printf 'test-public-log-paths.sh: PASS\n'
