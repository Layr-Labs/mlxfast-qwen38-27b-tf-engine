#!/usr/bin/env bash
# Offline security regression suite for the kept control-plane gates.
set -uo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
SCRIPTS="${ROOT}/.github/scripts"
EXPECTED_MIN_ASSERTIONS=201
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
assertions=0
failures=0

unset GITHUB_ENV GITHUB_OUTPUT GITHUB_PATH GITHUB_STEP_SUMMARY
export GIT_CONFIG_GLOBAL=/dev/null
export GIT_CONFIG_NOSYSTEM=1

pass() {
  assertions=$((assertions + 1))
  printf 'ok %d - %s\n' "${assertions}" "$1"
}

fail() {
  assertions=$((assertions + 1))
  failures=$((failures + 1))
  printf 'not ok %d - %s\n' "${assertions}" "$1" >&2
}

assert_status() {
  local name="$1" expected="$2"
  shift 2
  "$@" >"${WORK}/assert.out" 2>&1
  local actual=$?
  if [[ "${actual}" -eq "${expected}" ]]; then
    pass "${name}"
  else
    fail "${name} (exit ${actual}, expected ${expected}): $(tail -1 "${WORK}/assert.out")"
  fi
}

assert_status_contains() {
  local name="$1" expected="$2" pattern="$3"
  shift 3
  "$@" >"${WORK}/assert.out" 2>&1
  local actual=$?
  if [[ "${actual}" -ne "${expected}" ]]; then
    fail "${name} (exit ${actual}, expected ${expected}): $(tail -1 "${WORK}/assert.out")"
  elif grep -q -- "${pattern}" "${WORK}/assert.out"; then
    pass "${name}"
  else
    fail "${name} (missing diagnostic ${pattern}): $(tail -1 "${WORK}/assert.out")"
  fi
}

assert_equal() {
  local name="$1" actual="$2" expected="$3"
  if [[ "${actual}" == "${expected}" ]]; then
    pass "${name}"
  else
    fail "${name} (got ${actual}, expected ${expected})"
  fi
}

for driver in enforce-modifiable-surface.sh hardened-git.sh \
  overlay-editable-paths.sh submission-static-review-checks.sh; do
  if [[ -x "${SCRIPTS}/${driver}" ]]; then
    pass "harness/${driver} is executable"
  else
    fail "harness/${driver} is executable"
  fi
done
[[ -x "${ROOT}/tools/lint-benchmark-manifest.py" ]] \
  && pass "harness/manifest linter is executable" \
  || fail "harness/manifest linter is executable"

manifest_variant() {
  local name="$1" expression="$2"
  jq "${expression}" "${ROOT}/benchmark.json" > "${WORK}/${name}.json"
}

lint_manifest() {
  python3 "${ROOT}/tools/lint-benchmark-manifest.py" \
    --repo-root "${ROOT}" --manifest "$1"
}

assert_status "manifest/real manifest passes" 0 lint_manifest "${ROOT}/benchmark.json"

manifest_cases=(
  'missing-command|.setupCommand=["bash","-c","./missing-command.sh"]|target ./missing-command.sh does not exist'
  'trusted-optional|.optionalEditablePaths += [".github"]|trusted-scope exclusion'
  'bad-cap-string|.editableSurfaceByteBudget.maxGrowthBytes="large"|not a positive integer'
  'bad-cap-zero|.editableSurfaceByteBudget.maxGrowthBytes=0|not a positive integer'
  'bad-cap-negative|.editableSurfaceByteBudget.maxGrowthBytes=-1|not a positive integer'
  'bad-cap-float|.editableSurfaceByteBudget.maxGrowthBytes=1.5|not a positive integer'
  'bad-cap-bool|.editableSurfaceByteBudget.maxGrowthBytes=true|not a positive integer'
  'file-over-total|.editableSurfaceByteBudget.maxFileBytes=100000001|exceeds maxTotalBytes'
  'wrong-exponents|.scoring.scoredExponents.prefillGainExponent=0.5|prefillGainExponent'
  'duplicate-editable|.editablePaths += ["Vendor/TensorFold"]|duplicate entry'
  'nested-editable|.editablePaths += ["Vendor/TensorFold/src"]|nested inside'
  'optional-not-editable|.optionalEditablePaths += ["Vendor"]|not in editablePaths'
  'exempt-not-editable|.editableSurfaceByteBudget.exemptPaths=["Vendor"]|not an editable path'
  'bad-track-id|.staticReviewTrackId="wrong-track"|staticReviewTrackId'
  'bad-command-shape|.setupCommand=["./setup.sh"]|expected'
  'empty-command|.setupCommand=["bash","-c",""]|names no repo-relative entry point'
  'bad-score-path|.benchmarkCommand=["bash","-c","./tools/qwen38-27b-tf-yukon-run.sh"]|must set MLXFAST_SCORE_PATH'
)
for item in "${manifest_cases[@]}"; do
  IFS='|' read -r name expression pattern <<<"${item}"
  manifest_variant "${name}" "${expression}"
  assert_status_contains "manifest/${name} is refused" 1 "${pattern}" \
    lint_manifest "${WORK}/${name}.json"
done

inject_entry() {
  local name="$1" scope="$2" entry="$3"
  jq --arg scope "${scope}" --arg entry "${entry}" '
    if $scope == "editablePaths" then .editablePaths += [$entry]
    elif $scope == "optionalEditablePaths" then .optionalEditablePaths += [$entry]
    else .editableSurfaceByteBudget.exemptPaths = ((.editableSurfaceByteBudget.exemptPaths // []) + [$entry])
    end
  ' "${ROOT}/benchmark.json" > "${WORK}/${name}.json"
}

for root in benchd .gitmodules benchd.pin benchd-bin; do
  upper="$(printf '%s' "${root}" | tr '[:lower:]' '[:upper:]')"
  for shape in "${root}" "${root}/child" "${upper}" "./${root}" "${root}//child"; do
    key="forbidden-$(printf '%s' "${shape}" | shasum | cut -c1-12)"
    inject_entry "${key}" editablePaths "${shape}"
    assert_status_contains "manifest/forbidden path ${shape} is refused" 1 \
      "gitlink exclusion" lint_manifest "${WORK}/${key}.json"
  done
done

trusted_roots=(.github tools benchmark.json setup.sh fixtures correctness_prompts)
for root in "${trusted_roots[@]}"; do
  upper="$(printf '%s' "${root}" | tr '[:lower:]' '[:upper:]')"
  for shape in "${root}" "${root}/child" "${upper}" "./${root}" "${root}//child"; do
    pattern="trusted-scope exclusion"
    [[ "${shape}" == ./* ]] && pattern="gitlink exclusion"
    key="trusted-$(printf '%s' "${shape}" | shasum | cut -c1-12)"
    inject_entry "${key}" editablePaths "${shape}"
    assert_status_contains "manifest/trusted path ${shape} is refused" 1 \
      "${pattern}" lint_manifest "${WORK}/${key}.json"
  done
  for scope in optionalEditablePaths exemptPaths; do
    key="trusted-${scope}-$(printf '%s' "${root}" | shasum | cut -c1-12)"
    inject_entry "${key}" "${scope}" "${root}"
    assert_status_contains "manifest/${scope} cannot reach ${root}" 1 \
      "trusted-scope exclusion" lint_manifest "${WORK}/${key}.json"
  done
done

invalid_paths=(/tmp/escape ../escape src/../escape ./src src/./file :path 'src\file' '' . ./)
index=0
for entry in "${invalid_paths[@]}"; do
  index=$((index + 1))
  inject_entry "invalid-path-${index}" editablePaths "${entry}"
  assert_status "manifest/invalid path shape ${index} is refused" 1 \
    lint_manifest "${WORK}/invalid-path-${index}.json"
done

CONTRACT='{
  "editablePaths": ["src", "config.txt", "mtp-head.manifest.json"],
  "optionalEditablePaths": ["mtp-head.manifest.json"],
  "editableSurfaceByteBudget": {
    "maxTotalBytes": 700000,
    "maxFileBytes": 350000,
    "maxGrowthBytes": 262144
  },
  "staticReviewTrackId": "test-track-v1"
}'

new_fixture() {
  local dir
  dir="$(mktemp -d "${WORK}/fixture.XXXXXX")"
  mkdir -p "${dir}/trusted/src" "${dir}/sub/src"
  printf '%s\n' "${CONTRACT}" > "${dir}/trusted/benchmark.json"
  printf 'trusted kernel\n' > "${dir}/trusted/src/kernel.txt"
  printf 'trusted helper\n' > "${dir}/trusted/src/helper.txt"
  printf 'trusted config\n' > "${dir}/trusted/config.txt"
  printf '{}\n' > "${dir}/trusted/mtp-head.manifest.json"
  printf 'trusted readme\n' > "${dir}/trusted/README.md"
  printf 'submitted kernel\n' > "${dir}/sub/src/kernel.txt"
  printf 'submitted config\n' > "${dir}/sub/config.txt"
  printf '%s\n' "${dir}"
}

run_overlay() {
  local trusted="$1" sub="$2"
  env -C "${trusted}" SUBMISSION_WORKTREE="${sub}" CONTRACT_PATH=benchmark.json \
    "${SCRIPTS}/overlay-editable-paths.sh"
}

fx="$(new_fixture)"
assert_status_contains "overlay/benign archive overlays" 0 "trusted harness retained" \
  run_overlay "${fx}/trusted" "${fx}/sub"
assert_equal "overlay/editable file is replaced" \
  "$(cat "${fx}/trusted/src/kernel.txt")" "submitted kernel"
assert_equal "overlay/replace removes omitted editable files" \
  "$([[ -e "${fx}/trusted/src/helper.txt" ]] && echo present || echo absent)" "absent"
assert_equal "overlay/optional declaration keeps trusted copy" \
  "$(cat "${fx}/trusted/mtp-head.manifest.json")" '{}'
assert_equal "overlay/non-editable file stays trusted" \
  "$(cat "${fx}/trusted/README.md")" "trusted readme"

fx="$(new_fixture)"
rm "${fx}/sub/config.txt"
assert_status_contains "overlay/missing required path is refused" 1 "is missing" \
  run_overlay "${fx}/trusted" "${fx}/sub"

fx="$(new_fixture)"
ln -s "${fx}/trusted/README.md" "${fx}/sub/src/link"
assert_status_contains "overlay/symlink below an editable path is refused" 1 "symlinks" \
  run_overlay "${fx}/trusted" "${fx}/sub"

fx="$(new_fixture)"
rm -rf "${fx}/sub/src"
ln -s "${fx}/trusted/src" "${fx}/sub/src"
assert_status_contains "overlay/editable root symlink is refused" 1 "symlinks" \
  run_overlay "${fx}/trusted" "${fx}/sub"

fx="$(new_fixture)"
mkfifo "${fx}/sub/src/fifo"
assert_status_contains "overlay/FIFO is refused" 1 "regular files" \
  run_overlay "${fx}/trusted" "${fx}/sub"

fx="$(new_fixture)"
chmod u+s "${fx}/sub/src/kernel.txt"
assert_status "overlay/setuid source overlays with safe copy mode" 0 \
  run_overlay "${fx}/trusted" "${fx}/sub"
assert_equal "overlay/setuid bit does not land" \
  "$(find "${fx}/trusted/src/kernel.txt" -perm -4000 -print)" ""

fx="$(new_fixture)"
ln "${fx}/sub/src/kernel.txt" "${fx}/sub/src/hardlink.txt"
assert_status_contains "overlay/hardlinked file is refused" 1 "hardlinked" \
  run_overlay "${fx}/trusted" "${fx}/sub"

fx="$(new_fixture)"
printf 'attacker readme\n' > "${fx}/sub/README.md"
printf 'untouched\n' > "${fx}/sentinel.txt"
assert_status "overlay/non-editable archive payload is inert" 0 \
  run_overlay "${fx}/trusted" "${fx}/sub"
assert_equal "overlay/non-editable archive file does not land" \
  "$(cat "${fx}/trusted/README.md")" "trusted readme"
assert_equal "overlay/sentinel above trusted root stays unchanged" \
  "$(cat "${fx}/sentinel.txt")" "untouched"

for entry in "${invalid_paths[@]}"; do
  fx="$(new_fixture)"
  jq --arg entry "${entry}" '.editablePaths=[$entry]' \
    "${fx}/trusted/benchmark.json" > "${fx}/trusted/benchmark.tmp"
  mv "${fx}/trusted/benchmark.tmp" "${fx}/trusted/benchmark.json"
  assert_status "overlay/invalid contract path ${entry} is refused" 1 \
    run_overlay "${fx}/trusted" "${fx}/sub"
done

for root in benchd .gitmodules benchd.pin benchd-bin; do
  upper="$(printf '%s' "${root}" | tr '[:lower:]' '[:upper:]')"
  for entry in "${root}" "${root}/child" "${upper}"; do
    fx="$(new_fixture)"
    jq --arg entry "${entry}" '.editablePaths=["src","config.txt",$entry]' \
      "${fx}/trusted/benchmark.json" > "${fx}/trusted/benchmark.tmp"
    mv "${fx}/trusted/benchmark.tmp" "${fx}/trusted/benchmark.json"
    assert_status_contains "overlay/forbidden path ${entry} is refused" 1 \
      "measurement-harness surface" run_overlay "${fx}/trusted" "${fx}/sub"
  done
done

for mutation in empty missing malformed; do
  fx="$(new_fixture)"
  case "${mutation}" in
    empty) jq '.editablePaths=[]' "${fx}/trusted/benchmark.json" > "${fx}/trusted/benchmark.tmp"; mv "${fx}/trusted/benchmark.tmp" "${fx}/trusted/benchmark.json" ;;
    missing) jq 'del(.editablePaths)' "${fx}/trusted/benchmark.json" > "${fx}/trusted/benchmark.tmp"; mv "${fx}/trusted/benchmark.tmp" "${fx}/trusted/benchmark.json" ;;
    malformed) printf '{bad json\n' > "${fx}/trusted/benchmark.json" ;;
  esac
  assert_status "overlay/${mutation} contract is refused" 1 \
    run_overlay "${fx}/trusted" "${fx}/sub"
done

run_review() {
  local trusted="$1"
  shift
  env -C "${trusted}" CONTRACT_PATH=benchmark.json "$@" \
    "${SCRIPTS}/submission-static-review-checks.sh"
}

fx="$(new_fixture)"
assert_status_contains "static-review/benign whole surface passes" 0 \
  "deterministic checks passed" run_review "${fx}/trusted" \
  MLXFAST_SUBMISSION_REVIEW_MODE=whole-surface

fx="$(new_fixture)"
head -c 350001 /dev/zero > "${fx}/trusted/src/large.bin"
assert_status_contains "static-review/oversize file is refused" 1 "per-file" \
  run_review "${fx}/trusted" MLXFAST_SUBMISSION_REVIEW_MODE=whole-surface

fx="$(new_fixture)"
for i in 1 2 3; do head -c 250000 /dev/zero > "${fx}/trusted/src/chunk-${i}"; done
assert_status_contains "static-review/oversize total is refused" 1 "above static review limit" \
  run_review "${fx}/trusted" MLXFAST_SUBMISSION_REVIEW_MODE=whole-surface

fx="$(new_fixture)"
ln -s "${fx}/trusted/README.md" "${fx}/trusted/src/link"
assert_status_contains "static-review/symlink below editable path is refused" 1 "symlink" \
  run_review "${fx}/trusted" MLXFAST_SUBMISSION_REVIEW_MODE=whole-surface

fx="$(new_fixture)"
rm -rf "${fx}/trusted/src"
ln -s "${fx}/trusted/README.md" "${fx}/trusted/src"
assert_status_contains "static-review/editable root symlink is refused" 1 "symlink" \
  run_review "${fx}/trusted" MLXFAST_SUBMISSION_REVIEW_MODE=whole-surface

fx="$(new_fixture)"
assert_status_contains "static-review/no mode is refused" 1 "no review mode selected" \
  run_review "${fx}/trusted"
assert_status_contains "static-review/empty base is refused" 1 "set but empty" \
  run_review "${fx}/trusted" MLXFAST_SUBMISSION_REVIEW_BASE_SHA=
for mode in whole diff invalid; do
  assert_status "static-review/bad mode ${mode} with a base is refused" 1 \
    run_review "${fx}/trusted" MLXFAST_SUBMISSION_REVIEW_BASE_SHA=deadbeef \
    MLXFAST_SUBMISSION_REVIEW_MODE="${mode}"
done
assert_status_contains "static-review/track mismatch is refused" 1 "not the track" \
  run_review "${fx}/trusted" MLXFAST_SUBMISSION_REVIEW_MODE=whole-surface \
  MLXFAST_SUBMISSION_TRACK_ID=wrong
assert_status_contains "static-review/empty track is refused" 1 "set but empty" \
  run_review "${fx}/trusted" MLXFAST_SUBMISSION_REVIEW_MODE=whole-surface \
  MLXFAST_SUBMISSION_TRACK_ID=

for mutation in empty all-exempt; do
  fx="$(new_fixture)"
  if [[ "${mutation}" == empty ]]; then
    jq '.editablePaths=[]' "${fx}/trusted/benchmark.json" > "${fx}/trusted/benchmark.tmp"
  else
    jq '.editableSurfaceByteBudget.exemptPaths=.editablePaths' \
      "${fx}/trusted/benchmark.json" > "${fx}/trusted/benchmark.tmp"
  fi
  mv "${fx}/trusted/benchmark.tmp" "${fx}/trusted/benchmark.json"
  assert_status "static-review/${mutation} surface is refused" 1 \
    run_review "${fx}/trusted" MLXFAST_SUBMISSION_REVIEW_MODE=whole-surface
done

cap_values=('"large"' true false null 1.5 0 -1 '[]' '{}')
for cap in maxTotalBytes maxFileBytes maxGrowthBytes; do
  for value in "${cap_values[@]}"; do
    fx="$(new_fixture)"
    jq --arg cap "${cap}" --argjson value "${value}" \
      '.editableSurfaceByteBudget[$cap]=$value' \
      "${fx}/trusted/benchmark.json" > "${fx}/trusted/benchmark.tmp"
    mv "${fx}/trusted/benchmark.tmp" "${fx}/trusted/benchmark.json"
    assert_status "static-review/${cap} value ${value} is refused" 1 \
      run_review "${fx}/trusted" MLXFAST_SUBMISSION_REVIEW_MODE=whole-surface
  done
done

for value in null true '[]' '"large"'; do
  fx="$(new_fixture)"
  jq --argjson value "${value}" '.editableSurfaceByteBudget=$value' \
    "${fx}/trusted/benchmark.json" > "${fx}/trusted/benchmark.tmp"
  mv "${fx}/trusted/benchmark.tmp" "${fx}/trusted/benchmark.json"
  assert_status "static-review/budget object ${value} is refused" 1 \
    run_review "${fx}/trusted" MLXFAST_SUBMISSION_REVIEW_MODE=whole-surface
done

init_git_fixture() {
  local dir="$1"
  git -C "${dir}" init -q
  git -C "${dir}" config user.email test@example.com
  git -C "${dir}" config user.name test
  git -C "${dir}" add .
  git -C "${dir}" commit -qm base
}

fx="$(new_fixture)"
repo="${fx}/trusted"
init_git_fixture "${repo}"
base="$(git -C "${repo}" rev-parse HEAD)"
head -c 262145 /dev/zero > "${repo}/src/growth.bin"
git -C "${repo}" add src/growth.bin
git -C "${repo}" commit -qm growth
head_sha="$(git -C "${repo}" rev-parse HEAD)"
assert_status_contains "static-review/growth above 262144 is refused" 1 "growth limit 262144" \
  run_review "${repo}" MLXFAST_SUBMISSION_REVIEW_BASE_SHA="${base}" HEAD_SHA="${head_sha}"

fx="$(new_fixture)"
repo="${fx}/trusted"
init_git_fixture "${repo}"
base="$(git -C "${repo}" rev-parse HEAD)"
head -c 262144 /dev/zero > "${repo}/src/growth.bin"
git -C "${repo}" add src/growth.bin
git -C "${repo}" commit -qm growth
head_sha="$(git -C "${repo}" rev-parse HEAD)"
assert_status_contains "static-review/growth at 262144 passes" 0 "growth 262144 bytes" \
  run_review "${repo}" MLXFAST_SUBMISSION_REVIEW_BASE_SHA="${base}" HEAD_SHA="${head_sha}"

fx="$(new_fixture)"
repo="${fx}/trusted"
init_git_fixture "${repo}"
base="$(git -C "${repo}" rev-parse HEAD)"
printf 'small change\n' >> "${repo}/src/kernel.txt"
git -C "${repo}" add src/kernel.txt
git -C "${repo}" commit -qm small-change
head_sha="$(git -C "${repo}" rev-parse HEAD)"
assert_status_contains "static-review/small change passes" 0 "track test-track-v1" \
  run_review "${repo}" MLXFAST_SUBMISSION_REVIEW_BASE_SHA="${base}" HEAD_SHA="${head_sha}"

for attack in widen-cap self-exempt; do
  fx="$(new_fixture)"
  repo="${fx}/trusted"
  init_git_fixture "${repo}"
  base="$(git -C "${repo}" rev-parse HEAD)"
  case "${attack}" in
    widen-cap) jq '.editableSurfaceByteBudget.maxGrowthBytes=9999999' "${repo}/benchmark.json" > "${repo}/benchmark.tmp" ;;
    self-exempt) jq '.editableSurfaceByteBudget.exemptPaths=["src"]' "${repo}/benchmark.json" > "${repo}/benchmark.tmp" ;;
    rename-track) jq '.staticReviewTrackId="attacker"' "${repo}/benchmark.json" > "${repo}/benchmark.tmp" ;;
    drop-track) jq 'del(.staticReviewTrackId)' "${repo}/benchmark.json" > "${repo}/benchmark.tmp" ;;
  esac
  mv "${repo}/benchmark.tmp" "${repo}/benchmark.json"
  head -c 262145 /dev/zero > "${repo}/src/growth.bin"
  git -C "${repo}" add .
  git -C "${repo}" commit -qm "${attack}"
  head_sha="$(git -C "${repo}" rev-parse HEAD)"
  assert_status_contains "static-review/base contract defeats oversized ${attack}" 1 \
    "growth limit 262144" run_review "${repo}" \
    MLXFAST_SUBMISSION_REVIEW_BASE_SHA="${base}" HEAD_SHA="${head_sha}"
done

for attack in rename-track drop-track; do
  fx="$(new_fixture)"
  repo="${fx}/trusted"
  init_git_fixture "${repo}"
  base="$(git -C "${repo}" rev-parse HEAD)"
  case "${attack}" in
    rename-track) jq '.staticReviewTrackId="attacker"' "${repo}/benchmark.json" > "${repo}/benchmark.tmp" ;;
    drop-track) jq 'del(.staticReviewTrackId)' "${repo}/benchmark.json" > "${repo}/benchmark.tmp" ;;
  esac
  mv "${repo}/benchmark.tmp" "${repo}/benchmark.json"
  printf 'small change\n' >> "${repo}/src/kernel.txt"
  git -C "${repo}" add .
  git -C "${repo}" commit -qm "${attack}"
  head_sha="$(git -C "${repo}" rev-parse HEAD)"
  assert_status_contains "static-review/base contract defeats ${attack}" 0 \
    "track test-track-v1" run_review "${repo}" \
    MLXFAST_SUBMISSION_REVIEW_BASE_SHA="${base}" HEAD_SHA="${head_sha}"
done

run_surface() {
  local repo="$1" base="$2" head="$3"
  env -C "${repo}" BASE_SHA="${base}" HEAD_SHA="${head}" CONTRACT_PATH=benchmark.json \
    "${SCRIPTS}/enforce-modifiable-surface.sh"
}

fx="$(new_fixture)"
repo="${fx}/trusted"
init_git_fixture "${repo}"
base="$(git -C "${repo}" rev-parse HEAD)"
printf 'change\n' >> "${repo}/src/kernel.txt"
git -C "${repo}" add src/kernel.txt
git -C "${repo}" commit -qm inside
head_sha="$(git -C "${repo}" rev-parse HEAD)"
assert_status "surface-gate/edit inside the surface passes" 0 \
  run_surface "${repo}" "${base}" "${head_sha}"

fx="$(new_fixture)"
repo="${fx}/trusted"
init_git_fixture "${repo}"
base="$(git -C "${repo}" rev-parse HEAD)"
printf 'attack\n' >> "${repo}/README.md"
git -C "${repo}" add README.md
git -C "${repo}" commit -qm outside
head_sha="$(git -C "${repo}" rev-parse HEAD)"
assert_status_contains "surface-gate/edit outside the surface is refused" 1 \
  "outside the modifiable surface" run_surface "${repo}" "${base}" "${head_sha}"

fx="$(new_fixture)"
repo="${fx}/trusted"
init_git_fixture "${repo}"
base="$(git -C "${repo}" rev-parse HEAD)"
jq '.editablePaths += ["README.md"]' "${repo}/benchmark.json" > "${repo}/benchmark.tmp"
mv "${repo}/benchmark.tmp" "${repo}/benchmark.json"
printf 'attack\n' >> "${repo}/README.md"
git -C "${repo}" add .
git -C "${repo}" commit -qm self-widen
head_sha="$(git -C "${repo}" rev-parse HEAD)"
assert_status_contains "surface-gate/self-widened contract is refused" 1 \
  "outside the modifiable surface" run_surface "${repo}" "${base}" "${head_sha}"

for entry in benchd .gitmodules benchd.pin benchd-bin BENCHD .GITMODULES BENCHD.PIN BENCHD-BIN; do
  fx="$(new_fixture)"
  repo="${fx}/trusted"
  jq --arg entry "${entry}" '.editablePaths += [$entry]' \
    "${repo}/benchmark.json" > "${repo}/benchmark.tmp"
  mv "${repo}/benchmark.tmp" "${repo}/benchmark.json"
  init_git_fixture "${repo}"
  base="$(git -C "${repo}" rev-parse HEAD)"
  printf 'hostile\n' > "${repo}/${entry}"
  git -C "${repo}" add "${entry}"
  git -C "${repo}" commit -qm forbidden
  head_sha="$(git -C "${repo}" rev-parse HEAD)"
  assert_status_contains "surface-gate/forbidden write ${entry} is refused" 1 \
    "measurement-harness surface" run_surface "${repo}" "${base}" "${head_sha}"
done

EGRESS="${WORK}/egress"
mkdir -p "${EGRESS}"
printf '%s\n' 'name: ok' 'on: workflow_dispatch' 'permissions: {contents: read}' > "${EGRESS}/ok.yml"
assert_status "egress/benign workflow passes" 0 "${ROOT}/tools/ci-workflow-egress-scan.sh" "${EGRESS}"
printf '%s\n' 'jobs: {x: {env: {TOKEN: ${{ secrets.BAD }}}}}' > "${EGRESS}/secret.yml"
assert_status "egress/secret reference is refused" 1 "${ROOT}/tools/ci-workflow-egress-scan.sh" "${EGRESS}"
rm "${EGRESS}/secret.yml"
printf '%s\n' 'permissions:' '  id-token: write' > "${EGRESS}/oidc.yml"
assert_status "egress/OIDC write permission is refused" 1 "${ROOT}/tools/ci-workflow-egress-scan.sh" "${EGRESS}"
rm "${EGRESS}/oidc.yml"
printf '%s\n' 'steps:' '  - uses: actions/upload-artifact@0123456789012345678901234567890123456789' > "${EGRESS}/upload.yml"
assert_status "egress/artifact upload is refused" 1 "${ROOT}/tools/ci-workflow-egress-scan.sh" "${EGRESS}"

printf 'submission-security: %d distinct cases, %d failures\n' \
  "${assertions}" "${failures}"
if (( assertions < EXPECTED_MIN_ASSERTIONS )); then
  printf 'FAIL: suite size is %d; expected at least %d\n' \
    "${assertions}" "${EXPECTED_MIN_ASSERTIONS}" >&2
  exit 1
fi
(( failures == 0 ))
