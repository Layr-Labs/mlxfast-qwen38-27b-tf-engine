#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
WORK="$(mktemp -d)"
trap 'rm -rf -- "${WORK}"' EXIT
fail() { printf 'test-dflash-depth-guard.sh: FAIL: %s\n' "$*" >&2; exit 1; }

TREE="${WORK}/tree"
BASELINE="${WORK}/baseline"
GOLDENS="${WORK}/goldens"
mkdir -p "${TREE}/tools" "${TREE}/fixtures" "${TREE}/drafter_weights" \
  "${TREE}/Vendor/TensorFold/zig-out/bin" \
  "${TREE}/reference_weights/Qwen3.8-27B-MLX-4bit" \
  "${BASELINE}/drafter_weights" "${BASELINE}/Vendor/TensorFold/zig-out/bin" \
  "${BASELINE}/reference_weights/Qwen3.8-27B-MLX-4bit" "${GOLDENS}"
cp "${ROOT}/tools/qwen38-27b-tf-measure-and-score.sh" \
  "${ROOT}/tools/spec-declaration.sh" "${ROOT}/tools/orphan-guard.sh" "${TREE}/tools/"
chmod +x "${TREE}/tools/"*.sh
printf '#!/usr/bin/env bash\nexit 0\n' > "${TREE}/Vendor/TensorFold/zig-out/bin/bench-worker"
cp "${TREE}/Vendor/TensorFold/zig-out/bin/bench-worker" \
  "${BASELINE}/Vendor/TensorFold/zig-out/bin/bench-worker"
chmod +x "${TREE}/Vendor/TensorFold/zig-out/bin/bench-worker" \
  "${BASELINE}/Vendor/TensorFold/zig-out/bin/bench-worker"
printf '{"trackId":"test-track"}\n' > "${TREE}/benchmark.json"
printf '{}\n' > "${GOLDENS}/one.golden.json"
printf '{}\n' > "${GOLDENS}/one-dflash1.golden.json"
golden_sha="$(shasum -a 256 "${GOLDENS}/one.golden.json" | awk '{print $1}')"
golden_bytes="$(wc -c < "${GOLDENS}/one.golden.json" | tr -d '[:space:]')"
jq -n --arg sha "${golden_sha}" --argjson bytes "${golden_bytes}" '{
  official_scoring_enabled: true,
  live_goldens: ["one"],
  timed_prompt_pool: [{r2_path:"pool/one.golden.json", sha256:$sha, bytes:$bytes}],
  live_golden_speculative: {dflash1: {one: {
    r2_path:"pool/one-dflash1.golden.json", sha256:$sha, bytes:$bytes
  }}},
  dflash_head: {permitted_draft_depths:[1]}
}' > "${TREE}/fixtures/qwen3_8_27b_tf_mlx_v1_track.json"
printf '{}\n' > "${WORK}/calibration.json"

MOCK_BENCHD="${WORK}/benchd"
cat > "${MOCK_BENCHD}" <<'MOCK'
#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == iterate && "${2:-}" == --help ]]; then
  printf '%s\n' '--candidate-spec --control-golden'
  exit 0
fi
[[ "${1:-}" == iterate ]] || exit 2
score=""
shift
while [[ $# -gt 0 ]]; do
  if [[ "$1" == --score-path ]]; then score="$2"; shift 2; else shift; fi
done
printf '{"passed":true,"score":1,"metrics":{}}\n' > "${score}"
printf 'sealed\n' > "${score}.sha256"
printf '{}\n' > "$(dirname "${score}")/benchmark-integrity.json"
if [[ "${MOCK_ACCEPTANCE_LENGTHS}" == missing ]]; then
  printf '{"metrics":{}}\n' > "${BENCHD_PRIVATE_AUDIT_PATH}"
else
  printf '{"metrics":{"acceptance_lengths":%s}}\n' \
    "${MOCK_ACCEPTANCE_LENGTHS}" > "${BENCHD_PRIVATE_AUDIT_PATH}"
fi
MOCK
chmod +x "${MOCK_BENCHD}"

run_case() {
  local name="$1" depth="$2" lengths="$3" want="$4"
  local case_dir="${TREE}/${name}" score="${TREE}/${name}/score.json" out rc count
  mkdir -p "${case_dir}"
  if [[ "${depth}" == 1 ]]; then
    printf '{"spec":{"enabled":true,"tokens":1}}\n' > "${TREE}/mtp-head.manifest.json"
  else
    printf '{"spec":{"enabled":false,"tokens":0}}\n' > "${TREE}/mtp-head.manifest.json"
  fi
  set +e
  out="$(cd "${TREE}" && \
    HOME="${WORK}" BENCHD="${MOCK_BENCHD}" MLXFAST_GPU_WINDOW_HELD=1 \
    MLXFAST_SCORE_PATH="${score}" MLXFAST_QWEN38_GOLDEN_DIR="${GOLDENS}" \
    MLXFAST_PRIVATE_DIR="${GOLDENS}" MLXFAST_BASELINE_WORKSPACE="${BASELINE}" \
    MLXFAST_BASELINE_CALIBRATION="${WORK}/calibration.json" \
    MLXFAST_RESIDENT_LOG_ROOT="${WORK}/resident-logs" \
    MOCK_ACCEPTANCE_LENGTHS="${lengths}" \
    ./tools/qwen38-27b-tf-measure-and-score.sh 2>&1)"
  rc=$?
  set -e
  count="$(printf '%s\n' "${out}" | grep -c 'dflash-depth' || true)"
  if [[ "${want}" == pass ]]; then
    [[ ${rc} -eq 0 && -f "${score}" && -f "${score}.sha256" \
       && -f "${case_dir}/benchmark-integrity.json" && ${count} -eq 0 ]] \
      || fail "${name} did not pass"
  else
    [[ ${rc} -ne 0 && ${count} -eq 1 ]] || fail "${name} did not refuse dflash-depth once"
    [[ ! -e "${score}" && ! -e "${score}.sha256" \
       && ! -e "${case_dir}/benchmark-integrity.json" ]] \
      || fail "${name} left a sealed score file"
    [[ "$(<"${score}.refusal-reason")" == \
      'dflash-depth: the reported depth is not bounded by the committed tokens per step' ]] \
      || fail "${name} did not write the trusted refusal reason"
  fi
}

run_case depth1_three 1 '[3,1]' refuse
run_case depth1_two 1 '[2,2]' pass
run_case serial_two 0 '[2]' refuse
run_case depth1_missing 1 missing refuse

printf 'test-dflash-depth-guard.sh: PASS\n'
