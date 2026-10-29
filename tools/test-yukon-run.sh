#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
mkdir -p "${WORK}/tree/tools" "${WORK}/target"
cp "${ROOT}/tools/qwen38-27b-tf-yukon-run.sh" "${ROOT}/tools/spec-declaration.sh" "${WORK}/tree/tools/"
cp "${ROOT}/mtp-head.manifest.json" "${WORK}/tree/"
cat > "${WORK}/tree/tools/qwen38-27b-tf-measure-and-score.sh" <<'MOCK'
#!/usr/bin/env bash
printf 'ranked %s\n' "$*" > "${CAPTURE}"
MOCK
cat > "${WORK}/tree/benchmark.sh" <<'MOCK'
#!/usr/bin/env bash
[[ "${BENCH_WORKER_RESIDENT_SOCKET:-}" == "${MOCK_SOCKET}" ]] || exit 44
printf 'local %s %s\n' "${MLXFAST_WEIGHTS_PATH}" "${MLXFAST_CORRECTNESS_GOLDEN_PATH}" > "${CAPTURE}"
MOCK
cat > "${WORK}/tree/tools/resident-up.sh" <<'MOCK'
#!/usr/bin/env bash
[[ "$1" == --weights && -d "$2" ]] || exit 41
shift 2
[[ "$1" == -- ]] || exit 42
shift
export BENCH_WORKER_RESIDENT_SOCKET="${MOCK_SOCKET}"
exec "$@"
MOCK
chmod +x "${WORK}/tree/tools/"*.sh "${WORK}/tree/benchmark.sh"
printf '{}\n' > "${WORK}/golden.json"
printf '{}\n' > "${WORK}/target/config.json"
CAPTURE="${WORK}/ranked" GITHUB_ACTIONS=true "${WORK}/tree/tools/qwen38-27b-tf-yukon-run.sh" --preflight-only
grep -q 'ranked --preflight-only' "${WORK}/ranked"
# The local cases must not inherit the CI runner's GITHUB_ACTIONS=true (it selects the ranked path).
unset GITHUB_ACTIONS MLXFAST_QWEN38_GOLDEN_DIR MLXFAST_BASELINE_WORKSPACE MLXFAST_BASELINE_CALIBRATION
CAPTURE="${WORK}/local" MOCK_SOCKET="${WORK}/resident.sock" QWEN38_YUKON_IN_GPU_LOCK=1 \
  MLXFAST_REFERENCE_DIR="${WORK}/target" "${WORK}/tree/tools/qwen38-27b-tf-yukon-run.sh"
grep -q '^local ' "${WORK}/local"
! CAPTURE="${WORK}/bad" QWEN38_YUKON_IN_GPU_LOCK=1 MLXFAST_REFERENCE_DIR="${WORK}/missing" "${WORK}/tree/tools/qwen38-27b-tf-yukon-run.sh" >/dev/null 2>&1
! CAPTURE="${WORK}/bad" QWEN38_YUKON_IN_GPU_LOCK=1 MLXFAST_REFERENCE_DIR="${WORK}/target" "${WORK}/tree/tools/qwen38-27b-tf-yukon-run.sh" extra >/dev/null 2>&1
printf 'test-yukon-run.sh: PASS\n'
