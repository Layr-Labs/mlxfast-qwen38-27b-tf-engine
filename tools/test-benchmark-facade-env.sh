#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
mkdir -p "${WORK}/tools" "${WORK}/fixtures" "${WORK}/Vendor/TensorFold/zig-out/bin" "${WORK}/reference_weights/Qwen3.8-27B-MLX-4bit"
cp "${ROOT}/tools/benchmark.sh" "${ROOT}/tools/spec-declaration.sh" "${WORK}/tools/"
cp "${ROOT}/benchmark.json" "${ROOT}/mtp-head.manifest.json" "${WORK}/"
cp "${ROOT}/fixtures/qwen3_8_27b_tf_mlx_v1_track.json" "${WORK}/fixtures/"
cat > "${WORK}/Vendor/TensorFold/zig-out/bin/bench-worker" <<'MOCK'
#!/usr/bin/env bash
exit 0
MOCK
cat > "${WORK}/benchd" <<'MOCK'
#!/usr/bin/env bash
if [[ "$*" == "iterate --help" ]]; then printf '%s\n' '--candidate-spec'; exit 0; fi
printf '%s\n' "$*" > "${CAPTURE}"
MOCK
chmod +x "${WORK}/tools/"*.sh "${WORK}/benchd" "${WORK}/Vendor/TensorFold/zig-out/bin/bench-worker"
printf '{}\n' > "${WORK}/golden.json"
CAPTURE="${WORK}/serial.args" BENCHD="${WORK}/benchd" MLXFAST_ENGINE_BIN="${WORK}/Vendor/TensorFold/zig-out/bin/bench-worker" MLXFAST_WEIGHTS_PATH="${WORK}/reference_weights/Qwen3.8-27B-MLX-4bit" MLXFAST_CORRECTNESS_GOLDEN_PATH="${WORK}/golden.json" "${WORK}/tools/benchmark.sh"
grep -q -- '--mode local-iterate' "${WORK}/serial.args"
! grep -q -- '--candidate-spec' "${WORK}/serial.args"
jq '.spec.enabled=true | .spec.tokens=1' "${WORK}/mtp-head.manifest.json" > "${WORK}/enabled.json"
CAPTURE="${WORK}/dflash.args" BENCHD="${WORK}/benchd" MLXFAST_ENGINE_BIN="${WORK}/Vendor/TensorFold/zig-out/bin/bench-worker" MLXFAST_WEIGHTS_PATH="${WORK}/reference_weights/Qwen3.8-27B-MLX-4bit" MLXFAST_CORRECTNESS_GOLDEN_PATH="${WORK}/golden.json" "${WORK}/tools/benchmark.sh" --dflash-depth 1
grep -q -- '--candidate-spec {"mode":"dflash","dflash":{"depth":1}}' "${WORK}/dflash.args"
printf 'test-benchmark-facade-env.sh: PASS\n'
