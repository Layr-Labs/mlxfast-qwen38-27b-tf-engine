#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
TREE="${WORK}/tree"
mkdir -p "${TREE}/tools" "${TREE}/fixtures" "${TREE}/Vendor/TensorFold/zig-out/bin" "${TREE}/reference_weights/Qwen3.8-27B-MLX-4bit" "${WORK}/private/goldens" "${WORK}/outside"
cp "${ROOT}/tools/qwen38-27b-tf-measure-and-score.sh" "${ROOT}/tools/spec-declaration.sh" "${TREE}/tools/"
cp "${ROOT}/benchmark.json" "${ROOT}/mtp-head.manifest.json" "${TREE}/"
cat > "${TREE}/Vendor/TensorFold/zig-out/bin/bench-worker" <<'MOCK'
#!/usr/bin/env bash
[[ "$1" == self-test ]] && printf 'engine-protocol-v1 self-test: ok\n'
[[ "${1:-}" == iterate && "${2:-}" == --help ]] && printf '%s\n' '    --warmup-golden PATH'
exit 0
MOCK
chmod +x "${TREE}/tools/"*.sh "${TREE}/Vendor/TensorFold/zig-out/bin/bench-worker"
BASE="${WORK}/base"
mkdir -p "${BASE}/Vendor/TensorFold/zig-out/bin"
git -C "${BASE}" init -q
git -C "${BASE}" config user.email test@example.com
git -C "${BASE}" config user.name test
printf base > "${BASE}/README"
git -C "${BASE}" add README
git -C "${BASE}" commit -qm base
commit="$(git -C "${BASE}" rev-parse HEAD)"
printf '#!/usr/bin/env bash\nexit 0\n' > "${BASE}/Vendor/TensorFold/zig-out/bin/bench-worker"
chmod +x "${BASE}/Vendor/TensorFold/zig-out/bin/bench-worker"
for file in pool-01.golden.json pool-02.golden.json pool-01.dflash1.golden.json pool-02.dflash1.golden.json warmup-01.golden.json; do printf '%s\n' "${file}" > "${WORK}/private/goldens/${file}"; done
pin() { local f="$1"; jq -n --arg p "objects/$(basename "${f}")" --arg s "$(shasum -a 256 "${f}" | awk '{print $1}')" --argjson b "$(wc -c < "${f}" | tr -d ' ')" '{r2_path:$p,sha256:$s,bytes:$b}'; }
p1="$(pin "${WORK}/private/goldens/pool-01.golden.json")"
p2="$(pin "${WORK}/private/goldens/pool-02.golden.json")"
d1="$(pin "${WORK}/private/goldens/pool-01.dflash1.golden.json")"
d2="$(pin "${WORK}/private/goldens/pool-02.dflash1.golden.json")"
wu="$(pin "${WORK}/private/goldens/warmup-01.golden.json")"
jq --arg c "${commit}" --argjson p1 "${p1}" --argjson p2 "${p2}" --argjson d1 "${d1}" --argjson d2 "${d2}" --argjson wu "${wu}" '
  .official_scoring_enabled=true | .baseline_reference_commit=$c |
  .hidden_correctness_golden={sha256:"1111111111111111111111111111111111111111111111111111111111111111",bytes:1} |
  .timed_prompt_pool=[$p1,$p2] | .warmup_golden=$wu |
  .live_golden_speculative={dflash1:{"pool-01":$d1,"pool-02":$d2}}
' "${ROOT}/fixtures/qwen3_8_27b_tf_mlx_v1_track.json" > "${TREE}/fixtures/qwen3_8_27b_tf_mlx_v1_track.json"
jq -n --arg c "${commit}" '{track_id:"qwen3.8-27b-tf-mlx-v1",baseline_reference_commit:$c}' > "${WORK}/cal.json"
base=(env BENCHD="${TREE}/Vendor/TensorFold/zig-out/bin/bench-worker" MLXFAST_ENGINE_BIN="${TREE}/Vendor/TensorFold/zig-out/bin/bench-worker" MLXFAST_WEIGHTS_PATH="${TREE}/reference_weights/Qwen3.8-27B-MLX-4bit" MLXFAST_QWEN38_GOLDEN_DIR="${WORK}/private/goldens" MLXFAST_PRIVATE_DIR="${WORK}/private" MLXFAST_BASELINE_WORKSPACE="${BASE}" MLXFAST_BASELINE_CALIBRATION="${WORK}/cal.json")
"${base[@]}" "${TREE}/tools/qwen38-27b-tf-measure-and-score.sh" --preflight-only >/dev/null
! "${base[@]}" MLXFAST_PRIVATE_DIR="${WORK}/outside" "${TREE}/tools/qwen38-27b-tf-measure-and-score.sh" --preflight-only >/dev/null 2>&1
mv "${WORK}/private/goldens/pool-01.golden.json" "${WORK}/outside/pool-01.golden.json"
ln -s "${WORK}/outside/pool-01.golden.json" "${WORK}/private/goldens/pool-01.golden.json"
! "${base[@]}" "${TREE}/tools/qwen38-27b-tf-measure-and-score.sh" --preflight-only >/dev/null 2>&1
printf 'test-private-golden-dir.sh: PASS\n'
