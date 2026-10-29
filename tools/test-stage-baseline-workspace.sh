#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
TREE="${WORK}/tree"
SOURCE="${WORK}/source"
mkdir -p "${TREE}/tools" "${TREE}/fixtures" "${SOURCE}/Vendor/TensorFold"
mkdir -p "${WORK}/hub/models--fake/snapshots/revision" "${WORK}/hub/models--fake/blobs"
printf '{}\n' > "${WORK}/hub/models--fake/blobs/config"
printf 'fake shard\n' > "${WORK}/hub/models--fake/blobs/shard"
ln -s ../../blobs/config "${WORK}/hub/models--fake/snapshots/revision/config.json"
ln -s ../../blobs/shard "${WORK}/hub/models--fake/snapshots/revision/model.safetensors"
REFERENCE="${WORK}/hub/models--fake/snapshots/revision"
cp "${ROOT}/tools/stage-baseline-workspace.sh" "${TREE}/tools/"
git -C "${SOURCE}" init -q
git -C "${SOURCE}" config user.email test@example.com
git -C "${SOURCE}" config user.name test
printf source > "${SOURCE}/Vendor/TensorFold/source.zig"
git -C "${SOURCE}" add .
git -C "${SOURCE}" commit -qm source
commit="$(git -C "${SOURCE}" rev-parse HEAD)"
jq -n --arg c "${commit}" '{baseline_reference_commit:$c}' > "${TREE}/fixtures/qwen3_8_27b_tf_mlx_v1_track.json"
(cd "${WORK}/hub/models--fake/blobs" && shasum -a 256 config shard) | \
  sed -e 's/  config$/  config.json/' -e 's/  shard$/  model.safetensors/' \
  > "${TREE}/fixtures/reference_qwen3_8_27b_4bit.sha256"
plan="$(MLXFAST_REFERENCE_DIR="${REFERENCE}" "${TREE}/tools/stage-baseline-workspace.sh" --source "${SOURCE}" --dest "${WORK}/dest" --dry-run)"
[[ "${plan}" == *"git clone --no-local"* && "${plan}" == *"build bench-worker -Doptimize=ReleaseFast"* ]]
[[ "${plan}" == *"clone-copy"* ]]
[[ "${plan}" == *"mkdir-readonly"*"drafter_weights"* ]]
[[ ! -e "${WORK}/dest" ]]
cat > "${WORK}/zig" <<'MOCK'
#!/usr/bin/env bash
[[ "$1" == build && "$2" == bench-worker ]] || exit 2
mkdir -p zig-out/bin
printf '#!/usr/bin/env bash\nexit 0\n' > zig-out/bin/bench-worker
chmod +x zig-out/bin/bench-worker
MOCK
chmod +x "${WORK}/zig"
MLXFAST_REFERENCE_DIR="${REFERENCE}" ZIG="${WORK}/zig" "${TREE}/tools/stage-baseline-workspace.sh" --source "${SOURCE}" --dest "${WORK}/dest" >/dev/null
[[ -x "${WORK}/dest/Vendor/TensorFold/zig-out/bin/bench-worker" ]]
[[ -f "${WORK}/dest/reference_weights/Qwen3.8-27B-MLX-4bit/config.json" ]]
[[ ! -L "${WORK}/dest/reference_weights/Qwen3.8-27B-MLX-4bit" ]]
[[ ! -L "${WORK}/dest/reference_weights/Qwen3.8-27B-MLX-4bit/config.json" ]]
[[ ! -L "${WORK}/dest/reference_weights/Qwen3.8-27B-MLX-4bit/model.safetensors" ]]
[[ -d "${WORK}/dest/drafter_weights" ]]
[[ -z "$(find "${WORK}/dest/drafter_weights" -mindepth 1 -print -quit)" ]]
[[ "$(stat -f '%Lp' "${WORK}/dest/drafter_weights")" == 555 ]]
(cd "${WORK}/dest/reference_weights/Qwen3.8-27B-MLX-4bit" && \
  shasum -a 256 -c "${TREE}/fixtures/reference_qwen3_8_27b_4bit.sha256") >/dev/null
grep -q 'cp -cRL' "${ROOT}/.github/workflows/benchmark.yml"
grep -q 'find "${target_dir}" -type l' "${ROOT}/.github/workflows/benchmark.yml"
grep -q 'shasum -a 256 -c' "${ROOT}/.github/workflows/benchmark.yml"
baseline_commit="$(jq -r '.baseline_reference_commit' "${ROOT}/fixtures/qwen3_8_27b_tf_mlx_v1_track.json")"
reference_launcher="$(git -C "${ROOT}" show "${baseline_commit}:tools/resident-up.sh")"
[[ "${reference_launcher}" == *'"${MODE}" == "boot" && "${SPEC}" == "dflash"'* ]]
[[ "${reference_launcher}" == *'RESIDENT_COMMAND+=(--drafter "${DRAFTER_DIR}")'* ]]
! "${TREE}/tools/stage-baseline-workspace.sh" --source "${SOURCE}" --dest "${WORK}/dest" --dry-run >/dev/null 2>&1
printf 'test-stage-baseline-workspace.sh: PASS\n'
