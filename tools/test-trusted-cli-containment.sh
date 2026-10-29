#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
BASE="${WORK}/base"
mkdir -p "${BASE}/Vendor/TensorFold/zig-out/bin" "${WORK}/target" "${WORK}/goldens"
git -C "${BASE}" init -q
git -C "${BASE}" config user.email test@example.com
git -C "${BASE}" config user.name test
printf base > "${BASE}/README"
git -C "${BASE}" add README
git -C "${BASE}" commit -qm base
commit="$(git -C "${BASE}" rev-parse HEAD)"
printf '#!/usr/bin/env bash\nexit 0\n' > "${WORK}/outside-worker"
chmod +x "${WORK}/outside-worker"
ln -s "${WORK}/outside-worker" "${BASE}/Vendor/TensorFold/zig-out/bin/bench-worker"
printf '{}\n' > "${WORK}/goldens/pool-01.golden.json"
jq --arg c "${commit}" '.baseline_reference_commit=$c | .live_goldens=["pool-01"]' "${ROOT}/fixtures/qwen3_8_27b_tf_mlx_v1_track.json" > "${WORK}/fixture.json"
resolved="$(python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "${BASE}/Vendor/TensorFold/zig-out/bin/bench-worker")"
case "${resolved}" in "${BASE}"/*) exit 1;; esac
grep -q 'baseline worker resolves outside the baseline workspace' "${ROOT}/tools/calibrate-box.sh"
grep -q 'baseline worker resolves outside the baseline workspace' "${ROOT}/tools/qwen38-27b-tf-measure-and-score.sh"
printf 'test-trusted-cli-containment.sh: PASS\n'
