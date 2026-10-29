#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
mkdir -p "${WORK}/tree/tools" "${WORK}/tree/fixtures" "${WORK}/source"
cp "${ROOT}/tools/stage-baseline-workspace.sh" "${WORK}/tree/tools/"
git -C "${WORK}/source" init -q
git -C "${WORK}/source" config user.email test@example.com
git -C "${WORK}/source" config user.name test
printf base > "${WORK}/source/README"
git -C "${WORK}/source" add README
git -C "${WORK}/source" commit -qm base
commit="$(git -C "${WORK}/source" rev-parse HEAD)"
jq -n --arg c "${commit}" '{baseline_reference_commit:$c}' > "${WORK}/tree/fixtures/qwen3_8_27b_tf_mlx_v1_track.json"
plan="$("${WORK}/tree/tools/stage-baseline-workspace.sh" --source "${WORK}/source" --dest "${WORK}/dest" --dry-run)"
grep -q 'build bench-worker' <<<"${plan}"
jq '.baseline_reference_commit="0000000000000000000000000000000000000000"' "${WORK}/tree/fixtures/qwen3_8_27b_tf_mlx_v1_track.json" > "${WORK}/tree/fixtures/wrong"
mv "${WORK}/tree/fixtures/wrong" "${WORK}/tree/fixtures/qwen3_8_27b_tf_mlx_v1_track.json"
! "${WORK}/tree/tools/stage-baseline-workspace.sh" --source "${WORK}/source" --dest "${WORK}/dest" --dry-run >/dev/null 2>&1
printf 'test-reference-stamp.sh: PASS\n'
