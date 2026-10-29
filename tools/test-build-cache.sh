#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
mkdir -p "${WORK}/repo/tools" "${WORK}/repo/Vendor/TensorFold"
cp "${ROOT}/tools/build-cache.sh" "${WORK}/repo/tools/"
printf '#!/usr/bin/env bash\nexit 0\n' > "${WORK}/repo/tools/stage-bench-worker.sh"
printf 'source\n' > "${WORK}/repo/Vendor/TensorFold/source.zig"
chmod +x "${WORK}/repo/tools/"*.sh
git -C "${WORK}/repo" init -q
git -C "${WORK}/repo" config user.email test@example.com
git -C "${WORK}/repo" config user.name test
git -C "${WORK}/repo" add .
git -C "${WORK}/repo" commit -qm seed
export MLXFAST_BUILD_CACHE_DIR="${WORK}/cache"
k1="$(cd "${WORK}/repo" && tools/build-cache.sh key)"
printf 'changed\n' >> "${WORK}/repo/Vendor/TensorFold/source.zig"
k2="$(cd "${WORK}/repo" && tools/build-cache.sh key)"
[[ "${k1}" != "${k2}" ]]
git -C "${WORK}/repo" checkout -q -- Vendor/TensorFold/source.zig
mkdir -p "${WORK}/repo/Vendor/TensorFold/zig-out/bin"
printf '#!/usr/bin/env bash\nexit 0\n' > "${WORK}/repo/Vendor/TensorFold/zig-out/bin/bench-worker"
chmod +x "${WORK}/repo/Vendor/TensorFold/zig-out/bin/bench-worker"
(cd "${WORK}/repo" && tools/build-cache.sh save >/dev/null)
rm "${WORK}/repo/Vendor/TensorFold/zig-out/bin/bench-worker"
(cd "${WORK}/repo" && tools/build-cache.sh restore >/dev/null)
[[ -x "${WORK}/repo/Vendor/TensorFold/zig-out/bin/bench-worker" ]]
printf 'tampered\n' >> "${MLXFAST_BUILD_CACHE_DIR}/${k1}/Vendor/TensorFold/zig-out/bin/bench-worker"
rm "${WORK}/repo/Vendor/TensorFold/zig-out/bin/bench-worker"
! (cd "${WORK}/repo" && tools/build-cache.sh restore >/dev/null 2>&1)
printf 'test-build-cache.sh: PASS\n'
