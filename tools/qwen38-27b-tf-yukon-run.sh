#!/usr/bin/env bash
#
# qwen38-27b-tf-yukon-run.sh -- benchmark.json's benchmarkCommand.
#
# `yukon run` executes benchmarkCommand on the participant's machine. The
# ranked script, tools/qwen38-27b-tf-measure-and-score.sh, needs inputs
# that only a ranked box has: the staged goldens, the reference workspace and
# the box calibration. Off a ranked box it can only refuse. This script picks
# the path:
#
#   RANKED  GITHUB_ACTIONS is "true", or one or more of
#           MLXFAST_QWEN38_GOLDEN_DIR, MLXFAST_BASELINE_WORKSPACE and
#           MLXFAST_BASELINE_CALIBRATION is set. This script then replaces
#           itself with the ranked script and passes all arguments. The ranked
#           behavior does not change. A partial ranked environment goes to the
#           ranked script, which refuses by name. It never falls back.
#   LOCAL   none of the above. This script uses the pinned target snapshot and
#           runs ./benchmark.sh --local-submit at the depth that
#           mtp-head.manifest.json declares (tools/spec-declaration.sh).
#
# The ranked workflow (.github/workflows/benchmark.yml) calls the ranked script
# directly. It does not call this script.
#
# A LOCAL RESULT HAS NO SCORE. benchd seals a local run with "score": null,
# because a local run has no serial-control leg on this box. This script does
# not write a score of its own. Yukon reads the sealed file at scorePath.
#
# Env (LOCAL path only):
#   MLXFAST_SCORE_PATH     where benchd seals the result (benchmark.json sets
#                          score.json).
#   MLXFAST_REFERENCE_DIR  the existing pinned target snapshot.
#                          Default: reference_weights/Qwen3.8-27B-MLX-4bit.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ "${GITHUB_ACTIONS:-}" == "true" \
  || -n "${MLXFAST_QWEN38_GOLDEN_DIR:-}" \
  || -n "${MLXFAST_BASELINE_WORKSPACE:-}" \
  || -n "${MLXFAST_BASELINE_CALIBRATION:-}" ]]; then
  exec "${ROOT}/tools/qwen38-27b-tf-measure-and-score.sh" "$@"
fi

if [[ $# -ne 0 ]]; then
  echo "qwen38-27b-tf-yukon-run.sh: the local path takes no arguments (got: $*)." >&2
  exit 2
fi

echo "qwen38-27b-tf-yukon-run.sh: no ranked inputs. This is a LOCAL run. It has no score; the ranked runner gives the score." >&2

if [[ "${QWEN38_YUKON_IN_GPU_LOCK:-0}" != "1" ]]; then
  lock_path="${RESIDENT_UP_LOCK_PATH:-/tmp/mtplx-gpu-exclusive.lock}"
  lock_timeout="${MLXFAST_GPU_LOCK_TIMEOUT_S:-1800}"
  export RESIDENT_UP_LOCK_PATH="${lock_path}"
  export QWEN38_YUKON_IN_GPU_LOCK=1
  exec python3 -c '
import fcntl, os, signal, sys
fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o666)
os.set_inheritable(fd, True)
def timeout(*_):
    sys.stderr.write("qwen38-27b-tf-yukon-run.sh: GPU lock wait timed out; nothing was loaded\n")
    raise SystemExit(1)
signal.signal(signal.SIGALRM, timeout)
signal.alarm(int(sys.argv[2]))
fcntl.flock(fd, fcntl.LOCK_EX)
signal.alarm(0)
os.execv(sys.argv[3], sys.argv[3:])
' "${lock_path}" "${lock_timeout}" "${BASH}" "${BASH_SOURCE[0]}"
fi

reference="${MLXFAST_REFERENCE_DIR:-reference_weights/Qwen3.8-27B-MLX-4bit}"
[[ -f "${reference}/config.json" ]] || {
  echo "qwen38-27b-tf-yukon-run.sh: target snapshot is missing at ${reference}." >&2
  exit 1
}

depth="$(tools/spec-declaration.sh draft-len)"
depth_args=()
resident_args=(--weights "${reference}")
if [[ "${depth}" != "0" ]]; then
  depth_args=(--dflash-depth "${depth}")
  [[ -d "${ROOT}/drafter_weights" ]] || {
    echo "qwen38-27b-tf-yukon-run.sh: drafter snapshot is missing at ${ROOT}/drafter_weights." >&2
    exit 1
  }
  resident_args+=(--drafter "${ROOT}/drafter_weights")
fi
echo "qwen38-27b-tf-yukon-run.sh: declared DFlash depth ${depth}." >&2

exec tools/resident-up.sh "${resident_args[@]}" -- \
  env MLXFAST_ENGINE_BIN="${MLXFAST_ENGINE_BIN:-Vendor/TensorFold/zig-out/bin/bench-worker}" \
  MLXFAST_WEIGHTS_PATH="${MLXFAST_WEIGHTS_PATH:-${reference}}" \
  MLXFAST_CORRECTNESS_GOLDEN_PATH=correctness_prompts/public_longcopy_gate_english_1024_1024.json \
  ./benchmark.sh --local-submit ${depth_args[@]+"${depth_args[@]}"}
