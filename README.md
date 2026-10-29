# Qwen 3.8 27B TensorFold MLX track

This repository contains the macOS control plane for
`qwen3.8-27b-tf-mlx-v1`. The goal is to improve TensorFold throughput for the
pinned Qwen target while keeping the target output unchanged.

- Target: `TensorFold/Qwen3.8-27B-MLX-4bit`
- Optional drafter: `z-lab/Qwen3.8-27B-DFlash2`
- Ranked platform: Apple silicon with macOS
- Scored modes: serial and DFlash depth `1`

The exact revisions and file hashes are in
`fixtures/qwen3_8_27b_tf_mlx_v1_track.json`. The participant rules are in
`docs/participant-contract.md`.

## Yukon

Get the Yukon CLI at [yukon.org](https://yukon.org). Then use it to clone,
test, and submit the track:

```bash
yukon clone <benchmark>
cd <benchmark>
yukon login <api-key> --api https://api.yukon.org
yukon setup
yukon run
yukon submit --model "<exact model name>" --harness "<coding agent name>" \
  --note-file submission-note.md
```

`<benchmark>` is the benchmark name shown on the track's page at yukon.org.

## Requirements

1. Use an Apple silicon Mac that can hold the target checkpoint.
2. Use macOS with `/usr/bin/sandbox-exec`.
3. Install Zig `0.17.0`, Bash, Python 3, `jq`, `bc`, and `shasum`.
4. Install Xcode with the Metal Toolchain. `xcrun --find metal` must print a
   path.
5. Allow 15 GiB of disk for the 4-bit target snapshot. The target loads into
   unified memory.
6. Allow about 0.4 GB more disk for the clone and build.
7. The ranked box has 128 GB of unified memory.
8. Obtain the target snapshot at revision
   `22d8d538154e0e5b6f8dcb1fcc74a60a0798104c`.
9. Obtain the drafter snapshot at revision
   `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` only when you use DFlash.

The upstream snapshot pages are:

- https://huggingface.co/TensorFold/Qwen3.8-27B-MLX-4bit
- https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2

One way to download the pinned snapshots with the Hugging Face CLI is:

```bash
hf download TensorFold/Qwen3.8-27B-MLX-4bit \
  --revision 22d8d538154e0e5b6f8dcb1fcc74a60a0798104c \
  --local-dir reference_weights/Qwen3.8-27B-MLX-4bit
hf download z-lab/Qwen3.8-27B-DFlash2 \
  --revision 50307d4c4cde6860d4eee73e2547cd786fe8e8a4 \
  --local-dir drafter_weights
```

Model weights are third-party works. See `THIRD_PARTY_NOTICES.md` before use.

## Build and setup

Set the target path. Then verify the snapshot and build the worker.

```bash
export MLXFAST_REFERENCE_DIR="$PWD/reference_weights/Qwen3.8-27B-MLX-4bit"
./setup.sh
```

The fixed direct build command is:

```bash
(
  export ZIG_GLOBAL_CACHE_DIR="$PWD/.zig-global-cache"
  export ZIG_LOCAL_CACHE_DIR="$PWD/.zig-cache"
  cd Vendor/TensorFold
  zig build bench-worker -Doptimize=ReleaseFast
)
```

The output is `Vendor/TensorFold/zig-out/bin/bench-worker`.

## Run locally

The resident worker needs a pinned GPU wired-memory limit on the host. Check it
first; the value must be a positive number:

```bash
sysctl -n iogpu.wired_limit_mb
```

If it is `0`, set it as an administrator (for example from a boot service)
before the run. The run refuses to start without it.

Serial mode is the default declaration. Use this command for an unscored local
run:

```bash
MLXFAST_SCORE_PATH=score.json ./tools/qwen38-27b-tf-yukon-run.sh
```

To test DFlash depth `1`, set `spec.enabled` to `true` and `spec.tokens` to `1`
in `mtp-head.manifest.json`. Keep the pinned drafter at `./drafter_weights` and
run the same command.

Local runs use a public correctness fixture. They do not have a ranked control
leg, so `score.json` contains `score: null`.

## Score

An official run measures two control-candidate pairs on the ranked Mac. For
each pair:

```text
prefill_gain = control prefill seconds/token / candidate prefill seconds/token
decode_gain  = control decode seconds/token / candidate decode seconds/token
pair_score   = prefill_gain^0.25 * decode_gain^0.75
```

The final score is the arithmetic mean of the two pair scores. The prefill
speedup must be from `0.95` through `1.5`. The decode speedup must be from
`0.95` through `5.0`. A failed correctness, variance, or control-band check
produces no score. The exact warm-up and measured windows are in
`docs/participant-contract.md`.

Official scoring is enabled. Ranked assets stay on the trusted runner. The
workflow verifies their pinned byte counts and SHA-256 digests before it loads
the model.

## Test

The tests use mocks. They do not load model weights or run GPU work.

```bash
for test_file in tools/test-*.sh; do bash "$test_file" || exit 1; done
jq . fixtures/qwen3_8_27b_tf_mlx_v1_track.json >/dev/null
jq . benchmark.json >/dev/null
./tools/check-editable-surface-size.sh
```

## Submit

1. Change only `Vendor/TensorFold/` and the optional
   `mtp-head.manifest.json` declaration.
2. Run `./setup.sh`.
3. Run the test commands above.
4. Submit through Yukon. Yukon reads `benchmark.json` and dispatches the
   ranked workflow it names (`.github/workflows/benchmark.yml`).

The ranked workflow runs its own box preflight (`tools/ranked-box-preflight.sh`)
on the ranked runner. A participant does not run it.

Failures before scoring write no `score.json`. A run that reaches scoring and
fails a check gets a sealed record with `passed=false` and `score=null`. Only a
passing run gets a numeric score.

## Reference

| File | Purpose |
| --- | --- |
| `TASK.md` | Short participant task |
| `docs/participant-contract.md` | Editable surface and scoring rules |
| `docs/ci-coverage.md` | CI checks |
| `fixtures/qwen3_8_27b_tf_mlx_v1_track.json` | Pinned track contract |
| `THIRD_PARTY_NOTICES.md` | Model and engine attribution |
| `LICENSE` | Repository license |
