# Agent guide

## Result

This repository is the macOS control plane for `qwen3.8-27b-tf-mlx-v1`.

## Rules

1. Read `README.md` first.
2. Edit engine code only under `Vendor/TensorFold/`.
3. Edit `mtp-head.manifest.json` only to declare serial mode or DFlash depth
   `1`.
4. Keep the target checkpoint unchanged.
5. Keep the drafter checkpoint unchanged.
6. Keep the legacy declaration filename `mtp-head.manifest.json`.
7. Do not load model weights in tests.
8. Do not run GPU work in tests.
9. Use mocks for worker and resident tests.

## Build

Run the fixed build command from the repository root:

```bash
(
  export ZIG_GLOBAL_CACHE_DIR="$PWD/.zig-global-cache"
  export ZIG_LOCAL_CACHE_DIR="$PWD/.zig-cache"
  cd Vendor/TensorFold
  zig build bench-worker -Doptimize=ReleaseFast
)
```

## Track facts

- Target: `TensorFold/Qwen3.8-27B-MLX-4bit` at `22d8d538154e0e5b6f8dcb1fcc74a60a0798104c`.
- Drafter: `z-lab/Qwen3.8-27B-DFlash2` at `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`.
- TensorFold base: `eea9dc9c9c8b459c0a03de7f6fe0c6f4835d69ab`.
- Bench source: `8ce0ca7821e9c882f2ff1e1cbd1e2ce561040729`.
- Permitted DFlash depth: `1`.
- Official pairs: `2`.
- Decode floor: `0.95`.
- Prefill floor: `0.95`.

## Checks

```bash
for test_file in tools/test-*.sh; do bash "$test_file" || exit 1; done
jq . fixtures/qwen3_8_27b_tf_mlx_v1_track.json >/dev/null
jq . benchmark.json >/dev/null
```
