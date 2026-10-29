# Qwen 3.8 27B TensorFold MLX track notes

## Result

The macOS track uses one TensorFold Zig worker.

## Pins

1. The target revision is 22d8d538154e0e5b6f8dcb1fcc74a60a0798104c.
2. The drafter revision is 50307d4c4cde6860d4eee73e2547cd786fe8e8a4.
3. The TensorFold revision is eea9dc9c9c8b459c0a03de7f6fe0c6f4835d69ab.
4. The benchd source revision is 8ce0ca7821e9c882f2ff1e1cbd1e2ce561040729.
5. The vendored engine subset reference is the fixture's `baseline_reference_commit`.
   Maintainers check a new base tree with
   `tools/check-vendored-subset.sh <reference>`. It is not a submission check.
6. The TensorFold version is v1.1.6.

## Build

Run the fixed build command in Vendor/TensorFold.

    ZIG_GLOBAL_CACHE_DIR="$PWD/../../.zig-global-cache" ZIG_LOCAL_CACHE_DIR="$PWD/../../.zig-cache" zig build bench-worker -Doptimize=ReleaseFast

The output is Vendor/TensorFold/zig-out/bin/bench-worker.

## Scoring

1. The reference run has drafts off.
2. The candidate can use DFlash depth 1.
3. Each official cohort has two pairs.
4. Both speed floors are 0.95.
5. Official scoring is enabled.
6. The fixture is the authority for golden and baseline pins.
