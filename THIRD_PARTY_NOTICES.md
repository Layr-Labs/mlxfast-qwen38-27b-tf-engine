# Third-party notices

## Target model

`TensorFold/Qwen3.8-27B-MLX-4bit` is used at revision
`22d8d538154e0e5b6f8dcb1fcc74a60a0798104c`.

Source and license information:
https://huggingface.co/TensorFold/Qwen3.8-27B-MLX-4bit

## Drafter model

`z-lab/Qwen3.8-27B-DFlash2` is used at revision
`50307d4c4cde6860d4eee73e2547cd786fe8e8a4`.

Source and license information:
https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2

## TensorFold

`Vendor/TensorFold/` is a vendored engine subset of ashhart/TensorFold revision
`eea9dc9c9c8b459c0a03de7f6fe0c6f4835d69ab` (v1.1.6). The reference tree for
the subset is the commit named by `baseline_reference_commit` in
`fixtures/qwen3_8_27b_tf_mlx_v1_track.json`.

The subset does not include `.github/`, `.gitignore`, `AGENTS.md`,
`RELEASE-NOTES-1.0.0.md`, `SECURITY.md`, `ZIG-PREVIEW.md`, `app/`, `assets/`,
`docs/`, or `setup.py`. Without `app/`, Python helpers that import the
`tensorfold` package do not run from this tree.

### Changes to the vendored subset

Compared with upstream revision `eea9dc9c9c8b459c0a03de7f6fe0c6f4835d69ab`,
this repository changes 17 upstream files and adds 17 files. Every other
included file is unchanged from that revision. Each changed file starts with a
one-line notice that it was modified.

The changes add the Engine Protocol v1 bench worker and its build steps, a bench
session in the engine API and lanes, a DFlash audit journal, page and
pinned-memory release at phase drains in the Qwen CUDA engine, per-request draft
control in the native Qwen 27B lanes, and process-memory reporting on Linux.

Changed files (paths under `Vendor/TensorFold/`):

- `build.zig`
- `zig/src/core/engine_api.zig`
- `zig/src/core/lanes/engine.zig`
- `zig/src/core/lanes/fake.zig`
- `zig/src/core/lanes/logprob.zig`
- `zig/src/cuda/copy_source.zig`
- `zig/src/cuda/copy_source_spare_test.zig`
- `zig/src/families/qwen27/dflash/generation.zig`
- `zig/src/families/qwen27/dflash/runtime_frame.zig`
- `zig/src/families/qwen3_5/cuda_engine.zig`
- `zig/src/families/qwen3_5/cuda_native.zig`
- `zig/src/families/qwen3_5/cuda_pages.zig`
- `zig/src/families/qwen3_5/cuda_pages_lifetime_test.zig`
- `zig/src/native/cuda_drafted.zig`
- `zig/src/native/qwen27_lanes.zig`
- `zig/src/native/qwen27_slots.zig`
- `zig/src/server/process_memory.zig`

Added files (paths under `Vendor/TensorFold/`):

- `zig/build/bench_session_host.zig`
- `zig/build/dflash_audit_host.zig`
- `zig/src/bench/main.zig`
- `zig/src/bench/protocol.zig`
- `zig/src/bench/resident.zig`
- `zig/src/bench/resident_client.zig`
- `zig/src/bench/resident_test.zig`
- `zig/src/bench/self_test_test.zig`
- `zig/src/bench/worker.zig`
- `zig/src/bench/worker_dflash_test.zig`
- `zig/src/bench/worker_test.zig`
- `zig/src/core/bench_session.zig`
- `zig/src/core/bench_session_engine.zig`
- `zig/src/core/bench_session_test.zig`
- `zig/src/families/qwen27/dflash/audit.zig`
- `zig/src/families/qwen27/dflash/audit_test.zig`
- `zig/src/families/qwen3_5/cuda_native_phase_test.zig`

Upstream: https://github.com/ashhart/TensorFold

TensorFold is licensed under Apache License 2.0. See:

- `Vendor/TensorFold/LICENSE`
- `Vendor/TensorFold/NOTICE`
- `Vendor/TensorFold/THIRD_PARTY_NOTICES.md`
- `Vendor/TensorFold/LICENSES/`
