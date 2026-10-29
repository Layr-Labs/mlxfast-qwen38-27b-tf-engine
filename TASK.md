# Participant task

Improve the throughput of the pinned Qwen 3.8 27B target on Apple silicon.
Keep the target output unchanged.

## Permitted changes

1. Edit `Vendor/TensorFold/`.
2. Optionally edit `mtp-head.manifest.json`.
3. Keep the target and drafter checkpoints unchanged.
4. Use serial mode or DFlash depth `1`.

## Commands

```bash
export MLXFAST_REFERENCE_DIR=/path/to/Qwen3.8-27B-MLX-4bit
./setup.sh
MLXFAST_SCORE_PATH=score.json ./tools/qwen38-27b-tf-yukon-run.sh
for test_file in tools/test-*.sh; do bash "$test_file" || exit 1; done
```

The local result is not ranked. Yukon dispatches the ranked workflow. The
workflow measures two control-candidate pairs and writes a score only when all
correctness and performance guards pass.

The authoritative files are `benchmark.json` and
`fixtures/qwen3_8_27b_tf_mlx_v1_track.json`. See
`docs/participant-contract.md` for all restrictions and scoring details.
