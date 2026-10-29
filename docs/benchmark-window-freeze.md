# Benchmark window freeze

## Result

One official run uses one candidate commit and one baseline commit.

1. Stage the baseline with tools/stage-baseline-workspace.sh.
2. Verify both commits before the first model load.
3. Keep the GPU lock for all paired runs.
4. Stop each resident after its leg.
5. Reject changed pins, workers, or calibration files.

The ranked preflight enforces the stored commit and file pins.
