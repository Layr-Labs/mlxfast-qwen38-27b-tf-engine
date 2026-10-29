# New track procedure

## Result

Create a pinned TensorFold track from a local snapshot.

1. Copy the newest control-plane template.
2. Add Vendor/TensorFold at the approved revision.
3. Run `tools/new-track.sh` with `--snapshot-dir`.
4. Verify `benchmark.json` and the generated fixture with `jq`.
5. Run every `tools/test-*.sh` file.
6. Record goldens on the ranked machine.
7. Replace every `UNRECORDED` value in one reviewed change.

See `docs/new-track.md` for the command template.

The target, drafter, TensorFold, and benchd revisions must be full commit IDs.
