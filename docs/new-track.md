# New track

This page gives the command template. See
`docs/new-track-repo-procedure.md` for the full procedure.

Run this command from a clean seed.

```bash
tools/new-track.sh \
  --track-id qwen3.8-27b-tf-mlx-v1 \
  --fork-sha eea9dc9c9c8b459c0a03de7f6fe0c6f4835d69ab \
  --checkpoint TensorFold/Qwen3.8-27B-MLX-4bit@22d8d538154e0e5b6f8dcb1fcc74a60a0798104c \
  --bench-commit 8ce0ca7821e9c882f2ff1e1cbd1e2ce561040729 \
  --os macOS \
  --snapshot-dir /path/to/local/snapshot
```

The tool hashes local files.
It does not use the network.
