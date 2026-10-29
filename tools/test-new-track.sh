#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/test-new-track.XXXXXX")"
trap 'rm -rf "${TEMP_DIR}"' EXIT
mkdir -p "${TEMP_DIR}/tools" "${TEMP_DIR}/fixtures" "${TEMP_DIR}/snapshot"
cp "${ROOT}/tools/new-track.sh" "${TEMP_DIR}/tools/new-track.sh"
cp "${ROOT}/benchmark.json" "${TEMP_DIR}/benchmark.json"
jq '.trackId="seed-model-mlx-v0" | .contractPath="fixtures/seed_model_mlx_v0_track.json" | .leaderboard.namespace="seed-model-mlx-v0"' \
  "${ROOT}/benchmark.json" > "${TEMP_DIR}/benchmark.tmp"
mv "${TEMP_DIR}/benchmark.tmp" "${TEMP_DIR}/benchmark.json"
jq '.track_id="seed-model-mlx-v0" | .target={upstream_model_id:"seed/model",upstream_revision:("0"*40),files:[]}' \
  "${ROOT}/fixtures/qwen3_8_27b_tf_mlx_v1_track.json" > "${TEMP_DIR}/fixtures/seed_model_mlx_v0_track.json"
printf '%s\n' '{"architectures":["Qwen3_5ForConditionalGeneration"],"quantization_config":{"group_size":64,"bits":4,"mode":"affine"},"text_config":{"model_type":"qwen3_5_text","num_hidden_layers":2,"hidden_size":8,"intermediate_size":16,"vocab_size":32,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,"linear_num_key_heads":1,"linear_num_value_heads":2,"linear_key_head_dim":4,"linear_value_head_dim":4,"linear_conv_kernel_dim":4,"layer_types":["linear_attention","full_attention"],"max_position_embeddings":128}}' > "${TEMP_DIR}/snapshot/config.json"
printf 'weight\n' > "${TEMP_DIR}/snapshot/model.safetensors"
(cd "${TEMP_DIR}" && tools/new-track.sh --track-id test3.8-1b-mlx-v1 \
  --fork-sha eea9dc9c9c8b459c0a03de7f6fe0c6f4835d69ab \
  --checkpoint Test/Model@22d8d538154e0e5b6f8dcb1fcc74a60a0798104c \
  --bench-commit 8ce0ca7821e9c882f2ff1e1cbd1e2ce561040729 --os macOS \
  --snapshot-dir "${TEMP_DIR}/snapshot" --allow-dirty >/dev/null)
jq -e '.tensorfold_revision == "eea9dc9c9c8b459c0a03de7f6fe0c6f4835d69ab" and .benchd_source_commit == "8ce0ca7821e9c882f2ff1e1cbd1e2ce561040729" and (.target.files|length) == 2' \
  "${TEMP_DIR}/fixtures/test3_8_1b_mlx_v1_track.json" >/dev/null
printf 'test-new-track.sh: PASS\n'
