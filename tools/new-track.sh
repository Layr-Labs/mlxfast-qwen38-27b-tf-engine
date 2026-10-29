#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
TRACK_ID="" FORK_SHA="" CHECKPOINT="" BENCH_COMMIT="" OS_NAME="" SNAPSHOT_DIR=""
ALLOW_DIRTY=0

fail() { printf 'new-track.sh: %s\n' "$*" >&2; exit 1; }
usage() {
  printf '%s\n' 'usage: tools/new-track.sh --track-id ID --fork-sha SHA --checkpoint REPO@SHA --bench-commit SHA --os macOS|Linux --snapshot-dir DIR [--allow-dirty]'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --track-id) TRACK_ID="${2:-}"; shift 2 ;;
    --fork-sha) FORK_SHA="${2:-}"; shift 2 ;;
    --checkpoint) CHECKPOINT="${2:-}"; shift 2 ;;
    --bench-commit) BENCH_COMMIT="${2:-}"; shift 2 ;;
    --os) OS_NAME="${2:-}"; shift 2 ;;
    --snapshot-dir) SNAPSHOT_DIR="${2:-}"; shift 2 ;;
    --allow-dirty) ALLOW_DIRTY=1; shift ;;
    --hash-non-lfs) shift ;;
    -h|--help) usage; exit 0 ;;
    *) fail "unknown argument: $1" ;;
  esac
done

[[ "${TRACK_ID}" =~ ^[a-z0-9][a-z0-9.-]*-(mlx|cuda)-v[0-9]+$ ]] || fail "invalid --track-id"
[[ "${FORK_SHA}" =~ ^[0-9a-f]{40}$ ]] || fail "--fork-sha must be 40 lowercase hex characters"
[[ "${BENCH_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "--bench-commit is required and must be 40 lowercase hex characters"
[[ "${CHECKPOINT}" =~ ^[^/@]+/[^/@]+@[0-9a-f]{40}$ ]] || fail "--checkpoint must be REPO@40_HEX"
[[ "${OS_NAME}" == "macOS" || "${OS_NAME}" == "Linux" ]] || fail "--os must be macOS or Linux"
[[ -d "${SNAPSHOT_DIR}" && -f "${SNAPSHOT_DIR}/config.json" ]] || fail "--snapshot-dir must contain config.json"
if [[ "${ALLOW_DIRTY}" != 1 && -n "$(git -C "${ROOT}" status --porcelain)" ]]; then
  fail "worktree is dirty; use --allow-dirty only for the first seed conversion"
fi

OLD_CONTRACT="$(jq -er '.contractPath' "${ROOT}/benchmark.json")"
[[ -f "${ROOT}/${OLD_CONTRACT}" ]] || fail "contract is missing: ${OLD_CONTRACT}"
NEW_CONTRACT="fixtures/$(printf '%s' "${TRACK_ID}" | tr '.-' '__')_track.json"
CHECKPOINT_REPO="${CHECKPOINT%@*}"
CHECKPOINT_REV="${CHECKPOINT##*@}"
MODEL_PART="$(printf '%s' "${TRACK_ID}" | sed -E 's/-(mlx|cuda)-v[0-9]+$//' | tr -d '.')"
NAME="tensorfold-${MODEL_PART}"

python3 - "${ROOT}" "${OLD_CONTRACT}" "${NEW_CONTRACT}" "${TRACK_ID}" "${NAME}" \
  "${FORK_SHA}" "${BENCH_COMMIT}" "${CHECKPOINT_REPO}" "${CHECKPOINT_REV}" \
  "${SNAPSHOT_DIR}" "${OS_NAME}" <<'PY'
import hashlib, json, pathlib, sys

(root, old_rel, new_rel, track, name, fork_sha, bench_sha,
 repo, revision, snapshot, os_name) = sys.argv[1:]
root, snapshot = pathlib.Path(root), pathlib.Path(snapshot)

def digest_file(file_name):
    digest = hashlib.sha256()
    with file_name.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

skip = {".gitattributes", "LICENSE", "LICENSE.md", "LICENSE.txt", "qwen-logo.png"}
files = []
for item in sorted(snapshot.rglob("*")):
    if item.is_file() and item.name not in skip:
        files.append({"path": item.relative_to(snapshot).as_posix(),
                      "bytes": item.stat().st_size, "sha256": digest_file(item)})
if not files:
    raise SystemExit("new-track.sh: snapshot has no model files")

with (root / old_rel).open(encoding="utf-8") as stream:
    contract = json.load(stream)
with (snapshot / "config.json").open(encoding="utf-8") as stream:
    config = json.load(stream)
text = config.get("text_config", config)

contract.update({
    "track_id": track,
    "track_name": f"{track} paired composite speedup",
    "benchmark_name": name,
    "official_scoring_enabled": False,
    "official_pairs": 2,
    "baseline_reference_commit": "UNRECORDED",
    "tensorfold_revision": fork_sha,
    "benchd_source_commit": bench_sha,
    "vocab_size": text["vocab_size"],
    "num_hidden_layers": text["num_hidden_layers"],
    "golden_model_type": text["model_type"],
    "live_goldens": ["pool-01", "pool-02"],
    "timed_prompt_pool": [
        {"r2_path": f"correctness_prompts/{track}/pool-01.golden.json", "sha256": "UNRECORDED", "bytes": "UNRECORDED"},
        {"r2_path": f"correctness_prompts/{track}/pool-02.golden.json", "sha256": "UNRECORDED", "bytes": "UNRECORDED"}],
    "warmup_golden": {"r2_path": f"correctness_prompts/{track}/warmup-01.golden.json", "sha256": "UNRECORDED", "bytes": "UNRECORDED"},
    "hidden_correctness_golden": {"sha256": "UNRECORDED", "bytes": "UNRECORDED"},
    "live_golden_speculative": {"dflash1": {
        pool: {"r2_path": f"correctness_prompts/{track}/{pool}.dflash1.golden.json", "sha256": "UNRECORDED", "bytes": "UNRECORDED"}
        for pool in ("pool-01", "pool-02")}},
})
target = {
    "upstream_model_id": repo, "upstream_revision": revision,
    "architecture": config.get("architectures", [""])[0], "model_type": text["model_type"],
    "num_hidden_layers": text["num_hidden_layers"], "hidden_size": text["hidden_size"],
    "intermediate_size": text["intermediate_size"], "vocab_size": text["vocab_size"],
    "num_attention_heads": text["num_attention_heads"], "num_key_value_heads": text["num_key_value_heads"],
    "head_dim": text.get("head_dim", text["hidden_size"] // text["num_attention_heads"]),
    "linear_num_key_heads": text["linear_num_key_heads"], "linear_num_value_heads": text["linear_num_value_heads"],
    "linear_key_head_dim": text["linear_key_head_dim"], "linear_value_head_dim": text["linear_value_head_dim"],
    "linear_conv_kernel_dim": text["linear_conv_kernel_dim"], "layer_types": text["layer_types"],
    "max_position_embeddings": text["max_position_embeddings"], "quantization": config["quantization_config"],
    "files": files,
    "drafter": {"upstream_model_id": "z-lab/Qwen3.8-27B-DFlash2", "upstream_revision": "50307d4c4cde6860d4eee73e2547cd786fe8e8a4"},
}
manifest_rel = contract.get("target", {}).get("manifest_path")
if manifest_rel:
    target["manifest_path"] = manifest_rel
    manifest_file = root / manifest_rel
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    with manifest_file.open("w", encoding="utf-8") as stream:
        stream.write(f"# {repo}@{revision}\n")
        for record in files:
            stream.write(f"{record['sha256']}  {record['path']}\n")
contract["target"] = target
contract["dflash_head"] = {
    "role": "spec_decode_dflash2", "source": "pinned", "permitted_draft_depths": [1],
    "upstream_model_id": "z-lab/Qwen3.8-27B-DFlash2", "upstream_revision": "50307d4c4cde6860d4eee73e2547cd786fe8e8a4"}
contract["protocol"] = {"max_speculative_batch": 1, "spec_decoder": "dflash", "verification_mode": "rectangular"}
new_file = root / new_rel
new_file.parent.mkdir(parents=True, exist_ok=True)
with new_file.open("w", encoding="utf-8") as stream:
    json.dump(contract, stream, indent=2); stream.write("\n")
if old_rel != new_rel:
    (root / old_rel).unlink()

manifest_file = root / "benchmark.json"
with manifest_file.open(encoding="utf-8") as stream:
    manifest = json.load(stream)
manifest.update({"name": name, "trackId": track, "contractPath": new_rel,
                 "staticReviewTrackId": track, "description": f"Qwen3.8-27B TensorFold speed track on {os_name}."})
manifest["leaderboard"]["namespace"] = track
with manifest_file.open("w", encoding="utf-8") as stream:
    json.dump(manifest, stream, indent=2); stream.write("\n")
PY

printf 'new-track.sh: wrote %s\n' "${NEW_CONTRACT}"
printf 'new-track.sh: pinned benchd source %s\n' "${BENCH_COMMIT}"
printf 'new-track.sh: hashed local snapshot %s\n' "${SNAPSHOT_DIR}"
