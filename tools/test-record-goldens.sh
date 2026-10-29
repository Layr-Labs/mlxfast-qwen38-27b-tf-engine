#!/usr/bin/env bash
# Hermetic test for serial, warm-up, oracle, and apply phases.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
SEED_TOKENS="$(jq -er '.seed_tokens | select(type == "number" and . > 0 and . == floor)' \
  "${ROOT}/fixtures/qwen3_8_27b_tf_mlx_v1_track.json")"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
fail() { echo "test-record-goldens: FAIL -- $*" >&2; exit 1; }

TREE="${WORK}/tree"
mkdir -p "${TREE}/tools" "${TREE}/fixtures" "${TREE}/correctness_prompts" "${TREE}/Vendor/TensorFold/zig-out/bin" \
  "${TREE}/reference_weights/Qwen3.8-27B-MLX-4bit" "${WORK}/bin" "${WORK}/prompts" "${WORK}/out"
cp "${ROOT}/tools/record-goldens.sh" "${TREE}/tools/record-goldens.sh"
chmod 755 "${TREE}/tools/record-goldens.sh"
printf '#!/bin/sh\nexit 0\n' > "${TREE}/Vendor/TensorFold/zig-out/bin/bench-worker"
chmod 755 "${TREE}/Vendor/TensorFold/zig-out/bin/bench-worker"
printf '{"trackId":"qwen3.8-27b-tf-mlx-v1"}\n' > "${TREE}/benchmark.json"
cat > "${TREE}/fixtures/qwen3_8_27b_tf_mlx_v1_track.json" <<JSON
{
  "track_id": "qwen3.8-27b-tf-mlx-v1",
  "seed_tokens": ${SEED_TOKENS},
  "target": {"upstream_model_id": "TensorFold/Qwen3.8-27B-MLX-4bit", "upstream_revision": "22d8d538154e0e5b6f8dcb1fcc74a60a0798104c"},
  "dflash_head": {"permitted_draft_depths": [1]},
  "live_goldens": ["pool-01", "pool-02"],
  "timed_prompt_pool": [
    {"r2_path": "correctness_prompts/qwen3.8-27b-tf-mlx-v1/pool-01.golden.json", "sha256": "UNRECORDED", "bytes": "UNRECORDED"},
    {"r2_path": "correctness_prompts/qwen3.8-27b-tf-mlx-v1/pool-02.golden.json", "sha256": "UNRECORDED", "bytes": "UNRECORDED"}
  ],
  "warmup_golden": {"r2_path": "correctness_prompts/qwen3.8-27b-tf-mlx-v1/warmup-01.golden.json", "sha256": "UNRECORDED", "bytes": "UNRECORDED"},
  "hidden_correctness_golden": {"sha256": "UNRECORDED", "bytes": "UNRECORDED"},
  "live_golden_speculative": {
    "dflash1": {
      "pool-01": {"r2_path": "correctness_prompts/qwen3.8-27b-tf-mlx-v1/pool-01.dflash1.golden.json", "sha256": "UNRECORDED", "bytes": "UNRECORDED"},
      "pool-02": {"r2_path": "correctness_prompts/qwen3.8-27b-tf-mlx-v1/pool-02.dflash1.golden.json", "sha256": "UNRECORDED", "bytes": "UNRECORDED"}
    }
  }
}
JSON
for name in pool-01 pool-02 warmup-01; do printf '[1,2,3]\n' > "${WORK}/prompts/${name}.tokens.json"; done

cat > "${WORK}/bin/git" <<'SH'
#!/bin/sh
exit 0
SH
cat > "${WORK}/bin/record-correctness-golden" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "--help" ]]; then
  echo 'usage: --benchmark-spec <serial|dflash:<D>>'
  exit 0
fi
out="" name="" spec="" steps="" benchmark_steps="" benchmark_free_run=0 stop_token=0 prompt_tokens=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) out="$2"; shift 2 ;;
    --case-name) name="$2"; shift 2 ;;
    --benchmark-spec) spec="$2"; shift 2 ;;
    --steps) steps="$2"; shift 2 ;;
    --benchmark-steps) benchmark_steps="$2"; shift 2 ;;
    --benchmark-free-run) benchmark_free_run=1; shift ;;
    --stop-token) stop_token=1; shift 2 ;;
    --prompt-tokens) prompt_tokens="$2"; shift 2 ;;
    *) shift ;;
  esac
done
printf '%s %s\n' "${spec}" "${name}" >> "${MOCK_LOG}/recorder"
printf '%s %s %s %s %s %s %s\n' "${spec}" "${name}" "${steps}" "${benchmark_steps}" \
  "${benchmark_free_run}" "${stop_token}" "$(basename "${prompt_tokens}")" >> "${MOCK_LOG}/recorder-args"
printf '{"cases":[{"name":"%s","prompt_tokens":[1],"expected_tokens":[2]}],"benchmark":{"marker":"PRIVATE-GOLDEN-CONTENT","spec":"%s"}}\n' "${name}" "${spec}" > "${out}"
SH
cat > "${WORK}/bin/benchd" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
golden=""
while [[ $# -gt 0 ]]; do
  case "$1" in --golden) golden="$2"; shift 2 ;; *) shift ;; esac
done
[[ -s "${golden}" ]]
printf '%s\n' "$(basename "${golden}")" >> "${MOCK_LOG}/validate"
SH
cat > "${TREE}/tools/resident-up.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
mode="$1"; shift
if [[ "${mode}" == "--stop" ]]; then
  printf 'stop\n' >> "${MOCK_LOG}/resident"
  exit 0
fi
spec="" out=""
while [[ $# -gt 0 ]]; do
  case "$1" in --spec) spec="$2"; shift 2 ;; --socket-out) out="$2"; shift 2 ;; *) shift ;; esac
done
printf 'boot %s\n' "${spec}" >> "${MOCK_LOG}/resident"
printf '%s/mock.sock\n' "${MOCK_LOG}" > "${out}"
SH
chmod 755 "${WORK}/bin/git" "${WORK}/bin/record-correctness-golden" \
  "${WORK}/bin/benchd" "${TREE}/tools/resident-up.sh"

export MOCK_LOG="${WORK}/log"
mkdir -p "${MOCK_LOG}"
PATH="${WORK}/bin:${PATH}" BENCHD_BIN_DIR="${WORK}/bin" \
  bash "${TREE}/tools/record-goldens.sh" --prompts "${WORK}/prompts" --out "${WORK}/out" \
  --apply --in-lock >"${WORK}/stdout" 2>"${WORK}/stderr" \
  || fail "new-track recording failed: $(cat "${WORK}/stderr")"

[[ "$(cat "${MOCK_LOG}/recorder")" == $'serial pool-01\nserial pool-02\nserial warmup-01\ndflash:1 pool-01\ndflash:1 pool-02' ]] \
  || fail "recording order is wrong: $(cat "${MOCK_LOG}/recorder")"
[[ "$(wc -l < "${MOCK_LOG}/validate" | tr -d ' ')" == 5 ]] || fail "not every golden was validated"
[[ ! -e "${WORK}/out/.socket" ]] || fail "the socket handoff file was left behind"
grep -q 'PRIVATE-GOLDEN-CONTENT' "${WORK}/stdout" "${WORK}/stderr" && fail "golden content was printed"
[[ "$(stat -f '%Lp' "${WORK}/out")" == 700 ]] || fail "output directory is not mode 700"
for file in "${WORK}/out"/*.golden.json; do
  [[ "$(stat -f '%Lp' "${file}")" == 600 ]] || fail "${file} is not mode 600"
done

FIXTURE="${TREE}/fixtures/qwen3_8_27b_tf_mlx_v1_track.json"
jq -e '
  ([.timed_prompt_pool[].bytes, .warmup_golden.bytes, .hidden_correctness_golden.bytes,
     .live_golden_speculative.dflash1[].bytes] | all(type == "number" and . > 0))
  and ([.timed_prompt_pool[].sha256, .warmup_golden.sha256, .hidden_correctness_golden.sha256,
        .live_golden_speculative.dflash1[].sha256] | all(test("^[0-9a-f]{64}$")))
  and (.hidden_correctness_golden.sha256 == .timed_prompt_pool[0].sha256)
' "${FIXTURE}" >/dev/null || fail "--apply did not pin every fixture field"
echo "test-record-goldens: PASS -- new track records serial, warm-up, and oracle goldens, then pins every field"

: > "${MOCK_LOG}/recorder"
: > "${MOCK_LOG}/validate"
serial_before="$(shasum -a 256 "${WORK}/out/pool-02.golden.json" | awk '{print $1}')"
PATH="${WORK}/bin:${PATH}" BENCHD_BIN_DIR="${WORK}/bin" \
  bash "${TREE}/tools/record-goldens.sh" --prompts "${WORK}/prompts" --out "${WORK}/out" \
  --oracles-for pool-02 --apply --in-lock >"${WORK}/oracle-stdout" 2>"${WORK}/oracle-stderr" \
  || fail "--oracles-for failed: $(cat "${WORK}/oracle-stderr")"
[[ "$(cat "${MOCK_LOG}/recorder")" == "dflash:1 pool-02" ]] || fail "--oracles-for recorded a serial tape"
[[ "$(shasum -a 256 "${WORK}/out/pool-02.golden.json" | awk '{print $1}')" == "${serial_before}" ]] \
  || fail "--oracles-for changed the pinned serial tape"
[[ ! -e "${WORK}/out/.socket" ]] || fail "--oracles-for left the socket handoff file"
echo "test-record-goldens: PASS -- --oracles-for keeps the pinned serial tape and records only its oracle"

mkdir -p "${WORK}/public-out"
jq -n --argjson n "${SEED_TOKENS}" '[range(0; $n)]' > "${WORK}/public.tokens.json"
: > "${MOCK_LOG}/recorder"
: > "${MOCK_LOG}/recorder-args"
: > "${MOCK_LOG}/validate"
: > "${MOCK_LOG}/resident"
PATH="${WORK}/bin:${PATH}" BENCHD_BIN_DIR="${WORK}/bin" \
  bash "${TREE}/tools/record-goldens.sh" --public --out "${WORK}/public-out" \
  --public-tokens "${WORK}/public.tokens.json" --apply --in-lock \
  >"${WORK}/public-stdout" 2>"${WORK}/public-stderr" \
  || fail "--public failed: $(cat "${WORK}/public-stderr")"
[[ "$(cat "${MOCK_LOG}/recorder-args")" == \
  "$(printf 'serial public-longcopy-gate-english-%s 256 128 1 0 public_longcopy_gate_english_%s.tokens.json\nserial public-longcopy-gate-english-%s 1024 1024 1 0 public_longcopy_gate_english_%s.tokens.json' \
    "${SEED_TOKENS}" "${SEED_TOKENS}" "${SEED_TOKENS}" "${SEED_TOKENS}")" ]] \
  || fail "--public recorder arguments are wrong: $(cat "${MOCK_LOG}/recorder-args")"
[[ "$(cat "${MOCK_LOG}/resident")" == $'boot serial\nstop' ]] \
  || fail "--public did not use one serial resident: $(cat "${MOCK_LOG}/resident")"
[[ "$(wc -l < "${MOCK_LOG}/validate" | tr -d ' ')" == 2 ]] \
  || fail "--public did not validate both captures"
for suffix in 256 1024; do
  file="public_longcopy_gate_english_1024_${suffix}.json"
  [[ -s "${WORK}/public-out/${file}" ]] || fail "--public did not write ${file}"
  cmp -s "${WORK}/public-out/${file}" "${TREE}/correctness_prompts/${file}" \
    || fail "--apply did not copy ${file}"
done
jq -e --argjson n "${SEED_TOKENS}" 'type == "array" and length == $n' \
  "${WORK}/public-out/public_longcopy_gate_english_${SEED_TOKENS}.tokens.json" >/dev/null \
  || fail "--public did not write the ${SEED_TOKENS}-id token file"
echo "test-record-goldens: PASS -- --public records, validates, and applies both long-copy captures"

mkdir -p "${WORK}/bad-public-out"
BAD_PUBLIC_TOKENS=$((SEED_TOKENS - 1))
jq -n --argjson n "${BAD_PUBLIC_TOKENS}" '[range(0; $n)]' > "${WORK}/bad-public.tokens.json"
: > "${MOCK_LOG}/resident"
if PATH="${WORK}/bin:${PATH}" BENCHD_BIN_DIR="${WORK}/bin" \
  bash "${TREE}/tools/record-goldens.sh" --public --out "${WORK}/bad-public-out" \
  --public-tokens "${WORK}/bad-public.tokens.json" --in-lock \
  >"${WORK}/bad-public-stdout" 2>"${WORK}/bad-public-stderr"; then
  fail "--public accepted a token file with ${BAD_PUBLIC_TOKENS} ids"
fi
grep -q "exactly ${SEED_TOKENS} token ids" "${WORK}/bad-public-stderr" \
  || fail "--public gave the wrong invalid-token error: $(cat "${WORK}/bad-public-stderr")"
[[ ! -s "${MOCK_LOG}/resident" ]] || fail "--public booted a resident for invalid tokens"
echo "test-record-goldens: PASS -- --public refuses a token file that is not ${SEED_TOKENS} ids"

mkdir -p "${TREE}/correctness_prompts/nested"
ln -s "${TREE}/correctness_prompts/nested" "${WORK}/public-out-link"
: > "${MOCK_LOG}/recorder"
: > "${MOCK_LOG}/resident"
for forbidden_out in "${TREE}/correctness_prompts" "${WORK}/public-out-link"; do
  if PATH="${WORK}/bin:${PATH}" BENCHD_BIN_DIR="${WORK}/bin" \
    bash "${TREE}/tools/record-goldens.sh" --public --out "${forbidden_out}" \
    --public-tokens "${WORK}/public.tokens.json" --in-lock \
    >"${WORK}/forbidden-out-stdout" 2>"${WORK}/forbidden-out-stderr"; then
    fail "--public accepted forbidden --out ${forbidden_out}"
  fi
  grep -q -- "--out must be outside this tree's correctness_prompts directory" \
    "${WORK}/forbidden-out-stderr" \
    || fail "--public gave the wrong forbidden-output error: $(cat "${WORK}/forbidden-out-stderr")"
done
[[ ! -s "${MOCK_LOG}/recorder" ]] || fail "--public recorded into correctness_prompts"
[[ ! -s "${MOCK_LOG}/resident" ]] || fail "--public booted a resident for forbidden --out"
echo "test-record-goldens: PASS -- --public refuses correctness_prompts output paths"
echo "test-record-goldens: all cases pass"
