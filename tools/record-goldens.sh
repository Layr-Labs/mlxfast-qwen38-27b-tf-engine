#!/usr/bin/env bash
# record-goldens.sh -- record and pin this track's goldens on the ranked box.
#
# This Qwen3.8 contract uses stop token 248046
# (<|im_end|>) and reads depth 1 from dflash_head.permitted_draft_depths.
#
# ORGANIZER TOOL, box only. The candidate leg at DFlash depth D replays the
# engine's free run at depth D, and benchd checks it token for token against
# the oracle that live_golden_speculative[dflashD][<name>] pins. A depth oracle
# can equal the serial tape or differ from it. Each depth still has its own
# contract key and is recorded through the same free-run path the candidate
# leg replays (author-with-what-you-replay). The serial-control leg stays
# checked against the serial tape (--control-golden).
#
# One resident is booted per depth by this tree's tools/resident-up.sh --boot,
# using this tree's engine, weights, and DFlash head. Every live golden is
# recorded under that resident. Each oracle carries the pinned serial cases[]
# so every per-depth golden has the same token-fidelity cases as the serial
# tape. The MLX bench-worker takes the decoder per request, so the free run uses
# `record-correctness-golden --benchmark-spec dflash:<D>`.
#
# THE RECORDING TREE IS THIS CHECKOUT, NOT THE REFERENCE TREE. The reference
# tree at baseline_reference_commit runs the serial-control leg only. Each
# permitted depth needs the engine that serves it, so the organizer runs this
# tool from a built, clean checkout of the engine commit the oracles are pinned
# for. Its tracked files must match its HEAD.
#
# THE TOOL TAKES THE GPU LOCK ITSELF, for the whole window, the way
# tools/calibrate-box.sh does, and the resident boots check that it is held.
#
# Usage:
#   tools/record-goldens.sh --public --out <DIR> [--apply]
#                           [--public-tokens <FILE>]
#   tools/record-goldens.sh --prompts <DIR> --out <DIR> [--apply]
#   tools/record-goldens.sh --prompts <DIR> --out <DIR>
#                           --oracles-for <name|all> [--apply]
# The public mode records the 256- and 1024-step long-copy captures. It uses
# the staged target tokenizer unless --public-tokens supplies a seed-length file.
# The normal mode records serial pool tapes and the warm-up tape first. The
# oracle-only mode requires already-pinned serial tapes in --out.
#
# Env:
#   BENCHD_BIN_DIR              the benchd pair (default <repo>/benchd-bin); its
#                               record-correctness-golden must know
#                               --benchmark-spec.
#   RECORD_GOLDENS_STOP_TOKEN   the end-of-turn id the serial tapes were
#                               recorded with (default 248046, <|im_end|>).
#   RECORD_GOLDENS_PYTHON       Python with tokenizers (default python3).
#   RECORD_GOLDENS_TOKENIZER    tokenizer JSON (default staged target weights).
#   RESIDENT_UP_LOCK_PATH       the GPU lock (default /tmp/mtplx-gpu-exclusive.lock).
#   MLXFAST_GPU_LOCK_TIMEOUT_S  ceiling on the wait for it (default 1800).
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
FIXTURE="${ROOT}/fixtures/qwen3_8_27b_tf_mlx_v1_track.json"
log() { printf 'record-goldens: %s\n' "$*" >&2; }
die() { printf 'record-goldens: %s\n' "$*" >&2; exit 1; }

PROMPTS="" OUT="" APPLY=0 DEPTHS_ARG="" IN_LOCK=0 PUBLIC=0 PUBLIC_TOKENS="" ORACLES_FOR=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --public) PUBLIC=1; shift ;;
    --public-tokens) PUBLIC_TOKENS="$2"; shift 2 ;;
    --prompts) PROMPTS="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --depths) DEPTHS_ARG="$2"; shift 2 ;;
    --oracles-for) ORACLES_FOR+=("$2"); shift 2 ;;
    --apply) APPLY=1; shift ;;
    --in-lock) IN_LOCK=1; shift ;;
    *) die "unknown argument '$1'" ;;
  esac
done
if [[ "${PUBLIC}" == "1" ]]; then
  [[ -d "${OUT}" ]] || die "usage: --public --out <DIR> [--public-tokens <FILE>] [--apply]"
  [[ -z "${PROMPTS}" && -z "${DEPTHS_ARG}" && "${#ORACLES_FOR[@]}" -eq 0 ]] \
    || die "--public cannot be combined with --prompts, --depths, or --oracles-for"
  if [[ -n "${PUBLIC_TOKENS}" ]]; then
    [[ -f "${PUBLIC_TOKENS}" ]] || die "no --public-tokens file ${PUBLIC_TOKENS}"
    PUBLIC_TOKENS="$(cd -- "$(dirname -- "${PUBLIC_TOKENS}")" && pwd -P)/$(basename -- "${PUBLIC_TOKENS}")"
  fi
else
  [[ -z "${PUBLIC_TOKENS}" ]] || die "--public-tokens requires --public"
  [[ -d "${PROMPTS}" && -d "${OUT}" ]] || die "usage: --prompts <DIR> --out <DIR> [--depths 1,2,...] [--oracles-for <name|all>] [--apply]"
fi
OUT="$(cd -- "${OUT}" && pwd -P)"
[[ "${PUBLIC}" == "1" ]] || PROMPTS="$(cd -- "${PROMPTS}" && pwd -P)"
CORRECTNESS_PROMPTS="$(cd -- "${ROOT}/correctness_prompts" && pwd -P)"
case "${OUT}/" in
  "${CORRECTNESS_PROMPTS}/"*)
    die "--out must be outside this tree's correctness_prompts directory: ${CORRECTNESS_PROMPTS}"
    ;;
esac
chmod 700 "${OUT}"
umask 077
command -v jq >/dev/null 2>&1 || die "jq is required"
SEED_TOKENS="$(jq -er '.seed_tokens | select(type == "number" and . > 0 and . == floor)' "${FIXTURE}")" \
  || die "the fixture seed_tokens must be a positive integer"
TREE="${ROOT}"
git -C "${TREE}" diff --quiet HEAD -- 2>/dev/null \
  || die "the recording tree ${TREE} has tracked changes against its HEAD; record from a clean checkout"
[[ -x "${TREE}/Vendor/TensorFold/zig-out/bin/bench-worker" && -d "${TREE}/reference_weights/Qwen3.8-27B-MLX-4bit" ]] \
  || die "the recording tree ${TREE} is not built (no Vendor/TensorFold/zig-out/bin/bench-worker or reference target snapshot/)"
BIN="${BENCHD_BIN_DIR:-${ROOT}/benchd-bin}"
RECORDER="${BIN}/record-correctness-golden"
BENCHD="${BIN}/benchd"
[[ -x "${RECORDER}" && -x "${BENCHD}" ]] || die "no benchd pair in ${BIN} (run tools/fetch-benchd.sh)"
# --help exits non-zero, so read its text first; under pipefail a pipe into
# grep would report the recorder's exit status, not the match.
RECORDER_HELP="$("${RECORDER}" --help 2>&1 || true)"
grep -q -- '--benchmark-spec' <<<"${RECORDER_HELP}" \
  || die "${RECORDER} has no --benchmark-spec; it would record every depth serial"
grep -q -- 'dflash:<D>' <<<"${RECORDER_HELP}" \
  || die "${RECORDER} does not record dflash:<D>; fetch the channel benchd (tools/fetch-benchd.sh)"

TRACK="$(jq -r .trackId "${ROOT}/benchmark.json")"
REPO="$(jq -r .target.upstream_model_id "${FIXTURE}")"
REV="$(jq -r .target.upstream_revision "${FIXTURE}")"
NAMES=()
LIVE_NAMES=()
PERMITTED=()
if [[ "${PUBLIC}" == "0" ]]; then
  STOP="${RECORD_GOLDENS_STOP_TOKEN:-248046}"
  [[ "${STOP}" =~ ^[0-9]+$ ]] || die "RECORD_GOLDENS_STOP_TOKEN must be a token id"
  while IFS= read -r n; do NAMES+=("${n}"); done < <(jq -r '.timed_prompt_pool[].r2_path | split("/")[-1] | sub("\\.golden\\.json$"; "")' "${FIXTURE}")
  while IFS= read -r n; do LIVE_NAMES+=("${n}"); done < <(jq -r '(.live_goldens // [])[]' "${FIXTURE}")
  [[ "${#NAMES[@]}" -gt 0 && "${#LIVE_NAMES[@]}" -gt 0 ]] || die "the fixture declares no timed pool or live goldens"
  WARMUP_FILE="$(jq -r '.warmup_golden.r2_path | split("/")[-1]' "${FIXTURE}")"
  WARMUP_NAME="${WARMUP_FILE%.golden.json}"
  while IFS= read -r d; do PERMITTED+=("${d}"); done < <(jq -r '.dflash_head.permitted_draft_depths[]' "${FIXTURE}")
  if [[ -n "${DEPTHS_ARG}" ]]; then
    IFS=, read -ra DEPTHS <<<"${DEPTHS_ARG}"
    for d in "${DEPTHS[@]}"; do
      [[ " ${PERMITTED[*]} " == *" ${d} "* ]] || die "--depths ${d}: not a permitted draft depth (${PERMITTED[*]})"
    done
  else
    DEPTHS=("${PERMITTED[@]}")
  fi

  if [[ "${#ORACLES_FOR[@]}" -eq 0 ]]; then
    ORACLE_NAMES=("${LIVE_NAMES[@]}")
    for n in "${NAMES[@]}" "${WARMUP_NAME}"; do
      [[ -f "${PROMPTS}/${n}.tokens.json" ]] || die "no ${PROMPTS}/${n}.tokens.json"
    done
  else
    if [[ " ${ORACLES_FOR[*]} " == *" all "* ]]; then
      ORACLE_NAMES=("${LIVE_NAMES[@]}")
    else
      ORACLE_NAMES=("${ORACLES_FOR[@]}")
    fi
    for n in "${ORACLE_NAMES[@]}"; do
      [[ " ${LIVE_NAMES[*]} " == *" ${n} "* ]] || die "--oracles-for ${n}: not a live golden (${LIVE_NAMES[*]})"
      [[ -f "${PROMPTS}/${n}.tokens.json" ]] || die "no ${PROMPTS}/${n}.tokens.json"
      serial="${OUT}/${n}.golden.json"
      [[ -f "${serial}" ]] || die "--oracles-for ${n}: its pinned serial golden is not in ${OUT}"
      want="$(jq -r --arg f "/${n}.golden.json" 'first(.timed_prompt_pool[] | select(.r2_path | endswith($f))) | "\(.sha256) \(.bytes)"' "${FIXTURE}")"
      have="$(shasum -a 256 "${serial}" | cut -d' ' -f1) $(wc -c < "${serial}" | tr -d ' ')"
      [[ "${have}" == "${want}" ]] || die "--oracles-for ${n}: ${serial} does not match its timed_prompt_pool pin (${want})"
    done
  fi
fi

# THE LOCK WINDOW. The first invocation takes the lock on an inheritable
# descriptor and re-executes this script inside it.
if [[ "${IN_LOCK}" == "0" ]]; then
  LOCK="${RESIDENT_UP_LOCK_PATH:-/tmp/mtplx-gpu-exclusive.lock}"
  TIMEOUT="${MLXFAST_GPU_LOCK_TIMEOUT_S:-1800}"
  export RESIDENT_UP_LOCK_PATH="${LOCK}"
  log "taking the GPU lock ${LOCK}"
  if [[ "${PUBLIC}" == "1" ]]; then
    args=(--public --out "${OUT}" --in-lock)
    [[ -n "${PUBLIC_TOKENS}" ]] && args+=(--public-tokens "${PUBLIC_TOKENS}")
  else
    args=(--prompts "${PROMPTS}" --out "${OUT}" --in-lock)
    [[ -n "${DEPTHS_ARG}" ]] && args+=(--depths "${DEPTHS_ARG}")
    for n in ${ORACLES_FOR[@]+"${ORACLES_FOR[@]}"}; do args+=(--oracles-for "${n}"); done
  fi
  [[ "${APPLY}" == "1" ]] && args+=(--apply)
  exec python3 -c "
import fcntl, os, signal, sys
fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o666)
os.set_inheritable(fd, True)
def on_timeout(*_):
    sys.stderr.write('record-goldens: another owner holds ' + sys.argv[1] + '; nothing has been loaded\n')
    sys.exit(1)
signal.signal(signal.SIGALRM, on_timeout)
signal.alarm(int(sys.argv[2]))
fcntl.flock(fd, fcntl.LOCK_EX)
signal.alarm(0)
os.execv(sys.argv[3], sys.argv[3:])
" "${LOCK}" "${TIMEOUT}" "${BASH}" "${BASH_SOURCE[0]}" "${args[@]}"
fi

# Keep resident logs outside OUT because the arming tool accepts golden files only.
RESIDENT_LOGS="${OUT}.resident-logs"
mkdir -p "${RESIDENT_LOGS}"
chmod 700 "${RESIDENT_LOGS}"
SOCKET=""
TOKEN_TMP=""
stop_resident() {
  [[ -z "${SOCKET}" ]] && return 0
  RESIDENT_UP_LOG_DIR="${RESIDENT_LOGS}" "${TREE}/tools/resident-up.sh" --stop --socket "${SOCKET}" >&2
  SOCKET=""
  rm -f "${OUT}/.socket"
}
stop_resident_on_exit() {
  # A failed normal stop leaves SOCKET set, so the EXIT trap can retry it.
  [[ -z "${SOCKET}" ]] || RESIDENT_UP_LOG_DIR="${RESIDENT_LOGS}" "${TREE}/tools/resident-up.sh" --stop --socket "${SOCKET}" >&2 || true
}
# The recorder and validate-golden parse the contract strictly, and pool pins stay UNRECORDED until this
# tool arms them, so both read a private copy where only unrecorded pool pins are typed placeholders.
RECORD_CONTRACT="$(mktemp "${TMPDIR:-/tmp}/record-contract.XXXXXX")"
jq 'walk(if type == "object" and .bytes == "UNRECORDED" then .bytes = 1 | .sha256 = ("0" * 64) else . end)' \
  "${FIXTURE}" > "${RECORD_CONTRACT}"
trap 'stop_resident_on_exit; rm -f "${RECORD_CONTRACT}" "${OUT}/.socket" ${TOKEN_TMP:+"${TOKEN_TMP}"}' EXIT

boot() { # boot SPEC DEPTH
  RESIDENT_UP_LOG_DIR="${RESIDENT_LOGS}" "${TREE}/tools/resident-up.sh" \
    --boot --spec "$1" --draft-len "$2" --socket-out "${OUT}/.socket" >&2
  SOCKET="$(head -1 "${OUT}/.socket")"
}
BENCHMARK_SPEC="serial"
record() { # record NAME OUTFILE
  (cd "${TREE}" && BENCH_WORKER_RESIDENT_SOCKET="${SOCKET}" MLXFAST_QWEN_MTP_TRACK_ID="${TRACK}" "${RECORDER}" \
     --worker-bin "${TREE}/Vendor/TensorFold/zig-out/bin/bench-worker" --weights "${TREE}/reference_weights/Qwen3.8-27B-MLX-4bit" \
     --prompt-tokens "${PROMPTS}/$1.tokens.json" --case-name "$1" \
     --steps 1024 --stop-token "${STOP}" \
     --benchmark-free-run --benchmark-spec "${BENCHMARK_SPEC}" \
     --track "${TRACK}" --contract "${RECORD_CONTRACT}" \
     --model-provenance-repo "${REPO}" --model-provenance-rev "${REV}" --out "$2") >&2
  chmod 600 "$2"
}
validate() {
  "${BENCHD}" validate-golden --golden "$1" --track "${TRACK}" --contract "${RECORD_CONTRACT}" >&2 \
    || die "benchd refused $1"
}

if [[ "${PUBLIC}" == "1" ]]; then
  PUBLIC_NAME="public-longcopy-gate-english-${SEED_TOKENS}"
  PUBLIC_TOKEN_OUT="${OUT}/public_longcopy_gate_english_${SEED_TOKENS}.tokens.json"
  if [[ -n "${PUBLIC_TOKENS}" ]]; then
    jq -e --argjson n "${SEED_TOKENS}" \
      'type == "array" and length == $n and all(.[]; type == "number" and . >= 0 and . == floor)' \
      "${PUBLIC_TOKENS}" >/dev/null \
      || die "--public-tokens must be a JSON array of exactly ${SEED_TOKENS} token ids"
    if [[ "${PUBLIC_TOKENS}" != "${PUBLIC_TOKEN_OUT}" ]]; then
      TOKEN_TMP="${PUBLIC_TOKEN_OUT}.tmp"
      cp "${PUBLIC_TOKENS}" "${TOKEN_TMP}"
      mv "${TOKEN_TMP}" "${PUBLIC_TOKEN_OUT}"
      TOKEN_TMP=""
    fi
  else
    PUBLIC_PYTHON="${RECORD_GOLDENS_PYTHON:-python3}"
    command -v "${PUBLIC_PYTHON}" >/dev/null 2>&1 \
      || die "RECORD_GOLDENS_PYTHON is not executable: ${PUBLIC_PYTHON}"
    PUBLIC_PROMPT="${ROOT}/correctness_prompts/public_longcopy_gate_english_4096.txt"
    PUBLIC_TOKENIZER="${RECORD_GOLDENS_TOKENIZER:-${TREE}/reference_weights/Qwen3.8-27B-MLX-4bit/tokenizer.json}"
    [[ -f "${PUBLIC_PROMPT}" && -f "${PUBLIC_TOKENIZER}" ]] \
      || die "the public prompt or pinned target tokenizer is missing"
    TOKEN_TMP="${PUBLIC_TOKEN_OUT}.tmp"
    "${PUBLIC_PYTHON}" - "${PUBLIC_PROMPT}" "${PUBLIC_TOKENIZER}" "${TOKEN_TMP}" "${SEED_TOKENS}" <<'PY'
import json
import sys

from tokenizers import Tokenizer

prompt_path, tokenizer_path, output_path, seed_tokens_text = sys.argv[1:]
seed_tokens = int(seed_tokens_text)
tokenizer = Tokenizer.from_file(tokenizer_path)
with open(prompt_path, encoding="utf-8") as prompt_file:
    token_ids = tokenizer.encode(prompt_file.read(), add_special_tokens=False).ids
if len(token_ids) < seed_tokens:
    raise SystemExit(
        f"public prompt encoded to {len(token_ids)} ids; needs at least {seed_tokens}"
    )
with open(output_path, "w", encoding="utf-8") as output_file:
    json.dump(token_ids[:seed_tokens], output_file, separators=(",", ":"))
    output_file.write("\n")
PY
    mv "${TOKEN_TMP}" "${PUBLIC_TOKEN_OUT}"
    TOKEN_TMP=""
  fi
  chmod 600 "${PUBLIC_TOKEN_OUT}"

  record_public() { # record STEPS BENCHMARK_STEPS OUTFILE
    (cd "${TREE}" && BENCH_WORKER_RESIDENT_SOCKET="${SOCKET}" MLXFAST_QWEN_MTP_TRACK_ID="${TRACK}" "${RECORDER}" \
       --worker-bin "${TREE}/Vendor/TensorFold/zig-out/bin/bench-worker" --weights "${TREE}/reference_weights/Qwen3.8-27B-MLX-4bit" \
       --prompt-tokens "${PUBLIC_TOKEN_OUT}" --case-name "${PUBLIC_NAME}" \
       --steps "$1" --benchmark-free-run --benchmark-steps "$2" --benchmark-spec serial \
       --track "${TRACK}" --contract "${RECORD_CONTRACT}" \
       --model-provenance-repo "${REPO}" --model-provenance-rev "${REV}" --out "$3") >&2
    chmod 600 "$3"
  }

  PUBLIC_SHORT="${OUT}/public_longcopy_gate_english_1024_256.json"
  PUBLIC_LONG="${OUT}/public_longcopy_gate_english_1024_1024.json"
  log "serial resident: public long-copy captures"
  boot serial 0
  record_public 256 128 "${PUBLIC_SHORT}"
  validate "${PUBLIC_SHORT}"
  record_public 1024 1024 "${PUBLIC_LONG}"
  validate "${PUBLIC_LONG}"
  stop_resident
  if [[ "${APPLY}" == "1" ]]; then
    cp "${PUBLIC_SHORT}" "${ROOT}/correctness_prompts/$(basename "${PUBLIC_SHORT}")"
    cp "${PUBLIC_LONG}" "${ROOT}/correctness_prompts/$(basename "${PUBLIC_LONG}")"
    log "copied both public captures to ${ROOT}/correctness_prompts"
  fi
  log "recorded and validated both public captures in ${OUT}"
  exit 0
fi

if [[ "${#ORACLES_FOR[@]}" -eq 0 ]]; then
  log "serial resident: ${#NAMES[@]} pool golden(s) and one warm-up golden"
  boot serial 0
  for n in "${NAMES[@]}"; do
    record "${n}" "${OUT}/${n}.golden.json"
    validate "${OUT}/${n}.golden.json"
  done
  record "${WARMUP_NAME}" "${OUT}/${WARMUP_FILE}"
  validate "${OUT}/${WARMUP_FILE}"
  stop_resident
fi

for d in "${DEPTHS[@]}"; do
  log "dflash${d}: booting this tree's resident"
  boot dflash "${d}"
  BENCHMARK_SPEC="dflash:${d}"
  for n in "${ORACLE_NAMES[@]}"; do
    raw="${OUT}/${n}.dflash${d}.raw.json"
    record "${n}" "${raw}"
    jq --slurpfile serial "${OUT}/${n}.golden.json" '.cases = $serial[0].cases' "${raw}" > "${OUT}/${n}.dflash${d}.golden.json"
    chmod 600 "${OUT}/${n}.dflash${d}.golden.json"
    rm -f "${raw}"
    validate "${OUT}/${n}.dflash${d}.golden.json"
  done
  stop_resident
done

pins="{}"
if [[ "${#ORACLES_FOR[@]}" -eq 0 ]]; then
  for n in "${NAMES[@]}"; do
    f="${OUT}/${n}.golden.json"
    pins="$(jq -c --arg k "${n}.golden.json" --arg s "$(shasum -a 256 "${f}" | cut -d' ' -f1)" \
      --argjson b "$(wc -c < "${f}" | tr -d ' ')" '.[$k] = {sha256: $s, bytes: $b}' <<<"${pins}")"
  done
  f="${OUT}/${WARMUP_FILE}"
  pins="$(jq -c --arg k "${WARMUP_FILE}" --arg s "$(shasum -a 256 "${f}" | cut -d' ' -f1)" \
    --argjson b "$(wc -c < "${f}" | tr -d ' ')" '.[$k] = {sha256: $s, bytes: $b}' <<<"${pins}")"
fi
for d in "${DEPTHS[@]}"; do
  for n in "${ORACLE_NAMES[@]}"; do
    f="${OUT}/${n}.dflash${d}.golden.json"
    pins="$(jq -c --arg k "${n}.dflash${d}.golden.json" \
      --arg s "$(shasum -a 256 "${f}" | cut -d' ' -f1)" --argjson b "$(wc -c < "${f}" | tr -d ' ')" \
      '.[$k] = {sha256: $s, bytes: $b}' <<<"${pins}")"
  done
done
jq . <<<"${pins}"
log "upload each file to R2 at its r2_path; the bucket is append-only, so a key that exists with other bytes is a stop"

if [[ "${APPLY}" == "1" ]]; then
  jq --argjson p "${pins}" --arg warmup "${WARMUP_FILE}" '
    def pool_pin: . as $e | ($e.r2_path | split("/")[-1]) as $f | $e + ($p[$f] // {});
    .timed_prompt_pool |= map(pool_pin)
    | if $p[$warmup] then .warmup_golden += $p[$warmup] else . end
    | if $p[(.live_goldens[0] + ".golden.json")] then
        .hidden_correctness_golden = $p[(.live_goldens[0] + ".golden.json")]
      else . end
    | reduce ($p | to_entries[] | select(.key | test("^[a-z0-9_-]+\\.dflash[0-9]+\\.golden\\.json$"))) as $o (.;
        ($o.key | split(".")) as $k
        | (.live_golden_speculative[$k[1]][$k[0]].r2_path //
           ("correctness_prompts/" + .track_id + "/" + $o.key)) as $r2_path
        | .live_golden_speculative[$k[1]][$k[0]] =
            ({r2_path: $r2_path} + $o.value))' \
    "${FIXTURE}" > "${FIXTURE}.tmp"
  mv "${FIXTURE}.tmp" "${FIXTURE}"
  log "pinned ${FIXTURE}; review the diff, then commit"
fi
