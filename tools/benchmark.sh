#!/usr/bin/env bash
# benchmark.sh -- stable shell wrapper for the TensorFold benchd engine.
#
# benchd owns correctness checks, timing, and artifact sealing. This wrapper
# validates shell inputs, resolves the verified benchd binary, supplies the
# trusted track contract, and prints local diagnostics on standard error.
#
# The public shell contract keeps these behaviors stable:
#   - mode and path argument validation;
#   - explicit correctness-golden selection;
#   - local score and integrity path derivation;
#   - local baseline and score summaries;
#   - official sandbox and runtime-worker enforcement;
#   - benchd exit codes 0 and 1 pass through, while usage code 2 maps to 1.
#
# MLXFAST_ENGINE_BIN must name the Engine Protocol v1 worker that benchd
# starts. Official mode also needs a pinned correctness golden. Hidden ranked
# goldens are supplied by the trusted runner and verified before use.
set -euo pipefail

LOCAL_ITERATE=0
LOCAL_SUBMIT=0
OFFICIAL=0
LOCAL_COOL_GATE_ONLY=0
# Arguments are consumed by benchd. --official is a shell-level selector.
DFLASH_DEPTH=""
prev=""
for arg in "$@"; do
  if [[ "${prev}" == "--dflash-depth" ]]; then DFLASH_DEPTH="${arg}"; prev=""; continue; fi
  if [[ "${arg}" == "--dflash-depth" ]]; then prev="--dflash-depth"; continue; fi
  if [[ "${arg}" == --dflash-depth=* ]]; then DFLASH_DEPTH="${arg#--dflash-depth=}"; continue; fi
  prev=""
done
for arg in "$@"; do
  case "${arg}" in
    --weights|--weights=*|--golden|--golden=*|--score-path|--score-path=*)
      echo "benchmark.sh: use MLXFAST_WEIGHTS_PATH, MLXFAST_CORRECTNESS_GOLDEN_PATH, or MLXFAST_SCORE_PATH for shell path overrides" >&2
      echo "benchmark.sh: pass --weights/--golden/--score-path only through the documented environment variables" >&2
      exit 1
      ;;
    --official)
      OFFICIAL=1
      continue
      ;;
    --local-cool-gate-only)
      LOCAL_COOL_GATE_ONLY=1
      continue
      ;;
  esac
  if [[ "${arg}" == "--local-iterate" ]]; then
    LOCAL_ITERATE=1
  fi
  if [[ "${arg}" == "--local-submit" ]]; then
    LOCAL_SUBMIT=1
  fi
done

if [[ "${OFFICIAL}" == "1" && ( "${LOCAL_ITERATE}" == "1" || "${LOCAL_SUBMIT}" == "1" ) ]]; then
  echo "benchmark.sh: --official cannot be combined with --local-iterate/--local-submit" >&2
  exit 1
fi

if [[ "${LOCAL_ITERATE}" == "1" && "${LOCAL_SUBMIT}" == "1" ]]; then
  echo "benchmark.sh: --local-iterate and --local-submit cannot be used together" >&2
  exit 1
fi

# Bare invocations default to the participant-friendly local edit loop.
# The ranked full benchmark must be requested explicitly: with --official, or via the
# trusted workflow env (MLXFAST_OFFICIAL_BENCHMARK_RUN=1).
if [[ "${LOCAL_COOL_GATE_ONLY}" == "0" && "${LOCAL_ITERATE}" == "0" && "${LOCAL_SUBMIT}" == "0" && "${OFFICIAL}" == "0" ]]; then
  if [[ "${MLXFAST_OFFICIAL_BENCHMARK_RUN:-0}" == "1" ]]; then
    OFFICIAL=1
  else
    echo "benchmark.sh: no mode given; defaulting to --local-iterate (use --official for the ranked entrypoint, which requires the private oracle)"
    LOCAL_ITERATE=1
  fi
fi
# Tests pin the no-mode standard-output line above. "Private oracle" means the
# supplied correctness-golden input.

# Official mode routes to the benchd official backend. It requires a pinned
# correctness golden, the sandbox, and the runtime worker.

# The --local-cool-gate-only selector is a bare cool-gate probe in the reference;
# the facade does not run engine work in that mode, and benchd exposes its own
# `--local-cool-gate-only` entrypoint, so there is nothing further for the wrapper
# to dispatch here.
if [[ "${LOCAL_COOL_GATE_ONLY}" == "1" ]]; then
  exit 0
fi

if [[ -z "${MLXFAST_CORRECTNESS_GOLDEN_PATH:-}" ]]; then
  cat >&2 <<'EOF'
benchmark.sh: this track requires an explicit correctness golden.

Set MLXFAST_CORRECTNESS_GOLDEN_PATH. There is no default: a golden is never
selected implicitly. For a local run, use a checked-in public fixture --

  --local-iterate  correctness_prompts/public_longcopy_gate_english_1024_256.json
  --local-submit   correctness_prompts/public_longcopy_gate_english_1024_1024.json

Both public fixtures must be recorded against the pinned Qwen3.8 target
(model_type qwen3_5_text) before local measurement can run.
Official (--official) runs take an organizer-provisioned hidden golden
instead; those are not in this repository.
EOF
  exit 1
fi

if [[ "${LOCAL_ITERATE}" == "1" && -z "${MLXFAST_SCORE_PATH:-}" ]]; then
  SCORE_PATH="score.local-iterate.json"
else
  SCORE_PATH="${MLXFAST_SCORE_PATH:-score.json}"
fi
WEIGHTS_PATH="${MLXFAST_WEIGHTS_PATH:-reference_weights/Qwen3.8-27B-MLX-4bit}"
GOLDEN_PATH="${MLXFAST_CORRECTNESS_GOLDEN_PATH:-}"
if [[ "${LOCAL_ITERATE}" == "1" && -z "${MLXFAST_INTEGRITY_PATH:-}" ]]; then
  INTEGRITY_PATH="benchmark-integrity.local-iterate.json"
else
  INTEGRITY_PATH="${MLXFAST_INTEGRITY_PATH:-benchmark-integrity.json}"
fi

# Fail before run work when the golden fixture is missing.
if [[ ! -f "${GOLDEN_PATH}" ]]; then
  echo "benchmark.sh: correctness golden not found at ${GOLDEN_PATH}" >&2
  echo "benchmark.sh: check MLXFAST_CORRECTNESS_GOLDEN_PATH; there is no fallback golden." >&2
  exit 1
fi

# The supplied correctness golden is the official input. Ranked goldens come
# from read-only storage and have integrity pins. When the caller provides a
# SHA-256 digest or byte count, verify the
# supplied file against the pin BEFORE any run trusts it; a mismatch is fatal (never run
# against an unpinned/altered golden). When BOTH sha256 and bytes are pinned, the official
# dispatch also forwards them to benchd as --golden-sha256/--golden-bytes, which re-verifies
# the raw bytes before parsing — so this shell check is a loud early gate, not the only one.
if [[ -n "${MLXFAST_CORRECTNESS_GOLDEN_SHA256:-}" ]]; then
  if ! command -v shasum >/dev/null 2>&1; then
    echo "benchmark.sh: shasum is required to verify MLXFAST_CORRECTNESS_GOLDEN_SHA256" >&2
    exit 1
  fi
  golden_actual_sha256="$(shasum -a 256 "${GOLDEN_PATH}" | awk '{print $1}')"
  if [[ "${golden_actual_sha256}" != "${MLXFAST_CORRECTNESS_GOLDEN_SHA256}" ]]; then
    echo "benchmark.sh: correctness golden sha256 mismatch — pin=${MLXFAST_CORRECTNESS_GOLDEN_SHA256} actual=${golden_actual_sha256} (${GOLDEN_PATH}); refusing to run against an unpinned golden" >&2
    exit 1
  fi
fi
if [[ -n "${MLXFAST_CORRECTNESS_GOLDEN_BYTES:-}" ]]; then
  golden_actual_bytes="$(wc -c < "${GOLDEN_PATH}" | tr -d '[:space:]')"
  if [[ "${golden_actual_bytes}" != "${MLXFAST_CORRECTNESS_GOLDEN_BYTES}" ]]; then
    echo "benchmark.sh: correctness golden byte count mismatch — pin=${MLXFAST_CORRECTNESS_GOLDEN_BYTES} actual=${golden_actual_bytes} (${GOLDEN_PATH}); refusing to run against an unpinned golden" >&2
    exit 1
  fi
fi

# Local modes print the same-machine baseline snapshot, when one exists, before the
# run starts. Tests keep the "baseline to beat" line stable. Diagnostic only: any
# failure here must never fail the benchmark run.
report_local_baseline_context() {
  if [[ "${LOCAL_ITERATE}" != "1" && "${LOCAL_SUBMIT}" != "1" ]]; then
    return 0
  fi
  local baseline_path="${SCORE_PATH%.json}.baseline.json"
  if [[ ! -f "${baseline_path}" ]]; then
    return 0
  fi
  local context
  context="$(jq -r '
    def r3: . * 1000 | round / 1000;
    def r6: . * 1000000 | round / 1000000;
    (.metrics // {}) as $m
    | ($m.prefill_seconds_per_token // 0) as $p
    | ($m.decode_seconds_per_token // 0) as $d
    | select($p > 0 and $d > 0)
    | ($m.prefill_speedup // 0) as $ps
    | ($m.decode_speedup // 0) as $ds
    | (if $ps > 0 and $ds > 0 then pow($ds; 0.75) * pow($ps; 0.25) else 0 end) as $est
    | "prefill \($p | r6) s/token, decode \($d | r6) s/token"
      + (if $est > 0 then ", est score \($est | r3)" else "" end)
  ' "${baseline_path}" 2>/dev/null || true)"
  if [[ -n "${context}" ]]; then
    echo "benchmark.sh: local baseline to beat (${baseline_path}): ${context}" >&2
  fi
  return 0
}

# Local modes end with a compact human-readable summary on stderr. Tests keep
# this output stable. Diagnostic
# only: any failure here must never fail the benchmark run.
report_local_score_summary() {
  if [[ "${LOCAL_ITERATE}" != "1" && "${LOCAL_SUBMIT}" != "1" ]]; then
    return 0
  fi
  local mode_name="local-submit"
  if [[ "${LOCAL_ITERATE}" == "1" ]]; then
    mode_name="local-iterate"
  fi

  local summary
  summary="$(jq -r '
    def r3: . * 1000 | round / 1000;
    def r6: . * 1000000 | round / 1000000;
    (.metrics // {}) as $m
    | ($m.prefill_seconds_per_token // 0) as $p
    | ($m.decode_seconds_per_token // 0) as $d
    | select($p > 0 and $d > 0)
    | ($m.prefill_speedup // 0) as $ps
    | ($m.decode_speedup // 0) as $ds
    | (if $ps > 0 and $ds > 0 then pow($ds; 0.75) * pow($ps; 0.25) else 0 end) as $est
    | "  prefill \($p | r6) s/token  speedup \($ps | r3)x\n"
      + "  decode  \($d | r6) s/token  speedup \($ds | r3)x"
      + (if $est > 0
         then "\n  est score \($est | r3) (decode_speedup^0.75 * prefill_speedup^0.25; official score comes from the ranked runner)"
         else "" end)
  ' "${SCORE_PATH}" 2>/dev/null || true)"
  if [[ -z "${summary}" ]]; then
    return 0
  fi
  {
    echo "benchmark.sh: ${mode_name} summary"
    printf '%s\n' "${summary}"
  } >&2

  local baseline_path="${SCORE_PATH%.json}.baseline.json"
  if [[ ! -f "${baseline_path}" ]]; then
    if [[ "${LOCAL_ITERATE}" == "1" ]]; then
      echo "benchmark.sh: no local baseline at ${baseline_path}; run 'cp ${SCORE_PATH} ${baseline_path}' to compare future runs" >&2
    fi
    return 0
  fi
  local compare
  compare="$(jq -r -n --slurpfile cur "${SCORE_PATH}" --slurpfile base "${baseline_path}" '
    def r1: . * 10 | round / 10;
    def r3: . * 1000 | round / 1000;
    def r6: . * 1000000 | round / 1000000;
    def sign: if . >= 0 then "+" else "" end;
    ($cur[0].metrics // {}) as $c
    | ($base[0].metrics // {}) as $b
    | ($c.prefill_seconds_per_token // 0) as $cp
    | ($c.decode_seconds_per_token // 0) as $cd
    | ($b.prefill_seconds_per_token // 0) as $bp
    | ($b.decode_seconds_per_token // 0) as $bd
    | select($cp > 0 and $cd > 0 and $bp > 0 and $bd > 0)
    | (($cp - $bp) / $bp * 100) as $pdelta
    | (($cd - $bd) / $bd * 100) as $ddelta
    | ($c.prefill_speedup // 0) as $cps
    | ($c.decode_speedup // 0) as $cds
    | ($b.prefill_speedup // 0) as $bps
    | ($b.decode_speedup // 0) as $bds
    | (if $cps > 0 and $cds > 0 then pow($cds; 0.75) * pow($cps; 0.25) else 0 end) as $cest
    | (if $bps > 0 and $bds > 0 then pow($bds; 0.75) * pow($bps; 0.25) else 0 end) as $best
    | "    prefill \($bp | r6) -> \($cp | r6) s/token (\($pdelta | sign)\($pdelta | r1)%)\n"
      + "    decode  \($bd | r6) -> \($cd | r6) s/token (\($ddelta | sign)\($ddelta | r1)%)"
      + (if $cest > 0 and $best > 0
         then (((($cest - $best) / $best) * 100) as $edelta
           | "\n    est score \($best | r3) -> \($cest | r3) (\($edelta | sign)\($edelta | r1)%)")
         else "" end)
  ' 2>/dev/null || true)"
  if [[ -z "${compare}" ]]; then
    return 0
  fi
  {
    echo "benchmark.sh: vs ${baseline_path} (negative s/token deltas = faster)"
    printf '%s\n' "${compare}"
  } >&2
}

# Official mode uses the runtime-worker sandbox and refuses without it.
USE_RUNTIME_WORKER="${MLXFAST_USE_RUNTIME_WORKER:-1}"

# Official mode requires the sandbox and runtime worker. MLXFAST_PRIVATE_DIR
# is only a sandbox deny rule
# inside the worker profile, never a dependency, so it is NOT required here.
enforce_official_sandbox() {
  if [[ "${OFFICIAL}" != "1" ]]; then
    return 0
  fi
  if [[ "${MLXFAST_NO_SANDBOX:-0}" == "1" ]]; then
    echo "benchmark.sh: official GitHub benchmark runs must not set MLXFAST_NO_SANDBOX=1" >&2
    exit 1
  fi
  if [[ "${USE_RUNTIME_WORKER}" != "1" ]]; then
    echo "benchmark.sh: official GitHub benchmark runs must use the runtime worker sandbox" >&2
    exit 1
  fi
}

# ---- Dispatch to benchd -------------------------------------------------------
# The MLX engine binary (Engine Protocol v1) benchd spawns. Facade-specific and
# Required because benchd starts a separate engine process.
if [[ -z "${MLXFAST_ENGINE_BIN:-}" ]]; then
  echo "benchmark.sh: MLXFAST_ENGINE_BIN must point to the benchd engine binary" >&2
  exit 1
fi

FACADE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# TRACK ID. The variable name belongs to the harness protocol: benchd reads
# this exact spelling for each track it scores. `benchd iterate` requires
# MLXFAST_QWEN_MTP_TRACK_ID in every mode. Its `-{platform}-v{N}` suffix keys the
# official baseline pair and acceptance bands, and the official path resolves
# it before the gates-only branch. The value is the trusted manifest's
# `trackId` (the platform id, qwen3.8-27b-tf-mlx-v1 -- not the bench
# channel branch, which names the channel; see tools/fetch-benchd.sh).
# It is read from benchmark.json beside this file's parent, never taken from the
# caller: benchd treats env != contract as a hard error, so a caller that pre-set a
# DIFFERENT value is refused here rather than silently overridden, and a manifest
# with no trackId refuses rather than letting benchd guess a platform.
MANIFEST_TRACK_ID="$(jq -r '.trackId // empty' "${FACADE_DIR}/../benchmark.json" 2>/dev/null || true)"
if [[ -z "${MANIFEST_TRACK_ID}" ]]; then
  echo "benchmark.sh: benchmark.json carries no trackId; benchd iterate requires MLXFAST_QWEN_MTP_TRACK_ID and this facade will not guess one" >&2
  exit 1
fi
if [[ -n "${MLXFAST_QWEN_MTP_TRACK_ID:-}" && "${MLXFAST_QWEN_MTP_TRACK_ID}" != "${MANIFEST_TRACK_ID}" ]]; then
  echo "benchmark.sh: MLXFAST_QWEN_MTP_TRACK_ID is set to '${MLXFAST_QWEN_MTP_TRACK_ID}' but benchmark.json trackId is '${MANIFEST_TRACK_ID}'; the track id is ONE value and this facade will not override either" >&2
  exit 1
fi
export MLXFAST_QWEN_MTP_TRACK_ID="${MANIFEST_TRACK_ID}"

# BENCHD resolution. The default is the binary fetch-benchd.sh verifies against
# the channel manifest, so a bare run cannot score itself with an unpinned
# harness. fetch-benchd.sh is a no-op when
# benchd-bin/benchd is already present and matching, so this costs one hash on
# the warm path and works fully offline. An explicit BENCHD= override is still
# honoured (benchd development, bisecting a harness change) and is deliberately
# NOT hash-checked — it is the caller saying "I mean this other binary".
# BENCHCTL is accepted as a compatibility alias.
BENCHD="${BENCHD:-${BENCHCTL:-}}"
if [[ -z "${BENCHD:-}" ]]; then
  BENCHD="$("${FACADE_DIR}/fetch-benchd.sh")"
fi
if [[ "${OFFICIAL}" == "1" ]]; then
  MODE="official"
elif [[ "${LOCAL_ITERATE}" == "1" ]]; then
  MODE="local-iterate"
else
  MODE="local-submit"
fi

# Note the sidecar the facade would use for its own bookkeeping; benchd writes the
# authoritative per-mode integrity sidecar itself, so the facade does not re-seal it.
: "${INTEGRITY_PATH}"

# Print the same-machine baseline snapshot (when one exists) before the run starts,
# This is diagnostic only.
report_local_baseline_context || true

# Assemble the benchd iterate argv. official mode routes to benchd's official backend
# (`--mode official`) with the FULL methodology (sandboxed workers, correctness vs the supplied
# golden, official floor/band gating, timed + seal; MLXFAST_BENCHMARK_SKIP_TIMED=1 makes it a
# gates-only run). The benchd official backend loads --weights DIRECTLY — no reference-
# checkpoint regeneration — so the facade needs no cached reference weights. Official has no local
# cool gate (its gate is operator-side / thermal), so --cool-gate is LOCAL-only. When the golden
# is fully pinned (sha256 AND bytes), forward the pin so benchd re-verifies the raw bytes.
benchd_args=(
  --engine "${MLXFAST_ENGINE_BIN}"
  --weights "${WEIGHTS_PATH}"
  --golden "${GOLDEN_PATH}"
  --mode "${MODE}"
  --score-path "${SCORE_PATH}"
)
if [[ -n "${MLXFAST_CORRECTNESS_GOLDEN_SHA256:-}" && -n "${MLXFAST_CORRECTNESS_GOLDEN_BYTES:-}" ]]; then
  benchd_args+=(--golden-sha256 "${MLXFAST_CORRECTNESS_GOLDEN_SHA256}" --golden-bytes "${MLXFAST_CORRECTNESS_GOLDEN_BYTES}")
fi
# THE CONTRACT, ON EVERY MODE. benchd reads the measurement window, the model
# identity and the scoring values from the track fixture and refuses a run it
# was not given one for (WINDOW-SHAPE-UNDECLARED-FOR-TRACK), local modes
# included. The fixture is fixtures/qwen3_8_27b_tf_mlx_v1_track.json.
CONTRACT_FIXTURE="${FACADE_DIR}/../fixtures/qwen3_8_27b_tf_mlx_v1_track.json"
if [[ ! -f "${CONTRACT_FIXTURE}" ]]; then
  echo "benchmark.sh: every run requires the track contract fixture at ${CONTRACT_FIXTURE}" >&2
  exit 1
fi
CONTRACT_ARG="${CONTRACT_FIXTURE}"
if [[ "${OFFICIAL}" != "1" ]]; then
  # These guards describe the paired control run. Candidate-only local runs
  # have no control leg, so benchd refuses them. Keep the full fixture for
  # ranked and calibration paths; give local runs a short-lived view instead.
  LOCAL_RUN_TEMP="$(mktemp -d "${TMPDIR:-/tmp}/mlxfast-benchmark.XXXXXX")"
  chmod 700 "${LOCAL_RUN_TEMP}"
  trap 'rm -rf "${LOCAL_RUN_TEMP}"' EXIT
  CONTRACT_ARG="${LOCAL_RUN_TEMP}/contract.json"
  jq 'del(.warmup_golden, .decode_golden_offset)' \
    "${CONTRACT_FIXTURE}" > "${CONTRACT_ARG}"
fi
benchd_args+=(--contract "${CONTRACT_ARG}")
# THE SPECULATIVE LEG. Convert the DFlash selector to an explicit protocol spec.
if [[ -n "${DFLASH_DEPTH}" ]]; then
  [[ "${DFLASH_DEPTH}" == "1" ]] || { echo "benchmark.sh: --dflash-depth must be 1 (got '${DFLASH_DEPTH}')" >&2; exit 1; }
  benchd_args+=(--candidate-spec "{\"mode\":\"dflash\",\"dflash\":{\"depth\":${DFLASH_DEPTH}}}")
fi
if [[ "${OFFICIAL}" == "1" ]]; then
  enforce_official_sandbox
  export MLXFAST_USE_RUNTIME_WORKER="${USE_RUNTIME_WORKER}"
  # Official mode seals only when the fixture
  # declares official_scoring_enabled: true.
else
  benchd_args+=(--cool-gate)
fi

# Run benchd. Do NOT redirect its stdout: benchd emits the sealed JSON payload on
# stdout and the facade passes it through untouched (no cat / re-emit / re-seal).
# Capture the exit code without tripping `set -e`.
rc=0
"${BENCHD}" iterate "${benchd_args[@]}" || rc=$?

# Diagnostic-only summary — must NEVER change the run's exit code.
report_local_score_summary || true

# Exit-code mapping: benchd iterate returns 0 = pass, 1 =
# fail/exec-error, 2 = usage/bad-args. Pass 0 and 1 through; map benchd's native
# usage 2 -> 1 (the reference benchmark.sh uses exit 1 for usage).
if [[ "${rc}" == "2" ]]; then
  exit 1
fi
exit "${rc}"
