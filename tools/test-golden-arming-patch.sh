#!/usr/bin/env bash
# Unit test for tools/golden-arming-patch.py. Offline and hermetic: synthetic
# golden files in a temporary directory, a copy of the contract fixture for the
# --apply case, no R2, no box, no real golden.
#
# The synthetic set has the shape the depth rule must handle, over two live
# goldens. Alpha's dflash1 is byte-identical to its serial tape. Bravo's
# dflash1 differs from its tape. Each key gets a named per-depth object, even
# when its bytes equal the serial tape. The two public captures are not
# uploads: they ship in git.
#
# Usage: tools/test-golden-arming-patch.sh
# Exit:  0 all cases pass, 1 a case failed (printed with a FAIL prefix)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
TOOL="${SCRIPT_DIR}/tools/golden-arming-patch.py"
CONTRACT="${SCRIPT_DIR}/fixtures/qwen3_8_27b_tf_mlx_v1_track.json"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

failures=0
fail() { echo "FAIL: $1" >&2; failures=$((failures + 1)); }
pass() { echo "ok: $1"; }

# The marker is in every synthetic file. It must never reach the tool's output.
MARKER="SYNTHETIC-CONTENT-MARKER-7f3a"
golden() {
  local value
  value="$(printf '%s' "$2" | cksum | awk '{print ($1 % 1000) + 1}')"
  printf '{"seed_tokens":[%s],"reference_seed_token":%s,"rows":[{"sequential_argmax":%s}],"reference_self_consistent":true,"emitted_tokens":[%s]}\n' \
    "${value}" "${value}" "${value}" "${value}" > "$1"
}

make_set() { # make_set DIR
  mkdir -p "$1"
  local n
  for n in alpha bravo charlie; do
    golden "$1/${n}.golden.json" "${n}"
  done
  golden "$1/warmup-01.golden.json" "warmup"
  cp "$1/alpha.golden.json" "$1/alpha.dflash1.golden.json"
  golden "$1/bravo.dflash1.golden.json" "speculative-bravo"
  golden "$1/public_longcopy_gate_english_1024_256.json" "public-short"
  golden "$1/public_longcopy_gate_english_1024_1024.json" "public-long"
}

make_set "${WORK}/set"
q() { python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(eval(sys.argv[2]))" "$@"; }
unarm_contract() { # unarm_contract PATH
  python3 - "$1" <<'PY'
import json, sys
path = sys.argv[1]
document = json.load(open(path))
document["official_scoring_enabled"] = False
json.dump(document, open(path, "w"), indent=2)
PY
}

# --- case 1: the patch arms every key with its required named object ---------
LIVES=(--live alpha --live bravo)
if python3 "${TOOL}" --dir "${WORK}/set" "${LIVES[@]}" > "${WORK}/patch.json" 2> "${WORK}/case1.err"; then
  P="${WORK}/patch.json"
  pre="correctness_prompts/qwen3.8-27b-tf-mlx-v1"
  ok1=1
  [[ "$(q "${P}" 'd["official_scoring_enabled"]')" == "True" ]] || { fail "case 1: official_scoring_enabled is not true"; ok1=0; }
  [[ "$(q "${P}" 'len(d["timed_prompt_pool"])')" == "3" ]] || { fail "case 1: the pool is not 3 tapes"; ok1=0; }
  [[ "$(q "${P}" 'd["live_goldens"]')" == "['alpha', 'bravo']" ]] || { fail "case 1: live_goldens is not alpha, bravo"; ok1=0; }
  [[ "$(q "${P}" '[e for e in d["timed_prompt_pool"] if e["r2_path"].endswith("/alpha.golden.json")][0]["sha256"] == d["hidden_correctness_golden"]["sha256"]')" == "True" ]] \
    || { fail "case 1: the hidden oracle is not the live golden's own pin"; ok1=0; }
  [[ "$(q "${P}" 'd["warmup_golden"]["r2_path"].endswith("/warmup-01.golden.json") and isinstance(d["warmup_golden"]["bytes"], int)')" == "True" ]] \
    || { fail "case 1: the warm-up golden is not pinned separately"; ok1=0; }
  [[ "$(q "${P}" 'sorted(d["live_golden_speculative"])')" == "['dflash1']" ]] \
    || { fail "case 1: live_golden_speculative does not carry depth 1"; ok1=0; }
  [[ "$(q "${P}" 'all(sorted(v) == ["alpha", "bravo"] for v in d["live_golden_speculative"].values())')" == "True" ]] \
    || { fail "case 1: a depth does not carry one pin per live golden"; ok1=0; }
  [[ "$(q "${P}" 'd["live_golden_speculative"]["dflash1"]["alpha"]["r2_path"]')" == "${pre}/alpha.dflash1.golden.json" ]] \
    || { fail "case 1: alpha dflash1 does not name its oracle"; ok1=0; }
  [[ "$(q "${P}" 'd["live_golden_speculative"]["dflash1"]["alpha"]["sha256"] == [e for e in d["timed_prompt_pool"] if e["r2_path"].endswith("/alpha.golden.json")][0]["sha256"] and d["live_golden_speculative"]["dflash1"]["alpha"]["bytes"] == [e for e in d["timed_prompt_pool"] if e["r2_path"].endswith("/alpha.golden.json")][0]["bytes"]')" == "True" ]] \
    || { fail "case 1: serial-identical alpha pins differ"; ok1=0; }
  [[ "$(q "${P}" 'd["live_golden_speculative"]["dflash1"]["bravo"]["r2_path"]')" == "${pre}/bravo.dflash1.golden.json" ]] \
    || { fail "case 1: bravo dflash1 does not name its oracle"; ok1=0; }
  [[ "$(q "${P}" '"public_captures" in d')" == "False" ]] \
    || { fail "case 1: the patch carries public_captures; the captures ship in git, not in the fixture"; ok1=0; }
  grep -q "ship: public_longcopy_gate_english_1024_1024.json -> correctness_prompts/public_longcopy_gate_english_1024_1024.json" "${WORK}/case1.err" \
    || { fail "case 1: the local-submit capture is not listed at its repository path"; ok1=0; }
  grep -q "6 distinct object(s); 2 per-depth oracle object(s)" "${WORK}/case1.err" \
    || { fail "case 1: the upload summary is wrong ($(tail -1 "${WORK}/case1.err"))"; ok1=0; }
  if grep -q "${MARKER}" "${P}" "${WORK}/case1.err"; then
    fail "case 1: the output carries golden CONTENT"; ok1=0
  fi
  [[ "${ok1}" == "1" ]] && pass "the patch arms all keys with named objects and prints no content"
else
  fail "case 1: the tool refused a complete set ($(cat "${WORK}/case1.err"))"
fi

# --- case 2: serial-identical oracles keep their per-depth object names -------
make_set "${WORK}/serial"
for live in alpha bravo; do
  for depth in 1; do
    cp "${WORK}/serial/${live}.golden.json" "${WORK}/serial/${live}.dflash${depth}.golden.json"
  done
done
if python3 "${TOOL}" --dir "${WORK}/serial" "${LIVES[@]}" > "${WORK}/serial-patch.json" 2> "${WORK}/case2.err"; then
  if [[ "$(q "${WORK}/serial-patch.json" 'all(e["r2_path"].endswith("/" + live + ".dflash1.golden.json") for per_depth in d["live_golden_speculative"].values() for live, e in per_depth.items())')" == "True" ]]; then
    pass "all serial-identical DFlash oracles keep their per-depth object names"
  else
    fail "case 2: a DFlash oracle does not keep its per-depth object name"
  fi
else
  fail "case 2: the tool refused the current all-serial shape ($(cat "${WORK}/case2.err"))"
fi

# --- case 3: a missing per-depth oracle is refused by key --------------------
make_set "${WORK}/missing"
rm "${WORK}/missing/alpha.dflash1.golden.json"
if python3 "${TOOL}" --dir "${WORK}/missing" "${LIVES[@]}" > /dev/null 2> "${WORK}/case3.err"; then
  fail "case 3: the tool accepted a set with no dflash1 oracle"
elif grep -q "no per-depth oracle for: dflash1 (alpha)" "${WORK}/case3.err"; then
  pass "a missing per-depth oracle is refused by key"
else
  fail "case 3: refused for the wrong reason ($(cat "${WORK}/case3.err"))"
fi

# --- case 4: a re-indented hidden tape used as public is refused -------------
make_set "${WORK}/leak"
python3 - "${CONTRACT}" "${WORK}/leak/charlie.golden.json" "${WORK}/leak/public_longcopy_gate_english_1024_256.json" <<'LEAKEOF'
import json, sys
contract = json.load(open(sys.argv[1]))
d = {
    "version": 1,
    "model_type": contract["target"]["model_type"],
    "model_provenance": {
        "repository": contract["target"]["upstream_model_id"],
        "revision": contract["target"]["upstream_revision"],
    },
    "cases": [{
        "name": "hidden-charlie",
        "prompt_tokens": [101] * contract["seed_tokens"],
        "expected_tokens": [201] * contract["correctness_steps"],
    }],
}
json.dump(d, open(sys.argv[2], "w"), separators=(",", ":"))
json.dump(d, open(sys.argv[3], "w"), indent=4)
LEAKEOF
if python3 "${TOOL}" --dir "${WORK}/leak" "${LIVES[@]}" > /dev/null 2> "${WORK}/case4.err"; then
  fail "case 4: the tool accepted a re-indented hidden tape as a public capture"
elif grep -q "same .* as a hidden tape" "${WORK}/case4.err"; then
  pass "a re-indented hidden tape used as a public capture is refused"
else
  fail "case 4: refused for the wrong reason ($(cat "${WORK}/case4.err"))"
fi

# --- case 5: live count must divide official pairs ---------------------------
make_set "${WORK}/two-live"
cp "${WORK}/two-live/charlie.golden.json" "${WORK}/two-live/charlie.dflash1.golden.json"
if python3 "${TOOL}" --dir "${WORK}/two-live" --live alpha --live bravo --live charlie > /dev/null 2> "${WORK}/case5.err"; then
  fail "case 5: the tool accepted 3 live goldens for 2 official pairs"
elif grep -q "official_pairs 2 is not divisible by the 3 live goldens" "${WORK}/case5.err"; then
  pass "a live count that does not divide official_pairs is refused"
else
  fail "case 5: refused for the wrong reason ($(cat "${WORK}/case5.err"))"
fi

# --- case 6: a stored baseline pair is refused -------------------------------
make_set "${WORK}/pair"
printf '{"seed_tokens":[1],"reference_seed_token":2,"rows":[{"sequential_argmax":3}],"reference_self_consistent":true,"benchmark":{"baseline_decode_seconds_per_token":0.03}}\n' > "${WORK}/pair/delta.golden.json"
if python3 "${TOOL}" --dir "${WORK}/pair" "${LIVES[@]}" > /dev/null 2> "${WORK}/case6.err"; then
  fail "case 6: the tool accepted a tape that stores a baseline pair"
elif grep -q "stores a baseline pair" "${WORK}/case6.err"; then
  pass "a tape that stores a baseline pair is refused"
else
  fail "case 6: refused for the wrong reason ($(cat "${WORK}/case6.err"))"
fi

# --- case 7: --apply replaces disabled scoring and pending golden pins -------
# The ranked preflight refuses any PENDING-ORGANIZER in the contract, so an
# applied patch must leave none in the fields it owns. The target and head pins
# are not this tool's: while they are pending they carry the sentinel too, so
# the copy stands in synthetic values for them first, the way a re-stamp onto
# the published pack would.
python3 - "${CONTRACT}" "${WORK}/contract.json" <<'UNPENDEOF'
import json, sys
def unpend(node):
    if isinstance(node, dict):
        return {k: unpend(v) for k, v in node.items()}
    if isinstance(node, list):
        return [unpend(v) for v in node]
    if isinstance(node, str) and "PENDING-ORGANIZER" in node:
        return "synthetic-pin"
    return node
d = json.load(open(sys.argv[1]))
d = unpend(d)
d["official_scoring_enabled"] = False
d["live_goldens"] = ["PENDING-ORGANIZER"]
d["timed_prompt_pool"] = [{
    "r2_path": "PENDING-ORGANIZER",
    "sha256": "PENDING-ORGANIZER",
    "bytes": 0,
}]
d["hidden_correctness_golden"] = {
    "sha256": "PENDING-ORGANIZER",
    "bytes": 0,
}
d["warmup_golden"] = {
    "r2_path": "correctness_prompts/qwen3.8-27b-tf-mlx-v1/warmup-01.golden.json",
    "sha256": "PENDING-ORGANIZER",
    "bytes": 0,
}
d["live_golden_speculative"] = {"PENDING-ORGANIZER": {}}
json.dump(d, open(sys.argv[2], "w"), indent=2)
UNPENDEOF
if python3 "${TOOL}" --dir "${WORK}/set" "${LIVES[@]}" --apply --contract "${WORK}/contract.json" 2> "${WORK}/case7.err"; then
  if grep -q "PENDING-ORGANIZER" "${WORK}/contract.json"; then
    fail "case 7: the applied contract still carries a pending sentinel"
  elif [[ "$(q "${WORK}/contract.json" 'd["official_scoring_enabled"]')" != "True" ]]; then
    fail "case 7: --apply did not enable official scoring"
  elif [[ "$(q "${WORK}/contract.json" 'd["live_goldens"]')" != "['alpha', 'bravo']" ]]; then
    fail "case 7: --apply did not write the live goldens"
  elif [[ "$(q "${WORK}/contract.json" 'len(d["timed_prompt_pool"])')" != "3" ]]; then
    fail "case 7: --apply did not write the timed pool pins"
  elif [[ "$(q "${WORK}/contract.json" 'isinstance(d["warmup_golden"]["bytes"], int) and d["warmup_golden"]["bytes"] > 0')" != "True" ]]; then
    fail "case 7: --apply did not write the warm-up pin"
  elif [[ "$(q "${WORK}/contract.json" 'sorted(d["live_golden_speculative"])')" != "['dflash1']" ]]; then
    fail "case 7: --apply did not write the speculative pins"
  elif [[ "$(q "${WORK}/contract.json" 'd["track_id"]')" != "qwen3.8-27b-tf-mlx-v1" ]]; then
    fail "case 7: --apply lost a fixture key it does not own"
  else
    pass "--apply replaces disabled scoring and pending golden pins"
  fi
else
  fail "case 7: --apply refused ($(cat "${WORK}/case7.err"))"
fi

# --- case 8: marker-only JSON is not a golden and cannot arm scoring --------
make_set "${WORK}/missing-fields"
printf '{"version":1,"marker":"not-a-golden"}\n' > "${WORK}/missing-fields/charlie.golden.json"
if python3 "${TOOL}" --dir "${WORK}/missing-fields" "${LIVES[@]}" > /dev/null 2> "${WORK}/case8.err"; then
  fail "case 8: marker-only JSON armed scoring"
elif grep -q "lacks golden fields" "${WORK}/case8.err"; then
  pass "JSON without golden fields is refused"
else
  fail "case 8: refused for the wrong reason ($(cat "${WORK}/case8.err"))"
fi

assert_invalid_apply() { # NAME PYTHON_MUTATION EXPECTED_DIAGNOSTIC
  local name="$1" mutation="$2" diagnostic="$3"
  local dir="${WORK}/${name}"
  make_set "${dir}"
  cp "${CONTRACT}" "${dir}.contract.json"
  unarm_contract "${dir}.contract.json"
  python3 - "${dir}/alpha.golden.json" "${mutation}" <<'PY'
import json, sys
path, mutation = sys.argv[1:]
document = json.load(open(path))
exec(mutation, {"document": document})
json.dump(document, open(path, "w"))
PY
  if python3 "${TOOL}" --dir "${dir}" "${LIVES[@]}" --apply \
      --contract "${dir}.contract.json" > /dev/null 2> "${dir}.err"; then
    fail "${name}: invalid golden armed scoring"
  elif [[ "$(q "${dir}.contract.json" 'd["official_scoring_enabled"]')" != "False" ]]; then
    fail "${name}: failed arming changed official_scoring_enabled"
  elif grep -q "${diagnostic}" "${dir}.err"; then
    pass "${name} is refused and scoring stays disabled"
  else
    fail "${name}: refused for the wrong reason ($(cat "${dir}.err"))"
  fi
}

# --- cases 9-11: parent semantic validation runs before arming --------------
assert_invalid_apply "string token id" \
  'document["seed_tokens"][0] = "1"' "token ids must be non-negative integers"
assert_invalid_apply "negative token id" \
  'document["rows"][0]["sequential_argmax"] = -1' "token ids must be non-negative integers"
assert_invalid_apply "self-inconsistent reference" \
  'document["reference_self_consistent"] = False' "reference_self_consistent=false"

write_legacy_golden() { # PATH
  python3 - "${CONTRACT}" "$1" <<'PY'
import json, sys
contract = json.load(open(sys.argv[1]))
document = {
    "version": 1,
    "model_type": contract["target"]["model_type"],
    "model_provenance": {
        "repository": contract["target"]["upstream_model_id"],
        "revision": contract["target"]["upstream_revision"],
    },
    "cases": [{
        "name": "legacy-case",
        "prompt_tokens": [1] * contract["seed_tokens"],
        "expected_tokens": [2] * contract["correctness_steps"],
    }],
}
json.dump(document, open(sys.argv[2], "w"))
PY
}

assert_invalid_legacy_apply() { # NAME PYTHON_MUTATION EXPECTED_DIAGNOSTIC
  local name="$1" mutation="$2" diagnostic="$3"
  local slug dir
  slug="$(printf '%s' "${name}" | tr ' /' '--')"
  dir="${WORK}/legacy-${slug}"
  make_set "${dir}"
  write_legacy_golden "${dir}/alpha.golden.json"
  cp "${CONTRACT}" "${dir}.contract.json"
  unarm_contract "${dir}.contract.json"
  python3 - "${CONTRACT}" "${dir}/alpha.golden.json" "${mutation}" <<'PY'
import json, sys
contract = json.load(open(sys.argv[1]))
path, mutation = sys.argv[2:]
document = json.load(open(path))
exec(mutation, {"document": document, "contract": contract})
json.dump(document, open(path, "w"))
PY
  if python3 "${TOOL}" --dir "${dir}" "${LIVES[@]}" --apply \
      --contract "${dir}.contract.json" > /dev/null 2> "${dir}.err"; then
    fail "${name}: invalid legacy golden armed scoring"
  elif [[ "$(q "${dir}.contract.json" 'd["official_scoring_enabled"]')" != "False" ]]; then
    fail "${name}: failed arming changed official_scoring_enabled"
  elif grep -q "${diagnostic}" "${dir}.err"; then
    pass "${name} is refused and scoring stays disabled"
  else
    fail "${name}: refused for the wrong reason ($(cat "${dir}.err"))"
  fi
}

# --- parent legacy-golden checks: one refusal per validation rule ------------
legacy_cases=(
  'legacy version|document["version"] = 999|version must be 1'
  'mixed schema bypass|document.update(seed_tokens=[1], reference_seed_token=1, rows=[{"sequential_argmax": 1}], reference_self_consistent=True); document["version"] = 999; document["cases"].append(dict(document["cases"][0]))|mixes the tape schema and the cases schema'
  'legacy duplicate names|document["cases"].append(dict(document["cases"][0]))|duplicate correctness golden case name'
  'legacy unknown root|document["extra"] = 1|unknown golden field'
  'legacy missing case field|document["cases"][0].pop("expected_tokens")|missing expected_tokens'
  'legacy case name type|document["cases"][0]["name"] = 7|name: must be a string'
  'legacy case name whitespace|document["cases"][0]["name"] = " legacy"|surrounding whitespace'
  'legacy case name control|document["cases"][0]["name"] = "legacy\ncase"|control characters'
  'legacy prompt type|document["cases"][0]["prompt_tokens"] = "bad"|invalid legacy-case.prompt_tokens'
  'legacy prompt count|document["cases"][0]["prompt_tokens"] = [1]|need exactly'
  'legacy expected count|document["cases"][0]["expected_tokens"] = [2]|need at least'
  'legacy model type missing|document.pop("model_type")|model_type must be'
  'legacy model type wrong|document["model_type"] = "wrong"|model_type does not match'
  'legacy provenance missing|document.pop("model_provenance")|model_provenance must be a JSON object'
  'legacy provenance fields|document["model_provenance"].pop("revision")|requires only repository and revision'
  'legacy provenance revision|document["model_provenance"]["revision"] = "BAD"|40-character lowercase revision'
  'legacy provenance pin|document["model_provenance"]["repository"] = "wrong/model"|does not match the pinned reference model'
  'legacy gates null|document["correctness_gates"] = None|correctness_gates must be a JSON object'
  'legacy gates empty|document["correctness_gates"] = {}|at least one gate section'
  'legacy gates unknown|document["correctness_gates"] = {"anchorz": []}|unknown field'
  'legacy gate section empty|document["correctness_gates"] = {"anchors": []}|must be a non-empty array'
  'legacy anchor required field|document["correctness_gates"] = {"anchors": [{"name": "anchor"}]}|missing context_tokens, expected_token'
  'legacy anchor context empty|document["correctness_gates"] = {"anchors": [{"name": "anchor", "context_tokens": [], "expected_token": 1}]}|invalid anchor.context_tokens'
  'legacy anchor accepted empty|document["correctness_gates"] = {"anchors": [{"name": "anchor", "context_tokens": [1], "expected_token": 1, "accepted_tokens": []}]}|invalid anchor.accepted_tokens'
  'legacy anchor rank|document["correctness_gates"] = {"anchors": [{"name": "anchor", "context_tokens": [1], "expected_token": 1, "max_expected_rank": 9}]}|max_expected_rank must be'
  'legacy anchor delta dependency|document["correctness_gates"] = {"anchors": [{"name": "anchor", "context_tokens": [1], "expected_token": 1, "max_top_logit_delta": 1.0}]}|requires max_expected_rank'
  'legacy layered duplicate|document["correctness_gates"] = {"anchors": [{"name": "legacy-case", "context_tokens": [1], "expected_token": 1}]}|duplicate layered correctness case name'
  'legacy free-run prompt count|document["correctness_gates"] = {"free_run": [{"name": "free", "prompt_tokens": [1], "expected_tokens": [2]}]}|must have exactly'
  'legacy free-run expected empty|document["correctness_gates"] = {"free_run": [{"name": "free", "prompt_tokens": [1] * contract["seed_tokens"], "expected_tokens": []}]}|invalid free.expected_tokens'
  'legacy free-run prefix|document["correctness_gates"] = {"free_run": [{"name": "free", "prompt_tokens": [1] * contract["seed_tokens"], "expected_tokens": [2], "exact_prefix_tokens": 2}]}|outside expected_tokens'
  'legacy behavior prompt empty|document["correctness_gates"] = {"behavior": [{"name": "behavior", "prompt_tokens": [], "accepted_token_sequences": [[1]], "max_new_tokens": 1}]}|invalid behavior.prompt_tokens'
  'legacy behavior max|document["correctness_gates"] = {"behavior": [{"name": "behavior", "prompt_tokens": [1], "accepted_token_sequences": [[1]], "max_new_tokens": 129}]}|outside the permitted range'
  'legacy behavior sequences empty|document["correctness_gates"] = {"behavior": [{"name": "behavior", "prompt_tokens": [1], "accepted_token_sequences": [], "max_new_tokens": 1}]}|accepted_token_sequences must not be empty'
  'legacy behavior sequence long|document["correctness_gates"] = {"behavior": [{"name": "behavior", "prompt_tokens": [1], "accepted_token_sequences": [[1, 2]], "max_new_tokens": 1}]}|exceeds max_new_tokens'
  'legacy behavior semantic type|document["correctness_gates"] = {"behavior": [{"name": "behavior", "prompt_tokens": [1], "accepted_token_sequences": [[1]], "max_new_tokens": 1, "semantic_prompt": 7}]}|semantic_prompt must be a string'
  'legacy benchmark null|document["benchmark"] = None|benchmark must be a JSON object'
  'legacy benchmark fields|document["benchmark"] = {}|benchmark has invalid fields'
  'legacy benchmark seed count|document["benchmark"] = {"prefill_prompt_tokens": [1], "expected_prefill_token": 1, "decode_seed_tokens": [1] * contract["seed_tokens"], "expected_decode_seed_token": 1, "expected_decode_tokens": [1] * contract["benchmark_decode_steps"]}|prefill_prompt_tokens must have exactly'
  'legacy benchmark decode count|document["benchmark"] = {"prefill_prompt_tokens": [1] * contract["seed_tokens"], "expected_prefill_token": 1, "decode_seed_tokens": [1] * contract["seed_tokens"], "expected_decode_seed_token": 1, "expected_decode_tokens": [1]}|expected_decode_tokens is too short'
  'legacy benchmark half baseline|document["benchmark"] = {"prefill_prompt_tokens": [1] * contract["seed_tokens"], "expected_prefill_token": 1, "decode_seed_tokens": [1] * contract["seed_tokens"], "expected_decode_seed_token": 1, "expected_decode_tokens": [1] * contract["benchmark_decode_steps"], "baseline_prefill_seconds_per_token": 1.0}|baseline fields must be provided together'
  'legacy benchmark negative baseline|document["benchmark"] = {"prefill_prompt_tokens": [1] * contract["seed_tokens"], "expected_prefill_token": 1, "decode_seed_tokens": [1] * contract["seed_tokens"], "expected_decode_seed_token": 1, "expected_decode_tokens": [1] * contract["benchmark_decode_steps"], "baseline_prefill_seconds_per_token": -1.0, "baseline_decode_seconds_per_token": 1.0}|must be positive'
)
for item in "${legacy_cases[@]}"; do
  IFS='|' read -r name mutation diagnostic <<<"${item}"
  assert_invalid_legacy_apply "${name}" "${mutation}" "${diagnostic}"
done

echo
if (( failures > 0 )); then
  echo "${failures} case(s) failed" >&2
  exit 1
fi
echo "test-golden-arming-patch.sh: all cases passed"
