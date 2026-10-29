#!/usr/bin/env bash
set -u

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
DOC="${PARTICIPANT_CONTRACT_PATH:-${ROOT}/docs/participant-contract.md}"
FIXTURE="${TRACK_FIXTURE_PATH:-${ROOT}/fixtures/qwen3_8_27b_tf_mlx_v1_track.json}"
ERROR_FILE="$(mktemp "${TMPDIR:-/tmp}/test-participant-guard-docs.XXXXXX")" || {
  printf 'test-participant-guard-docs.sh: FAIL: cannot create a temporary file\n'
  exit 1
}
trap 'rm -f "${ERROR_FILE}"' EXIT

if python3 - "${DOC}" "${FIXTURE}" >"${ERROR_FILE}" 2>&1 <<'PY'
import json
import re
import sys

doc_path, fixture_path = sys.argv[1:]

try:
    with open(doc_path, encoding="utf-8") as handle:
        doc = handle.read()
    with open(fixture_path, encoding="utf-8") as handle:
        fixture = json.load(handle)
except (OSError, ValueError) as error:
    print(error)
    raise SystemExit(1)

guards = (
    ("Prefill speedup ceiling", "prefill_speedup_ceiling"),
    ("Decode speedup ceiling", "decode_speedup_ceiling"),
    ("Prefill speedup floor", "prefill_speedup_floor"),
    ("Decode speedup floor", "decode_speedup_floor"),
    ("Seed window maximum ratio", "seed_window_max_ratio"),
)

for label, key in guards:
    matches = re.findall(rf"^- {re.escape(label)}: `([^`]+)`\.$", doc, re.MULTILINE)
    expected = str(fixture.get(key))
    if matches != [expected]:
        actual = ", ".join(matches) if matches else "missing"
        print(f"{label} is {actual}; fixture is {expected}")
        raise SystemExit(1)

seed_ratio = str(fixture.get("seed_window_max_ratio"))
required_lines = (
    "- A pair above a ceiling refuses the run.",
    "- A pair below a floor refuses the run.",
    "- The ratio is seed seconds per token divided by the same leg's prefill seconds per token.",
    f"- A ratio above `{seed_ratio}` refuses the run.",
    "- Speed up the seed step together with prefill.",
)
for line in required_lines:
    if doc.count(line) != 1:
        print(f"required guard text is missing or repeated: {line}")
        raise SystemExit(1)

seed_line = "- Decode time includes the seed step."
if fixture.get("decode_window_includes_seed") is not True or doc.count(seed_line) != 1:
    print("decode seed accounting differs from the fixture")
    raise SystemExit(1)
PY
then
  printf 'test-participant-guard-docs.sh: PASS\n'
else
  error="$(tr '\n' ' ' < "${ERROR_FILE}")"
  printf 'test-participant-guard-docs.sh: FAIL: %s\n' "${error:-guard check failed}"
  exit 1
fi
