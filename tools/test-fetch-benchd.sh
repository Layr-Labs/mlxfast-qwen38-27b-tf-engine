#!/usr/bin/env bash
# test-fetch-benchd.sh -- fetch-benchd.sh resolves the channel pair.
#
# Hermetic: a stub `benchd` and a manifest in the current dist format
# (six top-level fields describing benchd, then the `binaries` map with a
# sha256/bytes entry per binary) are handed to fetch-benchd.sh through
# BENCHD_DIST_LOCAL. The script must install benchd + benchd.manifest.json into
# BENCHD_BIN_DIR, print the binary path, and refuse a pair whose bytes moved.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
fail() { echo "test-fetch-benchd: FAIL -- $*" >&2; exit 1; }

# The channel name is an input of fetch-benchd.sh (BENCHD_BRANCH); the test
# names its own so the manifest and the resolve agree by construction.
BRANCH="test-channel-v0"
export BENCHD_BRANCH="${BRANCH}"

DIST="${WORK}/dist"; mkdir -p "${DIST}"
printf '#!/bin/sh\necho stub-benchd\n' > "${DIST}/benchd"; chmod 755 "${DIST}/benchd"
printf '#!/bin/sh\necho stub-recorder\n' > "${DIST}/record-correctness-golden"; chmod 755 "${DIST}/record-correctness-golden"
sha() { shasum -a 256 "$1" | awk '{print $1}'; }
bytes() { wc -c < "$1" | tr -d '[:space:]'; }
{
  printf '{\n'
  printf '  "version": "0.0.0",\n'
  printf '  "branch": "%s",\n' "${BRANCH}"
  printf '  "source_commit": "3101418f95eb074cc9c100f8dc61cc85cc924155",\n'
  printf '  "target_triple": "aarch64-apple-darwin",\n'
  printf '  "sha256": "%s",\n' "$(sha "${DIST}/benchd")"
  printf '  "bytes": %s,\n' "$(bytes "${DIST}/benchd")"
  printf '  "binaries": {\n'
  printf '    "benchd": {"sha256": "%s", "bytes": %s},\n' "$(sha "${DIST}/benchd")" "$(bytes "${DIST}/benchd")"
  printf '    "record-correctness-golden": {"sha256": "%s", "bytes": %s}\n' "$(sha "${DIST}/record-correctness-golden")" "$(bytes "${DIST}/record-correctness-golden")"
  printf '  }\n'
  printf '}\n'
} > "${DIST}/benchd.manifest.json"

BIN="${WORK}/benchd-bin"
if ! out="$(BENCHD_DIST_LOCAL="${DIST}" BENCHD_BIN_DIR="${BIN}" bash "${ROOT}/tools/fetch-benchd.sh" 2>"${WORK}/err")"; then
  fail "resolve refused a good channel pair: $(cat "${WORK}/err")"
fi
[[ "${out}" == "${BIN}/benchd" ]] || fail "printed '${out}', expected ${BIN}/benchd"
[[ -x "${BIN}/benchd" && -x "${BIN}/record-correctness-golden" && -f "${BIN}/benchd.manifest.json" ]] \
  || fail "benchd, recorder, and manifest were not installed under ${BIN}"
grep -q "source_commit=3101418f95eb074cc9c100f8dc61cc85cc924155" "${WORK}/err" || fail "identity line missing: $(cat "${WORK}/err")"
echo "test-fetch-benchd: PASS -- channel pair resolves (binaries map tolerated)"

# The recorder has its own required manifest pin.
printf '#!/bin/sh\necho evil-recorder\n' > "${DIST}/record-correctness-golden"
BIN_RECORDER="${WORK}/benchd-bin-recorder-mismatch"
if BENCHD_DIST_LOCAL="${DIST}" BENCHD_BIN_DIR="${BIN_RECORDER}" bash "${ROOT}/tools/fetch-benchd.sh" >/dev/null 2>"${WORK}/recorder-err"; then
  fail "a recorder with a sha256 mismatch was accepted"
fi
[[ ! -e "${BIN_RECORDER}/benchd" ]] || fail "a pair with a bad recorder was partly installed"
grep -q "record-correctness-golden sha256 mismatch" "${WORK}/recorder-err" \
  || fail "recorder mismatch refusal was not clear: $(cat "${WORK}/recorder-err")"
printf '#!/bin/sh\necho stub-recorder\n' > "${DIST}/record-correctness-golden"
chmod 755 "${DIST}/record-correctness-golden"
echo "test-fetch-benchd: PASS -- a recorder sha256 mismatch is refused"

# A binary whose bytes moved after the manifest was written is refused.
printf '#!/bin/sh\necho tampered\n' > "${DIST}/benchd"
BIN2="${WORK}/benchd-bin-2"
if BENCHD_DIST_LOCAL="${DIST}" BENCHD_BIN_DIR="${BIN2}" bash "${ROOT}/tools/fetch-benchd.sh" >/dev/null 2>"${WORK}/err2"; then
  fail "a tampered binary was accepted"
fi
[[ ! -e "${BIN2}/benchd" ]] || fail "a tampered binary was installed"
echo "test-fetch-benchd: PASS -- a mismatching binary is refused and nothing is installed"

# A pair from another source revision is refused even when its bytes match.
sed 's/3101418f95eb074cc9c100f8dc61cc85cc924155/0123456789abcdef0123456789abcdef01234567/' \
  "${DIST}/benchd.manifest.json" > "${WORK}/wrong-source.json"
mv "${WORK}/wrong-source.json" "${DIST}/benchd.manifest.json"
BIN3="${WORK}/benchd-bin-3"
if BENCHD_DIST_LOCAL="${DIST}" BENCHD_BIN_DIR="${BIN3}" bash "${ROOT}/tools/fetch-benchd.sh" >/dev/null 2>"${WORK}/err3"; then
  fail "a manifest from another source revision was accepted"
fi
grep -q "expected pinned" "${WORK}/err3" || fail "wrong-source refusal was not clear"
echo "test-fetch-benchd: PASS -- another source revision is refused"
echo "test-fetch-benchd: all cases pass"
