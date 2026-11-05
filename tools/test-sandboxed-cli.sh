#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
if [[ "$(uname -s)" == Darwin ]] && ! /usr/bin/sandbox-exec -p '(version 1) (allow default)' /usr/bin/true >/dev/null 2>&1; then
  printf 'test-sandboxed-cli.sh: SKIP -- sandbox-exec is blocked by the outer sandbox\n'
  exit 0
fi
fail() { printf 'test-sandboxed-cli.sh: FAIL: %s\n' "$*" >&2; exit 1; }
WORK="$(mktemp -d)"
# Match the resolved path that the profile's exact process-exec rule uses.
WORK="$(cd -- "${WORK}" && pwd -P)"
# The self-test socket path must stay under 104 bytes, and the macOS $TMPDIR is
# too long for it. Cases that run self-test under RUNNER_TEMP use this short dir.
SHORT_TMP="$(mktemp -d /tmp/tfcli.XXXXXX)"
SHORT_TMP="$(cd -- "${SHORT_TMP}" && pwd -P)"
OFFICIAL_ROOT="/private/tmp/q38cli-$(/usr/bin/id -u).noindex"
REMOVE_OFFICIAL_ROOT=0
cleanup() {
  rm -rf "${WORK}" "${SHORT_TMP}"
  if [[ "${REMOVE_OFFICIAL_ROOT}" == 1 ]]; then
    rm -f "${OFFICIAL_ROOT}" 2>/dev/null || rmdir "${OFFICIAL_ROOT}" 2>/dev/null || true
  fi
}
trap cleanup EXIT
# The profile allows exec of the literal worker path only, so a #! script (which also execs its interpreter) is denied.
cat > "${WORK}/worker.c" <<'MOCK'
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static char *read_profile(const char *tmp) {
  char path[4096];
  snprintf(path, sizeof(path), "%s../profile.sb", tmp);
  FILE *file = fopen(path, "rb");
  if (!file) return NULL;
  fseek(file, 0, SEEK_END);
  long size = ftell(file);
  rewind(file);
  char *text = malloc((size_t)size + 1);
  if (!text || fread(text, 1, (size_t)size, file) != (size_t)size) return NULL;
  text[size] = '\0';
  fclose(file);
  return text;
}

static int count(const char *text, const char *part) {
  int found = 0;
  size_t length = strlen(part);
  while ((text = strstr(text, part))) { found++; text += length; }
  return found;
}

int main(int c, char **v) {
  const char *tmp = getenv("TMPDIR");
  char *profile = tmp ? read_profile(tmp) : NULL;
  if (tmp && getenv("MOCK_DIRTY_TMP")) {
    char locked[4096], file[4096];
    snprintf(locked, sizeof(locked), "%slocked", tmp);
    snprintf(file, sizeof(file), "%s/file", locked);
    mkdir(locked, 0700);
    FILE *dirty = fopen(file, "wb");
    if (dirty) fclose(dirty);
#ifdef __APPLE__
    chflags(file, UF_IMMUTABLE);
#endif
    chmod(locked, 0000);
  }
  if (getenv("MOCK_FAIL")) {
    fputs("CLI-SECRET\n", stderr);
    return 9;
  }
  if (getenv("MOCK_INVALID")) {
    fwrite("CLI-SECRET\0INVALID\n", 1, 19, stdout);
    return 0;
  }
  if (c > 1 && !strcmp(v[1], "self-test")) {
    const char *dir = getenv("BENCH_WORKER_SELF_TEST_DIR");
    char expected_dir[4096], socket[4096];
    if (!tmp || !dir || !profile) return 10;
    snprintf(expected_dir, sizeof(expected_dir), "%sself-test", tmp);
    snprintf(socket, sizeof(socket), "%s/resident.sock", dir);
    if (strcmp(dir, expected_dir) || strlen(socket) >= 104) return 11;
    if (count(profile, "unix-socket") != 2 || count(profile, socket) != 3) return 12;
    free(profile);
    if (getenv("MOCK_STDERR")) { fputs("bench-worker resident: run_free failed: SessionNotSeeded\n", stderr); fflush(stderr); }
    puts("engine-protocol-v1 self-test: ok");
    return 0;
  }
  if (c > 2 && !strcmp(v[1], "manifest") && !strcmp(v[2], "--digest")) {
    if (getenv("BENCH_WORKER_SELF_TEST_DIR") || !profile || strstr(profile, "unix-socket")) return 20;
    free(profile);
    if (getenv("MOCK_STDERR")) { fputs("CLI-SECRET diagnostic\n", stderr); fflush(stderr); }
    puts("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
    return 0;
  }
  free(profile);
  return 2;
}
MOCK
/usr/bin/cc -o "${WORK}/worker" "${WORK}/worker.c"
[[ "$(TMPDIR=/tmp MLXFAST_ENGINE_BIN="${WORK}/worker" "${ROOT}/tools/sandboxed-cli.sh" self-test)" == 'sandboxed-cli.sh: PASS: self-test' ]] \
  || fail "self-test profile or environment differs"
[[ "$(RUNNER_TEMP="${SHORT_TMP}" MOCK_DIRTY_TMP=1 MLXFAST_ENGINE_BIN="${WORK}/worker" "${ROOT}/tools/sandboxed-cli.sh" self-test)" == 'sandboxed-cli.sh: PASS: self-test' ]] \
  || fail "restricted CLI private temp cleanup failed"
[[ -z "$(find "${SHORT_TMP}" -mindepth 1 -maxdepth 1 -print -quit)" ]] \
  || fail "private CLI temp tree or cleanup log remained"
[[ "$(RUNNER_TEMP="${WORK}" MOCK_DIRTY_TMP=1 MLXFAST_ENGINE_BIN="${WORK}/worker" "${ROOT}/tools/sandboxed-cli.sh" manifest --digest)" == 'sandboxed-cli.sh: PASS: manifest' ]] \
  || fail "restricted CLI manifest temp cleanup failed"
[[ -z "$(find "${WORK}" -maxdepth 1 -name 'tf-sandboxed-cli.*' -print -quit)" ]] \
  || fail "private CLI temp tree remained"
[[ "$(TMPDIR=/tmp BENCH_WORKER_SELF_TEST_DIR=must-not-pass MLXFAST_ENGINE_BIN="${WORK}/worker" "${ROOT}/tools/sandboxed-cli.sh" manifest --digest)" == 'sandboxed-cli.sh: PASS: manifest' ]] \
  || fail "manifest received self-test access"
if MLXFAST_ENGINE_BIN="${WORK}/worker" "${ROOT}/tools/sandboxed-cli.sh" resident >/dev/null 2>&1; then
  fail "resident was accepted"
fi
if MLXFAST_ENGINE_BIN="${WORK}/worker" "${ROOT}/tools/sandboxed-cli.sh" manifest >/dev/null 2>&1; then
  fail "manifest without --digest was accepted"
fi
long_tmp="${WORK}/$(printf 'x%.0s' {1..90})"
mkdir "${long_tmp}"
out="$(RUNNER_TEMP="${long_tmp}" MLXFAST_ENGINE_BIN="${WORK}/worker" "${ROOT}/tools/sandboxed-cli.sh" self-test 2>&1 || true)"
[[ "${out}" == 'sandboxed-cli.sh: FAIL: socket-path' ]] || fail "long socket path was accepted"
if [[ ! -e "${OFFICIAL_ROOT}" && ! -L "${OFFICIAL_ROOT}" ]]; then
  mkdir -m 755 "${OFFICIAL_ROOT}"
  REMOVE_OFFICIAL_ROOT=1
  out="$(RUNNER_ENVIRONMENT=self-hosted MLXFAST_ENGINE_BIN="${WORK}/worker" \
    "${ROOT}/tools/sandboxed-cli.sh" self-test 2>&1 || true)"
  [[ "${out}" == 'sandboxed-cli.sh: FAIL: temp' ]] || fail "mode 755 work root was accepted"
  rmdir "${OFFICIAL_ROOT}"
  ln -s "${WORK}" "${OFFICIAL_ROOT}"
  out="$(RUNNER_ENVIRONMENT=self-hosted MLXFAST_ENGINE_BIN="${WORK}/worker" \
    "${ROOT}/tools/sandboxed-cli.sh" self-test 2>&1 || true)"
  [[ "${out}" == 'sandboxed-cli.sh: FAIL: temp' ]] || fail "symlink work root was accepted"
  rm "${OFFICIAL_ROOT}"
  REMOVE_OFFICIAL_ROOT=0
else
  printf 'test-sandboxed-cli.sh: INFO: existing official work root left unchanged\n'
fi
mkdir -p "${WORK}/goldens" "${WORK}/baseline" "${WORK}/benchd" "${WORK}/runner/work"
official_env=(
  RUNNER_ENVIRONMENT=self-hosted
  RUNNER_TEMP="${long_tmp}"
  MLXFAST_ENGINE_BIN="${WORK}/worker"
  MLXFAST_QWEN38_GOLDEN_DIR="${WORK}/goldens"
  MLXFAST_BASELINE_WORKSPACE="${WORK}/baseline"
  MLXFAST_BASELINE_CALIBRATION="${WORK}/calibration.json"
  BENCHD_BIN_DIR="${WORK}/benchd"
  RUNNER_WORKSPACE="${WORK}/runner/work"
)
out="$(env "${official_env[@]}" "${ROOT}/tools/sandboxed-cli.sh" self-test 2>&1)"
[[ "${out}" == 'sandboxed-cli.sh: PASS: self-test' ]] \
  || fail "self-hosted long RUNNER_TEMP did not use the short work root: ${out}"
out="$(TMPDIR=/tmp MLXFAST_ENGINE_BIN="${WORK}/worker" MLXFAST_OFFICIAL_BENCHMARK_RUN=1 "${ROOT}/tools/sandboxed-cli.sh" self-test 2>&1 || true)"
[[ "${out}" == 'sandboxed-cli.sh: FAIL: profile' || "${out}" == 'sandboxed-cli.sh: FAIL: sandbox-unavailable' ]] \
  || fail "official environment refusal changed"
out="$(TMPDIR=/tmp MOCK_FAIL=1 MLXFAST_ENGINE_BIN="${WORK}/worker" "${ROOT}/tools/sandboxed-cli.sh" self-test 2>&1 || true)"
[[ "${out}" == 'sandboxed-cli.sh: FAIL: command' ]] || fail "worker output reached the public result"
out="$(TMPDIR=/tmp MOCK_INVALID=1 MLXFAST_ENGINE_BIN="${WORK}/worker" "${ROOT}/tools/sandboxed-cli.sh" self-test 2>&1 || true)"
[[ "${out}" == 'sandboxed-cli.sh: FAIL: invalid-output' ]] || fail "invalid worker output reached the public result"
# Worker diagnostics on stderr (as the real worker writes before its ok line) stay private and do
# not break the stdout result check.
out="$(TMPDIR=/tmp MOCK_STDERR=1 MLXFAST_ENGINE_BIN="${WORK}/worker" "${ROOT}/tools/sandboxed-cli.sh" self-test 2>&1)"
[[ "${out}" == 'sandboxed-cli.sh: PASS: self-test' ]] || fail "worker stderr broke the self-test result (${out})"
out="$(TMPDIR=/tmp MOCK_STDERR=1 MLXFAST_ENGINE_BIN="${WORK}/worker" "${ROOT}/tools/sandboxed-cli.sh" manifest --digest 2>&1)"
[[ "${out}" == 'sandboxed-cli.sh: PASS: manifest' ]] || fail "worker stderr broke the manifest result (${out})"
# A full stderr buffer must not cut stdout. Truncated stdout must fail the run.
flood='import os, sys; os.write(2, b"e" * 1048543); os.close(2); sys.stdout.write("engine-protocol-v1 self-test: ok\n" + sys.argv[1]); sys.stdout.flush()'
python3 "${ROOT}/tools/bounded-run.py" --output "${WORK}/flood.out" --stderr "${WORK}/flood.err" -- python3 -c "${flood}" '' \
  || fail "stderr flood failed an honest stdout result"
[[ "$(cat "${WORK}/flood.out")" == 'engine-protocol-v1 self-test: ok' ]] || fail "stderr flood cut the stdout result"
python3 "${ROOT}/tools/bounded-run.py" --output "${WORK}/flood.out" --stderr "${WORK}/flood.err" -- python3 -c "${flood}" INVALID \
  || fail "stderr flood run failed"
[[ "$(cat "${WORK}/flood.out")" == $'engine-protocol-v1 self-test: ok\nINVALID' ]] || fail "stderr flood hid extra stdout"
python3 "${ROOT}/tools/bounded-run.py" --output "${WORK}/flood.out" --stderr "${WORK}/flood.err" --limit 34 -- python3 -c "${flood}" INVALID \
  && fail "cut stdout was accepted"
printf 'test-sandboxed-cli.sh: PASS\n'
