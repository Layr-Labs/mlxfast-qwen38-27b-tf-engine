#!/usr/bin/env bash
set -euo pipefail

fail() { printf 'zig-bin.sh: %s\n' "$*" >&2; exit 1; }

zig_bin=""
if [[ -n "${ZIG:-}" ]]; then
  if [[ "${ZIG}" == */* ]]; then
    zig_bin="${ZIG}"
  else
    zig_bin="$(command -v -- "${ZIG}" 2>/dev/null || true)"
  fi
  [[ -n "${zig_bin}" && -x "${zig_bin}" ]] || fail "ZIG does not name an executable: ${ZIG}"
else
  zig_bin="$(command -v zig 2>/dev/null || true)"
  [[ -n "${zig_bin}" && -x "${zig_bin}" ]] || fail "Zig 0.17.0 is required; set ZIG or add zig to PATH"
fi

version="$("${zig_bin}" version 2>/dev/null)" || fail "cannot read Zig version: ${zig_bin}"
[[ "${version}" == "0.17.0" ]] || fail "Zig 0.17.0 is required; ${zig_bin} reports ${version:-no version}"
printf '%s\n' "${zig_bin}"
