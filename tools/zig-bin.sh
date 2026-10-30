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
  [[ -n "${zig_bin}" && -x "${zig_bin}" ]] || fail "the configured Zig is missing or not executable"
else
  zig_bin="$(command -v zig 2>/dev/null || true)"
  [[ -n "${zig_bin}" && -x "${zig_bin}" ]] || fail "the configured Zig is missing or not executable"
fi

version="$("${zig_bin}" version 2>/dev/null)" || fail "cannot read the configured Zig version"
[[ "${version}" == "0.17.0" ]] || fail "the configured Zig is not 0.17.0"
printf '%s\n' "${zig_bin}"
