#!/usr/bin/env bash
set -euo pipefail

[[ $# -ge 1 && $# -le 3 ]] || exit 2
zig="$1"
root="${2:-/}"
owner_uid="${3:-0}"

real_path() {
  /usr/bin/python3 -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "$1"
}

path_uid() {
  if [[ "$(/usr/bin/uname -s)" == Darwin ]]; then
    /usr/bin/stat -f '%u' "$1"
  else
    /usr/bin/stat -c '%u' "$1"
  fi
}

# An extended ACL can give write access that the mode bits do not show.
# A path whose ACL cannot be read counts as having one (fail closed).
has_acl() {
  local out err rc=0
  err="$(mktemp)"
  if [[ "$(/usr/bin/uname -s)" == Darwin ]]; then
    out="$(/bin/ls -led "$1" 2>"${err}")" || rc=$?
  elif [[ -x /usr/bin/getfacl ]]; then
    out="$(/usr/bin/getfacl -p --omit-header "$1" 2>"${err}" | /usr/bin/grep -vE '^(user|group|other)::|^$')" || rc=$?
    [[ ${rc} -eq 1 && ! -s "${err}" ]] && rc=0
  else
    /bin/rm -f "${err}"
    return 0
  fi
  if [[ ${rc} -ne 0 || -s "${err}" ]]; then
    /bin/rm -f "${err}"
    return 0
  fi
  /bin/rm -f "${err}"
  if [[ "$(/usr/bin/uname -s)" == Darwin ]]; then
    [[ -n "$(printf '%s\n' "${out}" | /usr/bin/sed -n '2p')" ]]
  else
    [[ -n "${out}" ]]
  fi
}

path_mode() {
  if [[ "$(/usr/bin/uname -s)" == Darwin ]]; then
    /usr/bin/stat -f '%Lp' "$1"
  else
    /usr/bin/stat -c '%a' "$1"
  fi
}

resolved_zig="$(real_path "${zig}")"
resolved_root="$(real_path "${root}")"
if [[ "${resolved_root}" == / ]]; then
  virtual_zig="${resolved_zig}"
else
  case "${resolved_zig}" in
    "${resolved_root}"/*) virtual_zig="/${resolved_zig#"${resolved_root}"/}" ;;
    *) exit 1 ;;
  esac
fi
case "${virtual_zig}" in
  /Users/*|/private/var/folders/*|/private/tmp/*|/tmp/*) exit 1 ;;
esac

toolchain_dir="$(/usr/bin/dirname -- "${resolved_zig}")"
current="${toolchain_dir}"
while :; do
  [[ ! -L "${current}" && "$(path_uid "${current}")" == "${owner_uid}" ]] || exit 1
  ! has_acl "${current}" || exit 1
  mode="$(path_mode "${current}")"
  group_digit="${mode: -2:1}"
  other_digit="${mode: -1}"
  case "${group_digit}${other_digit}" in
    *[2367]*) exit 1 ;;
  esac
  [[ "${current}" != "${resolved_root}" ]] || break
  parent="$(/usr/bin/dirname -- "${current}")"
  [[ "${parent}" != "${current}" ]] || exit 1
  current="${parent}"
done

printf '%s\n' "${toolchain_dir}"
