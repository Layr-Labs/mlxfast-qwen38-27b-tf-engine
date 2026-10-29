#!/usr/bin/env bash
# install.sh -- install the root control plane and the user-owned runner.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null && pwd -P)"

fail() { echo "install.sh: $*" >&2; exit 78; }
info() { echo "install.sh: $*"; }

# --- Pinned actions-runner release ------------------------------------------
ACTIONS_RUNNER_VERSION="${ACTIONS_RUNNER_VERSION:-2.336.0}"

pinned_sha256() {
  case "$1" in
    osx-arm64) printf '%s' "8e8839c49b7060b6b2154f4931f815df330c27f167d53ef2239ee3dfce28b079" ;;
    osx-x64)   printf '%s' "f79c43232761ca495fc18df550bb2865aa99984b37c173c0aa1f8c09d0d548fe" ;;
    *)         return 1 ;;
  esac
}

platform_tag() {
  local os arch
  os="$(uname -s)"
  arch="$(uname -m)"
  [ "${os}" = "Darwin" ] || fail "unsupported OS '${os}' -- this installer targets macOS only"
  case "${arch}" in
    arm64) printf 'osx-arm64' ;;
    x86_64) printf 'osx-x64' ;;
    *) fail "unsupported architecture '${arch}'" ;;
  esac
}

compute_sha256() {
  local file="$1"
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "${file}" | awk '{print $1}'
  elif command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${file}" | awk '{print $1}'
  else
    fail "neither shasum nor sha256sum is on PATH"
  fi
}

# --- Installed paths ---------------------------------------------------------
# BOX_RUNNER_INSTALL_ROOT is a virtual root for tests. Production uses /.
INSTALL_ROOT="${BOX_RUNNER_INSTALL_ROOT:-/}"
INSTALL_OWNER_UID="${BOX_RUNNER_INSTALL_OWNER_UID:-0}"
RUNNER_USER_EXEC="${BOX_RUNNER_RUN_AS_USER:-}"
RUNNER_USER="${RUNNER_USER:-runner}"
RUNNER_DIR="${RUNNER_DIR:-/Users/runner/qwen38tf-actions-runner}"

install_path() {
  local relative="${1#/}"
  if [ "${INSTALL_ROOT}" = "/" ]; then
    printf '/%s' "${relative}"
  else
    printf '%s/%s' "${INSTALL_ROOT%/}" "${relative}"
  fi
}

CODE_DIR="$(install_path /usr/local/libexec/qwen38tf-box-runner)"
CONFIG_DIR="$(install_path /usr/local/etc/qwen38tf-box-runner)"
CONFIG_FILE="${CONFIG_DIR}/box-runner.env"
LOG_DIR="${LOG_DIR:-$(install_path /private/var/log/qwen38tf-box-runner)}"
STATE_DIR="${STATE_DIR:-$(install_path /private/var/db/qwen38tf-box-runner)}"
PLIST_FILE="$(install_path /Library/LaunchDaemons/com.qwen38tf.box-runner.plist)"

path_uid() {
  if [ "$(uname -s)" = "Darwin" ]; then
    stat -f '%u' "$1"
  else
    stat -c '%u' "$1"
  fi
}

path_mode() {
  if [ "$(uname -s)" = "Darwin" ]; then
    stat -f '%Lp' "$1"
  else
    stat -c '%a' "$1"
  fi
}

path_has_acl() {
  [ "$(uname -s)" = "Darwin" ] || return 1
  # shellcheck disable=SC2012  # macOS ls -led is the ACL interface required here.
  ls -led "$1" 2>/dev/null | tail -n +2 | grep -Eq '^[[:space:]]*[0-9]+:'
}

assert_trusted_component() {
  local path="$1" mode group_digit other_digit
  [ ! -L "${path}" ] || fail "unsafe install path: symlink ${path}"
  [ "$(path_uid "${path}")" = "${INSTALL_OWNER_UID}" ] \
    || fail "unsafe install path: owner of ${path} is not uid ${INSTALL_OWNER_UID}"
  mode="$(path_mode "${path}")"
  group_digit="${mode: -2:1}"
  other_digit="${mode: -1}"
  case "${group_digit}${other_digit}" in
    *[2367]*) fail "unsafe install path: group/other writable ${path} (mode ${mode})" ;;
  esac
  if path_has_acl "${path}"; then
    fail "unsafe install path: ACL on ${path}"
  fi
  return 0
}

# Check every existing component inside the virtual root. Missing components
# are allowed. create_trusted_dir creates them one at a time after this check.
assert_trusted_path() {
  local target="$1" relative current component
  case "${INSTALL_ROOT}" in
    /*) ;;
    *) fail "BOX_RUNNER_INSTALL_ROOT must be absolute: ${INSTALL_ROOT}" ;;
  esac
  case "${target}" in
    "${INSTALL_ROOT}"|"${INSTALL_ROOT%/}"/*) ;;
    *) fail "install target escapes BOX_RUNNER_INSTALL_ROOT: ${target}" ;;
  esac

  current="${INSTALL_ROOT%/}"
  [ -n "${current}" ] || current="/"
  [ -e "${current}" ] || fail "install root does not exist: ${current}"
  assert_trusted_component "${current}"

  relative="${target#"${INSTALL_ROOT%/}"/}"
  [ "${target}" = "${INSTALL_ROOT}" ] && relative=""
  while IFS= read -r component; do
    [ -n "${component}" ] || continue
    [ "${component}" != "." ] && [ "${component}" != ".." ] \
      || fail "unsafe install path component '${component}' in ${target}"
    if [ "${current}" = "/" ]; then
      current="/${component}"
    else
      current="${current}/${component}"
    fi
    if [ -e "${current}" ] || [ -L "${current}" ]; then
      assert_trusted_component "${current}"
    fi
  done < <(printf '%s\n' "${relative}" | tr '/' '\n')
}

create_trusted_dir() {
  local target="$1" final_mode="$2" relative current component
  assert_trusted_path "${target}"
  current="${INSTALL_ROOT%/}"
  [ -n "${current}" ] || current="/"
  relative="${target#"${INSTALL_ROOT%/}"/}"
  while IFS= read -r component; do
    [ -n "${component}" ] || continue
    if [ "${current}" = "/" ]; then
      current="/${component}"
    else
      current="${current}/${component}"
    fi
    if [ ! -e "${current}" ] && [ ! -L "${current}" ]; then
      assert_trusted_path "${current}"
      mkdir "${current}"
      chmod 0755 "${current}"
      [ "$(id -u)" -ne 0 ] || chown "${INSTALL_OWNER_UID}" "${current}"
    fi
    [ -d "${current}" ] || fail "install path component is not a directory: ${current}"
    assert_trusted_component "${current}"
  done < <(printf '%s\n' "${relative}" | tr '/' '\n')
  chmod "${final_mode}" "${target}"
  assert_trusted_component "${target}"
}

install_file() {
  local source="$1" target="$2" mode="$3" target_dir tmp sha
  target_dir="$(dirname -- "${target}")"
  assert_trusted_path "${target}"
  assert_trusted_component "${target_dir}"
  tmp="$(mktemp "${target_dir}/.box-runner-install.XXXXXX")"
  cp "${source}" "${tmp}"
  chmod "${mode}" "${tmp}"
  [ "$(id -u)" -ne 0 ] || chown "${INSTALL_OWNER_UID}" "${tmp}"
  mv -f "${tmp}" "${target}"
  assert_trusted_component "${target}"
  sha="$(compute_sha256 "${target}")"
  info "installed ${target} mode=${mode} sha256=${sha}"
}

install_control_plane() {
  local config_tmp plist_tmp
  create_trusted_dir "${CODE_DIR}" 0755
  create_trusted_dir "${CODE_DIR}/lib" 0755
  create_trusted_dir "${CONFIG_DIR}" 0755
  create_trusted_dir "${LOG_DIR}" 0755
  create_trusted_dir "${STATE_DIR}" 0755
  create_trusted_dir "$(dirname -- "${PLIST_FILE}")" 0755

  install_file "${SCRIPT_DIR}/supervisor.sh" "${CODE_DIR}/supervisor.sh" 0555
  install_file "${SCRIPT_DIR}/lib/pidtree.sh" "${CODE_DIR}/lib/pidtree.sh" 0555
  install_file "${SCRIPT_DIR}/gh-app-mint.sh" "${CODE_DIR}/gh-app-mint.sh" 0555

  if [ -e "${CONFIG_FILE}" ] || [ -L "${CONFIG_FILE}" ]; then
    assert_trusted_path "${CONFIG_FILE}"
    [ -f "${CONFIG_FILE}" ] || fail "installed config is not a regular file: ${CONFIG_FILE}"
    case "$(path_mode "${CONFIG_FILE}")" in
      600|644) ;;
      *) fail "installed config must have mode 0600 or 0644: ${CONFIG_FILE}" ;;
    esac
    info "kept ${CONFIG_FILE} sha256=$(compute_sha256 "${CONFIG_FILE}")"
  else
    config_tmp="$(mktemp "${CONFIG_DIR}/.box-runner-config.XXXXXX")"
    sed \
      -e "s#__RUNNER_DIR__#${RUNNER_DIR}#g" \
      -e "s#__GH_APP_MINT_PATH__#${CODE_DIR}/gh-app-mint.sh#g" \
      -e "s#__LOG_DIR__#${LOG_DIR}#g" \
      -e "s#__STATE_DIR__#${STATE_DIR}#g" \
      "${SCRIPT_DIR}/box-runner.env.example" > "${config_tmp}"
    install_file "${config_tmp}" "${CONFIG_FILE}" 0644
    rm -f "${config_tmp}"
  fi

  plist_tmp="$(mktemp "$(dirname -- "${PLIST_FILE}")/.box-runner-plist.XXXXXX")"
  sed \
    -e "s#__SUPERVISOR_PATH__#${CODE_DIR}/supervisor.sh#g" \
    -e "s#__LOG_DIR__#${LOG_DIR}#g" \
    -e "s#__CONFIG_ENV_PATH__#${CONFIG_FILE}#g" \
    -e "s#__SCRIPT_DIR__#${CODE_DIR}#g" \
    "${SCRIPT_DIR}/com.qwen38tf.box-runner.plist.template" > "${plist_tmp}"
  install_file "${plist_tmp}" "${PLIST_FILE}" 0644
  rm -f "${plist_tmp}"

  chmod 0555 "${CODE_DIR}/lib" "${CODE_DIR}"
}

run_as_runner() {
  if [ -n "${RUNNER_USER_EXEC}" ]; then
    "${RUNNER_USER_EXEC}" "${RUNNER_USER}" "$@"
  else
    /usr/bin/sudo -u "${RUNNER_USER}" -- "$@"
  fi
}

require_fresh_runner_dir() {
  [ ! -e "${RUNNER_DIR}" ] && [ ! -L "${RUNNER_DIR}" ] \
    || fail "RUNNER_DIR already exists; refusing to use shared or redirected path: ${RUNNER_DIR}"
  id "${RUNNER_USER}" >/dev/null 2>&1 || [ -n "${RUNNER_USER_EXEC}" ] \
    || fail "RUNNER_USER '${RUNNER_USER}' does not exist"
}

INSTALL_TMP_DIR=""
cleanup_install_tmp() {
  if [ -n "${INSTALL_TMP_DIR}" ] && [ -d "${INSTALL_TMP_DIR}" ]; then
    rm -rf -- "${INSTALL_TMP_DIR}"
  fi
}
trap cleanup_install_tmp EXIT

install_actions_runner() {
  local tag want_sha asset url tarball got_sha helper
  require_fresh_runner_dir

  tag="$(platform_tag)"
  want_sha="$(pinned_sha256 "${tag}")" \
    || fail "no pinned sha256 for platform '${tag}'"
  asset="actions-runner-${tag}-${ACTIONS_RUNNER_VERSION}.tar.gz"
  url="https://github.com/actions/runner/releases/download/v${ACTIONS_RUNNER_VERSION}/${asset}"
  [ -d "${STATE_DIR}" ] || fail "trusted state directory is missing: ${STATE_DIR}"
  assert_trusted_component "${STATE_DIR}"
  INSTALL_TMP_DIR="$(mktemp -d "${STATE_DIR}/.box-runner-install.XXXXXX")"
  chmod 0755 "${INSTALL_TMP_DIR}"
  tarball="${INSTALL_TMP_DIR}/${asset}"
  helper="${INSTALL_TMP_DIR}/extract-as-runner.sh"

  info "downloading ${url}"
  curl -fsSL --max-time 300 -o "${tarball}" "${url}" \
    || fail "download failed: ${url}"
  got_sha="$(compute_sha256 "${tarball}")"
  [ "${got_sha}" = "${want_sha}" ] \
    || fail "checksum mismatch for ${asset}: got ${got_sha}, want ${want_sha}"
  chmod 0444 "${tarball}"

  # This helper runs as RUNNER_USER. Root does not create, extract into, or
  # recursively change ownership below RUNNER_DIR.
  {
    printf '%s\n' '#!/usr/bin/env bash' 'set -euo pipefail'
    # shellcheck disable=SC2016  # these variables expand in the generated helper.
    printf '%s\n' 'runner_dir="$1"' 'tarball="$2"' 'version="$3"'
    # shellcheck disable=SC2016
    printf '%s\n' '[ ! -e "${runner_dir}" ] && [ ! -L "${runner_dir}" ]'
    # shellcheck disable=SC2016
    printf '%s\n' 'mkdir "${runner_dir}"'
    # shellcheck disable=SC2016
    printf '%s\n' 'tar -xzf "${tarball}" -C "${runner_dir}"'
    # shellcheck disable=SC2016
    printf '%s\n' 'printf "%s" "${version}" > "${runner_dir}/.box-runner-installed-version"'
  } > "${helper}"
  chmod 0555 "${helper}"
  run_as_runner "${helper}" "${RUNNER_DIR}" "${tarball}" "${ACTIONS_RUNNER_VERSION}" \
    || fail "RUNNER_USER could not create and extract ${RUNNER_DIR}"
  [ -x "${RUNNER_DIR}/run.sh" ] || fail "runner extraction did not create executable run.sh"
  info "RUNNER_USER ${RUNNER_USER} installed actions-runner at ${RUNNER_DIR}"
  cleanup_install_tmp
  INSTALL_TMP_DIR=""
}

main() {
  if [ "${INSTALL_ROOT}" = "/" ] && [ "$(id -u)" -ne 0 ]; then
    fail "production install must run as root"
  fi
  if [ "$(id -u)" -eq 0 ] && [ "${INSTALL_OWNER_UID}" != "0" ]; then
    fail "BOX_RUNNER_INSTALL_OWNER_UID is only for unprivileged staged tests"
  fi
  if [ "$(id -u)" -eq 0 ] && [ -n "${RUNNER_USER_EXEC}" ]; then
    fail "BOX_RUNNER_RUN_AS_USER is only for unprivileged staged tests"
  fi
  require_fresh_runner_dir
  install_control_plane
  install_actions_runner
  info "done. Edit ${CONFIG_FILE}, then run: launchctl bootstrap system ${PLIST_FILE}"
}

if [ "${BOX_RUNNER_INSTALL_SOURCE_ONLY:-0}" != "1" ]; then
  main "$@"
fi
