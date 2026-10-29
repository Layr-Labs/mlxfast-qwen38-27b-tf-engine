#!/usr/bin/env bash
#
# fetch-benchd.sh -- resolve the benchd measurement harness.
#
# The release manifest pins each binary by SHA-256 digest and byte count. The
# script also requires the manifest source commit below. It installs the
# verified manifest beside the binaries so every run records its provenance.
# Participants cannot edit this script, the destination, or the release pin.
#
# WHAT IT DOES, in order:
#   1. If benchd, record-correctness-golden, and benchd.manifest.json exist and
#      both binaries agree with their sha256 + bytes pins, use them without the
#      network. A missing or mismatched file refuses.
#      Set BENCHD_REFRESH=1 to discard the local pair and re-resolve the
#      channel tip.
#   2. Otherwise obtain all three files -- from BENCHD_DIST_LOCAL (file/dir, for
#      air-gapped boxes) or by downloading manifest-then-binary from the
#      channel -- verify the binary against its manifest, and install both.
#
# Prints the absolute path of the verified binary on STDOUT (diagnostics go to
# stderr), so callers can do:  BENCHD="$(./tools/fetch-benchd.sh)"
#
# Env:
#   BENCHD_BRANCH       channel branch. Default: main -- benchd is published
#                       from main, so there is no per-track release branch.
#                       THE BRANCH IS THE CHANNEL, NOT THE TRACK. The track id
#                       is qwen3.8-27b-tf-mlx-v1 and names the leaderboard
#                       namespace, runner labels, and R2 prefix. This variable
#                       and the manifest branch check are the only places the
#                       channel name belongs.
#   BENCHD_BIN_DIR      install directory. Default: <repo>/benchd-bin
#                       Both binaries and their manifest are installed here.
#   BENCHD_REFRESH      set to 1 to discard an already-installed pair and
#                       re-resolve the channel tip.
#   BENCHD_DIST_LOCAL   path to an already-obtained dist (a directory holding
#                       both binaries + benchd.manifest.json, or the benchd file
#                       with both files beside it). Verified, never trusted bare.
#   BENCHD_DIST_BASE_URL
#                       raw host and repository prefix. Default: the public
#                       benchd repository (no token needed). Set it only for a
#                       mirror; the manifest and source-commit pins still apply.
#   BENCHD_DIST_TOKEN / GITHUB_TOKEN
#                       bearer token for a private channel repo.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BRANCH="${BENCHD_BRANCH:-main}"
DEST_DIR="${BENCHD_BIN_DIR:-${REPO_ROOT}/benchd-bin}"
DEST="${DEST_DIR}/benchd"
DEST_MANIFEST="${DEST_DIR}/benchd.manifest.json"
RECORDER_NAME="record-correctness-golden"
DEST_RECORDER="${DEST_DIR}/${RECORDER_NAME}"
# The default is the public benchd channel described above.
BASE_URL="${BENCHD_DIST_BASE_URL:-https://raw.githubusercontent.com/Layr-Labs/benchd}"
# Two benchd commits are pinned, by design. PINNED_SOURCE_COMMIT is the benchd
# SOURCE commit that built the binaries (the dist manifest's source_commit).
# The track fixture's benchd_source_commit (8ce0ca78...) is the benchd DIST
# commit that published those binaries on top of that source commit.
PINNED_SOURCE_COMMIT="3101418f95eb074cc9c100f8dc61cc85cc924155"

die() {
  echo "fetch-benchd.sh: $*" >&2
  exit 1
}

command -v shasum >/dev/null 2>&1 || die "shasum is required to verify benchd."

# -- manifest -----------------------------------------------------------------
# benchd.manifest.json is values-only JSON, one key per line (build-dist.sh).
# Parsed with sed rather than jq/python3 so the OFFLINE path on the ranked box
# needs nothing but a shell and shasum.
# The FIRST match only: the six top-level fields come first and describe
# benchd; the `binaries` map that follows repeats sha256/bytes per binary,
# and a second match would make the field two lines and the resolve refuse
# a good manifest.
manifest_field() {
  sed -n "s/^[[:space:]]*\"$2\"[[:space:]]*:[[:space:]]*\"\{0,1\}\([^\",]*\)\"\{0,1\}[[:space:]]*,\{0,1\}[[:space:]]*\$/\1/p" "$1" | head -n 1
}

manifest_binary_field() {
  sed -n "s/^[[:space:]]*\"$2\"[[:space:]]*:[[:space:]]*{.*\"$3\"[[:space:]]*:[[:space:]]*\"\{0,1\}\([^\",}]*\)\"\{0,1\}.*}[[:space:]]*,\{0,1\}[[:space:]]*\$/\1/p" "$1"
}

# Load + validate a manifest file; sets MF_BRANCH/MF_COMMIT/MF_SHA256/MF_BYTES.
# An unparseable manifest must refuse, not degrade: empty fields would make
# every comparison below trivially "match".
load_manifest() {
  local path="$1"
  [[ -f "${path}" ]] || return 1
  MF_BRANCH="$(manifest_field "${path}" branch)"
  MF_COMMIT="$(manifest_field "${path}" source_commit)"
  MF_SHA256="$(manifest_field "${path}" sha256)"
  MF_BYTES="$(manifest_field "${path}" bytes)"
  MF_RECORDER_SHA256="$(manifest_binary_field "${path}" "${RECORDER_NAME}" sha256)"
  MF_RECORDER_BYTES="$(manifest_binary_field "${path}" "${RECORDER_NAME}" bytes)"
  if [[ "${#MF_SHA256}" -ne 64 || -n "${MF_SHA256//[0-9a-f]/}" ]]; then
    echo "fetch-benchd.sh:   manifest sha256 is not 64 lowercase hex characters: '${MF_SHA256}' (${path})" >&2
    return 1
  fi
  if [[ -z "${MF_BYTES}" || -n "${MF_BYTES//[0-9]/}" ]]; then
    echo "fetch-benchd.sh:   manifest bytes is not a positive integer: '${MF_BYTES}' (${path})" >&2
    return 1
  fi
  if [[ "${#MF_RECORDER_SHA256}" -ne 64 || -n "${MF_RECORDER_SHA256//[0-9a-f]/}" ]]; then
    echo "fetch-benchd.sh:   manifest ${RECORDER_NAME} sha256 is not 64 lowercase hex characters: '${MF_RECORDER_SHA256}' (${path})" >&2
    return 1
  fi
  if [[ -z "${MF_RECORDER_BYTES}" || -n "${MF_RECORDER_BYTES//[0-9]/}" ]]; then
    echo "fetch-benchd.sh:   manifest ${RECORDER_NAME} bytes is not a positive integer: '${MF_RECORDER_BYTES}' (${path})" >&2
    return 1
  fi
  if [[ "${MF_BRANCH}" != "${BRANCH}" ]]; then
    echo "fetch-benchd.sh:   manifest names branch '${MF_BRANCH}', expected '${BRANCH}' -- wrong channel (${path})" >&2
    return 1
  fi
  if [[ "${MF_COMMIT}" != "${PINNED_SOURCE_COMMIT}" ]]; then
    echo "fetch-benchd.sh:   manifest source_commit '${MF_COMMIT}', expected pinned '${PINNED_SOURCE_COMMIT}' (${path})" >&2
    return 1
  fi
  return 0
}

matches_recorder_manifest() {
  local path="$1" actual_bytes actual_sha
  [[ -f "${path}" ]] || return 1
  actual_bytes="$(wc -c < "${path}" | tr -d '[:space:]')"
  if [[ "${actual_bytes}" != "${MF_RECORDER_BYTES}" ]]; then
    echo "fetch-benchd.sh:   ${RECORDER_NAME} byte count mismatch: manifest=${MF_RECORDER_BYTES} actual=${actual_bytes} (${path})" >&2
    return 1
  fi
  actual_sha="$(shasum -a 256 "${path}" | awk '{print $1}')"
  if [[ "${actual_sha}" != "${MF_RECORDER_SHA256}" ]]; then
    echo "fetch-benchd.sh:   ${RECORDER_NAME} sha256 mismatch: manifest=${MF_RECORDER_SHA256} actual=${actual_sha} (${path})" >&2
    return 1
  fi
  return 0
}

# Verify a binary against the LOADED manifest. Bytes first (cheap, and a length
# mismatch is already disqualifying), then the digest. Reports on stderr so
# callers see WHICH half failed.
matches_manifest() {
  local path="$1" actual_bytes actual_sha
  [[ -f "${path}" ]] || return 1
  actual_bytes="$(wc -c < "${path}" | tr -d '[:space:]')"
  if [[ "${actual_bytes}" != "${MF_BYTES}" ]]; then
    echo "fetch-benchd.sh:   byte count mismatch: manifest=${MF_BYTES} actual=${actual_bytes} (${path})" >&2
    return 1
  fi
  actual_sha="$(shasum -a 256 "${path}" | awk '{print $1}')"
  if [[ "${actual_sha}" != "${MF_SHA256}" ]]; then
    echo "fetch-benchd.sh:   sha256 mismatch: manifest=${MF_SHA256} actual=${actual_sha} (${path})" >&2
    return 1
  fi
  return 0
}

announce() {
  echo "fetch-benchd.sh: benchd identity: branch=${MF_BRANCH} source_commit=${MF_COMMIT} sha256=${MF_SHA256} bytes=${MF_BYTES}" >&2
  echo "fetch-benchd.sh: ${RECORDER_NAME} identity: sha256=${MF_RECORDER_SHA256} bytes=${MF_RECORDER_BYTES}" >&2
}

# -- 1. already in place (the offline path) -----------------------------------
if [[ "${BENCHD_REFRESH:-0}" == "1" && -f "${DEST}" ]]; then
  echo "fetch-benchd.sh: BENCHD_REFRESH=1 -- discarding the installed pair to re-resolve the channel tip" >&2
  rm -f "${DEST}" "${DEST_MANIFEST}" "${DEST_RECORDER}"
fi

if [[ -f "${DEST}" ]]; then
  load_manifest "${DEST_MANIFEST}" \
    || die "${DEST} exists but ${DEST_MANIFEST} is missing or malformed; a binary with no manifest is unattributable and will not be run. Place the channel's benchd.manifest.json beside it, or delete the binary and re-run."
  matches_manifest "${DEST}" \
    || die "${DEST} does NOT match its manifest (see the mismatch above); refusing to run or replace it. Delete the pair deliberately, then re-run."
  [[ -f "${DEST_RECORDER}" ]] \
    || die "${DEST_MANIFEST} declares ${RECORDER_NAME}, but ${DEST_RECORDER} is missing."
  matches_recorder_manifest "${DEST_RECORDER}" \
    || die "${DEST_RECORDER} does NOT match its manifest entry; refusing to run or replace it."
  # A pinned pair the box staged (owner: the operator, mode 0750, group of the
  # runner) is already executable for the job, and the job's account may not
  # change its mode: chmod by a non-owner is EPERM. Set the mode only when the
  # binary is not executable for this caller.
  [[ -x "${DEST}" ]] || chmod 755 "${DEST}"
  [[ -x "${DEST_RECORDER}" ]] || chmod 755 "${DEST_RECORDER}"
  announce
  printf '%s\n' "${DEST}"
  exit 0
fi

# -- 2. obtain the pair -------------------------------------------------------
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT
STAGED="${TMP_DIR}/benchd"
STAGED_MANIFEST="${TMP_DIR}/benchd.manifest.json"
STAGED_RECORDER="${TMP_DIR}/${RECORDER_NAME}"

if [[ -n "${BENCHD_DIST_LOCAL:-}" ]]; then
  src="${BENCHD_DIST_LOCAL}"
  [[ -d "${src}" ]] && src="${src}/benchd"
  [[ -f "${src}" ]] || die "BENCHD_DIST_LOCAL does not resolve to a benchd file: ${src}"
  src_manifest="$(dirname "${src}")/benchd.manifest.json"
  [[ -f "${src_manifest}" ]] || die "BENCHD_DIST_LOCAL has no benchd.manifest.json beside the binary (${src_manifest}); an unattributable binary will not be installed."
  echo "fetch-benchd.sh: taking benchd from ${src}" >&2
  cp "${src}" "${STAGED}"
  cp "${src_manifest}" "${STAGED_MANIFEST}"
  load_manifest "${STAGED_MANIFEST}" \
    || die "BENCHD_DIST_LOCAL manifest is malformed or names the wrong channel; refusing."
  # One named source, so a mismatch is a hard refusal: silently continuing past
  # the file the operator explicitly pointed at would hide the real problem.
  matches_manifest "${STAGED}" \
    || die "BENCHD_DIST_LOCAL (${src}) does NOT match its manifest (see the mismatch above); refusing to install or run it."
  src_recorder="$(dirname "${src}")/${RECORDER_NAME}"
  [[ -f "${src_recorder}" ]] \
    || die "BENCHD_DIST_LOCAL manifest declares ${RECORDER_NAME}, but ${src_recorder} is missing."
  cp "${src_recorder}" "${STAGED_RECORDER}"
  matches_recorder_manifest "${STAGED_RECORDER}" \
    || die "BENCHD_DIST_LOCAL (${src_recorder}) does NOT match its manifest entry; refusing to install it."
else
  command -v curl >/dev/null 2>&1 || die "curl is required to download benchd (or set BENCHD_DIST_LOCAL)."

  # The channel: dist/ on the bench branch tip. The manifest is fetched FIRST,
  # then the binary from the same directory, and the binary must match the
  # manifest -- so a half-updated publish (one file at the old build) refuses
  # rather than installing a pair that disagrees. The `refs/heads/` prefix is
  # required, not cosmetic: branch names contain slashes, and without the
  # explicit ref namespace raw.githubusercontent cannot tell where the ref ends
  # and the path begins.
  DIST_DIR_URL="${BASE_URL}/refs/heads/${BRANCH}/dist"
  auth_token="${BENCHD_DIST_TOKEN:-${GITHUB_TOKEN:-}}"
  fetch() {
    local url="$1" out="$2"
    if [[ -n "${auth_token}" ]]; then
      curl -fsSL --retry 3 --retry-delay 2 -H "Authorization: Bearer ${auth_token}" -o "${out}" "${url}"
    else
      curl -fsSL --retry 3 --retry-delay 2 -o "${out}" "${url}"
    fi
  }

  echo "fetch-benchd.sh: resolving the ${BRANCH} channel tip (${DIST_DIR_URL})" >&2
  fetch "${DIST_DIR_URL}/benchd.manifest.json" "${STAGED_MANIFEST}" || {
    {
      echo "fetch-benchd.sh: could not fetch the channel manifest (${DIST_DIR_URL}/benchd.manifest.json)."
      if [[ -z "${auth_token}" ]]; then
        echo "  if the channel repo is private, set BENCHD_DIST_TOKEN/GITHUB_TOKEN."
      fi
      echo "  offline alternative: place benchd + benchd.manifest.json at ${DEST_DIR}/,"
      echo "  or point BENCHD_DIST_LOCAL at a directory holding both."
    } >&2
    exit 1
  }
  load_manifest "${STAGED_MANIFEST}" \
    || die "the channel manifest is malformed or names the wrong branch; refusing (this is a publish problem, not resolved by re-running)."
  fetch "${DIST_DIR_URL}/benchd" "${STAGED}" \
    || die "the channel manifest exists but the binary download failed (${DIST_DIR_URL}/benchd)."
  matches_manifest "${STAGED}" \
    || die "the downloaded benchd does NOT match the channel manifest (see the mismatch above) -- a half-updated publish; refusing. Republish dist bench-side, then re-run."
  fetch "${DIST_DIR_URL}/${RECORDER_NAME}" "${STAGED_RECORDER}" \
    || die "the channel manifest declares ${RECORDER_NAME}, but its download failed (${DIST_DIR_URL}/${RECORDER_NAME})."
  matches_recorder_manifest "${STAGED_RECORDER}" \
    || die "the downloaded ${RECORDER_NAME} does NOT match the channel manifest entry; refusing."
fi

# -- 3. install ONLY what was verified ----------------------------------------
mkdir -p "${DEST_DIR}"
chmod 755 "${STAGED}"
chmod 755 "${STAGED_RECORDER}"
mv "${STAGED}" "${DEST}"
mv "${STAGED_RECORDER}" "${DEST_RECORDER}"
mv "${STAGED_MANIFEST}" "${DEST_MANIFEST}"
echo "fetch-benchd.sh: installed benchd at ${DEST} (manifest beside it)" >&2
echo "fetch-benchd.sh: installed ${RECORDER_NAME} at ${DEST_RECORDER}" >&2
announce

printf '%s\n' "${DEST}"
