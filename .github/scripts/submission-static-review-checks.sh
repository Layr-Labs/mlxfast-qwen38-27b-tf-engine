#!/usr/bin/env bash
# Deterministic, offline half of the submission static review: the mechanical
# caps and path rules that must hold BEFORE any judge is asked anything, and
# before any submitted code is built or dispatched.
#
# Based on the static review in Layr-Labs/qwen-3.8-mtp-challenge revision
# bfab0de. The local restriction summary is in docs/submission-restriction-spec.md.
#
# WHAT IS NOT HERE. The LLM kernel-bypass judge needs a credential and a
# network, so it cannot run pre-dispatch on an offline box. This script covers
# the deterministic checks that can run as tests, and it fails closed on its
# own.
#
# Usage (exactly one mode must be selected, explicitly):
#   CONTRACT_PATH=benchmark.json \
#   MLXFAST_SUBMISSION_REVIEW_BASE_SHA=<sha> [HEAD_SHA=<sha>] \
#   [MLXFAST_SUBMISSION_STATIC_REVIEW_OUT_DIR=<dir>] \
#   .github/scripts/submission-static-review-checks.sh
#
#   CONTRACT_PATH=benchmark.json \
#   MLXFAST_SUBMISSION_REVIEW_MODE=whole-surface \
#   [MLXFAST_SUBMISSION_STATIC_REVIEW_OUT_DIR=<dir>] \
#   .github/scripts/submission-static-review-checks.sh
#
# Run from the checkout being reviewed (the trusted checkout after overlay, or
# the submission checkout in diff mode).
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null && pwd -P)"
HARDENED_GIT="${SCRIPT_DIR}/hardened-git.sh"

CONTRACT_PATH="${CONTRACT_PATH:-benchmark.json}"

if ! command -v jq >/dev/null 2>&1; then
  echo "::error::jq is required to read ${CONTRACT_PATH}" >&2
  exit 1
fi
if [[ ! -s "${CONTRACT_PATH}" ]]; then
  echo "::error file=${CONTRACT_PATH}::benchmark contract is missing or empty" >&2
  exit 1
fi

### Which contract states the rules ###########################################
#
# DIFF MODE (MLXFAST_SUBMISSION_REVIEW_BASE_SHA set) reviews an UNTRUSTED
# checkout before the overlay, so every byte of that work tree --
# benchmark.json included -- is attacker-controlled. Therefore every rule this
# script enforces is resolved from the review base
# blob and never from the work tree: the byte caps, the track id, the editable
# allowlist and the exemptions alike. Reading any of them from the work tree is
# a self-widening primitive -- a submission that raises its own maxGrowthBytes
# passes a growth its base contract refuses, and one that rewrites
# staticReviewTrackId selects which policy judges it.
#
# WHOLE-SURFACE MODE reads the contract from the WORK TREE, which is only safe
# because it runs in the TRUSTED checkout after the overlay. Nothing in this
# script can verify that precondition, so the mode requires this explicit
# opt-in: MLXFAST_SUBMISSION_REVIEW_MODE=whole-surface. A caller without a base
# SHA or the explicit opt-in gets a refusal.
#
# MODE SELECTION, in full:
#   BASE_SHA set-but-empty                    -> fatal (base computation failed)
#   BASE_SHA non-empty, MODE unset or "diff"  -> diff mode
#   BASE_SHA non-empty, MODE anything else    -> fatal (contradictory request)
#   BASE_SHA unset, MODE "whole-surface"      -> whole-surface mode
#   BASE_SHA unset, MODE anything else        -> fatal (no mode selected)
review_base="${MLXFAST_SUBMISSION_REVIEW_BASE_SHA:-}"
review_head="${HEAD_SHA:-HEAD}"
review_mode="${MLXFAST_SUBMISSION_REVIEW_MODE:-}"

# Set-but-empty means the caller intended diff-only mode but its base
# computation failed silently (a command substitution in a prefix assignment
# is invisible to set -e). Never degrade to whole-surface review over that.
if [[ -n "${MLXFAST_SUBMISSION_REVIEW_BASE_SHA+set}" && -z "${MLXFAST_SUBMISSION_REVIEW_BASE_SHA}" ]]; then
  echo "::error::MLXFAST_SUBMISSION_REVIEW_BASE_SHA is set but empty (did git merge-base fail?); refusing to fall back to whole-surface review" >&2
  exit 1
fi

if [[ -n "${review_base}" ]]; then
  if [[ -n "${review_mode}" && "${review_mode}" != "diff" ]]; then
    echo "::error::MLXFAST_SUBMISSION_REVIEW_MODE='${review_mode}' contradicts MLXFAST_SUBMISSION_REVIEW_BASE_SHA (a base sha selects diff mode); refusing an ambiguous review request" >&2
    exit 1
  fi
elif [[ "${review_mode}" != "whole-surface" ]]; then
  echo "::error::no review mode selected: set MLXFAST_SUBMISSION_REVIEW_BASE_SHA for diff mode, or MLXFAST_SUBMISSION_REVIEW_MODE=whole-surface to review the work-tree contract and the whole editable surface (only valid in the TRUSTED checkout, after the overlay). Got MLXFAST_SUBMISSION_REVIEW_MODE='${review_mode}'" >&2
  exit 1
fi

if [[ -n "${review_base}" ]]; then
  if ! command -v git >/dev/null 2>&1 || ! "${HARDENED_GIT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "::error::MLXFAST_SUBMISSION_REVIEW_BASE_SHA is set but this is not a git work tree" >&2
    exit 1
  fi
  if ! review_base="$("${HARDENED_GIT}" rev-parse --verify --quiet "${review_base}^{commit}")"; then
    echo "::error::submission review base '${MLXFAST_SUBMISSION_REVIEW_BASE_SHA}' is not a resolvable commit" >&2
    exit 1
  fi
  if ! review_head="$("${HARDENED_GIT}" rev-parse --verify --quiet "${review_head}^{commit}")"; then
    echo "::error::submission review head '${HEAD_SHA:-HEAD}' is not a resolvable commit" >&2
    exit 1
  fi
  # The diff selects paths from commits but collect_file reads the work tree;
  # those only agree when the work tree is the checkout of the review head.
  if [[ "$("${HARDENED_GIT}" rev-parse HEAD)" != "${review_head}" ]]; then
    echo "::error::review head ${review_head} is not the checked-out HEAD; work-tree content would not match the reviewed diff" >&2
    exit 1
  fi
  if ! contract_source="$("${HARDENED_GIT}" show "${review_base}:${CONTRACT_PATH}")"; then
    echo "::error::cannot read ${CONTRACT_PATH} from review base ${review_base}" >&2
    exit 1
  fi
  contract_origin="${CONTRACT_PATH} at review base ${review_base}"
else
  contract_source="$(cat "${CONTRACT_PATH}")"
  contract_origin="${CONTRACT_PATH}"
fi

# A contract that does not parse must be loud. Every read below goes through a
# process substitution or a command substitution, where a jq failure is
# invisible to set -e and reads as "the contract declares nothing" -- i.e. as
# fallback caps and an empty allowlist.
if ! jq -e 'type == "object"' >/dev/null 2>&1 <<<"${contract_source}"; then
  echo "::error::${contract_origin} does not parse as a JSON object" >&2
  exit 1
fi

### The byte caps #############################################################
#
# The caps come from the contract SOURCE resolved above, not from constants in
# this file. The manifest is the single source. The fallbacks below cover only
# a contract that
# declares none.
#
# A cap key that is PRESENT must be a positive integer JSON NUMBER. Anything
# else -- "2000000", false, null, 3.5 -- is malformed and fails closed, key for
# key. The parser keeps each JSON type distinct: false and null are invalid, and
# a string such as "2000000" cannot pass the positive-integer number check or
# widen the cap.
#
# Resolution is `manifest > constant`. The manifest is the single source. The
# environment cannot override the three byte caps. A cap restated out of band
# could differ from the reviewed contract without leaving a record.
#
# Do not add an environment override. The trusted contract must remain the
# only source of byte caps.
budget_kind="$(jq -r \
  'if has("editableSurfaceByteBudget") then (.editableSurfaceByteBudget | type) else "absent" end' \
  <<<"${contract_source}")"
if [[ "${budget_kind}" != "absent" && "${budget_kind}" != "object" ]]; then
  echo "::error::editableSurfaceByteBudget in ${contract_origin} is a ${budget_kind}, not an object; refusing to review under a malformed budget" >&2
  exit 1
fi

resolve_cap() { # resolve_cap VAR_NAME CONTRACT_KEY FALLBACK
  local var="$1" key="$2" fallback="$3" kind value
  if [[ "${budget_kind}" != "object" ]]; then
    printf -v "${var}" '%s' "${fallback}"
    return 0
  fi
  kind="$(jq -r --arg k "${key}" \
    'if (.editableSurfaceByteBudget | has($k)) then (.editableSurfaceByteBudget[$k] | type) else "absent" end' \
    <<<"${contract_source}")"
  case "${kind}" in
    absent)
      printf -v "${var}" '%s' "${fallback}"
      ;;
    number)
      # `. + 0` normalises the literal before printing it. jq preserves the
      # source spelling of a number otherwise, so `5e5` would stringify to
# "5E+5" and be rejected by the integer regex below. Normalize it to 500000.
# The contract parser and this script must
      # not disagree about a legal cap any more than about a malformed one.
      value="$(jq -r --arg k "${key}" '.editableSurfaceByteBudget[$k] | . + 0 | tostring' <<<"${contract_source}")"
      printf -v "${var}" '%s' "${value}"
      ;;
    *)
      echo "::error::editableSurfaceByteBudget.${key} in ${contract_origin} is a ${kind}, not a positive integer; refusing a malformed cap instead of substituting one" >&2
      exit 1
      ;;
  esac
}

# THE FALLBACKS BELOW MUST EQUAL benchmark.json's DECLARED CAPS. They cover only
# a contract that declares no caps, but a fallback that disagrees with the
# declaration is never harmless: a SMALLER one false-rejects a submission that
# the contract says is in budget, a LARGER one admits bytes the contract does
# not, and either way the two enforcers stop agreeing about what the budget is.
#
# The current TensorFold editable surface uses the caps declared in
# benchmark.json. The linter and security tests verify that these fallbacks
# stay equal to that declaration. Change them only with the contract.
resolve_cap MAX_BYTES maxTotalBytes 55829628
resolve_cap MAX_FILE_BYTES maxFileBytes 2500000
resolve_cap MAX_GROWTH_BYTES maxGrowthBytes 8388608

for cap_name in MAX_BYTES MAX_FILE_BYTES MAX_GROWTH_BYTES; do
  if ! [[ "${!cap_name}" =~ ^[1-9][0-9]*$ ]]; then
    echo "::error::${cap_name} must be a positive integer, got '${!cap_name}'" >&2
    exit 1
  fi
done
if (( MAX_FILE_BYTES > MAX_BYTES )); then
  echo "::error::maxFileBytes (${MAX_FILE_BYTES}) exceeds maxTotalBytes (${MAX_BYTES}); the per-file cap can never bind" >&2
  exit 1
fi

# Track selection. An unset variable uses the serial policy; a set-but-empty
# variable fails closed. Applying a speculative-decode policy that does not
# match the contract can invert the verdict, so an unknown id is fatal. This
# repository ranks one track and declares its id in the manifest. Read it, like
# the caps, from the trusted contract source and never from the reviewed work
# tree.
if [[ -n "${MLXFAST_SUBMISSION_TRACK_ID+set}" && -z "${MLXFAST_SUBMISSION_TRACK_ID}" ]]; then
  echo "::error::MLXFAST_SUBMISSION_TRACK_ID is set but empty (did the workflow track env fail to resolve?); refusing to fall back to a default review policy" >&2
  exit 1
fi
CONTRACT_TRACK_ID="$(jq -r '.staticReviewTrackId // empty' <<<"${contract_source}")"
if [[ -z "${CONTRACT_TRACK_ID}" ]]; then
  echo "::error::staticReviewTrackId is missing from ${contract_origin}; refusing to review under an unnamed policy" >&2
  exit 1
fi
TRACK_ID="${MLXFAST_SUBMISSION_TRACK_ID:-${CONTRACT_TRACK_ID}}"
if [[ "${TRACK_ID}" != "${CONTRACT_TRACK_ID}" ]]; then
  echo "::error::submission static-review track '${TRACK_ID}' is not the track this contract declares ('${CONTRACT_TRACK_ID}')" >&2
  exit 1
fi

OUT_DIR="${MLXFAST_SUBMISSION_STATIC_REVIEW_OUT_DIR:-}"
if [[ -z "${OUT_DIR}" ]]; then
  OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/submission-review.XXXXXX")"
  trap 'rm -rf "${OUT_DIR}"' EXIT
else
  mkdir -p "${OUT_DIR}"
fi
files_ndjson="${OUT_DIR}/files.ndjson"
: > "${files_ndjson}"
submission_diff_path="${OUT_DIR}/submission.diff"
: > "${submission_diff_path}"

validate_contract_path() {
  local path="$1"
  # A leading ':' would be git pathspec magic in the diff below, not a path.
  if [[ -z "${path}" || "${path}" == /* || "${path}" == :* || "${path}" == *\\* ]]; then
    echo "::error::invalid editable path '${path}' in ${contract_origin}" >&2
    exit 1
  fi
  case "/${path}/" in
    *"/../"*|*"/./"*)
      echo "::error::invalid editable path '${path}' in ${contract_origin}" >&2
      exit 1
      ;;
  esac
}

# This track has no byte-budget exemptions. Every editable path stays in the
# reviewed payload and counts against the byte budget.
exempt_paths=()
is_exempt_path() {
  local candidate="$1" exempt
  for exempt in ${exempt_paths[@]+"${exempt_paths[@]}"}; do
    if [[ "${candidate}" == "${exempt}" || "${candidate}" == "${exempt}/"* ]]; then
      return 0
    fi
  done
  return 1
}

total_bytes=0
file_count=0
deleted_path_count=0
stale_clone_hint=""

# Append one file to the review payload (size-capped, aborts on overflow).
collect_file() {
  local file_path="$1"
  [[ -f "${file_path}" ]] || return 0
  local bytes
  bytes="$(wc -c < "${file_path}" | tr -d ' ')"
  if ! [[ "${bytes}" =~ ^[0-9]+$ ]]; then
    echo "::error file=${file_path}::could not determine file size" >&2
    exit 1
  fi
  if (( bytes > MAX_FILE_BYTES )); then
    echo "::error file=${file_path}::editable file is ${bytes} bytes, above the per-file static review limit ${MAX_FILE_BYTES}; refusing an oversized file that could hide lookup tables" >&2
    exit 1
  fi
  total_bytes=$((total_bytes + bytes))
  file_count=$((file_count + 1))
  if (( total_bytes > MAX_BYTES )); then
    echo "::error::editable submission source is ${total_bytes} bytes, above static review limit ${MAX_BYTES}; refusing oversized source that could hide lookup tables${stale_clone_hint}" >&2
    exit 1
  fi
  jq -n \
    --arg path "${file_path}" \
    --argjson bytes "${bytes}" \
    --rawfile content "${file_path}" \
    '{path: $path, bytes: $bytes, content: $content}' >> "${files_ndjson}"
}

# Diff-only review: when a base commit is provided, review only the editable
# files this submission actually CHANGED versus its merge-base with main.
# Unchanged editable files are byte-identical to trusted main content, so
# feeding them to a judge only adds false-positive surface. Without a base,
# MLXFAST_SUBMISSION_REVIEW_MODE=whole-surface is required to review the whole
# editable surface. `review_base` and `review_head` were resolved above,
# together with the trusted contract source they select.

# The allowlist and the exemptions come from the same trusted contract source
# as the caps and the track id -- in diff mode the review BASE blob -- so
# nothing in the submitted work tree can steer which files are reviewed or
# which are held out of the budget.
editable_paths=()
while IFS= read -r editable_path; do
  editable_paths+=("${editable_path}")
done < <(jq -r '.editablePaths[]' <<<"${contract_source}")
while IFS= read -r exempt_path; do
  [[ -n "${exempt_path}" ]] && exempt_paths+=("${exempt_path}")
done < <(jq -r '(.editableSurfaceByteBudget.exemptPaths // [])[]' <<<"${contract_source}")

# A jq failure inside a process substitution is also invisible to set -e; an
# empty allowlist must be an error, never an accidental clean pass.
if (( ${#editable_paths[@]} == 0 )); then
  echo "::error::${contract_origin} lists no editablePaths for static review" >&2
  exit 1
fi

# Exemptions are paths, and a malformed one must be as loud here as a malformed
# editable path: they both end up in a git pathspec.
for exempt_path in ${exempt_paths[@]+"${exempt_paths[@]}"}; do
  validate_contract_path "${exempt_path}"
done

# The set actually reviewed. Exempt paths keep their editable status (the
# surface gate still admits edits to them) and simply never enter the payload,
# the diff or the growth arithmetic.
reviewed_paths=()
for editable_path in "${editable_paths[@]}"; do
  validate_contract_path "${editable_path}"
  if is_exempt_path "${editable_path}"; then
    echo "submission-review: editable path ${editable_path} is byte-budget exempt; excluded from the reviewed payload"
    continue
  fi
  reviewed_paths+=("${editable_path}")
done
if (( ${#reviewed_paths[@]} == 0 )); then
  echo "::error::${contract_origin} lists no reviewable editablePaths (every entry is byte-budget exempt)" >&2
  exit 1
fi

if [[ -n "${review_base}" ]]; then
  changed_paths_file="${OUT_DIR}/changed-paths.z"
  changed_present_paths_file="${OUT_DIR}/changed-present-paths.z"
  changed_path_count=0

  # Capture both the complete changed-path set (including deletions) and the
  # changed files still present in the head. Use regular temporary files so a
  # git failure cannot disappear inside process-substitution status handling.
  "${HARDENED_GIT}" diff --name-only -z "${review_base}" "${review_head}" -- "${reviewed_paths[@]}" \
    > "${changed_paths_file}"
  "${HARDENED_GIT}" diff --name-only -z --diff-filter=d "${review_base}" "${review_head}" -- "${reviewed_paths[@]}" \
    > "${changed_present_paths_file}"

  while IFS= read -r -d '' file_path; do
    changed_path_count=$((changed_path_count + 1))
  done < "${changed_paths_file}"

  # Editable files DELETED versus the trusted base are the signature of a
  # submission packaged from a stale clone: the submit pipeline rebuilds the
  # full editable surface from the archive on top of current main, so a clone
  # that predates an editable-surface expansion silently deletes every editable
  # file it never had. Those deletion hunks are trusted-base content, but they
  # still inflate the reviewed diff past the size cap, so surface the real
  # cause and the remedy instead of only the lookup-table refusal.
  deleted_paths_file="${OUT_DIR}/deleted-paths.z"
  "${HARDENED_GIT}" diff --name-only -z --diff-filter=D "${review_base}" "${review_head}" -- "${reviewed_paths[@]}" \
    > "${deleted_paths_file}"
  first_deleted_path=""
  while IFS= read -r -d '' file_path; do
    deleted_path_count=$((deleted_path_count + 1))
    if [[ -z "${first_deleted_path}" ]]; then
      first_deleted_path="${file_path}"
    fi
  done < "${deleted_paths_file}"
  if (( deleted_path_count > 0 )); then
    stale_clone_hint=". Note: this submission deletes ${deleted_path_count} editable file(s) that exist on the trusted base (first: ${first_deleted_path}); that usually means it was packaged from a stale clone that predates the current editablePaths surface, not deliberate deletion. Re-sync the clone with current main and resubmit"
    echo "submission-review: ${deleted_path_count} editable file(s) deleted versus base ${review_base} (first: ${first_deleted_path}); if unintentional, the submission likely came from a stale clone" >&2
  fi

  if (( changed_path_count == 0 )); then
    echo "submission-review: no editable files changed versus ${review_base}; nothing to review"
    exit 0
  fi

  # Growth cap: the TOTAL cap alone would let a submission that touches only
  # small files hide hundreds of kilobytes of lookup tables inside one edit.
  # Bound the net bytes this submission ADDS to the editable surface versus its
  # review base (committed blob sizes on both sides, so the work tree cannot
  # skew the arithmetic).
  base_surface_bytes=0
  head_surface_bytes=0
  while IFS= read -r -d '' changed_path; do
    if "${HARDENED_GIT}" cat-file -e "${review_base}:${changed_path}" 2>/dev/null; then
      base_surface_bytes=$((base_surface_bytes + $("${HARDENED_GIT}" cat-file -s "${review_base}:${changed_path}")))
    fi
    if "${HARDENED_GIT}" cat-file -e "${review_head}:${changed_path}" 2>/dev/null; then
      head_surface_bytes=$((head_surface_bytes + $("${HARDENED_GIT}" cat-file -s "${review_head}:${changed_path}")))
    fi
  done < "${changed_paths_file}"
  surface_growth_bytes=$((head_surface_bytes - base_surface_bytes))
  if (( surface_growth_bytes > MAX_GROWTH_BYTES )); then
    echo "::error::submission grows the editable surface by ${surface_growth_bytes} bytes (changed files: ${base_surface_bytes} -> ${head_surface_bytes}), above the growth limit ${MAX_GROWTH_BYTES}; refusing growth that could hide lookup tables" >&2
    exit 1
  fi
  echo "submission-review: editable surface growth ${surface_growth_bytes} bytes across ${changed_path_count} changed path(s) (limit ${MAX_GROWTH_BYTES})"

  while IFS= read -r -d '' file_path; do
    # Every non-deleted path the diff lists must agree with the review head.
    if [[ -h "${file_path}" || ! -f "${file_path}" ]]; then
      echo "::error file=${file_path}::changed editable path is missing or not a regular file in the checkout" >&2
      exit 1
    fi
    collect_file "${file_path}"
  done < "${changed_present_paths_file}"

  # Also capture the unified diff: changed FILES are collected whole (context),
  # but a verdict must be about what this submission CHANGED. It counts against
  # the same total cap.
  "${HARDENED_GIT}" diff "${review_base}" "${review_head}" -- "${reviewed_paths[@]}" > "${submission_diff_path}"
  diff_bytes="$(wc -c < "${submission_diff_path}" | tr -d ' ')"
  if ! [[ "${diff_bytes}" =~ ^[0-9]+$ ]]; then
    echo "::error::could not determine submission diff size" >&2
    exit 1
  fi
  total_bytes=$((total_bytes + diff_bytes))
  if (( total_bytes > MAX_BYTES )); then
    echo "::error::editable submission source plus diff is ${total_bytes} bytes, above static review limit ${MAX_BYTES}; refusing oversized source that could hide lookup tables${stale_clone_hint}" >&2
    exit 1
  fi
else
  for editable_path in "${reviewed_paths[@]}"; do
    if [[ ! -e "${editable_path}" ]]; then
      echo "::error file=${editable_path}::editable path missing after overlay" >&2
      exit 1
    fi
    # Whole-surface mode rejects symlinks before byte counting because
    # `find -type f` does not count a link or the payload it reaches. Reject any
    # symlink -- or any other non-regular, non-directory entry -- under the
    # reviewed surface, matching both the diff-mode check above and the overlay's
    # validate_overlay_tree.
    if find "${editable_path}" -type l -print -quit | grep -q .; then
      echo "::error file=${editable_path}::editable path is or contains a symlink; refusing a whole-surface review that would byte-count through a link" >&2
      exit 1
    fi
    if find "${editable_path}" ! -type f ! -type d -print -quit | grep -q .; then
      echo "::error file=${editable_path}::editable path contains a non-regular file; refusing a whole-surface review of a surface that is not plain files and directories" >&2
      exit 1
    fi
    while IFS= read -r -d '' file_path; do
      collect_file "${file_path}"
    done < <(find "${editable_path}" -type f -print0)
  done

  if (( file_count == 0 )); then
    echo "::error::editable paths selected no files for static review" >&2
    exit 1
  fi
fi

echo "submission-review: deterministic checks passed for track ${TRACK_ID} (files=${file_count} bytes=${total_bytes} limits total=${MAX_BYTES} file=${MAX_FILE_BYTES} growth=${MAX_GROWTH_BYTES})"
