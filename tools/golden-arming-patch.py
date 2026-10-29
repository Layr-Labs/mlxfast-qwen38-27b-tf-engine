#!/usr/bin/env python3
"""Write the arming patch for fixtures/qwen3_8_27b_tf_mlx_v1_track.json.

The organizer records the track's goldens on the track's box. This tool reads
a directory of those recorded files and writes the fixture values that arm the
track. It computes every {sha256, bytes} pin itself. It reads the files; it
never prints their contents.

The directory holds these files, and nothing else:

  <name>.golden.json              the timed-pool tapes, recorded serial, at
                                  least one and all distinct. Each --live
                                  names one of them.
  warmup-01.golden.json           the fixture-declared warm-up tape.
  <live>.dflashN.golden.json         one per-depth oracle for every live golden
                                  and every depth in
                                  dflash_head.permitted_draft_depths.
  public_longcopy_gate_english_1024_256.json
  public_longcopy_gate_english_1024_1024.json
                                  the two public captures for local-iterate and
                                  local-submit. They are not uploads: they ship
                                  in git under correctness_prompts/. The tool
                                  checks them and prints where to copy them.

THE PER-DEPTH SHAPE IS DECIDED BY THE RECORDING. A per-depth oracle can differ
from the serial tape (a speculative round verifies at M > 1, and MLX dispatches
a different kernel there), and it can equal it. Every live_golden_speculative
key gets its own pin and its own named R2 object. Equal bytes therefore keep
equal sha256 and bytes pins, but the per-depth object still uses the required
<live>.dflashN.golden.json name. The upload list goes to stderr, followed by one
`ship:` line per public capture naming the repository path to copy it to.

Usage:
  python3 tools/golden-arming-patch.py --dir DIR --live NAME [--live NAME ...]
  python3 tools/golden-arming-patch.py --dir DIR --live NAME [--live NAME ...] --apply

Without --apply the tool prints the patch (JSON, values only) on stdout. With
--apply it writes the values into the fixture in place.
Exit: 0 on success, 1 on any refusal (printed with a REFUSE prefix).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTRACT = os.path.join(ROOT, "fixtures", "qwen3_8_27b_tf_mlx_v1_track.json")
PUBLIC_CAPTURES = {
    "local_iterate": (
        "public_longcopy_gate_english_1024_256.json",
        "correctness_prompts/public_longcopy_gate_english_1024_256.json",
    ),
    "local_submit": (
        "public_longcopy_gate_english_1024_1024.json",
        "correctness_prompts/public_longcopy_gate_english_1024_1024.json",
    ),
}
FORBIDDEN_BENCHMARK_KEYS = (
    "baseline_prefill_seconds_per_token",
    "baseline_decode_seconds_per_token",
)
GOLDEN_ROOT_KEYS = {
    "version", "model_type", "model_provenance", "cases", "correctness_gates", "benchmark",
}
CORRECTNESS_TOP_LOGITS = 8
CORRECTNESS_MAX_ANCHOR_CONTEXT_TOKENS = 1024
CORRECTNESS_MAX_FREE_RUN_STEPS = 256
CORRECTNESS_MAX_BEHAVIOR_PROMPT_TOKENS = 2048
CORRECTNESS_MAX_BEHAVIOR_STEPS = 128


class Refusal(Exception):
    pass


def validate_token(value: object, field: str, name: str, vocab_size: int) -> None:
    """Apply the timed-golden token range used by benchd."""
    if type(value) is not int or value < 0 or value >= vocab_size:
        raise Refusal(
            f"{name} has invalid {field}: token ids must be non-negative integers "
            f"below vocab_size {vocab_size}"
        )


def validate_tokens(values: object, field: str, name: str, vocab_size: int,
                    *, nonempty: bool = True) -> list[int]:
    if not isinstance(values, list) or (nonempty and not values):
        raise Refusal(f"{name} has invalid {field}")
    for value in values:
        validate_token(value, field, name, vocab_size)
    return values


def validate_case_name(value: object, field: str, name: str) -> str:
    if not isinstance(value, str):
        raise Refusal(f"{name} has invalid {field}: must be a string")
    if not value.strip():
        raise Refusal(f"{name} has invalid {field}: must not be empty")
    if value != value.strip():
        raise Refusal(f"{name} has invalid {field}: must not have surrounding whitespace")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise Refusal(f"{name} has invalid {field}: must not contain control characters")
    return value


def validate_finite_number(value: object, field: str, name: str,
                           *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise Refusal(f"{name} has invalid {field}: must be finite")
    if positive and value <= 0:
        raise Refusal(f"{name} has invalid {field}: must be positive")
    return value


def validate_named_case(case: object, required: set[str], field: str, name: str) -> dict:
    if not isinstance(case, dict):
        raise Refusal(f"{name} has invalid {field}: must be an object")
    missing = sorted(required - case.keys())
    if missing:
        raise Refusal(f"{name} has invalid {field}: missing {', '.join(missing)}")
    validate_case_name(case["name"], f"{field}.name", name)
    return case


def validate_legacy_cases(document: dict, name: str, contract: dict) -> None:
    unknown = sorted(document.keys() - GOLDEN_ROOT_KEYS)
    if unknown:
        raise Refusal(f"{name} has unknown golden field: {', '.join(unknown)}")
    if type(document.get("version")) is not int or document["version"] != 1:
        raise Refusal(f"{name} correctness golden file version must be 1")

    model_type = document.get("model_type")
    expected_model_type = contract.get("target", {}).get("model_type")
    if not isinstance(model_type, str) or not model_type or model_type != model_type.strip():
        raise Refusal(f"{name} model_type must be a non-empty trimmed string")
    if model_type != expected_model_type:
        raise Refusal(f"{name} model_type does not match {expected_model_type}")

    provenance = document.get("model_provenance")
    if not isinstance(provenance, dict):
        raise Refusal(f"{name} model_provenance must be a JSON object")
    if set(provenance) != {"repository", "revision"}:
        raise Refusal(f"{name} model_provenance requires only repository and revision")
    repository = provenance.get("repository")
    revision = provenance.get("revision")
    if (not isinstance(repository, str) or not repository
            or not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None):
        raise Refusal(
            f"{name} model_provenance requires repository and a 40-character lowercase revision"
        )
    target = contract.get("target", {})
    if (repository, revision) != (
            target.get("upstream_model_id"), target.get("upstream_revision")):
        raise Refusal(f"{name} model_provenance does not match the pinned reference model")

    vocab_size = contract["vocab_size"]
    prompt_tokens = contract.get("seed_tokens")
    required_steps = contract.get("correctness_steps")
    if prompt_tokens <= 0 or type(required_steps) is not int or required_steps <= 0:
        raise Refusal("the contract has invalid golden validation dimensions")
    cases = document.get("cases")
    if not isinstance(cases, list) or not cases:
        raise Refusal(f"{name} must contain a non-empty cases array")
    names = set()
    for index, value in enumerate(cases):
        case = validate_named_case(
            value, {"name", "prompt_tokens", "expected_tokens"}, f"cases[{index}]", name
        )
        case_name = case["name"]
        if case_name in names:
            raise Refusal(f"{name} has duplicate correctness golden case name {case_name}")
        names.add(case_name)
        prompts = validate_tokens(
            case["prompt_tokens"], f"{case_name}.prompt_tokens", name, vocab_size,
            nonempty=False,
        )
        expected = validate_tokens(
            case["expected_tokens"], f"{case_name}.expected_tokens", name, vocab_size,
            nonempty=False,
        )
        if len(prompts) != prompt_tokens:
            raise Refusal(
                f"{name} {case_name}.prompt_tokens has {len(prompts)} tokens; "
                f"need exactly {prompt_tokens}"
            )
        if len(expected) < required_steps:
            raise Refusal(
                f"{name} {case_name}.expected_tokens has {len(expected)} tokens; "
                f"need at least {required_steps}"
            )

    gates = document.get("correctness_gates", ...)
    if gates is not ...:
        validate_correctness_gates(gates, names, name, vocab_size, prompt_tokens)
    benchmark = document.get("benchmark", ...)
    if benchmark is not ...:
        validate_benchmark(benchmark, name, contract, vocab_size)


def validate_correctness_gates(gates: object, base_names: set[str], name: str,
                               vocab_size: int, prompt_tokens: int) -> None:
    if not isinstance(gates, dict):
        raise Refusal(f"{name} correctness_gates must be a JSON object")
    unknown = sorted(gates.keys() - {"anchors", "free_run", "behavior"})
    if unknown:
        raise Refusal(f"{name} correctness_gates has unknown field: {', '.join(unknown)}")
    if not gates:
        raise Refusal(f"{name} correctness_gates must contain at least one gate section")
    layered_names = set(base_names)
    for section in ("anchors", "free_run", "behavior"):
        if section not in gates:
            continue
        values = gates[section]
        if not isinstance(values, list) or not values:
            raise Refusal(f"{name} correctness_gates.{section} must be a non-empty array")
        section_names = set()
        for index, value in enumerate(values):
            field = f"correctness_gates.{section}[{index}]"
            required = {
                "anchors": {"name", "context_tokens", "expected_token"},
                "free_run": {"name", "prompt_tokens", "expected_tokens"},
                "behavior": {"name", "prompt_tokens", "accepted_token_sequences", "max_new_tokens"},
            }[section]
            case = validate_named_case(value, required, field, name)
            case_name = case["name"]
            if case_name in section_names:
                raise Refusal(f"{name} has duplicate correctness {section} case name {case_name}")
            section_names.add(case_name)
            if case_name in layered_names:
                raise Refusal(f"{name} has duplicate layered correctness case name {case_name}")
            layered_names.add(case_name)
            if section == "anchors":
                validate_anchor_case(case, case_name, name, vocab_size)
            elif section == "free_run":
                validate_free_run_case(case, case_name, name, vocab_size, prompt_tokens)
            else:
                validate_behavior_case(case, case_name, name, vocab_size)


def validate_anchor_case(case: dict, case_name: str, name: str, vocab_size: int) -> None:
    context = validate_tokens(case["context_tokens"], f"{case_name}.context_tokens", name, vocab_size)
    if len(context) > CORRECTNESS_MAX_ANCHOR_CONTEXT_TOKENS:
        raise Refusal(f"{name} {case_name}.context_tokens exceeds the maximum")
    validate_token(case["expected_token"], f"{case_name}.expected_token", name, vocab_size)
    accepted = case.get("accepted_tokens")
    if accepted is not None:
        validate_tokens(accepted, f"{case_name}.accepted_tokens", name, vocab_size)
    rank = case.get("max_expected_rank")
    if rank is not None and (type(rank) is not int or rank < 1 or rank > CORRECTNESS_TOP_LOGITS):
        raise Refusal(f"{name} {case_name}.max_expected_rank must be in 1...{CORRECTNESS_TOP_LOGITS}")
    delta = case.get("max_top_logit_delta")
    if delta is not None:
        validate_finite_number(delta, f"{case_name}.max_top_logit_delta", name)
        if delta < 0:
            raise Refusal(f"{name} {case_name}.max_top_logit_delta must be non-negative")
        if rank is None:
            raise Refusal(f"{name} {case_name}.max_top_logit_delta requires max_expected_rank")


def validate_free_run_case(case: dict, case_name: str, name: str, vocab_size: int,
                           prompt_tokens: int) -> None:
    prompts = validate_tokens(
        case["prompt_tokens"], f"{case_name}.prompt_tokens", name, vocab_size, nonempty=False
    )
    if len(prompts) != prompt_tokens:
        raise Refusal(f"{name} {case_name}.prompt_tokens must have exactly {prompt_tokens} tokens")
    expected = validate_tokens(case["expected_tokens"], f"{case_name}.expected_tokens", name, vocab_size)
    if len(expected) > CORRECTNESS_MAX_FREE_RUN_STEPS:
        raise Refusal(f"{name} {case_name}.expected_tokens exceeds the maximum")
    prefix = case.get("exact_prefix_tokens")
    if prefix is not None and (type(prefix) is not int or prefix < 1 or prefix > len(expected)):
        raise Refusal(f"{name} {case_name}.exact_prefix_tokens is outside expected_tokens")


def validate_behavior_case(case: dict, case_name: str, name: str, vocab_size: int) -> None:
    prompts = validate_tokens(case["prompt_tokens"], f"{case_name}.prompt_tokens", name, vocab_size)
    if len(prompts) > CORRECTNESS_MAX_BEHAVIOR_PROMPT_TOKENS:
        raise Refusal(f"{name} {case_name}.prompt_tokens exceeds the maximum")
    maximum = case["max_new_tokens"]
    if type(maximum) is not int or maximum < 1 or maximum > CORRECTNESS_MAX_BEHAVIOR_STEPS:
        raise Refusal(f"{name} {case_name}.max_new_tokens is outside the permitted range")
    sequences = case["accepted_token_sequences"]
    if not isinstance(sequences, list) or not sequences:
        raise Refusal(f"{name} {case_name}.accepted_token_sequences must not be empty")
    for index, sequence in enumerate(sequences):
        tokens = validate_tokens(
            sequence, f"{case_name}.accepted_token_sequences[{index}]", name, vocab_size
        )
        if len(tokens) > maximum:
            raise Refusal(f"{name} {case_name}.accepted_token_sequences[{index}] exceeds max_new_tokens")
    for field in (
            "semantic_prompt", "semantic_answer_key", "semantic_reference_answer",
            "semantic_domain", "semantic_subdomain"):
        if case.get(field) is not None and not isinstance(case[field], str):
            raise Refusal(f"{name} {case_name}.{field} must be a string")


def validate_benchmark(benchmark: object, name: str, contract: dict, vocab_size: int) -> None:
    if not isinstance(benchmark, dict):
        raise Refusal(f"{name} benchmark must be a JSON object")
    allowed = {
        "prefill_prompt_tokens", "expected_prefill_token", "decode_seed_tokens",
        "expected_decode_seed_token", "expected_decode_tokens",
        "baseline_prefill_seconds_per_token", "baseline_decode_seconds_per_token",
    }
    required = allowed - set(FORBIDDEN_BENCHMARK_KEYS)
    unknown = sorted(benchmark.keys() - allowed)
    missing = sorted(required - benchmark.keys())
    if unknown or missing:
        raise Refusal(f"{name} benchmark has invalid fields")
    seed_tokens = contract["seed_tokens"]
    for field in ("prefill_prompt_tokens", "decode_seed_tokens"):
        values = validate_tokens(benchmark[field], f"benchmark.{field}", name, vocab_size,
                                 nonempty=False)
        if len(values) != seed_tokens:
            raise Refusal(f"{name} benchmark.{field} must have exactly {seed_tokens} tokens")
    validate_token(benchmark["expected_prefill_token"], "benchmark.expected_prefill_token", name,
                   vocab_size)
    validate_token(benchmark["expected_decode_seed_token"],
                   "benchmark.expected_decode_seed_token", name, vocab_size)
    decode = validate_tokens(
        benchmark["expected_decode_tokens"], "benchmark.expected_decode_tokens", name, vocab_size,
        nonempty=False,
    )
    required_decode = contract.get("benchmark_decode_steps")
    if type(required_decode) is not int or len(decode) < required_decode:
        raise Refusal(f"{name} benchmark.expected_decode_tokens is too short")
    prefill = benchmark.get("baseline_prefill_seconds_per_token")
    decode_time = benchmark.get("baseline_decode_seconds_per_token")
    if (prefill is None) != (decode_time is None):
        raise Refusal(f"{name} benchmark baseline fields must be provided together")
    if prefill is not None:
        validate_finite_number(prefill, "benchmark.baseline_prefill_seconds_per_token", name,
                               positive=True)
        validate_finite_number(decode_time, "benchmark.baseline_decode_seconds_per_token", name,
                               positive=True)


def validate_golden_fields(document: object, name: str, contract: dict) -> None:
    """Refuse JSON that the parent golden loaders refuse."""
    if not isinstance(document, dict):
        raise Refusal(f"{name} is not a golden object")
    vocab_size = contract["vocab_size"]
    tape_keys = ("seed_tokens", "reference_seed_token", "rows")
    if "cases" in document and any(key in document for key in tape_keys):
        raise Refusal(f"{name} mixes the tape schema and the cases schema")
    if all(key in document for key in tape_keys):
        validate_tokens(document["seed_tokens"], "seed_tokens", name, vocab_size)
        validate_token(document["reference_seed_token"], "reference_seed_token", name, vocab_size)
        if not isinstance(document["rows"], list) or not document["rows"]:
            raise Refusal(f"{name} has no rows")
        row_chain = []
        for index, row in enumerate(document["rows"]):
            if not isinstance(row, dict) or "sequential_argmax" not in row:
                raise Refusal(f"{name} has an invalid row {index}")
            validate_token(
                row["sequential_argmax"], f"rows[{index}].sequential_argmax", name, vocab_size
            )
            row_chain.append(row["sequential_argmax"])
            if "top2_tokens" in row:
                validate_tokens(
                    row["top2_tokens"], f"rows[{index}].top2_tokens", name, vocab_size,
                    nonempty=False,
                )
            for field in ("top1_logit",):
                if field in row and (
                    isinstance(row[field], bool)
                    or not isinstance(row[field], (int, float))
                    or not math.isfinite(row[field])
                ):
                    raise Refusal(f"{name} has invalid rows[{index}].{field}")
            if "top2_logits" in row and (
                not isinstance(row["top2_logits"], list)
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    for value in row["top2_logits"]
                )
            ):
                raise Refusal(f"{name} has invalid rows[{index}].top2_logits")
        consistent = document.get("reference_self_consistent")
        if consistent is False:
            raise Refusal(f"{name} reports reference_self_consistent=false")
        if consistent is not True:
            raise Refusal(f"{name} requires reference_self_consistent=true")
        if "emitted_tokens" in document:
            emitted = validate_tokens(
                document["emitted_tokens"], "emitted_tokens", name, vocab_size, nonempty=False
            )
            if emitted != row_chain:
                raise Refusal(f"{name} emitted_tokens do not match rows[].sequential_argmax")
        return
    if "cases" in document:
        validate_legacy_cases(document, name, contract)
        return
    raise Refusal(
        f"{name} lacks golden fields: expected cases, or seed_tokens/reference_seed_token/rows"
    )


def pin(path: str, contract: dict) -> dict:
    """The {sha256, bytes} pin of one recorded file, after a shape check."""
    with open(path, "rb") as fh:
        data = fh.read()
    try:
        document = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Refusal(f"{os.path.basename(path)} is not JSON: {exc.__class__.__name__}")
    validate_golden_fields(document, os.path.basename(path), contract)
    benchmark = document.get("benchmark") if isinstance(document, dict) else None
    if isinstance(benchmark, dict):
        stored = [key for key in FORBIDDEN_BENCHMARK_KEYS if key in benchmark]
        if stored:
            raise Refusal(
                f"{os.path.basename(path)} stores a baseline pair ({', '.join(stored)}); "
                "the pair is measured live on the box and never stored"
            )
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def case_token_fingerprints(path: str) -> dict[str, set[str]]:
    """Canonical hashes of the parsed token fields, without exposing tokens."""
    with open(path, encoding="utf-8") as fh:
        document = json.load(fh)
    fingerprints = {"prompt_tokens": set(), "expected_tokens": set()}
    cases = document.get("cases", []) if isinstance(document, dict) else []
    for case in cases:
        if not isinstance(case, dict):
            continue
        for field in fingerprints:
            if field in case:
                canonical = json.dumps(
                    case[field], sort_keys=True, separators=(",", ":"), ensure_ascii=False
                ).encode("utf-8")
                fingerprints[field].add(hashlib.sha256(canonical).hexdigest())
    return fingerprints


def build_patch(contract: dict, directory: str, lives: list[str]) -> tuple[dict, list[str], list[str]]:
    track = contract["track_id"]
    vocab_size = contract.get("vocab_size")
    if type(vocab_size) is not int or vocab_size <= 0:
        raise Refusal("the contract has no positive integer vocab_size")
    prefix = f"correctness_prompts/{track}/"
    public_names = {name for name, _ in PUBLIC_CAPTURES.values()}
    names = sorted(
        n for n in os.listdir(directory)
        if n.endswith(".golden.json") or n in public_names
    )
    others = sorted(
        n for n in os.listdir(directory)
        if not n.endswith(".golden.json") and n not in public_names
    )
    if others:
        raise Refusal(
            "the directory holds files that are not *.golden.json or a public capture: "
            f"{', '.join(others)}"
        )

    public_roles = {name: role for role, (name, _) in PUBLIC_CAPTURES.items()}
    warmup_entry = contract.get("warmup_golden")
    if not isinstance(warmup_entry, dict) or not isinstance(warmup_entry.get("r2_path"), str):
        raise Refusal("the contract has no warmup_golden.r2_path")
    warmup_name = os.path.basename(warmup_entry["r2_path"])
    if not warmup_name.endswith(".golden.json"):
        raise Refusal("the contract warmup_golden.r2_path is not a golden file")
    depth_keys = [f"dflash{d}" for d in contract["dflash_head"]["permitted_draft_depths"]]
    depth_re = re.compile(r"^(?P<name>.+)\.(?P<key>dflash[0-9]+)\.golden\.json$")

    pool_names, depth_files, public_files = [], {}, {}
    for name in names:
        match = depth_re.match(name)
        if name in public_roles:
            public_files[public_roles[name]] = name
        elif name == warmup_name:
            pass
        elif match:
            if match["name"] not in lives:
                raise Refusal(f"{name} is a per-depth oracle for {match['name']}, which is not a live golden")
            if match["key"] not in depth_keys:
                raise Refusal(f"{name} is for {match['key']}, which no decoder declares")
            depth_files[(match["key"], match["name"])] = name
        else:
            pool_names.append(name[: -len(".golden.json")])

    if not pool_names:
        raise Refusal("found no timed-pool tape")
    absent = [live for live in lives if live not in pool_names]
    if absent:
        raise Refusal(f"--live {', '.join(absent)} is not one of the timed-pool tapes")
    if len(set(lives)) != len(lives):
        raise Refusal("a live golden is named twice")
    official_pairs = contract["official_pairs"]
    if official_pairs % len(lives) != 0:
        raise Refusal(
            f"official_pairs {official_pairs} is not divisible by the "
            f"{len(lives)} live goldens; every job must measure every live golden"
        )
    missing = [f"{key} ({live})" for key in depth_keys for live in lives if (key, live) not in depth_files]
    if missing:
        raise Refusal(f"no per-depth oracle for: {', '.join(missing)} (the measure script refuses a declared depth without one)")
    missing = [role for role in PUBLIC_CAPTURES if role not in public_files]
    if missing:
        raise Refusal(f"no public capture for: {', '.join(missing)}")
    if warmup_name not in names:
        raise Refusal(f"no warm-up golden: {warmup_name}")

    pool = []
    live_entries = {}
    for name in pool_names:
        entry = {"r2_path": f"{prefix}{name}.golden.json", **pin(os.path.join(directory, f"{name}.golden.json"), contract)}
        pool.append(entry)
        if name in lives:
            live_entries[name] = entry
    if len({entry["sha256"] for entry in pool}) != len(pool):
        raise Refusal("two timed-pool tapes have the same bytes; pool entries must be distinct")

    speculative = {}
    for key in depth_keys:
        speculative[key] = {}
        for live in lives:
            own = pin(os.path.join(directory, depth_files[(key, live)]), contract)
            speculative[key][live] = {
                "r2_path": f"{prefix}{depth_files[(key, live)]}",
                **own,
            }
    oracle_entries = [entry for per_live in speculative.values() for entry in per_live.values()]
    warmup = {"r2_path": warmup_entry["r2_path"], **pin(os.path.join(directory, warmup_name), contract)}

    hidden_shas = ({entry["sha256"] for entry in pool}
                   | {entry["sha256"] for entry in oracle_entries}
                   | {warmup["sha256"]})
    hidden_tokens = {"prompt_tokens": set(), "expected_tokens": set()}
    hidden_files = ([f"{name}.golden.json" for name in pool_names]
                    + list(depth_files.values()) + [warmup_name])
    for name in hidden_files:
        fingerprints = case_token_fingerprints(os.path.join(directory, name))
        for field in hidden_tokens:
            hidden_tokens[field].update(fingerprints[field])
    ships = []
    for role, (_, repository_path) in PUBLIC_CAPTURES.items():
        public_path = os.path.join(directory, public_files[role])
        own = pin(public_path, contract)
        if own["sha256"] in hidden_shas:
            raise Refusal(f"the public capture {role} has the same bytes as a hidden tape or oracle; shipping it would publish that tape")
        public_tokens = case_token_fingerprints(public_path)
        for field in hidden_tokens:
            if public_tokens[field] & hidden_tokens[field]:
                raise Refusal(
                    f"the public capture {role} has the same parsed {field} as "
                    "a hidden tape or oracle; shipping it would publish that tape"
                )
        ships.append(
            f"{public_files[role]} -> {repository_path} "
            f"({own['sha256'][:12]}..., {own['bytes']} bytes)"
        )

    first = live_entries[lives[0]]
    patch = {
        "official_scoring_enabled": True,
        "live_goldens": list(lives),
        "timed_prompt_pool": pool,
        "warmup_golden": warmup,
        "hidden_correctness_golden": {"sha256": first["sha256"], "bytes": first["bytes"]},
        "live_golden_speculative": speculative,
    }

    uploads = []
    local_by_r2 = {entry["r2_path"]: f"{os.path.basename(entry['r2_path'])}" for entry in pool}
    local_by_r2[warmup["r2_path"]] = warmup_name
    for key in depth_keys:
        for live in lives:
            local_by_r2.setdefault(speculative[key][live]["r2_path"], depth_files[(key, live)])
    pins_by_r2 = {e["r2_path"]: e for e in pool + oracle_entries + [warmup]}
    for r2_path in sorted(local_by_r2):
        entry = pins_by_r2[r2_path]
        uploads.append(f"{local_by_r2[r2_path]} -> {r2_path} ({entry['sha256'][:12]}..., {entry['bytes']} bytes)")
    total = len(depth_keys) * len(lives)
    uploads.append(f"{len(local_by_r2)} distinct object(s); {total} per-depth oracle object(s)")
    return patch, uploads, ships


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", required=True, help="the directory of recorded golden files")
    ap.add_argument("--live", required=True, action="append", help="a pool tape the track scores over (its name without .golden.json); repeat for each live golden, the first is the hidden correctness golden")
    ap.add_argument("--apply", action="store_true", help="write the values into the fixture in place")
    ap.add_argument("--contract", default=CONTRACT, help=argparse.SUPPRESS)
    args = ap.parse_args()

    with open(args.contract, encoding="utf-8") as fh:
        contract = json.load(fh)
    try:
        patch, uploads, ships = build_patch(contract, args.dir, args.live)
    except Refusal as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 1

    for line in uploads:
        print(f"upload: {line}", file=sys.stderr)
    for line in ships:
        print(f"ship: {line}", file=sys.stderr)
    if args.apply:
        contract.update(patch)
        with open(args.contract, "w", encoding="utf-8") as fh:
            json.dump(contract, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        print(f"applied: {os.path.relpath(args.contract, ROOT)}", file=sys.stderr)
    else:
        json.dump(patch, sys.stdout, indent=2)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
