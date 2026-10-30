#!/usr/bin/env python3
"""Replace private runner values in the public benchmark stream."""

import json
import os
import re
import sys


def add(mapping, value, replacement):
    if value and value != "/":
        mapping[value] = replacement


def add_path(mapping, value, replacement):
    value = (value or "").rstrip("/")
    if not value:
        return
    spellings = {value, os.path.realpath(value)}
    for path in tuple(spellings):
        for private, short in (("/private/var", "/var"), ("/private/tmp", "/tmp")):
            if path == private or path.startswith(private + "/"):
                spellings.add(short + path[len(private) :])
            if path == short or path.startswith(short + "/"):
                spellings.add(private + path[len(short) :])
    for path in spellings:
        add(mapping, path, replacement)
        # benchd JSON output may escape "/" as "\/".
        add(mapping, path.replace("/", "\\/"), replacement)


mapping = {}
for name in (
    "MLXFAST_BASELINE_WORKSPACE",
    "MLXFAST_REFERENCE_DIR",
    "BENCHD_BIN_DIR",
    "MLXFAST_QWEN38_GOLDEN_DIR",
    "MLXFAST_BASELINE_CALIBRATION",
    "MLXFAST_PRIVATE_DIR",
    "MLXFAST_CORRECTNESS_GOLDEN_PATH",
):
    add_path(mapping, os.environ.get(name), "<staging>")

workspace = os.environ.get("GITHUB_WORKSPACE", "")
runner_temp = os.environ.get("RUNNER_TEMP", "")
runner_workspace = os.environ.get("RUNNER_WORKSPACE", "")
add_path(mapping, workspace, "<runner-work>")
add_path(mapping, runner_temp, "<runner-work>")
add_path(mapping, runner_workspace, "<runner-work>")
if "/_work/" in workspace:
    add_path(mapping, workspace.split("/_work/", 1)[0], "<runner-work>")

add_path(mapping, os.environ.get("HOME"), "<home>")
add(mapping, os.environ.get("RUNNER_NAME"), "<box>")
add_path(mapping, os.environ.get("MLXFAST_MACMON"), "<toolchain>")
add_path(mapping, os.environ.get("MLXFAST_MACMON_BIN"), "<toolchain>")
add_path(mapping, os.environ.get("BENCH_WORKER_RESIDENT_SOCKET"), "<resident-socket>")
add_path(
    mapping,
    os.environ.get("RESIDENT_UP_LOCK_PATH", "/tmp/mtplx-gpu-exclusive.lock"),
    "<gpu-lock>",
)

calibration = os.environ.get("MLXFAST_BASELINE_CALIBRATION", "")
try:
    with open(calibration, encoding="utf-8") as handle:
        box = json.load(handle).get("box")
except (AttributeError, OSError, json.JSONDecodeError):
    box = None
if isinstance(box, str):
    add(mapping, box, "<box>")

# A value followed by a letter, digit, "_" or "-" is a different name, so /var
# must not rewrite /various. No left boundary: a colour code or any prefix
# before a private value must not stop the redaction. Longest value wins.
values = sorted(mapping, key=len, reverse=True)
pattern = (
    re.compile(r"(?:" + "|".join(map(re.escape, values)) + r")(?![A-Za-z0-9_-])")
    if values
    else None
)
for raw_line in sys.stdin.buffer:
    line = raw_line.decode("utf-8", errors="surrogateescape")
    if pattern is not None:
        line = pattern.sub(lambda match: mapping[match.group(0)], line)
    sys.stdout.buffer.write(line.encode("utf-8", errors="surrogateescape"))
    sys.stdout.buffer.flush()
