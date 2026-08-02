#!/usr/bin/env python3
"""Reject unused project-prefixed variables reported by a CMake configure."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys


PROJECT_PREFIXES = ("GGML_", "LLAMA_")
UNUSED_HEADER = "Manually-specified variables were not used by the project:"
VARIABLE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*$")


def unused_project_variables(configure_log: Path) -> list[str]:
    lines = configure_log.read_text(encoding="utf-8", errors="replace").splitlines()
    unused: list[str] = []
    in_unused_block = False
    for line in lines:
        if UNUSED_HEADER in line:
            in_unused_block = True
            continue
        if not in_unused_block:
            continue
        match = VARIABLE.match(line)
        if match:
            name = match.group(1)
            if name.startswith(PROJECT_PREFIXES):
                unused.append(name)
            continue
        if line.lstrip().startswith("--"):
            in_unused_block = False
        elif line.strip() and not line.lstrip().startswith("CMake Warning"):
            in_unused_block = False
    return sorted(set(unused))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("configure_log", type=Path,
                        help="combined stdout/stderr from the CMake configure command")
    args = parser.parse_args()
    if not args.configure_log.is_file():
        parser.error(f"configure log does not exist: {args.configure_log}")

    unknown = unused_project_variables(args.configure_log)
    if unknown:
        print(
            "unused project-prefixed CMake variables: " + ", ".join(unknown),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
