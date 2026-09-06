#!/usr/bin/env python3
"""Compare KVarN CUDA objects using cuobjdump SASS and resource metadata."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import sys

SASS_FUNCTION_RE = re.compile(r"^\s*Function\s*:\s*(\S+)\s*$", re.MULTILINE)
RESOURCE_FUNCTION_RE = re.compile(r"^\s*Function\s+(\S+):\s*$", re.MULTILINE)
ENCODING_RE = re.compile(r"0x[0-9a-fA-F]{16}")
DECODE_COMBINE = "decode_combine_kernel"
WINDOW_KERNELS = ("window_dequant_kernel", "window_finalize_kernel")
OBJECT_SUFFIXES = {".o", ".obj", ".co"}


@dataclass(frozen=True)
class Kernel:
    name: str
    sass: tuple[str, ...]
    resources: tuple[str, ...]
    origin: str


def objects(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise ValueError(f"object path does not exist: {path}")
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in OBJECT_SUFFIXES)


def command_for(cuobjdump: Path, option: str, obj: Path) -> list[str]:
    command = [str(cuobjdump), option, str(obj)]
    if cuobjdump.suffix.lower() == ".py":
        command.insert(0, sys.executable)
    return command


def dump(cuobjdump: Path, option: str, obj: Path) -> str:
    try:
        result = subprocess.run(command_for(cuobjdump, option, obj), text=True, capture_output=True, check=False)
    except OSError as exc:
        raise ValueError(f"unable to execute cuobjdump {cuobjdump}: {exc}") from exc
    if result.returncode:
        raise ValueError(f"cuobjdump {option} failed for {obj}: {result.stderr.strip()}")
    return result.stdout


def sections(text: str, pattern: re.Pattern[str]) -> dict[str, str]:
    matches = list(pattern.finditer(text))
    result: dict[str, str] = {}
    for index, match in enumerate(matches):
        name = match.group(1)
        if name in result:
            raise ValueError(f"duplicate kernel section in cuobjdump output: {name}")
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        result[name] = text[match.end():end]
    return result


def parse_object(cuobjdump: Path, obj: Path) -> dict[str, Kernel]:
    sass_sections = sections(dump(cuobjdump, "--dump-sass", obj), SASS_FUNCTION_RE)
    resource_sections = sections(dump(cuobjdump, "--dump-resource-usage", obj), RESOURCE_FUNCTION_RE)
    if set(sass_sections) != set(resource_sections):
        missing_resources = sorted(set(sass_sections) - set(resource_sections))
        missing_sass = sorted(set(resource_sections) - set(sass_sections))
        raise ValueError(
            f"cuobjdump kernel inventory mismatch for {obj}: "
            f"missing resources={missing_resources}, missing SASS={missing_sass}"
        )

    result: dict[str, Kernel] = {}
    for name, body in sass_sections.items():
        encodings = tuple(word.lower() for word in ENCODING_RE.findall(body))
        if not encodings:
            raise ValueError(f"no SASS instruction encodings found for {name} in {obj}")
        resource_lines = tuple(
            " ".join(line.split())
            for line in resource_sections[name].splitlines()
            if line.strip()
        )
        if not resource_lines:
            raise ValueError(f"no resource metadata found for {name} in {obj}")
        result[name] = Kernel(name, encodings, resource_lines, str(obj))
    return result


def relative_name(root: Path, obj: Path) -> str:
    if root.is_file():
        return root.name
    return obj.relative_to(root).as_posix()


def read_tree(path: Path, cuobjdump: Path) -> dict[str, dict[str, Kernel]]:
    return {relative_name(path, obj): parse_object(cuobjdump, obj) for obj in objects(path)}


def read_common(paths: list[Path], cuobjdump: Path) -> tuple[dict[str, Kernel], set[str]]:
    result: dict[str, Kernel] = {}
    object_names: set[str] = set()
    for path in paths:
        for obj in objects(path):
            object_names.add(obj.name)
            for name, kernel in parse_object(cuobjdump, obj).items():
                if name in result:
                    raise ValueError(f"canonical kernel appears more than once: {name}")
                result[name] = kernel
    return result, object_names


def category(origin: str) -> str:
    name = Path(origin).name.lower()
    if "decode-pair" in name or ("decode-instance" in name and "common" not in name):
        return "decode"
    if any(word in name for word in ("geometry", "window", "case", "kvarn-instance-ncols")):
        return "geometry"
    return "other"


def compare_object(
    object_name: str,
    baseline: dict[str, Kernel],
    refactored: dict[str, Kernel],
    canonical: dict[str, Kernel],
) -> list[str]:
    errors: list[str] = []
    for name in sorted(set(refactored) - set(baseline)):
        errors.append(f"{object_name}: added kernel: {name}")

    removed_decode = [name for name in set(baseline) - set(refactored) if DECODE_COMBINE in name]
    if len(removed_decode) > 3:
        errors.append(f"{object_name}: more than three decode-combine kernels removed ({len(removed_decode)})")

    for name in sorted(set(baseline) - set(refactored)):
        old = baseline[name]
        allowed = (
            (DECODE_COMBINE in name and category(old.origin) == "decode")
            or (any(part in name for part in WINDOW_KERNELS) and category(old.origin) == "geometry")
        )
        if not allowed or name not in canonical:
            errors.append(f"{object_name}: unapproved removed kernel: {name} ({old.origin})")
        elif old.sass != canonical[name].sass or old.resources != canonical[name].resources:
            errors.append(f"{object_name}: removed kernel does not exactly match canonical kernel: {name}")

    for name in sorted(set(baseline) & set(refactored)):
        if baseline[name].sass != refactored[name].sass:
            errors.append(f"{object_name}: SASS differs: {name}")
        if baseline[name].resources != refactored[name].resources:
            errors.append(f"{object_name}: resource usage differs: {name}")
    return errors


def compare(
    baseline: dict[str, dict[str, Kernel]],
    refactored: dict[str, dict[str, Kernel]],
    canonical: dict[str, Kernel],
    common_object_names: set[str],
) -> list[str]:
    errors: list[str] = []
    baseline_names = set(baseline)
    refactored_names = {name for name in refactored if Path(name).name not in common_object_names}
    for name in sorted(baseline_names - refactored_names):
        errors.append(f"missing refactored object: {name}")
    for name in sorted(refactored_names - baseline_names):
        errors.append(f"added refactored object: {name}")
    for name in sorted(baseline_names & refactored_names):
        errors.extend(compare_object(name, baseline[name], refactored[name], canonical))
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("refactored", type=Path)
    parser.add_argument("--cuobjdump", required=True, type=Path)
    parser.add_argument(
        "--common",
        action="append",
        default=[],
        type=Path,
        help="canonical common object or directory (repeatable)",
    )
    args = parser.parse_args(argv)
    try:
        baseline = read_tree(args.baseline, args.cuobjdump)
        refactored = read_tree(args.refactored, args.cuobjdump)
        canonical, common_object_names = read_common(args.common, args.cuobjdump)
        errors = compare(baseline, refactored, canonical, common_object_names)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if errors:
        print("CUDA binary equivalence FAILED", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 1
    baseline_kernels = sum(len(kernels) for kernels in baseline.values())
    refactored_kernels = sum(
        len(kernels)
        for name, kernels in refactored.items()
        if Path(name).name not in common_object_names
    )
    print(
        f"CUDA binary equivalence OK "
        f"({len(baseline)} objects; {baseline_kernels} baseline / {refactored_kernels} refactored kernels)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
