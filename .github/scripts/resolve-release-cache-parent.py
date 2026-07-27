#!/usr/bin/env python3

import argparse
from collections.abc import Callable, Mapping
import re
import subprocess


VERSION_BRANCH = re.compile(r"^v([0-9]+)\.([0-9]+)\.([0-9]+)$")
REMOTE_PREFIX = "refs/remotes/origin/"


def parse_version_branch(name: str) -> tuple[int, int, int] | None:
    match = VERSION_BRANCH.fullmatch(name)
    if match is None:
        return None
    return tuple(int(component) for component in match.groups())


def select_cache_parent(
    source_ref: str,
    branches: Mapping[str, str],
    is_ancestor: Callable[[str], bool],
) -> str:
    source_version = parse_version_branch(source_ref)
    if source_version is None:
        return ""

    candidates = []
    for name, commit in branches.items():
        version = parse_version_branch(name)
        if version is not None and version < source_version:
            candidates.append((version, name, commit))

    for _version, name, commit in sorted(candidates, reverse=True):
        if is_ancestor(commit):
            return name
    return ""


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        check=check,
        text=True,
        stdout=subprocess.PIPE,
    )


def fetch_version_branches() -> dict[str, str]:
    git(
        "fetch",
        "origin",
        "+refs/heads/v*:refs/remotes/origin/v*",
        "--no-tags",
    )
    result = git(
        "for-each-ref",
        "--format=%(refname) %(objectname)",
        "refs/remotes/origin/v*",
    )

    branches = {}
    for line in result.stdout.splitlines():
        ref, commit = line.split(" ", 1)
        if ref.startswith(REMOTE_PREFIX):
            branches[ref.removeprefix(REMOTE_PREFIX)] = commit
    return branches


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-ref", required=True)
    parser.add_argument("--source-sha", required=True)
    args = parser.parse_args()

    branches = fetch_version_branches()

    def is_ancestor(commit: str) -> bool:
        result = git(
            "merge-base",
            "--is-ancestor",
            commit,
            args.source_sha,
            check=False,
        )
        if result.returncode not in (0, 1):
            result.check_returncode()
        return result.returncode == 0

    print(select_cache_parent(args.source_ref, branches, is_ancestor))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
