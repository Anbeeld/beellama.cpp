#!/usr/bin/env python3

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github/workflows"


def job_body(source: str, name: str) -> str:
    match = re.search(rf"^  {re.escape(name)}:\n(?P<body>(?:^(?:    |\s*$).*\n?)*)", source, re.MULTILINE)
    if not match:
        raise AssertionError(f"release workflow lacks job {name}")
    return match.group("body")


def main() -> None:
    for path in WORKFLOWS.glob("*.y*ml"):
        source = path.read_text(encoding="utf-8")
        if re.search(r"(?:^|/)master(?:$|['\" }])|^- master$", source, re.MULTILINE):
            raise AssertionError(f"stale master branch filter in {path.name}")

    release = (WORKFLOWS / "release.yml").read_text(encoding="utf-8")
    gates = (
        "test-portable-cpu",
        "test-windows-cuda-default",
        "test-cuda-all-quants",
        "test-cuda-kvarn-off",
    )
    for gate in gates:
        body = job_body(release, gate)
        if "needs: release-meta" not in body:
            raise AssertionError(f"{gate} does not consume release metadata")
        if "ref: ${{ needs.release-meta.outputs.source_sha }}" not in body:
            raise AssertionError(f"{gate} does not test the exact source SHA")
        if "LLAMA_BUILD_TESTS=ON" not in body:
            raise AssertionError(f"{gate} does not enable tests")
        if "if: always()" not in body or "actions/upload-artifact@" not in body:
            raise AssertionError(f"{gate} does not retain evidence on failure")

    aggregate = job_body(release, "behavioral-gates")
    for gate in gates:
        if f"- {gate}" not in aggregate:
            raise AssertionError(f"behavioral-gates does not require {gate}")

    package_jobs = (
        "macos-arm64", "ubuntu-cpu", "ubuntu-vulkan", "ubuntu-rocm", "ubuntu-sycl", "ubuntu-cuda",
        "windows-cpu", "windows-sycl", "windows-vulkan", "windows-cuda", "windows-hip", "package-assets",
        "docker-build", "docker-merge", "release",
    )
    for name in package_jobs:
        body = job_body(release, name)
        if "behavioral-gates" not in body and name not in ("package-assets", "docker-merge", "release"):
            raise AssertionError(f"package/publish job {name} bypasses behavioral gates")

    if "git merge-base --is-ancestor" not in release or "refs/remotes/origin/main" not in release:
        raise AssertionError("stable release no longer proves origin/main reachability")
    if "publish_release" not in release or "dry_run" not in release:
        raise AssertionError("release workflow lacks an explicit non-publishing dry run")


if __name__ == "__main__":
    main()
