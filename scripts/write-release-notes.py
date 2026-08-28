#!/usr/bin/env python3
import argparse
from pathlib import Path


def asset(release_url: str, tag: str, filename: str, label: str) -> str:
    return f"- [{label}]({release_url}/beellama-{tag}-{filename})"


def docker_line(image_repo: str, tag: str, label: str, suffix: str) -> str:
    return f"- {label}: `docker pull {image_repo}:server{suffix}-{tag}`"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--release-url", required=True)
    parser.add_argument("--image-repo", required=True)
    parser.add_argument("--package-url", required=True)
    parser.add_argument("--changelog", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    changelog = Path(args.changelog).read_text(encoding="utf-8").strip()
    tag = args.tag
    release_url = args.release_url.rstrip("/")

    lines = [
        "<details open>",
        "<summary>Changelog</summary>",
        "",
        changelog,
        "",
        "</details>",
        "",
        "**macOS:**",
        asset(release_url, tag, "bin-macos-arm64.tar.gz", "macOS Apple Silicon"),
        "",
        "**Linux:**",
        asset(release_url, tag, "bin-ubuntu-x64.tar.gz", "Ubuntu x64 CPU"),
        asset(release_url, tag, "bin-ubuntu-arm64.tar.gz", "Ubuntu arm64 CPU"),
        asset(release_url, tag, "bin-ubuntu-cuda-12.4-x64.tar.gz", "Ubuntu x64 CUDA 12.4"),
        asset(release_url, tag, "bin-ubuntu-cuda-13.3-x64.tar.gz", "Ubuntu x64 CUDA 13.3"),
        asset(release_url, tag, "bin-ubuntu-vulkan-x64.tar.gz", "Ubuntu x64 Vulkan"),
        asset(release_url, tag, "bin-ubuntu-rocm-7.2-x64.tar.gz", "Ubuntu x64 ROCm 7.2"),
        asset(release_url, tag, "bin-ubuntu-sycl-x64.tar.gz", "Ubuntu x64 SYCL"),
        "",
        "**Windows:**",
        asset(release_url, tag, "bin-win-cpu-x64.zip", "Windows x64 CPU"),
        asset(release_url, tag, "bin-win-vulkan-x64.zip", "Windows x64 Vulkan"),
        asset(release_url, tag, "bin-win-sycl-x64.zip", "Windows x64 SYCL"),
        (
            asset(release_url, tag, "bin-win-cuda-12.4-x64.zip", "Windows x64 CUDA 12.4")
            + f" - [DLLs]({release_url}/beellama-{tag}-cudart-win-cuda-12.4-x64.zip)"
        ),
        (
            asset(release_url, tag, "bin-win-cuda-13.3-x64.zip", "Windows x64 CUDA 13.3")
            + f" - [DLLs]({release_url}/beellama-{tag}-cudart-win-cuda-13.3-x64.zip)"
        ),
        asset(release_url, tag, "bin-win-hip-radeon-x64.zip", "Windows x64 HIP"),
        "",
        "**Docker:**",
        docker_line(args.image_repo, tag, "CPU", "-cpu"),
        docker_line(args.image_repo, tag, "CUDA", "-cuda"),
        docker_line(args.image_repo, tag, "CUDA 12", "-cuda12"),
        docker_line(args.image_repo, tag, "CUDA 13.3", "-cuda13"),
        docker_line(args.image_repo, tag, "ROCm", "-rocm"),
        docker_line(args.image_repo, tag, "Vulkan", "-vulkan"),
        docker_line(args.image_repo, tag, "SYCL", "-sycl"),
        "",
        f"[Browse all container images]({args.package_url})",
    ]

    Path(args.output).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
