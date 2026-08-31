#!/usr/bin/env python3

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github/workflows"
ACTIONS = ROOT / ".github/actions"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    expected = {
        "release-dispatch.yml",
        "release-preview-dispatch.yml",
        "release.yml",
    }
    actual = {path.name for path in WORKFLOWS.glob("*.y*ml")}
    require(
        actual == expected,
        "release workflow inventory diverged: "
        f"added={sorted(actual - expected)}, missing={sorted(expected - actual)}",
    )
    stale_rocwmma = [
        path.name
        for path in WORKFLOWS.glob("*.y*ml")
        if "GGML_HIP_ROCWMMA_FATTN" in path.read_text(encoding="utf-8")
    ]
    require(
        not stale_rocwmma,
        f"release workflows still pass removed GGML_HIP_ROCWMMA_FATTN: {stale_rocwmma}",
    )

    release = (WORKFLOWS / "release.yml").read_text(encoding="utf-8")
    preview_dispatch = (WORKFLOWS / "release-preview-dispatch.yml").read_text(encoding="utf-8")
    stable_dispatch = (WORKFLOWS / "release-dispatch.yml").read_text(encoding="utf-8")
    setup_ccache = (ACTIONS / "setup-ccache/action.yml").read_text(encoding="utf-8")

    require(
        "\n  push:\n" not in release,
        "the build workflow must only run through main-owned workflow_dispatch",
    )
    require(
        "cache_channel: ${{ steps.meta.outputs.cache_channel }}" in release,
        "release metadata must expose the v* cache channel",
    )
    require(
        "cache_parent_channel: ${{ steps.meta.outputs.cache_parent_channel }}" in release,
        "release metadata must expose the nearest compatible predecessor cache channel",
    )
    require("ccache_ref" not in release, "cache channel and cache owner ref must not be conflated")
    require(
        release.find("- name: Prune stale release caches") < release.find("- name: Clone"),
        "stale release caches must be pruned before checkout and cache restoration",
    )
    require(
        "github.rest.repos.listBranches" in release
        and "github.rest.actions.getActionsCacheList" in release
        and "github.rest.actions.deleteActionsCacheById" in release,
        "cache pruning must compare live GitHub branches with managed Actions caches",
    )
    require(
        'git merge-base --is-ancestor "${branch_commit}" "${source_sha}"' in release,
        "stable releases must verify that the matching v* branch is an ancestor of the tag",
    )
    require(
        "resolve-release-cache-parent.py" in release,
        "release metadata must resolve the predecessor cache channel from live version branches",
    )
    require(
        "cuda-architecture-compile" not in release,
        "release workflow must not run the exhaustive CUDA architecture matrix",
    )
    require(
        "cache-from: ${{ matrix.config.name != 'rocm' && format('type=registry,ref={0}:buildcache-runtime-{1}', needs.release-meta.outputs.image_repo, matrix.config.name) || '' }}"
        in release
        and "cache-to: ${{ matrix.config.name != 'rocm' && format('type=registry,ref={0}:buildcache-runtime-{1},mode=max', needs.release-meta.outputs.image_repo, matrix.config.name) || '' }}"
        in release,
        "ROCm Docker packaging must bypass the oversized GHCR registry cache",
    )
    require(
        "${{ inputs.publish_release && 'stable' || 'preview' }}" in release,
        "preview and stable releases must use separate concurrency groups",
    )
    require(
        "-DLLAMA_BUILD_IS_DEV=OFF" in release,
        "release binaries must report the release version without a -dev suffix",
    )
    require(
        "-DLLAMA_BUILD_UI=ON" in release and "-DLLAMA_USE_PREBUILT_UI=OFF" in release,
        "release jobs must build the UI from the checked-out source",
    )
    require(
        release.count("build/tools/ui/dist/index.html") == 6
        and release.count(r"build\tools\ui\dist\index.html") == 1,
        "all seven application builds must require source-built UI output",
    )
    require(
        'source_version="${version_major}.${version_minor}.${version_patch}"' in release
        and 'if [[ "${version_name#v}" != "${source_version}" ]]' in release,
        "release metadata must bind the v* version to CMakeLists.txt at source_sha",
    )
    require(
        release.count('cuda: "13.3"') == 2 and "13.1" not in release,
        "release CUDA 13 lanes must use CUDA 13.3",
    )
    require(
        release.count('cuda_cmake_args: "-DGGML_CUDA_CUB_3DOT2=ON"') == 2
        and release.count("${{ matrix.cuda_cmake_args }}") == 2,
        "only the CUDA 12.4 Linux and Windows matrix entries must request fetched CCCL",
    )
    for runtime in ("amdhip64_7.dll", "amd_comgr_3.dll"):
        require(
            f'--require-name "{runtime}"' in release,
            f"Windows HIP package must require {runtime}",
        )
    require(
        '$kpack = Join-Path $env:HIP_PATH "bin\\rocm_kpack.dll"' in release
        and "if (Test-Path $kpack)" in release
        and '--require-local-import "rocm_kpack.dll"' in release,
        "Windows HIP packaging must copy KPack when supplied and require it when imported",
    )
    require(
        '--require-name "amd_comgr.dll"' not in release,
        "ROCm 7.1 uses the versioned amd_comgr_3.dll runtime",
    )
    require(
        "find-msvc-openmp-runtime.ps1" not in release
        and '--require-name "libomp.dll"' in release
        and '--forbid-name "libomp140.x86_64.dll"' in release,
        "Windows packages must use only the fetched LLVM OpenMP runtime",
    )
    require(
        'foreach ($pattern in @("cudart64_*.dll", "cublas64_*.dll", "cublasLt64_*.dll"))'
        in release,
        "CUDA runtime packaging must fail when a required runtime family is absent",
    )
    require(
        'if [[ "${PREVIEW}" == "true" ]]; then\n              mapfile -t existing_assets' not in release,
        "stable release updates must reconcile stale assets as well as previews",
    )

    setup_cuda = (ACTIONS / "windows-setup-cuda/action.yml").read_text(encoding="utf-8")
    require("cuda_arch:" not in setup_cuda, "x64-only CUDA setup must not require cuda_arch")
    require("13.4" not in setup_cuda, "unused ARM64 CUDA 13.4 setup must not remain")

    removed_imports = (
        ACTIONS / "ccache-buckets/action.yml",
        ROOT / "scripts/ccache-clear.sh",
        ROOT / "scripts/release.sh",
        ROOT / "scripts/make-release-checks.sh",
        ROOT / "scripts/make-release-desc.sh",
        ROOT / "scripts/make-release-summary.txt",
        ROOT / "cmake/arm64-windows-msvc-cuda.cmake",
        ROOT / "scripts/find-msvc-openmp-runtime.ps1",
    )
    require(
        not any(path.exists() for path in removed_imports),
        "unused or incompatible upstream release helpers must not remain",
    )
    require(
        (ACTIONS / "linux-setup-vulkan/action.yml").exists(),
        "the dormant Vulkan setup action should remain aligned with the prior Bee release tree",
    )

    save_count = release.count("- name: Save ccache")
    require(save_count == 11, f"expected 11 rolling-cache save steps, found {save_count}")
    setup_count = release.count("uses: ./.github/actions/setup-ccache")
    require(setup_count == save_count, "every rolling cache must have one setup and one save step")
    require(
        release.count(
            "restore-keys: release-${{ needs.release-meta.outputs.cache_parent_channel }}-"
        )
        == setup_count,
        "every release cache must fall back to the same backend/toolchain key in the parent channel",
    )
    require(
        release.count("if: ${{ success() || cancelled() }}")
        == save_count,
        "every rolling-cache save must run for successful and cancelled builds, "
        "but skip failed builds so a crash cannot evict healthy entries",
    )
    require(
        "always() && needs.release-meta.outputs.preview == 'true'" not in release,
        "rolling-cache saves must not be limited to preview builds",
    )
    require(
        release.count("ref: refs/heads/main") == save_count,
        "every rolling cache must be replaced in main cache scope",
    )
    require(
        "needs.release-meta.outputs.cache_channel" in release,
        "release cache keys must use the branch cache channel",
    )
    require(
        'default: "3G"' in setup_ccache,
        "the shared ccache action must enforce the unified 3G limit",
    )
    require(
        "max-size: 5G" not in release,
        "release jobs must not override the unified ccache limit",
    )

    require(
        'branches:\n      - "v*"' in preview_dispatch,
        "preview dispatch must trigger for v* branch pushes",
    )
    require(
        "gh workflow run release.yml" in preview_dispatch
        and "--ref main" in preview_dispatch
        and '--field publish_release="false"' in preview_dispatch,
        "preview pushes must dispatch the main-owned build workflow in preview mode",
    )
    require(
        "gh workflow run release.yml" in stable_dispatch
        and "--ref main" in stable_dispatch
        and '--field publish_release="true"' in stable_dispatch,
        "stable tags must retain the main-owned release dispatch",
    )


if __name__ == "__main__":
    main()
