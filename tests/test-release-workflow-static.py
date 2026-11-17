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

    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    for component, value in (("MAJOR", 0), ("MINOR", 5), ("PATCH", 0)):
        require(
            f"set(LLAMA_VERSION_{component} {value})" in cmake,
            f"v0.5.0 metadata has the wrong {component.lower()} version",
        )

    require(
        'option(LLAMA_BUILD_IS_DEV "llama: dev build" OFF)' in cmake,
        "v0.5.0 must default to non-development build metadata",
    )
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    require(changelog.startswith("# Changelog\n\n## v0.5.0\n"),
            "the v0.5.0 entry must lead the changelog")
    require("## v0.4.10" not in changelog,
            "v0.4.10 changes belong to the combined v0.5.0 release")

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
        release.count('cuda: "13.5"') == 2 and release.count('cuda: "12.8"') == 2,
        "release CUDA lanes must retain CUDA 12.8 and 13.5",
    )
    require(
        release.count('cuda_cmake_args: "-DGGML_CUDA_CUB_3DOT2=ON"') == 2
        and release.count("${{ matrix.cuda_cmake_args }}") == 2,
        "only the CUDA 12.8 Linux and Windows matrix entries must request fetched CCCL",
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
    require(
        "Install Cuda Toolkit 13.4 for x64" in setup_cuda
        and "cuda_crt/windows-x86_64/cuda_crt-windows-x86_64-13.4" in setup_cuda
        and "windows-sbsa" not in setup_cuda,
        "CUDA 13.4 setup must remain x64-only, not restore ARM64 packaging",
    )
    require('cuda: "13.4"' not in release, "unused CUDA 13.4 must not enter the release matrix")

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

    checkpoint_build = (ACTIONS / "build-with-ccache/action.yml").read_text(encoding="utf-8")
    setup_count = release.count("uses: ./.ci-workflow/.github/actions/setup-ccache")
    build_count = release.count("uses: ./.ci-workflow/.github/actions/build-with-ccache")
    require(setup_count == build_count == 11,
            "all 11 compilation job definitions must restore and checkpoint ccache")
    require(release.count("- name: Clone workflow-owned build helpers") == 11,
            "every build must load its CI helpers independently of the source revision")
    require(release.count("restore-keys: release-${{ needs.release-meta.outputs.cache_parent_channel }}-") == setup_count,
            "every backend/toolchain cache must retain the parent-channel fallback")
    require("CACHE_SAVE_DEADLINE" not in release and "330 * 60" not in release,
            "the old job-start deadline must be removed")
    require(release.count("cache-build-deadline.py -- cmake --build") == build_count,
            "all compilation commands must use the bounded slice helper")
    for number in (1, 2, 3):
        require(f"id: slice{number}" in checkpoint_build,
                f"compilation slice {number} is missing")
        require(f"- name: Save ccache {number}" in checkpoint_build,
                f"compilation slice {number} must save its progress")
        require(checkpoint_build.index(f"- name: Verify completed build {number}")
                < checkpoint_build.index(f"- name: Save ccache {number}"),
                "completed-build checks must still pass before the final cache save")
        require(f"steps.slice{number}.outputs.checkpoint == 'false'" in checkpoint_build,
                "unfinished builds must not run completion-only checks")
    require(release.count("          verify-command: |") == 8,
            "all seven application job definitions and HIP runtime staging must retain their build checks")
    require(checkpoint_build.count("uses: ./.ci-workflow/.github/actions/save-rolling-ccache") == 3,
            "all checkpoints must reuse the original rolling-cache replacement action")
    require(checkpoint_build.count("success() && !cancelled()") == 3,
            "every save must exclude failed/cancelled compilation")
    for previous in (1, 2):
        require(checkpoint_build.count(f"steps.slice{previous}.outputs.checkpoint == 'true'") == 2,
                "later slices and their saves must run only after a controlled interruption")
    require(checkpoint_build.count("ref: refs/heads/main") == 3,
            "all checkpoints must retain main cache scope")
    require(checkpoint_build.count("evict-old-files: 1d") == 3,
            "all checkpoints must retain existing eviction policy")
    require("steps.slice3.outputs.checkpoint == 'true'" in checkpoint_build
            and "BEELLAMA_BUILD_INCOMPLETE:" in checkpoint_build,
            "the third interrupted slice must explicitly report incomplete")
    require("always() && needs.release-meta.outputs.preview == 'true'" not in release,
            "rolling-cache saves must not be limited to preview builds")

    gate = release.split("\n  retry-builds:\n", 1)[1].split("\n  package-assets:\n", 1)[0]
    require("if: ${{ always() }}" in gate,
            "retry evaluation must not be skipped after a failed/cancelled build")
    for job in ("release-meta", "macos-arm64", "ubuntu-cpu", "ubuntu-vulkan", "ubuntu-rocm",
                "ubuntu-sycl", "ubuntu-cuda", "windows-cpu", "windows-sycl", "windows-vulkan",
                "windows-cuda", "windows-hip"):
        require(f"      - {job}\n" in gate, f"retry evaluation must wait for {job}")
    require("      - package-assets\n" not in gate and "      - docker-build\n" not in gate,
            "retry evaluation must precede downstream packaging and publication")
    require("checks: read" in gate and "retry-release.cjs" in gate,
            "retry evaluation must be able to inspect runner-loss check annotations")
    require("RELEASE_INPUTS: ${{ toJSON(inputs) }}" in gate and "BUILD_RESULTS: ${{ toJSON(needs) }}" in gate,
            "the controller must receive original dispatch inputs and dependency results")
    for downstream, following in (("package-assets", "docker-build"), ("docker-build", "docker-merge")):
        block = release.split(f"\n  {downstream}:\n", 1)[1].split(f"\n  {following}:\n", 1)[0]
        require("      - retry-builds\n" in block and "needs.retry-builds.outputs.proceed == 'true'" in block,
                f"{downstream} must wait for the successful-build decision")
    require('options: ["0", "1", "2"]' in release and "inputs.retry_root || github.run_id" in release,
            "automatic retries must have a bounded attempt and stable chain identity")
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
