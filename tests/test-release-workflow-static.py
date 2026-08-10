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
