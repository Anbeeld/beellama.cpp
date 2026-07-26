#!/usr/bin/env python3

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github/workflows"


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

    release = (WORKFLOWS / "release.yml").read_text(encoding="utf-8")
    preview_dispatch = (WORKFLOWS / "release-preview-dispatch.yml").read_text(encoding="utf-8")
    stable_dispatch = (WORKFLOWS / "release-dispatch.yml").read_text(encoding="utf-8")

    require(
        "\n  push:\n" not in release,
        "the build workflow must only run through main-owned workflow_dispatch",
    )
    require(
        "cache_channel: ${{ steps.meta.outputs.cache_channel }}" in release,
        "release metadata must expose the v* cache channel",
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
        "restore-keys:" not in release,
        "release jobs must restore only their exact rolling branch cache",
    )

    save_count = release.count("- name: Save ccache")
    require(save_count == 11, f"expected 11 rolling-cache save steps, found {save_count}")
    require(
        release.count("if: ${{ always() && needs.release-meta.outputs.preview == 'true' }}")
        == save_count,
        "every rolling-cache save must be preview-only",
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
