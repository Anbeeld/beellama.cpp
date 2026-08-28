# BeeLlama.cpp release process

BeeLlama releases are built by the three workflows in `.github/workflows`:

- `release-preview-dispatch.yml` dispatches a preview when a `v*` branch is pushed.
- `release-dispatch.yml` dispatches a stable release when a `v*` tag is pushed.
- `release.yml` runs on `main` and builds, packages, and publishes the exact source commit supplied by the dispatcher.

The dispatchers record the source ref and SHA. Before publishing, `release.yml` verifies that a preview branch has not moved or that a stable tag resolves to the requested commit and remains reachable from `origin/main`. Packages and container metadata use the same resolved SHA.

## Version

The BeeLlama version is set at the top of `CMakeLists.txt`:

```cmake
set(LLAMA_VERSION_MAJOR 0)
set(LLAMA_VERSION_MINOR 4)
set(LLAMA_VERSION_PATCH 4)
```

Development builds use the default `LLAMA_BUILD_IS_DEV=ON` and report a `-dev` suffix. The release workflow passes `-DLLAMA_BUILD_IS_DEV=OFF` for previews and stable packages, and rejects a requested `vX.Y.Z` that does not match these source version fields.

## Release preparation

1. Create the `vX.Y.Z` branch from the intended `main` commit.
2. Update the version in `CMakeLists.txt` and add the matching section to `CHANGELOG.md`.
3. Complete the release build, test, runtime, and benchmark gates on that branch.
4. Push the branch to produce the rolling `preview-vX.Y.Z` build.
5. Merge the validated branch into `main`, then create and push the matching `vX.Y.Z` tag to dispatch the stable release.

Do not publish from a moved branch, an unmerged tag, or a locally inferred source SHA. The workflow validation is part of the release contract.
