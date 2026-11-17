# BeeLlama.cpp release process

This is BeeLlama's release automation, not upstream llama.cpp's release process.
The workflows in `.github/workflows` separate the source being released from the
CI definition that builds it:

- A push to a `v*` branch triggers `release-preview-dispatch.yml`, which sends a
  `workflow_dispatch` to `release.yml` on `main` with the exact source ref and
  SHA, using `preview-${source_ref}` as the preview tag.
- A push of a `v*` tag triggers `release-dispatch.yml`, which dispatches the
  stable release to the same main-owned workflow.
- `release.yml` builds, packages, and publishes that exact source commit. Source
  checkouts use the resolved source SHA; workflow-owned helpers are checked out
  separately at `github.workflow_sha` under `.ci-workflow`. Historical source
  therefore still uses the main-owned CI definition rather than its old helpers.

The workflow validates the requested version against the source's
`CMakeLists.txt`. A preview branch must still point to the requested SHA. For a
stable release, the source ref must match the tag, the tag must resolve to the
requested SHA and be reachable from `origin/main`, and the matching release
branch must be an ancestor of that tag. Publication rechecks the source identity;
packages and container metadata use the same resolved SHA.

## Version

The BeeLlama version is set by `LLAMA_VERSION_MAJOR`, `LLAMA_VERSION_MINOR`, and
`LLAMA_VERSION_PATCH` at the top of `CMakeLists.txt`. Read those fields at the
intended source commit rather than assuming the current `main` version applies
to an older release.

Builds configured with `LLAMA_BUILD_IS_DEV=ON` report a `-dev` suffix. The
release workflow passes `-DLLAMA_BUILD_IS_DEV=OFF` for previews and
stable packages, and rejects a requested `vX.Y.Z` that does not match the source
version fields. Here and below, `vX.Y.Z` denotes the intended release version.

## Release preparation

1. Create the `vX.Y.Z` branch from the intended `main` commit.
2. Update the version in `CMakeLists.txt` and add the matching section to `CHANGELOG.md`.
3. Complete the release build, test, runtime, and benchmark gates on that branch.
4. Push the branch to produce the rolling `preview-vX.Y.Z` build.
5. Merge the validated branch into `main`, then create and push the matching `vX.Y.Z` tag to dispatch the stable release.

Do not publish from a moved branch, an unmerged tag, or a locally inferred source
SHA. The workflow validation is part of the release contract.

## Compilation checkpoints and cache ownership

The compilation matrix has 14 lanes across 11 job definitions. Checkpointing does
not change their toolkits, matrix entries, compiler flags, or packaging steps.
Each lane configures once, then uses the workflow-owned
`.github/actions/build-with-ccache` composite for up to three 90-minute active
compilation slices. Setup and cache upload time do not count toward a slice.

At a slice deadline, the helper stops the compiler process tree before saving
ccache. After either of the first two deadlines, it resumes the existing build
directory without reconfiguring. After the third deadline, it saves the cache,
emits an explicit incomplete-build marker, and fails the job. A translation unit
interrupted at a deadline may need to compile again.

If compilation succeeds sooner, the composite makes the final cache save and
continues to the existing smoke tests, packaging, and artifact upload. A real
compilation error fails without an additional cache save; any checkpoints saved
earlier remain available. Lane archive creation and artifact upload remain
inside the compile jobs, before the retry gate.

Each lane keeps a `3G` ccache. Checkpoint and successful final saves use the
unchanged `save-rolling-ccache` action: evict files older than `1d`, delete the
previous exact key, and save the refreshed cache under that same key on
`refs/heads/main`. Keys use the release branch channel and platform lane, for
example `release-vX.Y.Z-macos-arm64`; previews and stable releases for that
channel share the cache.

Restore first uses the channel's own key, with a parent-channel fallback. The
parent resolver chooses the highest lower semantic-version branch whose tip is
an ancestor of the requested source SHA. Metadata setup prunes `release-`
caches outside `refs/heads/main` or without a live version-branch channel.

GitHub-hosted jobs still have a six-hour platform cap, including setup, cache
uploads, packaging, and artifact uploads. Checkpointing cannot remove that cap,
so fewer than three checkpoints may fit. Runner loss also loses unsaved work:
a dead runner cannot upload its cache. A fresh run restores saved compiler
outputs, not the previous runner's build directory.

## Retry gate and publication

The dedicated `retry-builds` job depends on publication metadata and every
compilation job, and runs with `if: always()` at their terminal boundary. It is
not an external `workflow_run` controller. Aggregate `package-assets`,
`docker-build`, and publication jobs are held behind its decision. The gate
outputs `proceed=true` only when all required jobs succeeded and the run is
neither cancelled nor superseded.

An incomplete build can trigger a fresh workflow run when the gate finds:

- the explicit marker after three compilation checkpoints;
- the known observed hosted-runner lost-communication annotation; or
- a hard hosted-job timeout.

Unknown failures and ordinary build failures alone do not trigger retries. If
one lane has a real compilation error while another has an incomplete-build
signal, the mixed result does trigger a retry. It does not allow the failed run
to publish.

A fresh run rebuilds all lanes with the original source SHA and inputs, the same
matrix and flags, and the usual main-owned workflow definition. Internal
`retry_attempt` values are `0`, `1`, and `2`: at most two automatic retries, or
three runs total. `retry_root` carries the root run ID, and run titles end with
`[chain <root> attempt <n>]` so the controller can identify successors and
suppress duplicate dispatches. Manually rerunning the original after a successor
exists must not create another successor.

Newer preview commits cancel and replace older runs as before. The controller
checks both the current branch SHA and newer preview runs; cancelled or
superseded runs must not resurrect an older preview. Reaching the retry limit
leaves the build failed and publication blocked rather than extending the chain.

## Troubleshooting and verification

When publication is blocked, inspect the compilation jobs and `retry-builds`
first. Distinguish an ordinary compiler error from an incomplete-build marker,
a lost-communication annotation, or a hosted-job timeout. Check the title's
chain and attempt before manually rerunning a job, and confirm that the branch
still points to the requested source SHA. For cache misses, inspect the lane's
channel key, selected parent channel, restore log, and last successful save;
runner loss cannot preserve work after that save.

Run these targeted local checks when changing the checkpoint or retry machinery:

```sh
python tests/test-cache-build-deadline.py
python tests/test-release-workflow-static.py
node --test tests/test-release-retry.cjs
python tests/test-release-cache-parent.py
```

These commands are verification steps, not a statement that the checks have
passed. Local tests also do not demonstrate a hosted runner interruption or a
complete release publication; those require observing the corresponding Actions
run.
