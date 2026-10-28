# v0.4.9 upstream integration

## Revisions and ownership

The `v0.4.9` branch starts from Bee
`ce1a9994fad8cf7706749666249398bbb7c600b1` and integrates upstream master at
`fd55cb98fe29d3df0054d3f3e6e6f15958ea72ab`. The merge base is
`c06f84160a30c66d7b5a2829ae9b3ea15275cbc3`.

This is an integration branch, not a release-version bump. Bee's `0.4.8`
identity, README and release workflows remain maintained by the fork.
Deleted generic upstream workflows stay deleted. New upstream model docs use
Bee build instructions where they describe this checkout.

## Integration choices

| Route | Upstream change and Bee obligation | Resolution and regression coverage |
| --- | --- | --- |
| `common/chat.h`, `tools/server/server-context.cpp` | Upstream prompt/session/parser-arena APIs must retain input/template provenance and media-injection protection. | Use `common_chat_prompt` and `common_chat_session`; remove duplicate prompt-part storage, preserve individual provenance parts through continuation and auto-parser prefixes. Covered by chat, PEG, prompt-part and server-component suites. |
| `src/llama-graph.cpp`, `src/llama-model.cpp` | Central graph-output construction, Hadamard storage and MoE expert caching replace duplicated plumbing. Bee tails and per-layer KV shapes must still reach attention. | Use upstream `hdmd`, including cached experts and one inverse embedding transform. Preserve generic Bee-aware attention for Qwen4Exp, upstream PLE placement, Qwen3Next per-layer heads and DFlash selector placement. `test-nextn-flags` exercises all eight trunk/head/load combinations and ordinary-model resets. |
| `common/speculative.cpp`, `src/llama-batch.cpp` | Upstream state embeddings and scheduling remain canonical; owned/shared cache state and adaptive limits remain distinct. | Port callers onto upstream batch/session APIs. DFlash2 and LiLiCorr retain fixed trained blocks. LiLiCorr's projected frontier uses a bounded per-sequence `LCRR v1` payload inside the existing `SPEC v1` envelope; missing/reset frontiers decline drafting rather than reuse stale rows. Synthetic transaction tests cover validation, repeated commit, reset and sequence isolation. |
| `common/arg.cpp`, `common/common.h` | Bee's omitted-option default is bounded; upstream's explicit zero means target inheritance. | Omission retains automatic capacity of at least 128, including parallel block capacity. Positive values share one draft capacity. Explicit `--spec-draft-n-ubatch 0` inherits target `-ub`; the legacy positive-only spelling remains supported. Parser tests cover defaults, environment, independence, rejection and option ordering. |
| `src/llama-memory-hybrid.cpp`, `src/llama-kv-cache-kvarn.cpp` | Immutable compressed records do not preserve exact rows overwritten by later requests. Recurrent and exact-tail state must describe the same checkpoint. | Keep prepared atomic restore and the bounded optional `SCKP v1` appendix, not a duplicate upstream appendix or mutating probe. Ordinary overlays and KVarN exact suffixes participate in hybrid partial checkpoints; immutable bodies remain reference-only. The public-API synthetic regression compares recurrent-only F16 state with ordinary low-bit and KVarN overlays and round-trips each payload. |
| `ggml/src/ggml-cuda/mmq.cu`, `ggml/src/ggml-cuda/top-k.cu` | Upstream explicit RHS precision and preallocation tile selection must coexist with Bee low-bit formats and tuning. | Select one `J_best` before allocation and reuse it through launch/fixup, with a 128-wide compiled limit and environment cap. Retain low-bit loaders, RHS deduplication and Bee per-expert width formulas. Use upstream general radix top-k with Bee tiled/HIP paths rather than duplicate radix families. |
| `ggml/src/ggml-backend-meta.cpp`, `ggml/src/ggml-backend.cpp` | Upstream host views and two-pass copies must not reinterpret Bee descriptors or proxy addresses. | Discover recursive Meta owners first and preserve hidden KVarN dependencies, arenas and synchronization. CPU-backed Meta buffers also advertise host storage, so they are explicitly excluded from physical external-host classification. ASan reproduced the old proxy-address reads in SET_ROWS/KVARN_STORE; Meta reuse, exact-tail and KVarN tests pass after the correction. |
| `ggml/src/ggml-vulkan/ggml-vulkan.cpp`, CPU/CUDA dispatch | Upstream BF16/type dispatch and CPU Q2 helpers must retain Bee writers and placement constraints. | Preserve Vulkan misaligned-RHS staging and padded strides, clamp memory budgets before saturating usage subtraction, keep CPU PQ2/PTQ symbols, and combine CUDA attention barrier conditions. Cache-facing `q2_0` remains `GGML_TYPE_Q2_0S`, distinct from the upstream serialized weight type. |
| `tests/CMakeLists.txt`, `tests/test-backend-ops.cpp` | Upstream consolidated context/component suites do not replace distinct Bee state checks. | Retain legacy save/load, tail, recurrent rollback, fragmented restore, fixtures and static keepers alongside upstream suites. The keeper changes follow independently tested NextN helpers and per-stream iSWA query counts. |

### Historical-tail regression

The unchanged real-model nonunified hybrid cache test initially restored a
640-token checkpoint but produced token 220 instead of the cold reference's
8029. Its recurrent checkpoint was byte-identical to the cold-prefix state,
and post-commit serialization matched all 52,691,548 bytes. Independently
disabling fusion, graph reuse or CUDA graphs did not remove the failure.
Increasing only the diagnostic exact-tail capacity from 512 to 768 did.

The longer branch had overwritten part of the checkpoint's exact window.
Restoring recurrent state alone silently substituted compressed history for
those rows. Including the existing tail/stage payload in partial checkpoints
makes the original 512-token fixture pass without changing its assertions.
The always-available synthetic test also failed before each ordinary/KVarN
admission correction and passed afterward. Full sequence-state format
versions are unchanged.
Legacy recurrent-only partial checkpoints cannot supply omitted exact rows;
restore preparation rejects them and safely reprocesses the prompt rather
than silently substituting lower-precision history.

### Vulkan fault reporting

Device-fault reporting previously supplied a null `pVendorBinaryData` while
retaining the driver's nonzero `vendorBinarySize`. ASan traced a secondary
NVIDIA driver write to that null pointer. Only textual/address diagnostics are
consumed, so the query now requests zero binary bytes and accepts
`VK_INCOMPLETE` for the omitted dump. This follows the
[Khronos query contract](https://docs.vulkan.org/refpages/latest/refpages/source/vkGetDeviceFaultInfoEXT.html).
It repairs diagnostics, not the underlying device-loss condition.

## Reproduction inputs

The Windows checks use two RTX 3090 GPUs, driver 616.56, CUDA 13.3, MSVC
19.44.35226 and Vulkan SDK 1.4.350.0. Model-backed checks use local files; no
model download is needed:

- `Qwen3.5-9B-UD-Q5_K_XL.gguf`, SHA256
  `bb0aa4bf2acf4b6d97eca051a7af91200729e9e97449ce98785af9e70f6d703c`.
- `qwen35-9b-dflash-Q4_K_M.gguf`, SHA256
  `56bf9df07d6f8b140c818d387d4758530bee1ba18f10d54982dba602471035e7`.
- Thread-safety fixture `stories15M-q4_0.gguf`, SHA256
  `66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739`.

`-AllTests` builds test targets; it does not execute CTest. Integration evidence
and exact commands are retained locally under `tmp/merge-v049/WORKLOG.md` and
its adjacent logs. Build labels and model results must not be treated as a
cross-hardware performance claim.

## Verification and qualifications

Fresh full CTest runs pass **135/135 per configuration** for CPU, default CUDA,
expanded CUDA and Vulkan. CUDA cache files confirm both settings of
`GGML_CUDA_FA_ALL_QUANTS`; these test builds use static libraries and disable
dynamic backend loading. `test-nextn-flags`, LiLiCorr state transactions, the
new partial-tail regression and the retained legacy state suites are included.
CUDA-only paths naturally skip in builds without CUDA; the named GPU checks
below provide their separate runtime coverage.

Additional checks:

- The local standard/KVarN server cache module passes all four tests, including
  the original nonunified historical-prefix fixture.
- The two-slot MTP cache/oracle workload completes 48 requests with 44 cache
  hits, 153,344 planned and committed reuse tokens, no first-target-token
  mismatches, no restore failures and no degraded-tail sequences. Nine later
  generated sequences differ from their cold references; full-sequence
  identity is not claimed. The identical baseline workload fails earlier on
  committed-versus-planned reuse accounting.
- Both revisions pass all eight default DFlash memory profiles, including
  `kvarn4:kvarn2`, with actual draft generation and native KVarN routes.
- Selected CUDA MMQ low-bit/view/ID-offset cases pass: 339 supported cases;
  33 are reported unsupported. Ordinary sparse/low-bit attention passes 42
  supported cases; 20 are unsupported. Top-k, writers and tail-equivalence
  families pass 1,083 supported cases; 719 are unsupported. Required-backend
  assertions remain intact. Selected Vulkan RHS-view cases also pass.
- UI typechecking reports zero errors and warnings; 736 unit tests, 165
  headless client tests and 39 UI-story tests pass. The production build
  succeeds, and the rebuilt server embeds 70 generated assets. The initial
  headed-browser failure receives no coverage credit.
- Python conversion sources compile. The existing Qwen4Exp converter
  assertion at line 66 fails identically on the pinned baseline and merge;
  `conversion/qwen4exp.py` is unchanged.

The strict two-slot DFlash checkpoint script was executed for all seven
profiles on both revisions without weakening its assertions. Baseline
`q8_0` fails full restored-versus-cold content equality at token index 7; the six
baseline KVarN profiles fail earlier on suffix-removal fallback/cache-source
accounting. All seven merged profiles reach committed historical checkpoints
and cold comparison, retaining the first target token, but later generated
text still differs. These strict scripts therefore remain **failed**, not
passed checkpoint qualifications; the cause of the later differences is not
established here.

Vulkan multicontext reliability remains a separate qualification. An earlier
native thread-safety invocation failed; subsequent complete native suites
pass. Matching ASan controls, including `/EHsc`, reproduce device loss in both
the pinned baseline and merge. The diagnostic correction exposes the primary
Vulkan1 device-loss error instead of the null binary-buffer write. A
validation-layer experiment also reports VUIDs and an SDK-unknown NVIDIA
extension; it is not a clean-validation result. No driver/device-loss fix is
claimed.

LiLiCorr serialization is qualified by synthetic transactions, not a local
LiLiCorr weight file. Metal, HIP and other unavailable hardware are not
runtime-qualified by these Windows runs. No throughput or KLD improvement is
claimed.

A separate fresh Windows default-CUDA shared-library build also succeeds with
`BUILD_SHARED_LIBS=ON` and `GGML_BACKEND_DL=ON`. Its server passes the same four
cache tests and serves an HTML index byte-for-byte identical to the newly
built UI asset. This checks dynamic backend/helper loading in addition to the
static test matrix.

Cross-version state was exercised with the pinned baseline writer and merged
shared-library reader: a 248,240,339-byte complete slot state containing 794
saved tokens loads with identical reported byte/token counts. An unusable
legacy partial checkpoint is rejected before commit; prompt reprocessing
reports zero reused tokens and matches the cold first target token. This is
safe legacy fallback, not preservation of the omitted historical exact rows.
