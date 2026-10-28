# v0.4.9 upstream master refresh

## Source pins and ownership

- Bee parent: `17026e696f55da254ade56b8745148cc42bb0ddd`.
- Upstream master target: `d327bf1f0780f87b8c01b367b61853b0c291d7c4`.
- Merge base: `fd55cb98fe29d3df0054d3f3e6e6f15958ea72ab`.

This refresh follows `beellama.doc/UPSTREAM-MERGE.md`. Shared implementations
follow upstream architecture; Bee cache representations, precision tails,
transactional state, loop guard, and capability hooks remain separate
extensions. Root README, AGENTS, release policy, and previously removed generic
workflows retain Bee ownership. The new generic behavior and WASI workflows
are not adopted; the WASI build documentation links to the pinned upstream
workflow instead. The release version remains 0.4.9.

## Integration decisions

| Route | Upstream change | Bee obligation and resolution | Evidence/check |
| --- | --- | --- | --- |
| ggml operators | `TOP_K_MASK` and backend dispatch | Keep four KVarN operators and their registrations; operation count is 106 | CPU/CUDA operator checks and native CTest |
| CUDA | Attention normalization, runtime warp predicates, graph changes, BF16 normalization, Q6_K vector update | Retain low-bit/KVarN representation dispatch, tail cases, RHS layouts, and guarded HIP helpers | Fresh default and expanded builds; backend cases |
| Other backends | OpenCL I32 rows, SYCL/backend refactors, Vulkan alignment/precision | Keep typed cache rows and capability predicates alongside upstream paths | Vulkan CTest and RHS cases; static keepers for unavailable runtimes |
| Qwen4Exp QSA | Selection replaces the ordinary mask | Retain the ordinary planning mask when exact tails exist; gather selection into tail coordinates and combine masks | Synthetic CPU QSA/exact-tail oracle: original error 2.04146e-6, corrected error 0 |
| Speculative acceptance | Upstream moves accepted-token batch indices and adds target logprobs | Use upstream verification and the moved batch indices for Bee callbacks; no private verifier | Real-model draft-vs-target logprobs and cache tests |
| Historical MTP | Upstream prunes checkpoints against the shared input prefix | Also retain Bee's restored `n_past` bound for MTP continuation state | Native seed4301 RED/GREEN; checkpoint-policy assertion RED/GREEN |
| RAM candidate selection | Lexical ties could select a snapshot with zero restorable prefix | Reject such candidates beyond recurrent rollback capacity or MTP continuation bounds; retain ordinary body-backed truncation and valid checkpoints | Selection unit RED/GREEN; targeted DFlash2 four-slot and MTP one-slot runs |
| Disk slots | Upstream appends live draft state | Prepare target file, owned draft, speculative state, and prompt metadata before publication; never restore shared draft KV independently | Corrupt/truncated state rejection, repeated valid restore, legacy reevaluation |
| Cache accounting | Common rollback can reduce a lexical prefix to a descriptor boundary | Publish cached/planned counts and cumulative cached metrics only after the actual common boundary commits | DFlash1/KVarN seed4301: rollback1786→1664 and2635→2560; 197 apparent excess prompt tokens eliminated |
| Router/downloads | Upstream server entry-point and sidecar-plan refactor | Preserve explicit authenticated POST reload, sensitive-option filtering, and private token propagation into plan options | Local authenticated router/UI smoke; source trace for private Hub options |
| Quantized matmul tests | Upstream retires amax overflow rescaling (`3f03b1772`) | Adopt graph scaling and remove the two obsolete overflow tests; retain Bee RHS/offset and upstream skip-ID cases | Backend tests and deterministic-seed keeper |
| RPC | New operator insertion shifts Bee wire operation IDs | Deliberately bump Bee protocol to 9.0.0; patch-only changes cannot reject incompatible op numbering | New/new CPU discovery succeeds; old/new and new/old reject HELLO |

The MTP regression appeared as `no_restorable_checkpoint` after a historical
edit. Upstream's new `n_prefix` pruning condition had replaced Bee's MTP
`n_past` condition. Restoring that additional bound, without removing the
upstream shared-prefix check, changed the same native seed4301 workload from
failure to success. The policy now has a focused executable regression in
`test-server-prompt-checkpoint`.

## Local verification evidence

Evidence is retained in the integration workspace under `tmp/`, not committed
as model weights, generated binaries, or large request transcripts.

- Static keepers: `tmp/upstream-static/summary.json`.
- UI: `tmp/upstream-ui/`; typecheck, 763 unit tests, 199 client tests,
  40 component tests, production/PWA build. Package and lockfile hashes were
  unchanged by validation. npm reported 10 dependency vulnerabilities
  (7 low, 1 moderate, 2 high); this merge does not fix them.
- Transaction and logprob tests:
  `tools/server/tests/unit/test_slot_draft_transaction.py` uses the local
  ReasonLite 0.6B BF16 model, including the same model as draft-simple.
- Prompt-cache tests use ReasonLite for standard cache and Qwen3.5-4B BF16
  for KVarN. CPU and CUDA runs cover unified and non-unified cache contexts.
- RPC peer-boundary evidence: `tmp/upstream-rpc/`.
- Router smoke: `tmp/upstream-router-smoke.py` and its local artifacts check
  GET reload has no side effect, authentication, POST reload, child completion
  and unload, sensitive options, and embedded UI bytes.

### Matched performance and memory

`tmp/upstream-performance/manifest.json`, `runs.json`, `comparison.json`, and
`memory.json` retain exact commands and samples. Both binaries used the same
`C:/models/Qwen3-4B-GGUF/Qwen3-4B-BF16.gguf`, physical RTX 3090 GPU0,
CUDA 13.3, SM86, four CPU threads, batch512/ubatch128, depth2048, pp512 and
tg64. Each result is the median of three interleaved runs, each with three
repetitions and normal warmup. Tokens are llama-bench's C-RNG-generated
benchmark input; there is no text sampler. The baseline is the Bee parent;
the candidate application source was staged as tree
`11daae338d62a6a187534b5d923d735bafeb39c7` before the later server-only
checkpoint-pruning correction.

| Cache / F16 tail | Baseline pp tok/s | Candidate pp tok/s | Baseline tg tok/s | Candidate tg tok/s |
| --- | ---: | ---: | ---: | ---: |
| q4_0 / 0 | 3922.23 | 4009.10 | 88.62 | 88.77 |
| q4_0 / 512 | 2151.27 | 2094.41 | 60.46 | 61.32 |
| kvarn4 / 512 | 1891.73 | 1910.27 | 63.82 | 64.39 |

These measurements do not establish a universal speedup. Standard-cache
medians did not exceed the 5% regression gate in this workload.

For the separate `--kv-memory` runs (depth2048, pp0/tg1, one repetition,
no warmup), reported KV resident bytes were identical between builds:
95,551,488 for q4_0/tail0, 171,196,416 for q4_0/tail512, and 225,165,312 for
kvarn4/tail512. CUDA peak/runtime-overhead counters differed and remain in
`memory.json`; equal KV accounting does not mean equal total driver usage.

### Cached-request and restore timing

The targeted eight-turn DFlash2 q4_0 comparison used Qwen3.8-27B UD-Q4_K_M
and its DFlash2 Q4_K_M drafter, four slots, seed4303, context65536,
batch512/ubatch128, draft-max8, F16 tail512, RAM4096MiB, and top-logprobs5.
Both runs used physical GPU0. Each had 24 requests with committed cache hits.
These are aggregate medians, not paired-request speedup claims; the candidate
run precedes the final RAM-selection correction. The native endpoint is
non-streamed, so its recorded response latency is not true first-token latency.
Raw records are in `tmp/dflash2-miss-{baseline,merged}-targeted/`.

| Build | Hit response ms | RAM prepare / commit ms | Checkpoint prepare / commit ms |
| --- | ---: | ---: | ---: |
| Bee parent | 12644.917 | 17.530 / 549.253 | 22.603 / 499.981 |
| Candidate | 12850.357 | 16.705 / 532.702 | 22.991 / 497.876 |

### Matched quality

`tmp/upstream-quality/comparison.json` retains the results. The model is the
same Qwen3-4B BF16 file; WikiText test text, context4096, two chunks,
batch512/ubatch128. Both builds compare against the pinned Bee F16 logits.
The baseline F16 perplexity was 13.9196 +/- 0.72811.

| Profile | Baseline mean KLD | Candidate mean KLD | Baseline same-top % | Candidate same-top % |
| --- | ---: | ---: | ---: | ---: |
| F16 / tail0 | -0.000000 | 0.000052 | 100.000 | 99.560 |
| q4_0 / tail0 | 0.032632 | 0.035392 | 93.161 | 92.843 |
| q4_0 / tail512 | 0.005489 | 0.005401 | 97.167 | 97.044 |
| kvarn4 / tail512 | 0.000537 | 0.000565 | 99.218 | 99.194 |

The builds are not numerically identical, including the F16 reference path.
These are measured comparisons, not a claim of unchanged quality or improved
quality across all models.

## Existing failures and qualification limits

The user explicitly chose to preserve documented body-backed historical
rollback, including reported reductions in exact-tail coverage, rather than
change rollback behavior to satisfy a blanket no-degradation gate. The standing
merge guide and linked state guide now describe this approval-bound exception.
Strict test failures remain failures; the decision does not waive matrix
coverage, truthful accounting, isolation, atomic restore, or token-oracle checks.

- `test-qwen4exp-mtp-conversion.py` fails the same assertion at line66 on the
  clean Bee parent and the merged tree. It was not weakened or silently fixed.
- The no-spec Qwen3-4B q4_0 four-slot seed4301 workload fails its tail-degradation
  assertion on both builds. Each has 40 cached plus 40 oracle requests and
  24 full-sequence differences. The merged no-cache repeat control has zero
  first-target differences but three full-sequence differences. This does not
  establish that all cached differences are harmless.
- DFlash1 one-slot q4_0 runs for seeds4301-4303 fail tail-degradation checks on
  both builds. Before the accounting correction, KVarN4 runs failed the same
  excess prompt-work checks (197, 201, 197 tokens respectively). The corrected
  seed4301 candidate reports zero excess work but still fails its explicit
  tail-degradation assertion. Each original cell produced 10 cached and
  10 oracle requests; these failed cells are not qualification passes.
- Synthetic QSA/tail coverage does not qualify the local 93GB Qwen4Exp target
  as a real-model GPU configuration.
- HIP, Metal, SYCL, OpenCL, and CANN runtime qualification is not supplied by
  this Windows CPU/CUDA/Vulkan matrix. Source/keeper checks are not runtime
  or performance evidence for those backends.

## Closing verification scope

The user stopped the extended matrix and requested targeted fixing/testing
instead of multi-hour matrices. No test assertion was relaxed. The exploratory
cached phase completed 45 configurations (2160 cached/oracle requests), with
zero first-token differences and 338 full-sequence differences. Of its 21
nonzero exits, 20 were explicit tail-degradation qualifications and one was the
DFlash2 RAM-selection case investigated below. Eight full-depth no-cache
control configurations completed with zero first-token differences and 19
full-sequence differences. The remaining controls and fixed-order probe matrix
were not completed; this is not a blanket full-token-equivalence qualification.

### Final targeted corrections and checks

A zero-restorable RAM candidate could win a lexical tie and be installed even
though recurrent state could not reach the requested boundary. Selection now
checks recurrent rollback capacity before that publication. MTP requires a
checkpoint for a partial prefix. Ordinary body-backed attention keeps its
existing deep-truncation behavior, and positive checkpoint candidates remain
eligible. The focused unit test first failed because the unusable candidate
mutated the destination; it now verifies rejection without restore attempts,
plus preservation of valid checkpoints and unrestricted native rollback.

- The original DFlash2 q4_0 four-slot seed4303 workload now passes all unchanged
  assertions: 40 cached plus 40 oracle requests, reuse efficiency 1.0,
  zero first-token differences, zero restore failures, four busy slots, and
  overlap 0.9823. Full-sequence differences remain 20. Evidence:
  `tmp/dflash2-selection-fixed-full/`.
- Targeted MTP q4_0 one-slot seed4301 passes: 10 cached plus 10 oracle requests,
  nine cache hits, zero first-token differences and restore failures.
  Evidence: `tmp/upstream-model-matrix/cache-selection-mtp-targeted/`.
- The DFlash2/KVarN4 four-slot profile rejects draft checkpoints because its
  physical KV stream is shared (`target_status=0`, `draft_status=3`). It falls
  back to full reevaluation and fails the reuse qualification, with zero
  first-token differences. A two-turn pinned-baseline control reproduces the
  same no-reuse failure and shared-stream warning. This unsupported historical
  draft-state configuration is not declared fixed or qualified. Evidence:
  `tmp/upstream-model-matrix/cache-selection-kvarn-targeted/` and
  `tmp/dflash2-shared-draft-baseline/`.
- Two-GPU DFlash2/KVarN4 memory and historical-checkpoint tests pass for the
  single selected profile. The memory run uses context64000, 20 prompt repeats,
  one request and 32 generated tokens. The checkpoint test uses its unchanged
  one-slot workload and restores 12000 cached tokens before processing four
  prompt tokens. Evidence: `tmp/two-gpu-targeted-memory*` and
  `tmp/two-gpu-targeted-checkpoint*`. This does not qualify every bit pair or
  multi-slot draft-state configuration.
- DSpark q4_0/KVarN4 and probabilistic DFlash2 q4_0/KVarN4 each passed three
  32-token requests with seeds4301-4303, actual drafting, and finite target
  logprobs. These are execution smokes, not distribution or speed claims.
  Evidence: `tmp/upstream-spec-smoke/`.

Fresh default CUDA, expanded CUDA, CPU, and Vulkan builds each passed the full
139-test CTest suite after the earlier integration/accounting fixes
(`tmp/upstream-post-accounting-checks/`). After the final RAM-selection change,
the owning server targets were rebuilt for all four backends and all four
server CTest tests passed in each tree, 16/16 total
(`tmp/cache-selection-final-checks/`). The full 139-test suites were not rerun
after that final server-only change.

The final CPU shared-library and CUDA local slot/speculative-logprob suites
both pass 5/5. The standard/KVarN prompt-cache suite passes 4/4 without missing
model skips. The shared build also exports the file/data prepare and
commit/free transaction APIs. Static keepers pass 31/31. Relevant logs are
`tmp/cache-selection-{shared,cuda}-slots.log`,
`tmp/cache-selection-final-model-cache.log`, `tmp/upstream-shared-api.json`,
and `tmp/upstream-static-terminal-final/`.
