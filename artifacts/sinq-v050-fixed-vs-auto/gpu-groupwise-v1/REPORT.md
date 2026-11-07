# BeeLlama v0.5.0: GPU Groupwise KLD and completed A/B extension

## Result

CUDA Groupwise KLD is working with verified **33/33 layer offload**. Six new selector runs produced usable GGUFs, including bounded 27-evaluation GPU selections for both recipes. All six new models were then measured on the frozen full WikiText test reference and PTB test stream, alongside six read-only controls. No held-out result was used to change a selection.

GPU decode was 14.38x faster for T340 and 17.11x for T425, but complete 13-evaluation selection was **0.985x and 1.018x** as fast, respectively. Assembly and integrity hashing, not inference, dominate the elapsed time. This fixes the backend restriction; it does **not** resolve the practical full-search bottleneck.

The primary new quants are:

| Recipe | Model, relative to this directory | SHA-256 |
|---|---|---|
| T340 | `T340/cuda27-v2/best.gguf` | `3364bb34a9b58dbff0653317fb7ece1a55f0093701c9f8ec5d6710fc4f612bb0` |
| T425 | `T425/cuda27-v2/best.gguf` | `9192cdd67b1a1e0bffd35637d6e08d0c1de1721e155236a7f032cea698cad5d7` |

T340 CUDA-27 selected exactly the same bytes as CUDA-13 and the historical budgeted CPU selection. T425 CUDA-27 is a distinct new model with three accepted group changes. The CPU-13 and CUDA-13 GGUFs also remain under each recipe's corresponding directory. [final-models-frozen.json](final-models-frozen.json) records all model paths, hashes and frozen assignments.

## Scope, machine and provenance

Worktree: `C:/Users/anbee/projects/beellama-v050-sinq`, branch `v0.5.0`. Starting HEAD was `cba08b5d3457c16638697a77e615cce483035e89`, matching the requested reference. The historical experiment locks instead name `adf8aa6b9250f49e031a416b8d13b14b2e0f18e6`; those histories are not conflated.

The machine reported Ryzen 9 9950X, 32 logical processors, two RTX 3090 24 GB GPUs, and driver 616.56. **Actual reported physical RAM was 100,579,684,352 bytes, about 93.67 GiB, not the supplied 32 GB.** Selection peak working sets exceeded 32 GiB. These measurements therefore do not establish practical operation under a 32 GB host-memory constraint. See [current-hardware.json](current-hardware.json).

Selection used GPU UUID `GPU-5e87a39a-204c-4054-14a8-5b97ae4f16fd`, native PCI identity `0000:01:00.0`. Held-out T425 evaluation used the other GPU, UUID `GPU-5f3a3f44-a860-109d-a6e5-c9972f0f9842`, PCI `0000:03:00.0`.

Build: Release, CUDA 13.3, architecture 86, Flash Attention enabled, `GGML_NATIVE=OFF`, BLAS off, neither forced MMQ nor forced cuBLAS. The historical CPU-only executable was a different native build. Executable drift was established rather than assuming the historical binary was still installed.

| Binding | SHA-256 |
|---|---|
| Original BF16 source, `C:/models/Qwen3.5-9B-bf16/Qwen3.5-9B-text-BF16.gguf` | `da9df48e961a191aafca383628f983da4552a0a61c1257b93e1d678966f1c4af` |
| Current `build-sinq-clean/bin/llama-perplexity.exe` | `50d96fe6dd501cba9a7b92a7238d3f0b68c9494357ac887909d684925d832eb2` |
| Current `llama-sinq-model.exe` | `157049127f722fc2059daadba97cb314b1c7619a7e590ce96458fafbc52594ae` |
| Python implementation closure used by all six selections | `f15c64efc5019fec3a321b6c4a39968ac2fb5831ec90f4a432763c14e9a8063b` |
| T340 original candidate manifest | `b13c23622c54e14992ba3a8caa69e2fb108b1af0f2bf467c599daebc1f00ae2c` |
| T425 original candidate manifest | `75e86b85f88c1a76911b9833edf72385125341177b1eb1d4d6ed9807a82ca1f9` |
| Original 565-token selection stream, `../data/selection.u32` | `2934b74af4cc47e19db6b56b2a1383fc25727dfc80586b41c9970b9f3199c001` |
| Selection native prompt, `../data/selection.valid-prefix.native-prompt.raw` | `b2ad9c44ace4c1c61fd974ed25ceec90cfe4c4f86bb4606f2b34fcd36f9781bf` |
| Original CPU selection reference, preserved | `4172910d566b2a4ddbe908d773f87f6689b5d9f77e1b7e9e7b690ffc520c1b0a` |
| New strict CPU BF16 selection reference | `6a202edcd4bb37d5551c48077f1ad8b384e08b7c7f70e5b8cb8a1c928264d1e5` |
| New CUDA BF16 selection reference | `a4dc5cceb3c995cb03734bc9db817929bc50a91b35c0e60db3f8424dc9618b34` |
| Frozen full WikiText BF16 reference, unchanged | `b0963bb182cfb65b9f317aab36203718d5730c21f82eb57ca401bdb89969d89b` |

[provenance.json](provenance.json) contains source-file, executable, orchestration and tracked-diff hashes plus build settings. Each selection's `run-identity.json` contains its complete configuration and input bindings. [measurements.json](measurements.json) consolidates those identities, verified scores, timings and held-out evidence paths.

## Implementation

The existing `NativeKLEvaluator` in `tools/sinq/optimizer/kl.py` now accepts a validated `execution` object in the existing evaluator JSON. Schema-v2 optimizer configurations and CLI orchestration remain in use. CPU is still the default; CUDA is opt-in.

```json
{
  "backend": "cuda",
  "cuda_device": 0,
  "gpu_layers": "all",
  "cache_type_k": "bf16",
  "cache_type_v": "bf16",
  "flash_attention": "on",
  "cublas_compute_type": "auto",
  "mmq_precision": "auto"
}
```

This is an evaluator-file fragment, not a complete optimizer configuration. CUDA discovery resolves the explicit physical index through `nvidia-smi`, pins its UUID, selects native `CUDA0`, disables splitting/fitting, and requests all layers. Native logs must prove full offload and the matching physical PCI identity. Missing support, process failure, partial offload and CPU-only fallback are rejected. Capacity errors are not converted into a CPU retry. CPU explicitly selects `--device none --no-op-offload -ngl 0`.

Policies bind backend, UUID/PCI/driver, declared full offload, K/V types, FA, compute overrides, context, batching, threads, executable and adjacent shared-runtime hashes. The reference manifest additionally binds actual offloaded/total layers; candidate executions must have the same full layout. This evidence enters the reference-manifest digest used by score keys, and the existing selection/checkpoint identities bind that policy and reference. Runtime bytes are checked again before cached-score use and native execution. Old references and scores are intentionally not silently upgraded.

Inherited numerical/device overrides are removed. Context/batch/microbatch, timeout and thread controls are reused. CPU defaults remain F16 K/V and automatic FA when execution settings are omitted. Controlled measurements here explicitly used BF16 K/V and FA on for both backends. A new reference is required for either expanded policy; no candidate regeneration is needed.

No coordinate-descent, assembly, normalization, quantization kernel, candidate encoding, reference encoding or KL arithmetic was changed. Native TSV output appends `candidate_top1`, allowing actual prediction comparison instead of comparing only agreement flags. Existing numeric helpers still calculate full-vocabulary KL; the new top-1 helper preserves the existing first-maximum tie rule.

### Diagnostic failures and recovery

1. The first CUDA reference attempt failed closed because default native logging did not expose offload/device proof. Explicit verbose logging supplied the required evidence; the check was not relaxed.
2. Exploratory same-executable runs appeared almost identical across backends. Native logs revealed a CUDA compute buffer even with `-ngl 0` and an empty CUDA visibility variable on Windows. Those were not strict CPU baselines. Explicit device-none/no-operation-offload fixed that route, and CPU mode now also rejects leaked CUDA execution evidence.
3. The initial GPU selector pilot was stopped during input verification before publication so the final implementation identity could be frozen. No result from that attempt was accepted.
4. An assembly regression initially failed because `llama-completion.exe` was stale and rejected current SINQ metadata. Rebuilding that existing target resolved it without changing model semantics or tests.

`references/`, `numerics/`, `configs/` and the early logs remain diagnostic artifacts. **Accepted results use `references-v2/`, `numerics-v2/`, `configs-v2/` and `runtime-v2/`.** Unused 37-evaluation configs were never run. The larger budget was reduced to 27 based on pilot runtime alone, before held-out evaluation.

## Experiment reuse and integrity

Both recipes reused the original manifests, caches, source, stock-baseline locks, tensor allocation, singleton groups, locks, corpus partitions and selection configurations. Each has 248 eligible tensors and 2,976 conditioned candidates: widths 32/64/128/256, strengths 0.5/0.75/1.0, 16 iterations. No candidates were generated or requantized.

Selection retained context 65,536, batch 512, microbatch 256, 32 threads, 564 aligned targets, vocabulary 248,320, one pass and zero improvement tolerance. The T340 starting assignment was W128/S1; T425 was W64/S0.5. New references used exactly the existing 565-token selection stream, `_logitF1` encoding and reset-every-context policy. Reference generation was separate from selection: **35.62 s CPU and 28.11 s CUDA**, including source/reference hashing; native process components were 17.41 s and 10.07 s.

[final-integrity.json](final-integrity.json) records fresh verification of both entire candidate closures and all **34 historical GGUF hashes** against their existing audits. All matched. Source/candidate/selection closure checks also ran throughout each selector invocation. Historical results, historical models and the frozen final reference were not overwritten. [final-reference-integrity.json](final-reference-integrity.json) also confirms the original source, old CPU reference and full WikiText reference hashes after all evaluations.

## CPU versus CUDA performance

Seconds unless GiB is stated. These are fresh model-score caches, not cold filesystem caches. The two native timing numbers are cumulative **loading / decode-and-logits**; native process wall time, which includes scoring, reference I/O, output and other overhead, is shown separately below.

| Recipe | Backend | Evaluations | Total selection time | Assembly/hash time | Loading/inference time | Time/evaluation | Peak RAM | Peak VRAM |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| T340 | CPU | 13 | 2238.47 | 1908.25 | 8.59 / 56.42 | 172.19 | 39.88 GiB | 0.23 GiB background |
| T340 | CUDA | 13 | 2272.21 | 1955.94 | 22.87 / 3.92 | 174.79 | 39.88 GiB | 5.83 GiB |
| T425 | CPU | 13 | 2315.57 | 1917.47 | 9.15 / 62.05 | 178.12 | 41.66 GiB | 0.23 GiB background |
| T425 | CUDA | 13 | 2274.39 | 1958.13 | 27.94 / 3.63 | 174.95 | 41.65 GiB | 6.63 GiB |

Assembly/hash is the existing runner's assembly wall time excluding native validation. It includes assembly-local integrity work but not every hash elsewhere in selection. Native validation added 11.79/13.07 s for T340 CPU/CUDA and 13.36/12.72 s for T425. Do not sum inclusive nested hash phases or concurrent hash-worker times as if they were exclusive wall time.

RAM is the Python process lifetime high-water working set; native child working sets are recorded separately and were smaller. It is not instantaneous combined machine memory. VRAM is global device usage sampled every two seconds, including background allocations; it can miss short transients and is not an exact per-process peak. CPU logs prove no CUDA compute despite background VRAM usage.

| Recipe | CPU native process time | CUDA native process time | Native-process speedup | Decode/logits speedup | End-to-end speedup |
|---|---:|---:|---:|---:|---:|
| T340 | 76.90 | 37.81 | 2.034x | 14.378x | 0.985x |
| T425 | 85.30 | 43.09 | 1.979x | 17.113x | 1.018x |

GPU model loading is slower than CPU mmap loading, so the process-level gain is far smaller than the decode gain. Native full-vocabulary probability scoring added about 1.15–1.42 s per 13-evaluation run; token output added 0.24–0.36 s. Reference-read time was 2.22–2.84 s and overlapped decoding. These native timing categories are not all additive.

Backend decisions can change assignments. Exact fixed-file comparisons were therefore also run through the existing parity evaluator, with the same executable, corpus, reference, batching, threads and explicit numerical settings:

| Identical fixed model | CPU decode/logits | CUDA decode/logits | CPU process wall | CUDA process wall |
|---|---:|---:|---:|---:|
| T340 stock | 4.244 | 0.294 | 5.831 | 3.064 |
| T340 SINQ W128/S1 | 4.306 | 0.309 | 5.882 | 3.068 |
| T425 SINQ W64/S0.5 | 4.595 | 0.271 | 6.364 | 3.543 |

Each of the six selection runs had zero model-KL cache hits: 13/13/27 misses per backend/budget configuration. Registry verification receipts were reused within each invocation, exactly as in the existing implementation. Completed CUDA-13 checkpoint replays took 164.33 s for T340 and 149.47 s for T425 with **zero new native evaluations**. That is checkpoint-assisted reuse, not an inference speedup. A repeated native T340 numerical run also used new processes and reproduced GPU token output exactly. No OS cache eviction was attempted, so true cold-storage performance is unmeasured.

## CPU versus CUDA numerical comparison

All rows below compare the same model, corpus and **common new CUDA selection reference**, with the same current executable. This comparison is separate from actual CPU selection, which uses its own CPU reference. Therefore the CPU values below must not be substituted for CPU selector scores.

| Recipe/model | Mean KL CPU | Mean KL CUDA | Mean absolute ΔKL/token | Max absolute ΔKL/token | Top-1 disagreements |
|---|---:|---:|---:|---:|---:|
| BF16 reference | 0.00003296196 | 0.000000002949 | 0.00003295901 | 0.001512019 | 2 / 564 |
| T340 stock | 0.072383140 | 0.064806199 | 0.013425287 | 0.966634258 | 16 / 564 |
| T340 SINQ | 0.064163909 | 0.063441264 | 0.008486188 | 0.261345062 | 9 / 564 |
| T425 SINQ | 0.019797102 | 0.018371355 | 0.004233997 | 0.133607982 | 8 / 564 |

These are actual `candidate_top1` token-ID differences, not agreement-flag differences. Native aligned TSVs and raw logs are retained under `numerics-v2/<model>-matched/attempt-01/`.

| Model | Mean absolute Δcandidate-NLL/token | Maximum absolute Δcandidate-NLL/token |
|---|---:|---:|
| BF16 | 0.004816926 | 0.121927994 |
| T340 stock | 0.054361922 | 1.031372229 |
| T340 SINQ | 0.042616086 | 0.805369439 |
| T425 SINQ | 0.039393808 | 0.547335861 |

The historical comparison was reproduced using its original CPU-only executable (`e1d10c5c756f61a7f9f431e047c1eb61cb7a5eb65994be251f46be203b371fdd`), original selection reference and T340 SINQ model: mean absolute ΔKL **0.008797916260873478**, mean absolute ΔNLL **0.042236311065440235**, maximum ΔNLL **0.6356077419725672**, and **11 differing `top1_agree` flags**. The old CPU executable does not emit actual prediction IDs; the historical 11/564 statistic must not be relabeled as an independently measured prediction-ID count.

Controlled T340 SINQ changes did not eliminate the divergence:

| Common CPU/CUDA setting | Mean absolute ΔKL/token | Actual prediction differences |
|---|---:|---:|
| BF16 K/V, FA on, default compute | 0.008486188 | 9 |
| BF16 K/V, FA off | 0.008148893 | 10 |
| F16 K/V, FA on | 0.008608894 | 10 |
| BF16 K/V, FA on, CUDA cuBLAS F32 override | 0.008486188 | 9 |

Thus neither KV dtype nor Flash Attention alone explains the discrepancy. It persists with the same executable, so differing executable builds are not its sole cause either. The cuBLAS override does not control every quantized MMQ/recurrent operation; unchanged output under that override does not isolate an individual kernel. Remaining backend matrix/recurrent numerical differences were not fully attributed and are disclosed, not corrected by score normalization.

GPU repeat output and the GPU 16-versus-32-thread comparison were identical for every emitted per-token metric and prediction on this stream. Selection's GPU numerical controls agree with the historical final CUDA policy: BF16 K/V, FA on, full single-GPU offload, context 65,536, batch 512, microbatch 256, automatic compute, no warmup or fitting. Final tests used 16 threads, as before. The current fixed-anchor full-test results also reproduced their historical KLD/PPL values exactly. This establishes the tested policy's consistency, not universal bit identity across hardware or backends.

## Groupwise selection progress

Groups visited counts groups with at least one actually evaluated non-baseline assignment, not merely the next group encountered at budget exhaustion. All budgets exhausted; none of the accepted runs recorded a failed evaluation.

| Recipe | Backend | Evaluations | Groups visited | Accepted changes | Start KL | Final selection KL | Budget exhausted |
|---|---|---:|---:|---:|---:|---:|---|
| T340 | CPU historical | 13 | 2 | 1 | 0.063497254 | 0.062356239 | Yes |
| T340 | CPU controlled | 13 | 2 | 2 | 0.064135167 | 0.062504781 | Yes |
| T340 | CUDA | 13 | 2 | 1 | 0.063441264 | 0.062489739 | Yes |
| T340 | CUDA extended | 27 | 3 | 1 | 0.063441264 | 0.062489739 | Yes |
| T425 | CPU historical | 13 | 2 | 1 | 0.018161554 | 0.017815021 | Yes |
| T425 | CPU controlled | 13 | 2 | 1 | 0.019752540 | 0.019050500 | Yes |
| T425 | CUDA | 13 | 2 | 1 | 0.018371355 | 0.017932594 | Yes |
| T425 | CUDA extended | 27 | 3 | 3 | 0.018371355 | 0.017441887 | Yes |

**Scores across these policies/references are not directly interchangeable.** Historical CPU selection used its old reference and defaults, whereas controlled CPU and CUDA each use a newly generated policy-matched reference.

At 13 evaluations, both recipes considered the baseline plus 11 assignments in `blk.0.attn_gate.weight` and one in `blk.0.attn_qkv.weight`. At 27 they considered the baseline plus 11 assignments in each of those groups and four in `blk.0.ffn_down.weight`. This is 26 considered non-baseline assignments, not a completed search over 248 groups.

Extended CUDA runs took **4511.24 s / 4626.25 s**, or 167.08 / 171.34 s per evaluation, with peak Python working sets 39.89 / 41.66 GiB and sampled device VRAM 5.83 / 6.67 GiB. Native processes consumed only 82.08 / 94.65 s. Model-score caches again had zero hits. Synthetic interruption/continuation tests prove completed CUDA-provenance scores are retained and changed policies are rejected; actual completed-run checkpoint replays also succeeded without reevaluation. No real GPU OOM was deliberately induced.

## Completed held-out A/B measurements

All assignments were frozen in `final-models-frozen.json` **before** the new held-out runs. Existing final runners were extended with an explicit external frozen-model manifest and a new execution lock; the old input locks and old results were not edited to accommodate executable drift. The current executable identity is bound by `final-execution-lock.json`. Both current controls and new models used the same current CUDA policy.

WikiText: all **297,193 targets**, context 65,536, frozen original BF16 distribution reference. PTB: all **98,776 targets**, identical original normalized text/token lock, full-corpus NLL/PPL. There is no frozen PTB BF16 distribution reference, so **PTB KLD is not claimed**. Every emitted row was checked for target/index/context alignment and finite metrics. Each final runner verified 33/33 offload, the selected GPU's physical identity, and unchanged model bytes.

| Recipe | Frozen arm | Full WikiText KLD | WikiText PPL | PTB PPL |
|---|---|---:|---:|---:|
| T340 | Stock control | 0.104081454 | 7.209603 | 12.113470 |
| T340 | Fixed anchor W128/S1 | 0.096561983 | 7.112571 | 12.106593 |
| T340 | Historical CPU groupwise-13 | 0.096052488 | 7.101471 | 12.082488 |
| T340 | New controlled CPU groupwise-13 | 0.095814915 | 7.092430 | 12.092060 |
| T340 | New CUDA groupwise-13 | 0.096052488 | 7.101471 | 12.082488 |
| T340 | New CUDA groupwise-27 | 0.096052488 | 7.101471 | 12.082488 |
| T425 | Stock control | 0.044583063 | 6.601768 | 10.934203 |
| T425 | Fixed anchor W64/S0.5 | 0.038698239 | 6.693302 | 11.040010 |
| T425 | Historical CPU groupwise-13 | 0.038978606 | 6.692550 | 11.043071 |
| T425 | New controlled CPU groupwise-13 | 0.038795733 | 6.692891 | 11.045178 |
| T425 | New CUDA groupwise-13 | 0.038978606 | 6.692550 | 11.043071 |
| T425 | New CUDA groupwise-27 | 0.038755844 | 6.688022 | 11.033516 |

T340's extended GPU search did not change its model or held-out metrics. T425's extended search improved PPL against its fixed anchor and GPU-13 model, but its WikiText KLD remained slightly worse than the fixed anchor. T425 stock retained better PPL than these SINQ arms despite worse KL. There is no universal quality win, and no further selection was performed in response to these results.

All 24 model/corpus measurements passed. `held-out/<corpus>/<recipe>/progress.json` binds each `metrics.json`, token TSV, model hash and log. `measurements.json` supplies exact values and evidence paths; `table-B-groupwise-extension-v2.csv` is the compact A/B extension, with separate backend/reference identities for selection scores. The original 34-model study remains a separate historical record, not silently replaced by these current-executable rows.

## Tests and reproducibility

The final targeted run passed **25/25 CTest entries in 136.79 s**, including native full-vocabulary math, CPU/CUDA SINQ backend and storage tests, assembly, candidate/cache integrity, corpus isolation, optimizer config/selection, checkpoint recovery and the end-to-end optimizer workflow. The new native execution and provenance tests cover CPU placement, command generation, both GPU indices, unavailable devices/support, process/capacity failure, fallback rejection, policy/reference separation, changed executable/reference cache invalidation, aligned finite outputs and interrupted CUDA-provenance continuation. They do not claim a deliberately induced hardware OOM or a full repository-wide test run. See `final-verification-tests.log`. The source patch passes its whitespace check; whole-patch `git diff --check` additionally flags trailing whitespace emitted by the preserved native logs. Those evidence bytes were not edited to satisfy formatting checks. Scoped Git attributes disable line-ending conversion for hash-bound research artifacts, without disabling their diffs or checks.

Build and test commands are recorded in `build.ps1` and the corresponding build/test logs. The relevant invocation was:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File artifacts/sinq-v050-fixed-vs-auto/gpu-groupwise-v1/build.ps1
```

The existing research scripts, rather than a new evaluator subsystem, ran the experiment. From the worktree root, representative archived commands were:

```text
python artifacts/sinq-v050-fixed-vs-auto/create_selection_reference.py --output-directory artifacts/sinq-v050-fixed-vs-auto/gpu-groupwise-v1/references-v2/cuda --backend cuda --threads 32
python artifacts/sinq-v050-fixed-vs-auto/run_measured_selection.py artifacts/sinq-v050-fixed-vs-auto/gpu-groupwise-v1/configs-v2/T425-cuda27.json --report-directory artifacts/sinq-v050-fixed-vs-auto/gpu-groupwise-v1/runtime-v2
python artifacts/sinq-v050-fixed-vs-auto/run_measured_selection.py artifacts/sinq-v050-fixed-vs-auto/gpu-groupwise-v1/configs-v2/T425-cuda13.json --resume --report-directory artifacts/sinq-v050-fixed-vs-auto/gpu-groupwise-v1/runtime-v2
python artifacts/sinq-v050-fixed-vs-auto/run_final_wikitext.py T425 1 --models-manifest artifacts/sinq-v050-fixed-vs-auto/gpu-groupwise-v1/final-models-frozen.json --execution-lock artifacts/sinq-v050-fixed-vs-auto/gpu-groupwise-v1/final-execution-lock.json --output-directory artifacts/sinq-v050-fixed-vs-auto/gpu-groupwise-v1/held-out/wikitext2
python artifacts/sinq-v050-fixed-vs-auto/run_final_ptb.py T425 1 --models-manifest artifacts/sinq-v050-fixed-vs-auto/gpu-groupwise-v1/final-models-frozen.json --execution-lock artifacts/sinq-v050-fixed-vs-auto/gpu-groupwise-v1/final-execution-lock.json --output-directory artifacts/sinq-v050-fixed-vs-auto/gpu-groupwise-v1/held-out/ptb
python artifacts/sinq-v050-fixed-vs-auto/gpu-groupwise-v1/summarize.py
```

CPU and T340 variants use the corresponding archived configs. Exact numerical experiment commands are in `numerics-v2-commands.json`; native commands are also embedded in score-cache/reference evidence and final metrics records. References, selection outputs and summaries are immutable. Reproduction requires **new output paths**, or explicit resume for an existing selection; rebuilding a different executable requires new matching references and identities. Do not rerun generation or alter the frozen test reference. Large GGUFs, native binaries, reference distributions and raw TSV/log evidence remain local; committed manifests and summaries identify them by hashes and paths.

## Final assessment

- **Was forced CPU inference dominant?** No. Historical assembly already dominated, and current measurements confirm that result.
- **How much faster is GPU inference?** Decode/logits was 14.38x / 17.11x faster; complete native processes were only about 2x faster because loading and other work remain.
- **How much faster is selection?** T340 was about 1.5% slower; T425 about 1.8% faster in these single-run comparisons. There is no evidence of a substantial end-to-end acceleration.
- **Where does time go?** CUDA-13 assembly consumed 86.08% / 86.09% of total time, before counting other hashing. GPU model loading contributed another 1.01% / 1.23%; decode contributed only about 0.17% / 0.16%. Each 13-evaluation stage hashed roughly 2.24 / 2.34 TB of file input, plus tensor hashing. Nested/concurrent phase totals overlap and must not be summed. Extended runs remained 87–88% assembly.
- **Does the GPU policy match final A/B evaluation?** Yes for the tested numerical controls and models. The frozen original final reference was reused unchanged, and control results were reproduced. GPU repeats were internally reproducible; CPU/GPU logits are not asserted to be identical.
- **Is divergence fully explained?** No. Its historical magnitude was reproduced; strict CPU/backend, KV, FA and cuBLAS controls narrow but do not fully attribute it. CUDA uses its own matched selection reference, not recycled CPU scores.
- **Can substantially more of 248 groups now finish cheaply?** The selector now demonstrably reaches a third group, but three visited groups are only about 1.2% of the full set. At the measured assembly cadence, a full pass would still take many days. Neither a full search nor practical 32 GB host operation was established.
- **Separate optimization target:** reduce repeated full-model/baseline hashing and assembly I/O while preserving fail-closed integrity. Persistent loading or tensor hot-swapping was not introduced. The remaining bottleneck is not a reason to weaken validation or claim a cache-assisted inference speedup.
