# Tier-C implementation validation

CPU and CUDA FP32/BF16 execution passed the available-device checks. The original
Qwen3.5-9B reference also produced blockwise and Fisher observations in all four
CPU/two-GPU dtype combinations. This report is an engineering result, not a
quantization-quality study or a production-default recommendation.

## Repository and environment

- Branch: `v0.5.0`; base: `c0a156838148769b121ead96be8db2ab7cf390fa`.
- Implementation and this report are committed together; the delivery message gives the resulting commit hash.
- Windows 11 build 26200; Python 3.13.2; Torch 2.6.0+cu124; Transformers 5.6.2;
  NumPy 1.26.4; safetensors 0.7.0. Torch CUDA runtime 12.4; native CUDA toolkit
  13.3; MSVC 19.44; NVIDIA driver 616.56.
- Two detected RTX 3090 devices, 24 GiB each. CPU execution/build requests used
  32 threads. Native CUDA regressions ran on physical GPU 0 and GPU 1 in parallel.
- Schema version remains 1. See [execution configuration](sinq-tier-c-execution.md)
  and the four complete `docs/examples/sinq-score-*.json` configs.

## Checks

| Check | Result |
|---|---|
| CPU native build, `--parallel 32` | PASS |
| CUDA native build, `--parallel 32`, sm_86 | PASS |
| `ctest --test-dir build-sinq-cpu-threaded -R "sinq|perplexity-full" --output-on-failure -j 1` | 33 PASS, 0 FAIL, 0 SKIP |
| Python unittest cases within that CTest run | 165 PASS across 19 suites, 0 SKIP |
| Native QK/SINQ CUDA backend regressions | PASS on both physical GPUs |
| Real tiny Qwen3.5 backend integration | 10 configurations PASS; 20 score operations, 40 candidate records |
| Complete example configs | 4 schema PASS |
| Python compilation / `git diff --check` | PASS |

The CTest programs cover native SINQ math/activation/storage, full perplexity math,
exporter/source identity, blockwise/Fisher, configs, candidate cache, hybrid
assembly, groupwise selection, stock baselines, resume/repair, implementation
fingerprints, corpora, threading, native codecs and the end-to-end workflow.
Threading-contract tests retain their deliberate edge-count/serial comparisons;
32 is the requested count for the actual workload fixtures, builds and smoke runs.

Negative checks include malformed/duplicate/unavailable devices, unsupported
BF16 capability, local-scoring controls, insufficient resources, stale captures,
changed buffers/settings, detached downstream graphs, nonfinite weights, lossy
projection storage, and restoration after exceptions, injected OOM and cancellation.
The final-poll failure race and aggregate baseline decode bounds have regressions.

Initial engineering failures were diagnosed rather than hidden: BF16 hashing,
pre-subtraction precision and normalization needed changes; the original 9B
checkpoint required exact FP32 norm storage. An initial native build without
`vcvars64.bat` failed on missing standard headers. An attempted shared CPU build
exposed an existing `ggml_graph_next_uid` export/link failure; the supported
static CPU and CUDA builds passed without native numerical/source changes.

## Exact BF16 reference policy

The original checkpoint has 24 FP32 linear-attention norm tensors that are not
BF16-representable. The explicit `bf16-bulk-exact-fp32-qwen-norms-v1` policy keeps
those 12,288 bytes exactly in FP32 while the bulk model and activations execute
in BF16. Stock norm operators already compute in FP32 and return the activation
dtype. Every exception is recorded in reference metadata. No approximate-reference
mode, checkpoint editing, lossy projection acceptance or silent backend fallback
was used. A lossy projection still fails closed.

All four original-model runs verified the same 427 GGUF source tensors, with
one identical verified-tensor digest and identical sampled pseudo-target hashes.
Full source SHA-256:

`da9df48e961a191aafca383628f983da4552a0a61c1257b93e1d678966f1c4af`

## Tiny-graph numerical comparison

Real recurrent/full-attention layers use native Q5_0/Q8_0 joint substitutions.
Each backend had exact repeated logits under its own locked settings, zero-delta
reconstruction, finite nonzero VJPs and unchanged parameter storage after injected
failures. The two-GPU route demonstrated gradient flow across a layer boundary;
detaching that downstream boundary was rejected. Parameter gradients stayed absent.

| Execution | Maximum absolute logit difference from CPU FP32 | Candidate rank agreement, blockwise / Fisher |
|---|---:|---|
| cpu-fp32 | 0 | True / True |
| cpu-bf16 | 0.007223367691 | True / True |
| cuda-0-fp32 | 5.438923836e-07 | True / True |
| cuda-0-bf16 | 0.007610440254 | True / True |
| cuda-1-fp32 | 5.438923836e-07 | True / True |
| cuda-1-bf16 | 0.007610440254 | True / True |
| cuda-fp32 | 5.438923836e-07 | True / True |
| cuda-bf16 | 0.007610440254 | True / True |
| cuda-layerwise-fp32 | 5.438923836e-07 | True / True |
| cuda-layerwise-bf16 | 0.007610440254 | True / True |

Rank agreement concerns only these two fixture candidates. These measurements
are not a universal cross-backend tolerance or a quality conclusion.

## Bounded original 9B observations

Each run used `blk.0.ffn_up.weight`, the same permitted existing Tier-C candidate,
eight explicitly selected original token IDs, reference inputs, fresh cache-free
sequences, two MC VJPs, seed 617, and only the last output position. The original
candidate manifests, corpus locks and historical controls were not modified.
This declared engineering subset is not a rerun of the historical protocol.

| Execution | Block output MSE | Fisher quadratic | Host process peak, GiB | GPU allocator peaks, GiB (0 / 1) |
|---|---:|---:|---:|---:|
| cpu-fp32 | 3.45708040856e-06 | 0.00014132722134 | 53.238 | not applicable / not applicable |
| cpu-bf16 | 3.57233234683e-06 | 0.000150608421841 | 36.265 | not applicable / not applicable |
| layerwise-fp32 | 3.4570808093e-06 | 0.000141256943374 | 53.123 | 17.256 / 17.078 |
| layerwise-bf16 | 3.56990443053e-06 | 0.000158648049632 | 34.987 | 8.730 / 8.728 |

Per-phase host sampled current peaks and exact Torch allocated/reserved peaks,
device-global free bytes, elapsed times and measurement sources are included in
[the machine-readable report](sinq-tier-c-validation.json). Host OS high-water
values are cumulative within the process, not independently reset phase peaks.
GPU free memory includes the driver and other processes; it is not attributed
entirely to this process. Native-child peaks were not measured.

The plan's 77,224,834,944-byte preallocation number was an estimate, not a measured
working set. The revised loader and these original-model runs stayed below the
historical 66-GiB host limit and reached both observation types, clearing that
pre-observation blocker for this bounded subset. The full historical context,
candidate set and probe workload were not rerun, so their capacity is not established.

## Review, provenance and remaining limits

The complete changed-code paths were reviewed for duplicate models, unbounded
host conversions, stale CPU assumptions, detached transfers, resource accounting
and hidden precision substitution. Confirmed defects were repaired and affected
checks rerun. No native SINQ/A-SINQ, imatrix, production recipe, historical record
or production-default changes are part of this implementation.

Full local score/reference manifests and raw logs are under
`artifacts/sinq-tier-c-backend-implementation/`. They bind source/candidate hashes,
assignments, storage exceptions, placement, software/implementation fingerprints,
determinism and memory policy. The JSON report records artifact checksums and a
public projection; original tokens and physical GPU UUIDs remain local.

Residual limits: host RSS guards are sampled rather than hard OS RSS quotas;
unknown kernel workspace still needs physical headroom. Unsupported hardware,
other Transformers versions, more than two GPUs, nondefault attention kernels,
and full historical workloads were not integration-tested. Use the generic
capability checks and tiny integration test on a new host before a bounded real
workload; use process/container limits when hard host isolation is required.
The initial shared-native export issue remains outside this Python backend change.

## Changed-file inventory

- `docs/examples/sinq-score-cpu-bf16.json`
- `docs/examples/sinq-score-cpu-fp32.json`
- `docs/examples/sinq-score-cuda-bf16.json`
- `docs/examples/sinq-score-cuda-fp32.json`
- `docs/sinq-tier-c-execution.md`
- `docs/sinq-tier-c-validation.json`
- `docs/sinq-tier-c-validation.md`
- `tests/CMakeLists.txt`
- `tests/test-sinq-activation-parity.py`
- `tests/test-sinq-backend-integration.py`
- `tests/test-sinq-backend-scoring.py`
- `tests/test-sinq-blockwise.py`
- `tests/test-sinq-execution-policy.py`
- `tests/test-sinq-fisher.py`
- `tests/test-sinq-generation-resume.py`
- `tests/test-sinq-groupwise-kl.py`
- `tests/test-sinq-hybrid.py`
- `tests/test-sinq-native-bridge.py`
- `tests/test-sinq-reference-loader.py`
- `tests/test-sinq-research-e2e.py`
- `tests/test-sinq-research-repairs.py`
- `tests/test-sinq-stock-baseline.py`
- `tools/sinq/README.md`
- `tools/sinq/export_activations.py`
- `tools/sinq/research/blockwise.py`
- `tools/sinq/research/capture.py`
- `tools/sinq/research/cli.py`
- `tools/sinq/research/device_placement.py`
- `tools/sinq/research/execution_backend.py`
- `tools/sinq/research/fisher.py`
- `tools/sinq/research/memory_accounting.py`
- `tools/sinq/research/qwen35_adapter.py`
- `tools/sinq/research/reference_loader.py`
- `tools/sinq/research/schemas.py`
