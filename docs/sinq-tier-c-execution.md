# Tier-C execution backends

Advanced v2 `score` stages (`blockwise_output_l1`, `blockwise_output_mse` and
`fisher_vjp_mc`) support explicit
CPU/CUDA placement and FP32/BF16 parameter storage. CPU FP32 remains the default.
The native SINQ codec, candidate assignments and scoring objectives are unchanged.
This is an engineering backend, not a new quantization recipe or quality result.

## Configuration

Only strict optimizer schema version 2 is supported. Put placement and precision
in `execution.scoring`, CPU threads in `execution.threads`, and the scorer in
`scoring.kind`. There is no user JSON `operation` field. `device` accepts `cpu`,
`cuda` (the current CUDA device), or `cuda:N`. `reference_dtype` accepts `fp32`
or `bf16`, defaulting to `fp32`.

For example, save this score configuration after supplying the local inputs:

```json
{
  "schema_version": 2,
  "inputs": {
    "candidate_manifest": "fixtures/generated/run-manifest.json",
    "capture_manifest": "fixtures/block-capture.json",
    "checkpoint": "fixtures/checkpoint",
    "native_bridge": "build-sinq/bin/llama-sinq-native"
  },
  "scoring": {"kind": "blockwise_output_mse", "context": "reference_inputs"},
  "execution": {
    "threads": 8,
    "scoring": {
      "device": "cpu", "reference_dtype": "fp32",
      "memory_policy": {"host_limit_bytes": 8589934592, "reserve_bytes": 67108864}
    }
  },
  "limits": {
    "workspace_bytes": 134217728, "max_artifact_bytes": 67108864,
    "max_files": 100, "max_tensors": 4, "max_tokens": 16, "max_block_scores": 8
  },
  "output": {"directory": "new-tierc-score"}
}
```

Run `python -m tools.sinq.optimizer.cli score --config FILE`. For the four
execution combinations, use these fields inside `execution.scoring`:

| Backend | Fields |
|---|---|
| CPU FP32 | `"device": "cpu", "reference_dtype": "fp32"` |
| CPU BF16 | `"device": "cpu", "reference_dtype": "bf16"` |
| CUDA FP32 | `"device": "cuda", "reference_dtype": "fp32"` |
| CUDA BF16 | `"device": "cuda", "reference_dtype": "bf16"` |

Complete schema-checked configurations are in
[`examples/sinq-score-cpu-fp32.json`](examples/sinq-score-cpu-fp32.json),
[`examples/sinq-score-cpu-bf16.json`](examples/sinq-score-cpu-bf16.json),
[`examples/sinq-score-cuda-fp32.json`](examples/sinq-score-cuda-fp32.json), and
[`examples/sinq-score-cuda-bf16.json`](examples/sinq-score-cuda-bf16.json).
Change output directories between runs; score outputs are immutable. On other
platforms, use the actual native binary path. For Fisher, set `scoring.kind` to
`fisher_vjp_mc`, add `scoring.probes` and a nonnegative integer `scoring.seed`,
and include `limits.max_fisher_vjps` as well as `max_block_scores`. An optional
`scoring.fisher_positions` list selects explicit `[batch,time]` output coordinates
without changing the capture attention mask; every coordinate must be unmasked
and `len(fisher_positions) * probes` must fit `max_fisher_vjps`.
Without it, the capture mask selects positions as before. The same capture,
provenance and candidate rules apply.

For explicit layerwise placement, replace `device` with an ordered
`cuda_devices` list, for example `"cuda_devices": ["cuda:0", "cuda:1"]` on a
host where both exist. An accompanying `device: "cuda"` is permitted; `cpu` or
an explicit indexed device conflicts with the list. Lists must contain distinct
absolute indices. There is no assumed device count or card capacity.

Optional `memory_policy.cuda_limit_bytes` maps selected absolute device names
to positive byte limits. `reserve_bytes` is nonnegative and defaults to 256 MiB.
An omitted host limit is resolved from measured available physical memory plus
current process RSS minus reserve, separately from operation workspace. Set an
explicit host limit for reproducible resource policy. CUDA limits default to
the device's capacity; placement also checks current free memory.

Local scoring kinds accept an explicit `execution.scoring.device: "cpu"`, but reject CUDA,
`reference_dtype`, `cuda_devices`, and `memory_policy` as unused advanced controls.
Schema validation does not import Torch, initialize CUDA, or inspect devices.
Runtime validation separately checks device availability and native BF16 support.

## Exact reference storage and precision

The loader creates only the text graph and language-model head on meta, then
loads safetensors one tensor at a time. Conditional checkpoint vision/MTP
namespaces are not loaded into the text graph. Tied output embeddings stay tied;
untied heads must be present. Nonpersistent RoPE frequency buffers retain their
stock FP32 values, even with BF16 parameters.

There is no approximate-reference mode. BF16 uses the explicit storage policy
`bf16-bulk-exact-fp32-qwen-norms-v1`: bulk weights use BF16, while original FP32
parameters of stock Qwen3.5 RMS normalization modules retain FP32 storage. These
operators already compute in FP32 and return the activation dtype. Every such
exception is named and byte-counted in `parameter_storage_exceptions`; this is
not a rounded or silently substituted reference. FP32 mode remains uniform FP32.

Other checkpoint values that cannot be represented exactly in the requested
storage dtype fail with their tensor names. Complete GGUF verification checks
all source tensors, shapes, converter transforms, biases, aliases and
permutations in bounded slabs. Requesting BF16 for an F16/F32 source never
authorizes rounding the reference. A nonrepresentable projection still requires
explicit FP32 execution; do not edit the checkpoint to make verification pass.

Source-storage identity and execution arithmetic are separate metadata. Candidate
effective weights, unlike the original reference, are rounded to the actual
parameter dtype before substitution. Their original native-decoded values and
assignments retain their meaning. Temporary substitutions save only the affected
parameters and restore exact storage on exceptions, OOM and cancellation.

Blockwise losses promote both outputs to FP32 **before** subtraction and reduce
in CPU F64. Fisher normalizes low-precision logits in FP32, samples pseudo-targets
with a CPU-seeded generator, differentiates through stock downstream layers and
cross-device transfers, and stores bounded CPU F64 VJPs. Parameter gradients are
disabled. No dense Fisher, NLL-gradient substitute, autocast, compilation,
inference offload or automatic backend/precision fallback is enabled.

## Placement, identities and resource limits

Placement assigns complete layers in order according to measured device budgets.
Embeddings, normalization, rotary state and the output head have explicit
placements. Recursive `.to(device)` argument transfers retain autograd. Attention
masks, RoPE operands and positional inputs follow each layer; every sequence uses
`use_cache=False` and fresh recurrent state.

The execution identity binds dtype, absolute device indices and UUIDs, placement,
full reference configuration, source GGUF, implementation/dependency fingerprints,
Torch/Transformers/CUDA/cuDNN versions, CPU threads, deterministic algorithms,
TF32/CUBLAS settings and memory policy. Captures bind that identity. Changed
parameters, buffers or execution settings invalidate a locked reference. Old
optimizer configs and artifacts are not accepted as v2 inputs.
No bit-exact CPU/CUDA or FP32/BF16 equivalence is claimed.

Resource records distinguish conservative loader estimates from measurements:

- Host current/high-water working set is process-scoped, including mapped pages.
- CUDA allocated/reserved/peak bytes come from the Torch allocator. Per-phase
  allocator peaks reset at phase entry; sampled phase peaks are labeled separately.
- Device free memory comes from `cudaMemGetInfo` and includes driver allocations
  and other processes. It is not attributed to this process's Torch allocations.
- Polling runs during expensive phases, with cooperative checks at loader/slab
  boundaries and CLI layer forwards. Explicit CUDA allocator caps are installed
  for each phase and restored afterward.

Workspace preflights cover decoded baselines, swaps, captures, Fisher output and
probe buffers, and bounded diagnostics. Kernel/autograd workspace is not exactly
predictable. Host RSS guards are sampled, not OS-enforced hard RSS quotas; a
running native allocation/kernel cannot be interrupted by the polling thread.
Available host physical memory is also monitored against reserve where measurable.
Keep physical headroom and use process/container limits for stronger isolation.
An observed violation prevents final score publication, including a violation in
the last poll. There is no automatic lower-precision or CPU retry.

## Reproduce the engineering checks

Use a local environment with Torch, Transformers Qwen3.5 support, NumPy,
safetensors and converter dependencies. CPU imports do not require a CUDA toolkit
or GPU-specific Python add-ons. The validated environment is recorded with the
run evidence, not enforced as a package installation by the tool.

On Windows, use an x64 Visual Studio developer environment for native builds
and put the selected CUDA toolkit's `bin` directory on `PATH` for CUDA binaries.

```powershell
$env:OMP_NUM_THREADS = "32"
$env:MKL_NUM_THREADS = "32"
cmake -B build-sinq-cpu-threaded -DLLAMA_BUILD_SINQ_OPTIMIZER=ON
cmake --build build-sinq-cpu-threaded --parallel 32
ctest --test-dir build-sinq-cpu-threaded -R "sinq|perplexity-full" --output-on-failure -j 1
python tests/test-sinq-backend-integration.py --bin build-sinq-cpu-threaded/bin --output artifacts/sinq-tier-c-backend-implementation/tiny-verified
```

The integration test uses real tiny Qwen3.5 recurrent/full-attention layers,
source-verified weights, native Q5_0/Q8_0 candidates, joint substitutions and CLI
score publication. It exercises CPU and every detected CUDA device, bare `cuda`,
and a layerwise route when multiple devices exist. It checks exact same-backend
repeats, nonzero cross-layer VJPs, detachment rejection, zero delta, restoration,
provenance checksums, numerical differences and candidate rank agreement. It
creates separate output artifacts and never changes historical research locks.
See [Tier-C validation](sinq-tier-c-validation.md) for the test results, measured
memory and bounded existing-model observations. Full local provenance and raw
logs remain under `artifacts/sinq-tier-c-backend-implementation/` (ignored).
Those historical observations are not a completed v2 hardware qualification.
The full six-arm T340/T425 checks remain incomplete; local surrogate scores are
not model-KL selection results or untouched-test quality claims.
