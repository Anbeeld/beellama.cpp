# SINQ weight preconditioning and offline optimization

SINQ (Sinkhorn Normalized Quantization) reduces differences in weight scale
before low-bit quantization. BeeLlama adapts the normalization procedure from
[Huawei SINQ](https://github.com/huawei-csl/SINQ/tree/42ca83edf8fe1afc48c40fca6acc973c5d2bb59f)
to existing GGML IQ/Q codecs. It is an offline weight-conditioning feature,
not a new weight format, KV-cache compressor, or inference-time calibration step.

Use direct conditioning for a fixed recipe, or the optional offline optimizer to
generate and select per-tensor recipes. Ordinary SINQ requires no activation
samples. Activation-aware SINQ adds optional AWQ corrections using captured
reference inputs. Neither approach guarantees better model quality: compare the
finished model against a matched stock quantization on data not used for selection.

**Conditioned GGUFs require BeeLlama.** Other engines can recognize the standard
weight type IDs while ignoring their required scale companions, producing wrong
output. Name exports `*.beellama-qk-sinq.gguf`; the suffix warns users but does not
force third-party loaders to reject them. Do not strip the companions or metadata.

## Contents

- [Support and compatibility](#support-and-compatibility)
- [How conditioning works](#how-conditioning-works)
- [Direct quantization](#direct-quantization)
- [Offline optimizer](#offline-optimizer)
- [Activation export and file format](#activation-export-and-file-format)
- [Advanced scoring execution](#advanced-scoring-execution)
- [GGUF runtime contract](#gguf-runtime-contract)
- [Validation and troubleshooting](#validation-and-troubleshooting)

## Support and compatibility

The source must be an original F32, F16 or BF16 Qwen3.5 GGUF (`qwen35`).
Already conditioned sources cannot be copied or requantized through the
quantizer. Quantized sources and Hadamard-transformed sources are not suitable
conditioning inputs. No absorbed or companion-free export is provided.

Conditioning supports the following primary-trunk projection roles, with exact
names `blk.<layer>.<role>.weight`:

| Family | Roles |
| --- | --- |
| Full attention | `attn_q`, `attn_k`, `attn_v`, `attn_output` |
| Linear attention / SSM | `attn_qkv`, `attn_gate`, `ssm_alpha`, `ssm_beta`, `ssm_out` |
| Feed-forward | `ffn_gate`, `ffn_up`, `ffn_down` |

Embeddings, output heads, norms, convolution/state tensors, vision and auxiliary
MTP projections are outside the conditioning set. They retain the ordinary
quantization allocation; excluding them from conditioning does not mean retaining
all of them as floating-point weights.

Supported destination codecs are `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q4_0`, `Q4_1`,
`Q5_0`, `Q5_1`, `Q6_K`, `Q8_0`,
`IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`, `IQ4_NL` and
`IQ4_XS`. Selected matrices must be contiguous, have at least two output rows,
and have input dimensions divisible by both the native codec block size and the
chosen normalization width. Explicit incompatible assignments fail; they do not
silently switch width or codec.

Preset names such as `IQ2_M`, `IQ3_XS` and `IQ3_M` describe mixed native-codec
allocations, not additional tensor codecs. Direct quantization preserves the
ordinary preset allocation and conditions eligible projections using their actual
codec. Optimizer `quantization.recipe` entries require exact tensor codec names,
not preset names. Retained F32/F16/BF16 tensors remain unconditioned.

CPU and qualified CUDA inference use existing matrix kernels. CUDA tensor
parallel inference supports `--split-mode tensor --device CUDA0,CUDA1
--tensor-split 1,1`, including owned Qwen3.5 MTP draft contexts. Input-channel
scales follow the base matrix's exact partition; output-channel projections
replicate their input scales. Unequal split ratios use the same codec and head
alignment as ordinary weights, including ranks with no assigned channels. Row
split remains unsupported; use tensor, layer or none instead.

Tensor mode retains the backend's ordinary numerical policies, not bit-exact
single-device results. The CUDA internal AllReduce defaults to BF16 wire
rounding for F32 tensors. For tight numerical comparisons, set
`GGML_CUDA_AR_BF16_THRESHOLD=0` to retain F32 communication. This is an existing
backend control, not a SINQ requirement; the serving configuration needs no new
SINQ option.

Do not infer support for other backends or placements from the standard weight
IDs. Each companion must be placed with its base matrix. Offline Torch scoring
on multiple CUDA devices is a separate capability from native model serving.
LoRA deltas continue to receive the original, unscaled inputs.

The current optimizer/config and GGUF metadata schemas are version 2. Old
unreleased interfaces and v1 conditioned models are rejected rather than
implicitly migrated. Activation capture files have their own independent
version-1 schema, described below.

## How conditioning works

For a weight matrix `W` with shape `[N,K]`, SINQ balances row and column
statistics within full-height strips of input columns. Width counts columns,
not codec blocks. Normalization statistics use F16-rounded source values and
FP32 log-space updates; the best intermediate supplies a positive column factor
`t`. Row factors contribute to optimization but are not runtime parameters.

For ordinary SINQ, the effective diagonal is `u = F16(t^strength)`. BeeLlama uses
that exact serialized diagonal when compensating the original source weights:

```text
stored weights = native_quantizer(W / u)
base output = native_matmul(stored weights, u * x)
effective reconstructed weights = native_dequantizer(stored weights) * u
```

Without quantization the diagonal cancels in exact arithmetic. With quantization,
changing the weight distribution changes codec error. The native encoder, block
layout and type IDs remain unchanged. An importance matrix `I` is transformed to
`u*u*I` for encoding; this does not recapture activations through a quantized model.

Normalization widths are 8, 16, 32, 64, 128, 256 or 512; iteration counts are 4, 8 or 16.
Strength is finite in `[0,1]`. Strength zero produces an identity normalization diagonal but remains a
conditioned candidate, distinct from a stock candidate. Optional post-encoding
scale refinement can still change that diagonal when explicitly enabled. Normalizer
worker buffers have an aggregate 128 MiB bound. A strip that cannot fit fails
instead of being resized.

### Activation-aware correction

Optimizer candidates with `generator: "awq"` start from ordinary SINQ factors
and apply a correction derived from mean absolute reference activations. They
still use the destination native codec, not upstream SINQ's row-scale RTN codec.

`correction: "weight_activation"` combines activation means raised to `alpha`
with inverse weight means raised to `1-alpha`; `activation_only` uses only the
activation term. Factors are stabilized, normalized by the geometric mean of
their minimum and maximum, and raised to `lambda`. The strength-powered SINQ
factor is divided by that correction **before** final F16 rounding. `lambda: 0`
disables correction; `alpha: 0` also disables activation-only correction.
All of `strength`, `alpha` and `lambda` must be finite in `[0,1]`.

AWQ candidates require verified activation inputs. There is no implicit alpha
grid, automatic strength, or automatically added ordinary candidate: declare
all candidate recipes you want to compare. Local reconstruction losses are
surrogates, not full-model KL or perplexity.

## Direct quantization

No optimizer helper build or Python execution is needed for direct conditioning.
Save this configuration as `direct.json`:

```json
{
  "schema_version": 2,
  "defaults": {"generator": "sinq", "strip_width": 64, "iterations": 16, "strength": 1.0},
  "targets": {"include": "eligible"},
  "overrides": [
    {"selector": {"layer": 0}, "parameters": {"strength": 0.5}}
  ]
}
```

```sh
llama-quantize --pure --sinq-config direct.json source-bf16.gguf model.beellama-qk-sinq.gguf Q4_K
llama-quantize --pure source-bf16.gguf stock.gguf Q4_K
llama-cli -m model.beellama-qk-sinq.gguf --split-mode none -p "Explain quantization."
```

The second command creates a matched stock control. Use the same ordinary codec
allocation and imatrix for both conversions. `--pure` disables conventional
K-quant mixture heuristics, but inspect final types for ordinary shape fallbacks;
a selected SINQ tensor requiring codec fallback fails.

### Quantizer arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `--sinq-config FILE` | Absent/off | Strict v2 direct configuration; ordinary SINQ only |
| `--sinq-diagnostics-json FILE` | Absent | External deterministic per-tensor report, also usable for stock conversion |
| `--pure` | Off | Disable normal K-quant mixture allocation |
| `--imatrix FILE` | Absent | Existing importance-matrix input, transformed for conditioned weights |
| `--include-weights NAME`, `--exclude-weights NAME` | Unused | Existing imatrix inclusion/exclusion controls, not SINQ target selection |
| `--tensor-type NAME=TYPE` | Unused | Existing exact/regex destination codec allocation |
| `--tensor-type-file FILE` | Unused | Existing codec allocation rules read from a file |
| `--output-tensor-type TYPE`, `--token-embedding-type TYPE` | Ordinary allocation | Choose types for these unconditioned tensors |
| `--keep-split` | Off | Keep input sharding, with companions in their base matrix's shard |
| Final `nthreads` positional argument | Quantizer default | Quantization thread count |

Both `--sinq-config=FILE` and space-separated values are accepted, likewise for
diagnostics. `llama-quantize --help` lists the full stock quantizer interface.
No SINQ-specific inference switch or environment variable is required: the
loader detects the GGUF metadata.

The config path must be nonempty and actual quantization is required, not COPY.
Diagnostics are incompatible with COPY and `--dry-run`. Their path cannot alias
the input, output or configuration. Report-write failure fails quantization even
if the GGUF was already written; use a separate writable destination and inspect
the exit status before treating the output as complete.

### Direct JSON schema

Only `schema_version`, `defaults`, `targets`, optional `overrides` and optional
`encoding` are accepted at the root. `schema_version`, `defaults` and `targets` are required.
Unknown/duplicate keys and nonfinite numbers fail.

| Object | Fields and rules |
| --- | --- |
| `defaults` | Optional `generator: "sinq"`, `strip_width: 64`, `iterations: 16`, `strength: 1.0`; omitted values take these defaults |
| `targets` | Required `include`, either `"eligible"` or a nonempty exact-name array; optional `exclude` exact-name array |
| Override | Required `selector` and `parameters` |
| `selector` | Exactly one of `tensor` (exact name), `layer` (nonnegative integer), `role` (one supported projection role) |
| Override `parameters` | Partial direct defaults, applied over the root defaults |

Names must resolve. Explicit included tensors must be eligible; exclusions must
exist. Each selector must match selected eligible tensors, and override rules
must be disjoint even when they assign the same value. No selected tensors is an
error. For example, to override only down projections, replace the layer rule
with `{"selector":{"role":"ffn_down"},"parameters":{"strip_width":128}}`.
Do not combine it with an overlapping layer rule.

### Optional encoding controls

The existing `--sinq-config` file can additionally contain a global `encoding`
object. These controls affect direct quantization only, not inference or offline
optimizer candidates. They do not belong inside normalization `defaults` or
per-tensor `overrides`:

```json
{
  "schema_version": 2,
  "defaults": {"generator": "sinq", "strip_width": 64, "iterations": 16, "strength": 1.0},
  "targets": {"include": "eligible"},
  "encoding": {
    "scale_refinement_lambda": 0.17,
    "imatrix_exponent": 1.73
  }
}
```

- `scale_refinement_lambda` is any finite floating-point value in `[0,1]`,
  default **0** (disabled). After encoding all rows with the original serialized
  diagonal `u0`, positive per-column least squares fits the original weights
  against the fixed native decoded payload `B`. Only the companion changes:
  `F16((1-lambda)*u0 + lambda*sum(W*B)/sum(B*B))`. Row slabs contribute to one
  whole-tensor fit; no re-encoding or alternating optimization occurs. This is
  distinct from normalization strength and AWQ's `lambda`, and is not
  activation-aware calibration. Zero/nonfinite denominators, nonpositive or
  nonrepresentable F16 solutions retain `u0`, with counts in logs/diagnostics.
- `imatrix_exponent` is any finite floating-point value, default **2**. Conditioned
  tensors with importance entries use `I_prime = I * u0^p`. At `p=2` the original
  FP32 multiplication order is preserved exactly. Missing optional entries remain
  unaffected; required low-bit entries still fail closed. Invalid/overflowed
  transformed values fail rather than falling back. Without an imatrix, changing
  this exponent has no effect and produces a warning.

Neither control is restricted to the historical experimental grid: `0.17`,
`0.63`, `1.73`, and `2.12` are ordinary valid values in their respective domains.
Omitting `encoding`, or explicitly using lambda 0 and exponent 2, preserves the
previous payloads, companions and metadata. Non-default controls are recorded as
F64 producer provenance in `quantize.sinq.scale_refinement_lambda` and
`quantize.sinq.imatrix_exponent`, outside the closed inference namespace. The v2
companion layout, normalization recipes, codec allocations, and runtime math are
unchanged. Non-default refinement may change model quality adversely even when
weight-reconstruction error improves; opt in only after matched evaluation.

The public API exposes `llama_model_quantize_params.sinq_config` and
`sinq_diagnostics_json`. Initialize the structure with
`llama_model_quantize_default_params()` before setting them.

## Offline optimizer

Run from the repository root, using original floating Qwen3.5 GGUF weights and
local inputs. The optimizer does not download checkpoints or corpora. Native
code owns normalization, codec encoding/decoding, imatrix transformation and AWQ
correction; Python orchestrates immutable candidates, scoring and selection.

```sh
cmake -B build-sinq -DLLAMA_BUILD_SINQ_OPTIMIZER=ON
cmake --build build-sinq --target llama-quantize llama-sinq-native llama-sinq-model llama-sinq-groupwise
python -m tools.sinq.optimizer.cli plan --config generate.json
python -m tools.sinq.optimizer.cli generate --config generate.json
python -m tools.sinq.optimizer.cli score --config score.json
```

Supply your own original model, configuration files and captures. Adapt helper
paths to the actual build output (`.exe` on Windows).
Optional Python dependencies include NumPy/gguf and, for capture or advanced
scoring, Torch, safetensors and Transformers with Qwen3.5 support.

### Commands and stage schemas

```text
python -m tools.sinq.optimizer.cli <plan|generate|score|select|assemble|validate> --config FILE [--resume RUN_DIR]
```

Only `generate` and `select` accept `--resume`. Every configuration has integer
`schema_version: 2`, `inputs`, `limits`, and only its applicable groups below.
Unknown or duplicate keys, old schemas and nonfinite numbers fail. Do not put an
`operation` field in user JSON. `output.directory` is required except for
assembly (`output.gguf`) and validation (no output required).

| Stage | Required inputs | Other required groups |
| --- | --- | --- |
| `plan`, `generate` | `source_gguf`, `native_bridge` | `quantization`, `targets`, `candidate_generators` |
| `score` | `candidate_manifest`, `capture_manifest`, `native_bridge` | `scoring` |
| `select` | `source_gguf`, `candidate_manifest` | `selection` |
| `assemble` | `source_gguf`, `candidate_manifest`, `native_loader` | `selection.assignment` |
| `validate` | `artifact_manifest` | None |

`plan` shares the generation schema and preflights without encoding. Generation
uses exact uppercase qtypes in `quantization.recipe`, with an optional
`imatrix_path`. `targets` uses the direct resolver's exact include/exclude rules.
`parameters` optionally carries `defaults`, disjoint selector `overrides`, and
`locks`. See [direct settings](#direct-json-schema).

Every stage requires positive `limits.workspace_bytes`, `max_artifact_bytes` and
`max_files`; `max_tensors` is optional. Generation and native local scoring
accept at most 1 GiB of workspace, matching the native bridge limit; larger
values fail during configuration validation. Reference-backend scoring does not
have this native workspace ceiling. Additional limits are stage-specific:

- Generation: `max_candidates_per_tensor`, including stock.
- Scoring: `max_local_scores` for local scoring; `max_block_scores` for blockwise
  or Fisher; Fisher also requires `max_fisher_vjps`. `max_tokens` is optional.
- Selection: `max_model_evaluations` and `max_tokens` bound model-KL work.
- Assembly and validation accept no generation/scoring/selection counters.

Use `execution.threads` (1..1024) to override the logical-CPU default. This count
reaches native helpers, PyTorch and subprocess OpenMP/MKL controls and enters
run identities. Baseline verification also uses this limit for independent
file/tensor hash workers; it retains complete before/after SHA-256 checks.
GGUF metadata parsing and each individual SHA-256 stream remain serial.
Advanced placement and precision belong in `execution.scoring`,
not at the top level. See [advanced scoring execution](#advanced-scoring-execution).

### Complete candidates and locks

Ordinary candidates require `generator: "sinq"`, `strip_width`, `iterations`
and `strength`. AWQ candidates additionally require `generator: "awq"`,
`correction` (`weight_activation` or `activation_only`), `alpha` and `lambda`.
All three floating parameters are finite in `[0,1]`. Widths are
8/16/32/64/128/256/512; iterations are 4/8/16. AWQ requires verified activations via
`inputs.activation_manifest`; ordinary SINQ needs neither activations nor local
scores. There is no implicit ordinary candidate or AWQ alpha grid.

Each `candidate_generators` entry supplies `type` and either explicit complete
`configurations` (the type supplies `generator`) or `expansion: "cartesian"`
with an array for every applicable parameter. For example:

```json
{
  "type": "awq",
  "configurations": [
    {"strip_width": 64, "iterations": 16, "strength": 0.75,
     "correction": "weight_activation", "alpha": 0.5, "lambda": 1.0}
  ]
}
```

Declare ordinary SINQ candidates explicitly and add AWQ only with verified
captures. Automatic widths that do not divide a
tensor's input dimension are recorded as skipped with reasons; invalid manual
assignments fail. Generated stock is separate from strength-zero conditioning.
Candidate configuration identity remains distinct even when packed payloads match.
The registry binds source, tensor, codec, recipe, imatrix, complete parameters,
relevant calibration and producer fingerprints plus packed/F16 scale hashes.

Generation `parameters.locks` maps exact tensor names to complete parameters or
immutable candidate IDs. Selection cannot change these assignments. Selection
`locks` must agree with `baseline`; split any group mixing locked and unlocked
tensors before running. Resume requires the configured existing output directory
and verifies the complete input/producer/candidate closure before skipping work.
Changed inputs or settings require a new run directory.

### Surrogate scoring and model-KL selection

`scoring.kind` supports `local_output_l1`, `local_output_mse`,
`blockwise_output_l1`, `blockwise_output_mse` and `fisher_vjp_mc`. Blockwise and
Fisher require `inputs.checkpoint` and a versioned token/layer capture spec;
local scoring uses an activation manifest. Explicit local `scoring.assignments`
aggregate the sum of member tensor mean losses; their `max_local_scores` cap
counts primitive tensor/candidate scores, not just group labels. When a corpus
lock exists, scorer tokens must belong to its calibration/validation streams. Local held-out scoring uses
`selection_policy: "held_out"` plus validation and pairing manifests. Advanced
held-out capture provenance is not supported. Scores bind scorer, objective,
direction, scope, source, capture/reference and execution context. Do not compare
scores across metrics or contexts.

`selection` has `policy`, complete `baseline` (tensor-to-candidate-ID mapping),
and `groups` (group-to-tensor-list mapping). Groups must partition the complete
run assignment. Joint unlocked groups require explicit complete `alternatives`;
there is no implicit Cartesian product of joint alternatives.

- `manual` retains the baseline. `per_group_score` requires a `scores` artifact
  and chooses the minimum compatible surrogate score. Ties retain the initial
  assignment, then canonical candidate order. Both require `inputs.native_loader`.
- `groupwise_kl` evaluates complete model assignments under a hard model-evaluation
  budget. It requires `inputs.evaluator`, `selection_manifest` and
  `corpus_partitions_manifest`, with source/reference and selection-split locks.
- `pipeline` requires explicit `stages`. Every filter has exactly `type: "filter"`,
  `scores`, positive `keep` and `context: "frozen"`; the last stage is exactly
  `{"type":"select","method":"groupwise_kl"}`. Filters retain deterministic
  ties and locks, but are not rescored in each changing model context.

#### Native Groupwise KLD execution

The JSON file named by `inputs.evaluator` requires `schema_version: 2`,
`executable`, `native_loader`, `reference`, `corpus`, `context`, `batch`,
`microbatch`, `max_tokens` and `timeout`. These fields have no config defaults;
budgets must be positive, timeout finite and positive, and microbatch no larger
than batch. Model-KL reference generation requires original BF16/F32 weights,
not F16 or quantized weights. Optional `prepared_executable` enables prepared
evaluation; `reference_policy_sha256` explicitly binds a reused reference policy.
Prepared cross-device reuse is allowed only with unchanged numerical/runtime
policy and matching GPU model and driver. Its optional `execution` object
selects the native backend, independently of Torch surrogate scoring:

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

This object is a fragment of the evaluator file, not a complete select config.
Omitting it selects CPU placement, F16 K/V and automatic Flash Attention.
CPU mode explicitly uses `--device none --no-op-offload`; an empty CUDA visibility
variable alone is not sufficient to prevent operation offload on every platform.
`backend` accepts `cpu` or `cuda`; CUDA requires an explicit physical
`nvidia-smi` index. The evaluator resolves that index to a UUID, pins visibility
to that UUID and selects native `CUDA0` with no tensor splitting. CUDA supports
only `gpu_layers: "all"`; partial/zero offload and missing or mismatched native
PCI-device/full-layer evidence fail rather than silently accepting CPU inference.
Unavailable capacity remains a native process error, not a fallback policy.
K/V types are deliberately limited to `f32`, `f16` and `bf16` here; Flash
Attention accepts `on`, `off`, `auto`. CPU rejects CUDA placement/compute controls.
`cublas_compute_type` accepts `auto`, `f32`, `f16`, `bf16`; `mmq_precision` accepts
`auto`, `q4`, `q8`. These select existing native compute controls, not new kernels.

Native evaluation disables warmup and automatic fitting and explicitly binds
context, batching and stage threads. Inherited `LLAMA_ARG_*`, `GGML_*`,
`CUDA_*`, `CUBLAS_*` and `NVIDIA_TF32_*` overrides are removed from its environment;
only the declared native CUDA controls are installed. Policies bind the GPU
UUID/PCI identity and driver, executable and adjacent shared-runtime hashes,
KV/FA/compute settings and context/batching. Actual layer counts, native timing
lines and command evidence accompany references and uncached scores.

Create a **new selection-only reference** with
`NativeKLEvaluator(..., execution=the_same_object).create_reference(...)`, using
the existing selection token alignment. Never overwrite a CPU or final-test
reference. Expanded policy identities intentionally reject older reference
manifests and score/checkpoint identities; regenerate references into new paths,
not candidate matrices. CPU and CUDA KL values with different references are
not interchangeable. Resume uses the existing `select --resume RUN_DIR` path.
Native token TSV keeps its original columns and appends `candidate_top1` for
comparison rows (NA for reference generation), so numerical investigations can
compare actual predictions rather than only reference-agreement flags.

`select` publishes `selection.json` and `best.gguf`. For separate assembly, copy
the selected assignment into `selection.assignment` in an `assemble` config.
`validate` checks the candidate artifact manifest and its content closure; it is
not a substitute for a model forward or a quality evaluation. No surrogate loss
or selection KL establishes untouched-test improvement.

### Source, stock and corpus tooling

Keep a verified original source and complete stock allocation. The library API
`tools.sinq.optimizer.baseline.lock_baseline(source, baseline, recipe, output,
quantizer=..., native_validate=..., imatrix=..., workspace_bytes=...,
max_output_bytes=..., threads=..., timeout=...)` independently reencodes and checks
every stock payload before publishing a lock. `recipe` names every source tensor
with its exact uppercase qtype, including retained floating tensors;
`native_validate` is a required callable that checks the model with the native
loader. Pass the lock as `inputs.stock_baseline_manifest`. Unselected tensors
must come from a verified stock allocation, not accidentally retained source
floats. A partial allocation without that baseline is not a full-model recipe.

`tools.sinq.optimizer.corpora.lock_corpora(token_sources, partitions, output,
max_tokens=..., max_output_bytes=..., max_files=...)` is also a library API, not
an optimizer subcommand. It hashes local raw little-endian uint32 token files.
`token_sources` maps labels to files; partition chunks use `source`, `start` and
`end` original-token ranges. Supply disjoint `calibration`, `selection` and `final_test`
partitions (and optional `validation`); final-test data must never drive generation,
scoring, filtering or selection. Pass the lock as `corpus_partitions_manifest`
where applicable. Neither lock API is a CLI operation.

### Optional fields and defaults

Only the input names below supplement the required-input table above.

| Stage | Optional `inputs` keys |
| --- | --- |
| Plan/generate | `activation_manifest`, `validation_manifest`, `pairing_manifest`, `stock_baseline_manifest`, `corpus_partitions_manifest` |
| Score | `checkpoint`, `validation_manifest`, `pairing_manifest`, `stock_baseline_manifest` |
| Select | `selection_manifest`, `evaluator`, `stock_baseline_manifest`, `corpus_partitions_manifest`, `native_loader` |
| Assemble | `stock_baseline_manifest` |
| Validate | `source_gguf`, `native_loader`, `stock_baseline_manifest` |

`parameters` accepts only `defaults`, `overrides`, `locks`. Optimizer defaults
must be complete parameter records (unlike partial direct-quantizer defaults);
overrides use the direct selector schema. Automatic candidates are complete and have no
implicit width; discovery uses the minimum codec-aligned width only to resolve
eligible names, never to replace a requested candidate. A lock is either a
complete registered parameter record or an exact immutable ID reproduced by
that same run's generator specifications and source/producer binding.

`scoring` accepts `kind`, `context`, `probes`, `seed`, `fisher_positions`,
`assignments`, `selection_policy`. Context is `reference_inputs` (default) or
`selected_quantized_inputs` for advanced scorers. Fisher defaults to one probe
and seed zero; positions are distinct `[batch,time]` pairs. `assignments` maps
names to explicit tensor-to-candidate-ID objects. Local policy defaults to
`in_sample`; `held_out` requires both validation/pairing inputs. Unused advanced
controls and unrelated scorer limits fail validation.

`selection` accepts `policy`, `baseline`, `groups`, `locks`, `max_passes`,
`improvement_tolerance`, `scores`, `stages`, `alternatives`. Defaults are one pass,
zero tolerance and no locks. `alternatives` maps group names to lists of complete
member assignments. KLD/pipeline requires an explicit `max_model_evaluations`;
non-KLD selectors do not evaluate full-model KL.

Generation defaults to at most 128 candidates per tensor including stock
(hard native maximum 1024). Shared `max_tensors` defaults to 100000; scoring and
KLD token limits default to 4096. Other shared limits have no implicit defaults.
Advanced execution placement/precision/memory defaults are documented in
[advanced scoring execution](#advanced-scoring-execution). `execution.scoring` is
accepted only by scoring, not silently ignored by generation/assembly.

Every CLI invocation prints `stage_summary` with config/input hashes, exact
output file hashes and sizes, observed registry/model-KL cache hits/misses,
execution controls, elapsed time and Python process lifetime peak RSS. Native
workspace and advanced CUDA phase measurements are separate; lifetime RSS is
not a stage-specific hard quota. Immutable run/score/selection artifacts exclude
wall-clock summaries so resumption retains their identities.

## Activation export and file format

`export_activations.py` captures real Linear pre-inputs from a local unmodified
Qwen3.5 conditional checkpoint. It disables remote code and downloads. The
reference environment used Transformers 5.6.2; the older default conversion pin
4.57.6 does not provide that Qwen3.5 environment.

```sh
python tools/sinq/export_activations.py --checkpoint local-hf-model --gguf source-bf16.gguf --tensor-list selected.txt --text calibration.txt --output activations --device cpu --dtype bf16
```

This exporter's `--tensor-list` is an exact-name file, not a direct-quantizer
flag. Instead of `--text`, use `--tokens` with a local JSON integer-token array.

| Exporter argument | Default | Accepted input |
| --- | --- | --- |
| `--checkpoint` | Required | Local unmodified conditional Qwen3.5 checkpoint directory |
| `--gguf` | Required | Original floating GGUF matching the checkpoint |
| `--tensor-list` | Required | Unique exact base names, one per nonblank line |
| `--output` | Required | New output directory |
| `--text`, `--tokens` | Exactly one required | Local text or JSON integer-token array; no pad IDs or implicit truncation |
| `--device` | `cpu` | Torch device string, validated when placing the model |
| `--dtype` | `bf16` | `bf16` or `f16` |
| `--rows` | 128 | Reservoir rows per tensor, 1..256 |
| `--seed` | 0 | Unsigned 32-bit integer |
| `--context` | 512 | Fresh prefill chunk length, 1..4096 |
| `--max-tokens` | 4096 | Input token bound, 1..1048576 |
| `--threads` | Logical CPU count | Integer 1..1024 |

Default limits are 4096 tokens, fresh cache-free 512-token chunks and 128 rows
per tensor; maximum reservoir size is 256 rows. Every selected module must
execute. Full text-model storage and graph configuration are verified against
the original GGUF, including unselected tensors, aliases and converter
permutations, before capture. Transformed or prequantized inputs fail.

The new output directory contains `activations.json` and little-endian F32
sample files. Capture manifests bind source, checkpoint, corpus, token stream,
site and sampling identities. Keep activations local: they can contain sensitive
data. Failed exports may leave partial files but no final manifest; diagnose the
failure and use a new directory. The capture payload format is specified below.

This producer-only format stores actual reference linear inputs for optional
native-codec A-SINQ. It is not an imatrix format or GGUF runtime extension.
The consumer reads JSON and raw little-endian float32, never pickle.

### Files and validation

Pass the `activations.json` path as `inputs.activation_manifest` for optimizer
generation or `inputs.capture_manifest` for local scoring. This independent
export format remains version 1; it is not a v1 optimizer cache/config reader.
The root object requires:

| Field | Value |
| --- | --- |
| `schema_version` | Integer `1` |
| `architecture` | HF adapter identity `qwen3_5` (the source GGUF architecture is `qwen35`) |
| `source_sha256` | SHA-256 of the complete original single-file GGUF |
| `source_revision` | Nonempty reference checkpoint identity |
| `sampling` | Object described below |
| `tensors` | Nonempty array, at most 4096 entries |
| `provenance` | Optional object, recorded by the exporter |

`sampling` contains `contract: "valid-token-prefill-reservoir-v1"`, a uint32
`seed`, `max_rows` in `1..256`, and `corpus_sha256`/`token_sha256` digests.
Tokens are unpadded; every chunk is fresh prefill with no recurrent/KV cache
carried from another chunk. Token identity hashes uint32 little-endian IDs.
The exporter uses Algorithm R with independent PCG64 streams: SHA-256 of the
four little-endian seed bytes followed by the exact UTF-8 tensor name,
taking the first 128 digest bits as a little-endian integer seed. Defaults
are seed zero, 128 retained rows and 512-token chunks.

Each tensor object requires these fields, with no unknown fields:

| Field | Value |
| --- | --- |
| `name` | Exact GGUF base tensor, for example `blk.0.ffn_up.weight` |
| `site` | Exact adapter site, for example `qwen3_5.layers.0.ffn_up.input` |
| `reference_module` | Actual HF named-module path |
| `channel_layout` | `gguf-input` |
| `input_dim`, `output_dim` | Positive integer column/output-row counts |
| `count` | Integer `1..sampling.max_rows` |
| `dtype`, `order` | `<f4`, `row-major` |
| `data_file` | Contained relative filename |
| `sha256` | Actual binary digest, 64 lowercase hexadecimal digits |

The binary is exactly `count * input_dim * 4` bytes, without a header or
trailer, and contains finite IEEE binary32 samples shaped `[count,input_dim]`.
The exact base tensor and input/output dimensions must match the original
GGUF. The site is `qwen3_5.layers.{layer}.{role}.input` for its exact GGUF role.

The parser rejects duplicate JSON keys/tensor names, unknown required-format
fields, unsupported versions/sites/layouts/dtypes, nonintegral dimensions,
bad digests, nonfinite samples and missing selected tensors. Manifest size
is limited to 8 MiB, nesting to 64 levels and each sample file to 256 MiB.
Relative sample paths must remain inside the canonical manifest directory;
absolute paths, traversal and symlink escapes fail. All entries are checked
sequentially with a 64 KiB read buffer, then selected samples are revalidated
when loaded for scoring. Outputs cannot overwrite the source, manifest,
selection file or activation binaries.

### Qwen3.5 capture sites

For the supported conditional HF model, prefix every module below with
`model.language_model.layers.{layer}.`. Capture the actual `nn.Linear`
pre-input, not the decoder layer's generic hidden state.

| GGUF role | HF module | Input semantics |
| --- | --- | --- |
| `ffn_gate` | `mlp.gate_proj` | Post-attention normalized residual |
| `ffn_up` | `mlp.up_proj` | Same normalized residual, not gate output |
| `ffn_down` | `mlp.down_proj` | `SiLU(gate(x))*up(x)` |
| `attn_q` | `self_attn.q_proj` | Normalized attention input, joint per-head query and output-gate rows |
| `attn_k` | `self_attn.k_proj` | Before K normalization and rotary embedding |
| `attn_v` | `self_attn.v_proj` | Normalized attention input |
| `attn_output` | `self_attn.o_proj` | Flattened attention result after sigmoid output gating |
| `attn_qkv` | `linear_attn.in_proj_qkv` | Masked normalized input before convolution |
| `attn_gate` | `linear_attn.in_proj_z` | Same linear-attention input, Z projection |
| `ssm_alpha` | `linear_attn.in_proj_a` | Before bias, softplus and negative exponential multiplier |
| `ssm_beta` | `linear_attn.in_proj_b` | Before sigmoid |
| `ssm_out` | `linear_attn.out_proj` | Recurrent result after gated RMS normalization |

Full attention has no separate gate module: those rows are interleaved in
`q_proj`. The `attn_gate` GGUF role above is linear-attention Z.

The converter reorders linear-attention V heads from grouped to tiled order.
For `ssm_out`, captured input channels must undergo the same permutation as
GGUF input columns: reshape the head axis as
`[key_heads,value_heads/key_heads,value_head_dim]`, swap its first two axes,
then flatten. QKV's V rows, Z rows and alpha/beta rows receive corresponding
weight-output permutations; their input channels are unchanged. The exporter
verifies the entire text-model parameter set against GGUF,
not only selected projections. This includes embeddings, regular norm `1+w`,
negative exponentiation of `A_log`, convolution squeezing, SSM head reordering
and the output head (or its verified tied alias). Comparisons are exact after
converter transformations and rounding to each GGUF tensor's storage dtype.
Large matrices, including embeddings, are compared in bounded row chunks;
packed BF16 words are sliced before conversion. Graph-defining configuration
is also checked, and unsupported activation/RoPE variants fail explicitly.

The mapping follows the converter's tensor transformations and the supported
Qwen3.5 reference graph. Structural correspondence does not imply bitwise HF
versus native activation equality. The exporter records graph/module identity,
checkpoint file hashes, corpus/token identity, dtype and chunking policy.
Provenance includes `reference_identity.contract` equal to
`qwen35-full-text-storage-v1`, verified tensor count/digest, configuration fields,
output alias handling and precision policy. The GGUF, corpus and checkpoint
hashes are checked again before writing the manifest. Captures made without
this attestation must be re-exported, not relabelled. This remains
producer-only provenance within format v1, not a new required runtime field.

Hadamard-transformed, already conditioned/quantized sources, fused/missing
reference modules and other architectures are rejected rather than guessed.
MTP and vision projections are outside the supported primary-trunk set.


## Advanced scoring execution

Advanced v2 `score` stages (`blockwise_output_l1`, `blockwise_output_mse` and
`fisher_vjp_mc`) support explicit
CPU/CUDA placement and FP32/BF16 parameter storage. CPU FP32 remains the default.
The native SINQ codec, candidate assignments and scoring objectives are unchanged.
This is an engineering backend, not a new quantization recipe or quality result.

### Configuration

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
    "candidate_manifest": "generated-candidates/run-manifest.json",
    "capture_manifest": "block-capture.json",
    "checkpoint": "local-hf-model",
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
  "output": {"directory": "blockwise-scores"}
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

### Exact reference storage and precision

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

### Placement, identities and resource limits

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


## GGUF runtime contract

The closed `beellama.qk_sinq.` namespace contains exactly five keys:

| Key | GGUF type | Value |
| --- | --- | --- |
| `version` | UINT32 | `2` |
| `transform` | STRING | `f16_input_diagonal` |
| `tensors` | ARRAY of STRING | Strictly sorted base tensor names |
| `recipes` | ARRAY of STRING | Aligned complete canonical candidate JSON records |
| `producer` | STRING | Nonempty producer identity, at most 1024 bytes |

Recipes use sorted keys, no whitespace and finite numbers. Each is at most
4096 bytes; there are at most 100000 conditioned tensors and 16 MiB of aggregate
recipe/name metadata. There are no global width, iteration, algorithm or numerics
keys. Version-1 metadata and unknown namespace keys fail, not migrate implicitly.
Each `<base>.sinq_scale2` companion is contiguous positive finite F16 `{K}` in
the same shard as its base. Missing coverage, orphans and invalid per-tensor
role, codec, shape or alignment fail.

Inference still computes `(W/u) * (u*x)`; LoRA receives original inputs.
Ordinary SINQ forms `u = F16(t^strength)` and uses the serialized diagonal for
weight compensation and imatrix transformation. Per-tensor production settings
do not introduce different inference math. Source, corpus and recipe hashes and
detailed evidence belong in external manifests.

The loader checks companion values even when optional general tensor checking is
disabled. It promotes them once into placement-matched F32 runtime buffers,
shares equal vectors outside tensor mode and omits exact all-one diagonals from
the activation graph. Tensor mode keeps separate runtime vectors per base matrix
so equal values cannot accidentally reuse a different channel partition.
Identical input/scale products can be reused before ordinary matrix multiplication.
The serialized companion costs `2*K` bytes per matrix, or `16/N` idealized bits
per original weight, excluding metadata and alignment.

## Validation and troubleshooting

Build the helpers with `LLAMA_BUILD_SINQ_OPTIMIZER=ON`. `llama-sinq-model` accepts:

```text
llama-sinq-model --load-only MODEL
llama-sinq-model --forward MODEL [--gpu-layers N] [--threads N] TOKEN...
```

The forward command takes 1..32 integer token IDs, not text. `--threads` accepts
1..1024 and `--gpu-layers` a positive integer. CPU is the default;
explicit GPU offload fails if no GPU backend is available. Device visibility
follows the caller's environment. Loading checks the runtime contract; a forward
also exercises the model graph. Neither measures quality.

`llama-sinq-native` is the optimizer's request-file bridge. Its entry points are:

```text
llama-sinq-native request.json
llama-sinq-native --plan-config direct.json inventory.json
llama-sinq-native --validate-activations activations.json source.gguf [THREADS]
```

The Python optimizer constructs codec requests; use its stage configuration
rather than treating the bridge's internal request format as a quantizer CLI.
`llama-sinq-groupwise` provides prepared native model-KL evaluation and uses the
perplexity argument parser. Supply it through evaluator `prepared_executable`;
it is not a seventh optimizer subcommand.

Check a stage schema without model execution:

```python
import json
from tools.sinq.optimizer.schemas import validate_config

with open("generate.json", encoding="utf-8") as stream:
    validate_config(json.load(stream), "generate")
```

Actual stage loading additionally rejects duplicate JSON keys. Schema validation
alone does not verify files, devices, checkpoints or runtime compatibility.

For advanced scorers, `inputs.capture_manifest` is a separate schema-2 capture
specification, not an activation-export manifest. Required fields are
`schema_version`, `source_sha256`, `tokens` (nonempty 2D integer token array) and
`layers` (unique valid block indices). Optional `attention_mask` and `position_ids`
match the token shape; masks are binary. `reference_inputs` rejects baseline
weights. `selected_quantized_inputs` requires an explicit `baseline` affecting
upstream blocks only. Fisher `seed` must fit uint32 at runtime.

### Generation configuration example

This bounded example generates candidates for one projection. Replace input
paths with your local files; it is not a complete full-model codec allocation.
For full-model assembly supply the complete exact recipe and verified stock
baseline described above.

```json
{
  "schema_version": 2,
  "inputs": {
    "source_gguf": "source-bf16.gguf",
    "native_bridge": "build-sinq/bin/llama-sinq-native"
  },
  "quantization": {"recipe": {"blk.0.ffn_up.weight": "Q4_K"}},
  "targets": {"include": ["blk.0.ffn_up.weight"]},
  "candidate_generators": [
    {"type": "sinq", "expansion": "cartesian", "parameters": {
      "strip_width": [64, 128], "iterations": [16], "strength": [0.5, 1.0]
    }}
  ],
  "execution": {"threads": 8},
  "limits": {
    "workspace_bytes": 1073741824,
    "max_artifact_bytes": 1073741824,
    "max_files": 1000,
    "max_tensors": 1,
    "max_candidates_per_tensor": 5
  },
  "output": {"directory": "generated-candidates"}
}
```

Use `plan` first: source dimensions, workspace and file budgets can reject even
a small candidate count. The optimizer currently requires a single-file source.
`IQ1_S`, `IQ2_XXS` and `IQ2_XS` candidates require an imatrix. Its native input
contains F32 `<tensor>.in_sum2` channel sums and `<tensor>.counts` scalar counts;
sums must be finite/nonnegative and counts finite/positive.

### Common failures

| Failure | Action |
| --- | --- |
| Unknown field, old schema or obsolete option | Use the v2 fields in this guide; do not relabel old artifacts |
| Overlapping override | Make selectors disjoint or combine the assignment |
| Selected tensor alignment or codec fallback | Choose a compatible width and explicit supported codec |
| Missing activation sample/site | Capture that actual Linear input or explicitly reduce the selected target set |
| Checkpoint/source mismatch | Use the original matching checkpoint and GGUF; re-export, do not edit provenance |
| BF16 reference cannot represent a parameter exactly | Use FP32 scoring; do not round the reference to make checks pass |
| Workspace, file or memory bound exceeded | Adjust declared budgets with physical headroom, or reduce the explicit workload |
| Existing output or resume identity mismatch | Resume only unchanged supported stages; changed inputs require a new directory |
| Missing/orphan companion, wrong shard or unknown metadata key | Recreate the model from original floats with a supported producer |
| Base/companion placement mismatch | Apply matching placement overrides; row split is unsupported |
| Native CUDA offload evidence missing | Check build, device selection and capacity; CPU fallback is not accepted as CUDA evaluation |

Run targeted regressions with
`ctest --test-dir build-sinq -R 'sinq|perplexity-full' --output-on-failure`.
Optional-dependency skips and unavailable devices do not establish coverage.
For quality evaluation, keep calibration, selection and final-test data disjoint;
use matching source, codec allocation, imatrix, tokens, context, batch/ubatch,
sampling and backend policy for the stock and conditioned models. Keep detailed
run records outside the source repository.

The numerical adaptation and reference fixtures retain Apache-2.0 attribution
in [LICENSE-SINQ](../licenses/LICENSE-SINQ) and
[NOTICE-SINQ](../licenses/NOTICE-SINQ). BeeLlama is not the original Python SINQ
serving implementation.
