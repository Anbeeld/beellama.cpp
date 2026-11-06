# Offline SINQ optimizer v2

Run from the repository root, using original floating Qwen3.5 GGUF weights and
local inputs. The optimizer does not download checkpoints or corpora. Native
code owns normalization, codec encoding/decoding, imatrix transformation and AWQ
correction; Python orchestrates immutable candidates, scoring and selection.

```sh
cmake -B build-sinq -DLLAMA_BUILD_SINQ_OPTIMIZER=ON
cmake --build build-sinq --target llama-quantize llama-sinq-native llama-sinq-model
python -m tools.sinq.optimizer.cli plan --config ../beellama.doc/docs/records/worklogs/sinq-optimizer-refactor/examples/generate-small-grid.json
python -m tools.sinq.optimizer.cli generate --config ../beellama.doc/docs/records/worklogs/sinq-optimizer-refactor/examples/generate-small-grid.json
python -m tools.sinq.optimizer.cli score --config ../beellama.doc/docs/records/worklogs/sinq-optimizer-refactor/examples/score-local.json
```

These example paths refer to inputs you must supply, not bundled models or
captures. Adapt helper paths to the actual build output (`.exe` on Windows).
Optional Python dependencies include NumPy/gguf and, for capture or advanced
scoring, Torch, safetensors and Transformers with Qwen3.5 support.

## Commands and stage schemas

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
`locks`. See [direct settings](../../docs/sinq-settings.md).

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
not at the top level. See [Tier-C execution](../../docs/sinq-tier-c-execution.md).

## Complete candidates and locks

Ordinary candidates require `generator: "sinq"`, `strip_width`, `iterations`
and `strength`. AWQ candidates additionally require `generator: "awq"`,
`correction` (`weight_activation` or `activation_only`), `alpha` and `lambda`.
All three floating parameters are finite in `[0,1]`. Widths are
32/64/128/256/512; iterations are 4/8/16. AWQ requires verified activations via
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

[generate-small-grid.json](../../../beellama.doc/docs/records/worklogs/sinq-optimizer-refactor/examples/generate-small-grid.json) explicitly
contains ordinary SINQ candidates; add the optional AWQ entry above only with verified captures. Automatic widths that do not divide a
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

## Surrogate scoring and model-KL selection

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

`select` publishes `selection.json` and `best.gguf`. For separate assembly, copy
the selected assignment into `selection.assignment` in an `assemble` config.
`validate` checks the candidate artifact manifest and its content closure; it is
not a substitute for a model forward or a quality evaluation. No surrogate loss,
selection KL, or tiny test establishes untouched-test improvement. Current verification and the six-model T340/T425 gate are recorded in
[REPORT.md](../../../beellama.doc/docs/records/worklogs/sinq-optimizer-refactor/REPORT.md).

## Source, stock and corpus tooling

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

## Local activation export

`export_activations.py` captures real Linear pre-inputs from a local unmodified
Qwen3.5 conditional checkpoint. It disables remote code and downloads. The
reference environment used Transformers 5.6.2; the older default conversion pin
4.57.6 does not provide that Qwen3.5 environment.

```sh
python tools/sinq/export_activations.py --checkpoint local-hf-model --gguf source-bf16.gguf --tensor-list selected.txt --text calibration.txt --output activations --device cpu --dtype bf16
```

This exporter's `--tensor-list` is an exact-name file, not a direct-quantizer
flag. Instead of `--text`, use `--tokens` with a local JSON integer-token array.
Default limits are 4096 tokens, fresh cache-free 512-token chunks and 128 rows
per tensor; maximum reservoir size is 256 rows. Every selected module must
execute. Full text-model storage and graph configuration are verified against
the original GGUF, including unselected tensors, aliases and converter
permutations, before capture. Transformed or prequantized inputs fail.

The new output directory contains `activations.json` and little-endian F32
sample files. Capture manifests bind source, checkpoint, corpus, token stream,
site and sampling identities. Keep activations local: they can contain sensitive
data. Failed exports may leave partial files but no final manifest; diagnose the
failure and use a new directory. See [activation format](../../docs/sinq-activation-format.md)
for the export payload format, which is separate from optimizer config version 2.

## Validation scope

Schema examples can be checked with
`tools.sinq.optimizer.schemas.validate_config(raw, stage)` without model execution.
Native helper, exporter, optimizer and backend tests live under `tests/test-sinq-*`.
Optional-dependency skips and unavailable hardware are not runtime coverage.
The [v2 contract](../../../beellama.doc/docs/records/worklogs/sinq-optimizer-refactor/sinq-optimizer-contract.md) defines the implementation
target; it does not claim that every verification gate passed.

## Complete optional fields and defaults

Only the input names below supplement the required-input table above.

| Stage | Optional `inputs` keys |
| --- | --- |
| Plan/generate | `activation_manifest`, `validation_manifest`, `pairing_manifest`, `stock_baseline_manifest`, `corpus_partitions_manifest` |
| Score | `checkpoint`, `validation_manifest`, `pairing_manifest`, `stock_baseline_manifest` |
| Select | `selection_manifest`, `evaluator`, `stock_baseline_manifest`, `corpus_partitions_manifest`, `native_loader` |
| Assemble | `stock_baseline_manifest` |
| Validate | `source_gguf`, `native_loader`, `stock_baseline_manifest` |

`parameters` accepts only `defaults`, `overrides`, `locks`. Manual defaults and
overrides use the direct schema. Automatic candidates are complete and have no
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
[Tier-C execution](../../docs/sinq-tier-c-execution.md). `execution.scoring` is
accepted only by scoring, not silently ignored by generation/assembly.

Every CLI invocation prints `stage_summary` with config/input hashes, exact
output file hashes and sizes, observed registry/model-KL cache hits/misses,
execution controls, elapsed time and Python process lifetime peak RSS. Native
workspace and advanced CUDA phase measurements are separate; lifetime RSS is
not a stage-specific hard quota. Immutable run/score/selection artifacts exclude
wall-clock summaries so resumption retains their identities.

New complete direct, automatic, locked-hybrid and scorer examples are kept in
[beellama.doc](../../../beellama.doc/docs/records/worklogs/sinq-optimizer-refactor/examples/).
The native checker supports `--forward MODEL [--gpu-layers N] TOKEN...`; CPU is
the default, explicit GPU offload fails when no GPU backend exists. Device
visibility follows the caller's environment. `--load-only MODEL` independently
checks the runtime contract without running a forward.
