# SINQ quantizer settings

SINQ sidecars require BeeLlama. The weights retain their ordinary native GGML
codec IDs, but other GGUF engines do not apply the companion diagonals and can
silently produce incorrect results. Use a `.beellama-qk-sinq.gguf` filename.
Always quantize from original F32, F16 or BF16 weights, not a conditioned or
quantized artifact.

## Controls and defaults

| CLI flag | Default | Meaning and compatibility |
| --- | --- | --- |
| `--sinq-precondition none\|sidecar\|identity` | `none` | `sidecar` enables conditioning; `identity` is test-only. |
| `--sinq-strip-width 32\|64\|128\|256\|512` | `64` | Full-height normalization strip width, not the codec block width. Selected input dimensions must be divisible by it. |
| `--sinq-iterations 4\|8\|16` | `16` | Best pre-update iterate selection. |
| `--sinq-strength FLOAT` | `1.0` | Finite value in `[0,1]`. Requires conditioning; nondefault values require `sidecar`, not test-only `identity`. |
| `--sinq-calibration PATH` | absent | Opt-in real-activation manifest or directory. Native-codec A-SINQ at fixed SINQ strength; requires `sidecar`. |
| `--sinq-tensor-list PATH` | absent | UTF-8 file of exact base tensor names, one per line. Requires conditioning. Cannot be combined with historical coverage. |
| `--sinq-diagnostics-json PATH` | absent | External deterministic JSON. Also valid for stock quantization. Not valid with COPY or `--dry-run`. Write failures fail the quantization call. |
| `--sinq-coverage` | off | Historical reproduction only. Preserves noneligible source types, changing ordinary quantization allocation. Never use it as a conditioning mask. |

Both `--sinq-option value` and `--sinq-option=value` forms are accepted for
value-taking options. Conditioning requires actual quantization, not COPY.

Conditioning covers only Qwen3.5 primary-trunk two-dimensional attention, SSM
and FFN projections. Embeddings, the output head, norms and auxiliary layers
remain outside the conditioning set. Supported actual target codecs are
`Q3_K`, `Q4_K`, `Q5_0`, `Q5_1`, `Q6_K`, `Q8_0`, `IQ1_S`, `IQ1_M`,
`IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`, `IQ4_NL` and `IQ4_XS`.
Automatic full coverage retains the existing exclusion of unsupported codecs;
an explicit list naming one fails instead. A selected shape requiring codec
fallback or incompatible strip alignment fails rather than changing the type
or reducing coverage.

## Strength and compatibility

For the selected best-iterate diagonal `t`, strength produces `u = t^strength`.
The optimizer rounds the result to F16 and uses the serialized values for all
three operations: `W/u`, `u*x`, and `imatrix*u²`. Strength 1 uses the full
normalization diagonal; strength 0 produces identity in ordinary SINQ. With
calibration enabled, strength 0 disables only the Sinkhorn component; AWQ
correction can still change the companion. LoRA continues to use the
original activations; only the base weight receives compensated inputs.

The public fields in `llama_model_quantize_params` are
`float sinq_strength`, `const char * sinq_tensor_list`,
`const char * sinq_diagnostics_json`, and `const char * sinq_calibration`.
Initialize with `llama_model_quantize_default_params()` to obtain strength
`1.f` and null optional pointers. The API rejects nondefault strength or a tensor list without
conditioning. Unlike the CLI, it cannot distinguish an explicitly assigned
strength of 1 from the default. Nondefault strength with test-only identity is
rejected; use sidecar with strength 0 for that control.

Strength and mask provenance stays external. No new GGUF metadata fields are
added: the serialized diagonal already defines the runtime transform. Omitting
the new options, or using strength 1 and an explicit full eligible mask, keeps
the normalization-only metadata and companion format. Byte-for-byte
compatibility still requires the same input, allocation, settings, toolchain
and numerical code.

The supported `beellama.qk_sinq.` metadata fields are `version`, `tensors`,
`algorithm`, `strip_width`, `iterations`, `producer` and `numerics`. Any other
field under this prefix is rejected, even without a version field. There is
no compatibility interpretation of unsupported fields.

## Exact-name masks

For example, a list can contain:

```text
blk.0.ffn_gate.weight
blk.0.ffn_up.weight
blk.0.ffn_down.weight
```

Names are case-sensitive. LF and CRLF line endings are accepted, including a
final newline. There is no whitespace trimming, comment syntax, regex, BOM,
fuzzy matching or alias expansion. Empty files, blank lines, duplicates,
malformed names, missing names, unsupported codecs, ineligible shapes,
embedding names and output names fail. Eligible names are ASCII, a subset of
UTF-8. A list must select at least one tensor.

Absent a list, all eligible supported projections receive companions. With a
list, only named tensors receive them. Ordinary codec allocation is unchanged;
unselected payloads follow the stock native quantization path. Complementary
masks therefore partition conditioning, not quantization coverage.

## External diagnostics

The report contains `schema_version`, `tensor_list_sha256`, `tensors` and
`totals`. The optional list digest hashes sorted exact names, each followed by
one LF byte. An absent list has a null digest. Tensor records are sorted by
exact name, independent of processing order; JSON object keys are stable.
Reports contain no timestamps or input/output paths.

Every structurally eligible projection is recorded, including stock and
excluded tensors. Records include actual target `qtype`, shape, parameter
count, settings, inclusion and reason (`included`, `stock`,
`unsupported_codec`, or `not_in_tensor_list`). Included records report
identity, each strip's chosen iteration and initial/best/terminal joint
imbalance, chosen-iteration histogram, scale min/median/max, selected-identity
count, companion payload bytes and SHA-256 of the exact serialized F16 scale
bytes. Importance-matrix presence and transformation are reported separately.

Excluded records explicitly use `statistics_status: "not_computed"` with null
numerical diagnostics and scale digest, zero companion bytes, and no importance
transformation. They do not pretend to measure serialized scales. Totals
include model and eligible counts/parameters, included counts/parameters,
companion bytes, payload bytes and reason counts.

Choose a diagnostics path outside the GGUF, input model and mask files, with an
existing parent directory. Diagnostics are written after model processing;
a diagnostics failure returns failure even if an output GGUF was already
written. Do not treat that GGUF as a successfully locked candidate.

## Optional native-codec A-SINQ

A-SINQ uses actual reference linear-input samples, not diagonal imatrix values.
It is adapted to the selected native GGUF codec, not a reproduction of the
paper's uniform RTN quantizer. End-to-end KLD/PPL improvements on Thireus
recipes have not been measured. A lower local calibration loss is not a
quality guarantee.

First export from a local, unmodified Qwen3.5 conditional HF checkpoint and
your own calibration text or token IDs. Optional Python dependencies are
confined to this tool. No corpus or model is downloaded:

```sh
PYTHONPATH=gguf-py python tools/sinq/export_activations.py --checkpoint local-hf-model --gguf source-bf16.gguf --tensor-list selected.txt --text calibration.txt --output activations --rows 128 --seed 0 --device cuda:0
llama-quantize --pure --sinq-precondition sidecar --sinq-tensor-list selected.txt --sinq-calibration activations --sinq-diagnostics-json calibration-report.json source-bf16.gguf model.beellama-qk-sinq.gguf Q4_K
```

Every selected tensor needs its own samples. An absent tensor list selects all
eligible supported tensors, so a partial activation manifest requires an
explicit list. Missing samples, wrong source hashes/shapes/sites, malformed
files, and nonfinite values fail. Calibration currently requires one original
GGUF source file and rejects layer pruning/remapping and graph metadata
overrides. The CLI's four imatrix provenance fields remain supported.
Ordinary SINQ retains its existing split and selection behavior.

At the user's fixed strength, calibration tests the ordinary F16 diagonal plus
20 AWQ factors with alpha `0, 0.05, ..., 0.95`, beta zero and weight scaling.
For channel means `mX=mean(abs(X))`, `mW=mean(abs(W))`, the upstream formula is
`a=clamp(clamp(mX^alpha,1e-4)/(mW^(1-alpha)+1e-4),1e-4)` followed by
`a/=sqrt(min(a)*max(a))`. Candidate compensation is
`u=F16((t^strength)/a)`, with one F16 rounding after AWQ correction. The
ordinary candidate keeps its original F16 bytes. Each candidate uses `W/u`,
`I*u*u` when available, and the ordinary native encoder/dequantizer. The
objective is `mean(abs(X @ (Wq*u-W).T))`, accumulated in ordered compensated
double precision. Ties within `64*DBL_EPSILON*max(1,losses)` keep the earlier
candidate; ordinary SINQ is first.

Full-height normalization uses at most 128 MiB of aggregate worker buffers;
an indivisible strip exceeding that bound fails. Calibration row/sample
buffers are capped at the smaller of the configured quantizer buffer limit
and 64 MiB. Native codec setup tables, source mappings and thread stacks are
not part of that row-buffer estimate. Only one tensor and one candidate are
scored at a time. The producer logs time and buffer estimates. Deterministic
external JSON adds source/manifest identities, per-tensor sample count,
codec, dimensions, activation digest/site, candidate losses and selected AWQ
alpha (`null` means ordinary SINQ). Identity and strip iteration statistics
describe the Sinkhorn stage; scale ranges/digests describe the final serialized
companion.

The seven-field version-1 runtime metadata namespace, F16 companion shape,
graph operators, LoRA input separation and inference kernels are unchanged.
See [activation format and mapping](sinq-activation-format.md) and the
[offline exporter](../tools/sinq/README.md) for source requirements and limits.

## Verification boundaries

The registered `test-qk-sinq-model` fixture covers complementary FFN/attention
masks, exact companion association, unchanged stock types and excluded bytes,
strength-zero identity payloads, strength-one/full-mask compatibility,
deterministic external reports, invalid API values and malformed selections.
Numeric normalization and strength tests, locked artifact comparisons,
LoRA/CPU/CUDA checks and the full registered CTest suite remain required before
experimental candidate generation. Additional registered tests cover activation-format rejection, native-codec
calibration, real Linear capture, and a tiny exporter/importer roundtrip.
