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
| `--sinq-strip-width 64\|128` | `64` | Full-height normalization strip width, not the codec block width. Selected input dimensions must be divisible by it. |
| `--sinq-iterations 4\|8\|16` | `16` | Best pre-update iterate selection. The settings ablation uses only 4 and 16. |
| `--sinq-gate on\|off` | `on` | Whole-matrix kurtosis greater than 5 selects identity. The completed Thireus baseline explicitly used `off`. |
| `--sinq-strength FLOAT` | `1.0` | Finite value in `[0,1]`. Requires conditioning; nondefault values require `sidecar`, not test-only `identity`. |
| `--sinq-tensor-list PATH` | absent | UTF-8 file of exact base tensor names, one per line. Requires conditioning. Cannot be combined with historical coverage. |
| `--sinq-diagnostics-json PATH` | absent | External deterministic JSON. Also valid for stock quantization. Not valid with COPY or `--dry-run`. Write failures fail the quantization call. |
| `--sinq-coverage` | off | Historical reproduction only. Preserves noneligible source types, changing ordinary quantization allocation. Never use it as a conditioning mask or for the settings ablation. |

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
three operations: `W/u`, `u*x`, and `imatrix*u²`. Strength 1 retains the legacy
rounding path; strength 0 produces identity. LoRA continues to use the original
activations; only the base weight receives compensated inputs.

The new public API fields, appended to `llama_model_quantize_params`, are
`float sinq_strength`, `const char * sinq_tensor_list`, and
`const char * sinq_diagnostics_json`. Initialize with
`llama_model_quantize_default_params()` to obtain `1.f`, `nullptr`, and
`nullptr`. The API rejects nondefault strength or a tensor list without
conditioning. Unlike the CLI, it cannot distinguish an explicitly assigned
strength of 1 from the default. Nondefault strength with test-only identity is
rejected; use sidecar with strength 0 for that control.

Strength and mask provenance stays external. No new GGUF metadata fields are
added: the serialized diagonal already defines the runtime transform. Omitting
the new options, or using strength 1 and an explicit full eligible mask, keeps
the existing metadata and companion format. Byte-for-byte compatibility still
requires the same input, allocation, settings, toolchain and numerical code.
Locked T340/T425 artifact comparisons are separate verification gates, not a
claim established by these source changes. In the current reproduction,
the main agent reports the locked T340 SINQ producer as `438829a44` and the
configured build commit as `e19c75f54`. The producer field must describe the
actual build, not be rewritten to force a whole-file hash match. Treat this
expected provenance-only difference separately from payload comparisons and
obtain the required owner exception before quality scoring.

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
`unsupported_codec`, or `not_in_tensor_list`). Included records report gate
outcome, double-precision kurtosis, degeneracy, identity, each strip's chosen
iteration and initial/best/terminal joint imbalance, chosen-iteration
histogram, scale min/median/max, selected-identity count, companion payload
bytes and SHA-256 of the exact serialized F16 scale bytes. Importance-matrix
presence and transformation are reported separately.

Excluded records explicitly use `statistics_status: "not_computed"` with null
numerical diagnostics and scale digest, zero companion bytes, and no importance
transformation. They do not pretend to measure a gate outcome or serialized
scales. Collect whole-model gate counts independently when comparing excluded
or stock tensors. Totals include model and eligible counts/parameters,
included counts/parameters, companion bytes, payload bytes and reason counts.

Choose a diagnostics path outside the GGUF, input model and mask files, with an
existing parent directory. Diagnostics are written after model processing;
a diagnostics failure returns failure even if an output GGUF was already
written. Do not treat that GGUF as a successfully locked candidate.

## Verification boundaries

The registered `test-qk-sinq-model` fixture covers complementary FFN/attention
masks, exact companion association, unchanged stock types and excluded bytes,
strength-zero identity payloads, strength-one/full-mask compatibility,
deterministic external reports, invalid API values and malformed selections.
Numeric strength and gate tests, locked artifact comparisons, LoRA/CPU/CUDA
checks and the full registered CTest suite remain required before experimental
candidate generation. The existing CTest registration is unchanged.
