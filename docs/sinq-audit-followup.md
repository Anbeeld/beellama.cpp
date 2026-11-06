# A-SINQ audit follow-up

> Historical pre-refactor report. Its removed interfaces and v1 artifacts are
> not current instructions or v2 verification. Use [current SINQ settings](sinq-settings.md).

Reviewed against `5336b2b5ddad878af523609b6b07b3c660a064f7`. The checkpoint
identity concern was valid. The activation and additional-codec requests were
validation gaps, not demonstrated errors in the candidate scorer. This update
changes offline verification and tests, not inference or the optimization grid.

## Complete reference identity

Export now verifies every text-model parameter against the source GGUF before
capturing any activations. Matching selected projections alone is insufficient.
The comparison includes embeddings, unselected projections, normalization,
SSM convolution/state parameters and the output head or verified tied alias.
Unknown/missing/extra text tensors and mismatches fail closed. Vision parameters
are excluded because the exporter executes the text network directly.

The check applies the converter's `1+w` regular norms, `-exp(A_log)`, convolution
squeeze, grouped-to-tiled projection/state permutations and declared GGUF float
storage rounding. Large weights are streamed through bounded row comparisons,
including packed BF16 decoding, rather than expanding entire embeddings to F32.
The conservative identity workspace limit is 64 MiB, excluding reference-model
storage, source mappings and small permutation indices.

Graph-defining metadata is checked too: dimensions, heads, norm epsilon, SSM
configuration, recurrent layers and supported RoPE settings. Unsupported graph
variants are rejected. The manifest's optional producer provenance records the
`qwen35-full-text-storage-v1` contract, verified tensor count/digest, configuration
fields, alias handling and precision policy. GGUF, corpus and checkpoint hashes
are checked again before publishing the manifest. This does not change activation
format v1 or the runtime metadata namespace.

Regression tests reproduce the audited baseline accepting changed embeddings
while its selected projection remains identical. The updated exporter rejects
that case before creating the output directory. Tests also cover unselected
norm changes, configuration mismatch, missing/extra weights, nonfinite values,
forced row chunking, input-file mutation, and a complete locally generated
27-tensor Qwen3.5 text graph converted by the repository's actual converter.

Existing calibration captures are not retroactively attested. Re-export them
before relying on this stronger identity guarantee. Manually supplied activation
files still require trustworthy acquisition; JSON provenance is not a signature.

## CPU cross-runtime activation checks

A new test uses the actual Transformers 5.6.2 Qwen3.5 implementation and actual
repository converter. Its tiny two-layer network contains linear attention,
full attention and both FFNs. Linear key/value head counts 2/4 deliberately
exercise a nonidentity grouped-to-tiled permutation. Seven tokens are compared
at 15 matmul input sites covering all 12 projection roles.

A test-only native runner uses the existing scheduler evaluation callback to
read each actual matmul input with physical tensor strides. All-token logits
prevent last-layer row cropping. Both recurrent and KV state are cleared before
a second native prefill; repeated native captures match exactly. No product
capture facility, inference operator or kernel was added.

Comparisons use CPU F32 execution and F32 KV. Two additional cases round model
parameters to F16/BF16 and then execute in F32. They are weight-rounding smoke
checks, not claims about native half-precision or GPU execution.

| Weight rounding | Maximum absolute input difference | Maximum relative RMS difference |
| --- | ---: | ---: |
| F32 | 2.146e-6 | 8.756e-7 |
| F16 | 2.205e-6 | 8.796e-7 |
| BF16 | 1.907e-6 | 7.810e-7 |

Each case reports all per-site differences and retains `metrics.json` and raw
captures. The predeclared bound remains `max_abs <= 2e-5 + 2e-4 * reference_rms`;
it was not widened after observing results. Initial storage-smoke failures
exposed a fixture mistake: casting the whole HF module also rounded its
nonpersistent RoPE frequency buffers, while GGUF reconstructed those from the
original configuration. The fixture now rounds parameters only. This was not
an exporter permutation defect or a change to either runtime.

These fixtures establish representative numerical correspondence, not equality
for every model size, sequence length, precision policy or backend.

## Native-codec end-to-end coverage

The exporter/importer-to-native-quantizer fixture now covers `Q3_K`, `IQ3_S`,
`IQ4_XS`, `Q5_0` and `Q6_K`, at one and four threads. A native GGUF sums/counts
imatrix fixture exercises the real importance path. Checks assert actual final
qtypes, positive finite F16 companions, version-1 metadata, thread-reproducible
payloads/diagnostics and selected output L1 independently reconstructed with
NumPy dequantization and float64 matrix multiplication. Existing malformed-file
and input-protection cases remain.

## Verification and limits

The CPU-only build explicitly disables CUDA, Vulkan and dynamic backend loading.
Eleven affected registered tests passed, including the locked numeric fixtures,
CPU graph/LoRA checks, complete identity tests, all-role parity and five-codec
integration. Flake8 core checks and whitespace/syntax checks pass on touched
Python files. Local logs are in `artifacts/sinq-audit-followup/`; reproducible
test sources and CMake registrations are committed, not model binaries.

```sh
ctest --test-dir build-sinq-cpu -R '^(test-qk-sinq-(numeric|model|graph|cpu|python-storage)|test-sinq-(activation|activation-parity|calibration|calibration-e2e|exporter|reference-identity))$' --output-on-failure
```

No GPU tests, performance runs, full-model calibration or KLD/PPL experiments
were run for this follow-up. The earlier performance report remains historical.
The existing incompatible Flake8/no-print hook dependency pins were not changed.
