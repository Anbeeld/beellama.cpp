# SINQ v2 settings

SINQ-conditioned GGUFs require BeeLlama. Other engines may recognize the native
weight codec but ignore the companion diagonal and produce incorrect output.
Use a `.beellama-qk-sinq.gguf` filename and original F32, F16 or BF16 source
weights, never a conditioned or quantized artifact.

## Direct conditioning

Pass one strict JSON configuration to `llama-quantize`:

```sh
llama-quantize --pure --sinq-config direct.json source-bf16.gguf model.beellama-qk-sinq.gguf Q4_K
llama-quantize --pure source-bf16.gguf stock.gguf Q4_K
```

Save this as `direct.json`:

```json
{
  "schema_version": 2,
  "defaults": {"generator": "sinq", "strip_width": 64, "iterations": 16, "strength": 1.0},
  "targets": {"include": ["blk.0.ffn_up.weight", "blk.0.ffn_down.weight"]},
  "overrides": [
    {"selector": {"tensor": "blk.0.ffn_up.weight"}, "parameters": {"strength": 0.5}}
  ]
}
```

The direct interface supports ordinary `sinq`, not AWQ search. `defaults` and
`targets` are required; omitted normalization values default to width 64,
16 iterations and strength 1. `overrides` defaults to an empty array.
Unknown or duplicate fields and nonfinite numbers fail. Width must be
32/64/128/256/512, iterations 4/8/16 and strength finite in `[0,1]`.
Strength zero is still a conditioned candidate, not a stock alias.

`targets.include` is `"eligible"` or a nonempty exact-name array; optional
`targets.exclude` is an exact-name array. Explicit unknown or ineligible targets
fail, and exclusions must resolve. Each override selector has exactly one of
`tensor` (exact name), `layer` (nonnegative integer) or `role` (supported
projection role), and must match a selected eligible tensor. Override parameters
are partial defaults. Rules must be disjoint even if they assign equal values.
Selected input dimensions must fit both the codec block and requested width;
manual assignments fail rather than silently changing width or codec.

Eligibility covers original floating Qwen3.5 primary-trunk attention, SSM and
FFN projection matrices. Embeddings, output heads, norms and auxiliary layers
are not conditioned. Supported target codecs are `Q3_K`, `Q4_K`, `Q5_0`, `Q5_1`,
`Q6_K`, `Q8_0`, `IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`,
`IQ3_S`, `IQ4_NL` and `IQ4_XS`. Conditioning selection does not alter ordinary
codec allocation. Inspect actual final types when using `--pure`, since shape
fallbacks can affect stock quantization.

The public quantization API uses `sinq_config` and the optional shared
`sinq_diagnostics_json` destination; initialize parameters with
`llama_model_quantize_default_params()`. `--sinq-diagnostics-json FILE` is also
available for stock quantization. Use a separate writable destination; report
write failures fail the quantization call even if the GGUF was already written.

## Runtime format

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

## Offline optimizer

Build the optional helpers with `-DLLAMA_BUILD_SINQ_OPTIMIZER=ON`, then use
`python -m tools.sinq.optimizer.cli` from the repository root. The six subcommands
are `plan`, `generate`, `score`, `select`, `assemble` and `validate`, each with
`--config FILE`. Only `generate` and `select` accept `--resume RUN_DIR`.
There is no JSON `operation` field. See the
[optimizer guide](../tools/sinq/README.md) for strict stage groups, ordinary/AWQ
candidate spaces, locks, score policies and examples, and
[Tier-C execution](sinq-tier-c-execution.md) for advanced scoring controls.

Local, blockwise and Fisher scores are surrogate objectives, not model KL or
untouched-test quality measurements. Full six-arm T340/T425 model and hardware
verification remains incomplete; tiny integration checks do not establish that
gate or a quality improvement.
