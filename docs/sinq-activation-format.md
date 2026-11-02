# SINQ activation format version 1

This producer-only format stores actual reference linear inputs for optional
native-codec A-SINQ. It is not an imatrix format or GGUF runtime extension.
The consumer reads JSON and raw little-endian float32, never pickle.

## Files and validation

Pass an explicit JSON file or a directory containing `activations.json` to
`--sinq-calibration`. The root object requires:

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

## Verified Qwen3.5 sites

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
verifies selected checkpoint weights against GGUF weights after these exact
converter transformations. It correctly decodes packed BF16 GGUF storage.

Mapping was traced through `conversion/qwen.py`, `conversion/base.py`, GGUF
tensor mappings, `src/models/qwen35.cpp` and installed Transformers 5.6.2
`modeling_qwen3_5.py`. Structural correspondence is not a claim of bitwise HF
versus native activation equality. The exporter records graph/module identity,
checkpoint file hashes, corpus/token identity, dtype and chunking policy.
Hadamard-transformed, already conditioned/quantized sources, fused/missing
reference modules and other architectures are rejected rather than guessed.
MTP and vision projections are outside the supported primary-trunk set.

## Native adaptation

The mathematical orientation follows upstream
[`awq.py`](https://github.com/huawei-csl/SINQ/blob/42ca83edf8fe1afc48c40fca6acc973c5d2bb59f/sinq/awq.py)
and [`dual_shift.py`](https://github.com/huawei-csl/SINQ/blob/42ca83edf8fe1afc48c40fca6acc973c5d2bb59f/sinq/dual_shift.py):
AWQ multiplies normalized weights and divides column compensation. BeeLlama
uses the original weights and actual activation means to construct factors,
but substitutes the destination native GGML codec for upstream row-scale RTN.
The native codec handles its own internal row/block scaling. See
[SINQ settings](sinq-settings.md) for the fixed grid, F16 rounding, objective,
tie policy and memory bounds. No claim of improved global KLD/PPL follows.
