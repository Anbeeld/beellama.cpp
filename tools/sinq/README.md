# Local Qwen3.5 activation export

`export_activations.py` captures actual `torch.nn.Linear` pre-inputs from a local,
unquantized Hugging Face Qwen3.5 conditional checkpoint. Install optional
`torch`, `transformers`, `numpy`, and `gguf` dependencies separately. The tool
never downloads checkpoints, tokenizer files, or corpora and disables remote code.
The audited reference implementation is Transformers 5.6.2. Use a separate
optional environment: the repository's older default conversion-script pin
(4.57.6) does not provide this tested Qwen3.5 environment.

Run from the repository root with your own local calibration text:

```powershell
python tools/sinq/export_activations.py --checkpoint C:/models/Qwen3.5-9B-bf16 --gguf C:/models/Qwen3.5-9B-bf16/Qwen3.5-9B-text-BF16.gguf --tensor-list selected-tensors.txt --text calibration.txt --output local-activations --device cpu --dtype bf16
```

`--tensor-list` contains exact GGUF names, one per line. All selected modules
must execute. Instead of `--text`, `--tokens` accepts a local JSON array of integer
token IDs. Padding IDs, invalid IDs, empty input, and implicit truncation are
rejected. Default input limit is 4,096 tokens, with fresh, cache-free 512-token
prefill chunks. `--context` supports 1..4,096 and `--max-tokens` supports
1..1,048,576. No built-in text or corpus selection is provided.

Output is a new directory containing `activations.json` (activation format v1)
and headerless little-endian float32 files. Existing directories or files are
never overwritten. Failed exports can leave partial files, but no final manifest;
use a new output directory after diagnosing the failure.

## Capture and validation

Supported sites are FFN gate/up/down, full-attention Q/K/V/output, and linear
attention QKV/Z/a/b/output. Actual module paths start with
`model.language_model.layers`. Fused, missing, or non-Linear sites are rejected.
Before capture, every text-model parameter is mapped and compared against the
original floating GGUF, including embeddings, both normalization stages,
unselected projections, SSM state/conv parameters and the output head. The
check applies converter normalization/exponential/head-layout transforms and
compares at the GGUF storage precision. Missing, extra or mismatched text
tensors fail even when the selected projection matches. Tied output aliases
are checked explicitly. Vision parameters are outside this text-only forward.
Graph-defining configuration is checked as well; unsupported activation/RoPE
variants fail rather than being guessed. Verification runs on CPU in bounded
row chunks before moving the model to the requested device. Linear-attention V rows,
Z rows, a/b rows, and output columns are permuted as appropriate. Only
linear-attention output activations need the grouped-to-tiled V-head permutation.
BF16 GGUF bytes are decoded as 16-bit words, not numerical byte values.
Transformed or prequantized GGUF/checkpoint inputs are rejected.

Each tensor uses a separate disk-backed Algorithm R reservoir, default 128 rows,
maximum 256. PCG64 is seeded by the first 128 bits of
`SHA256(uint32-LE seed || UTF8 exact tensor name)`, interpreted little-endian.
Sampling does not depend on hook traversal order. A binary file cannot exceed
256 MiB. Hooks process one selected row at a time on the host; they do not retain
all forward activations. Model memory and the current prefill's temporary tensors
still depend on the checkpoint, context, dtype, and device. Complete identity verification uses at most 64 MiB of conservative row-buffer
workspace, excluding the loaded reference model, source mappings and small
permutation indices; it does not expand a complete embedding matrix to F32.

The manifest records corpus and little-endian uint32 token-stream SHA256 hashes,
checkpoint config/safetensors hashes, actual module names/classes, dtype, device,
context, token count, and sampling details. Raw text is not printed or included in
the manifest. The provenance also records `reference_identity`: its
`qwen35-full-text-storage-v1` contract, verified tensor count/digest, checked
configuration fields, alias handling and storage-precision policy. Source,
corpus and checkpoint hashes are rechecked before the manifest is published.
Older captures do not gain this verification retroactively; re-export them
before relying on the stronger identity contract. Calibration activation files can themselves contain sensitive data;
keep them local and exclude them from version control.

## Small CPU test

```powershell
python tests/test-sinq-exporter.py
```

These tests use real tiny `torch.nn.Linear` forwards, not a full checkpoint. They
check capture counts/bytes, deterministic reservoir selection, matrix-product
parity, all audited permutation families, packed BF16 decoding, malformed bounds,
nonfinite activations, missing supported names, and overwrite rejection. Missing
optional NumPy or Torch produces an explicit unittest skip, not runtime coverage.
No full-model export or GPU validation is implied by these tests.
