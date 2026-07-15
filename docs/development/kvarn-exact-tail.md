# KVarN exact-tail architecture and validation record

This record defines how BeeLlama combines KVarN records with the shared exact
KV-tail contract. The user-facing behavior is documented in
`docs/beellama-features.md`; this page records the invariants, backend behavior,
state boundary, and reproducible validation evidence.

## Logical policy

KVarN uses a 128-token group and retains three attention regions: the permanent
non-SWA 128-token sink, compressed body records, and an exact recent suffix.
For each canonical cache group:

```text
intrinsic = min(128, effective_window)
explicit  = min(align_up(raw_request, 128), effective_window)
effective = max(intrinsic, explicit)
```

An omitted request and numeric zero therefore preserve the paper-faithful
128-token suffix. Positive numeric, positional, named, structural-ID, and
`auto` requests enlarge it. The physical ubatch does not participate in this
formula. When `effective == effective_window`, the component becomes one native
F16/BF16 cache and allocates neither KVarN records nor an exact overlay.

## Storage and attention

KVarN staging is incomplete-group workspace, not precision policy. A completed
128-token K/V group is quantized eagerly on CPU, CUDA, and the Vulkan store
implementation, including when its rows remain in the exact suffix. This makes
record validity independent of future ubatch segmentation or exact-row
eviction.

The overlay stores canonical post-RoPE K and canonical V in the selected F16 or
BF16 type. KVarN records retain their rotated K domain and original V domain.
KVarN defaults to the paper-faithful F16 representation; an explicit BF16
request remains supported and takes precedence over that default.
Each query receives exact source indices for its own logical suffix. The body
route excludes sink/tail overlap and exports its final FP32 row maximum and
softmax denominator. CUDA merges the exact suffix against that metadata, so
sink, body, and suffix form one softmax and each physical key contributes once.
The same generic tail descriptor and lifecycle logic remains KVarN-neutral.

Head dimensions 128, 256, and 512 are handled as one, two, or four KVarN slices.
The WHT/store paths accept F16 and BF16 canonical inputs. CUDA supports native
KVarN attention plus the exact merge. CPU supplies storage/reference oracles.
Vulkan supports eager storage but fails KVarN context placement closed because
it does not yet consume native KVarN views in FlashAttention.

## Lifecycle and state

Exact rows follow the shared `(stream, cell, generation)` identity model.
Sequence copy, remove, keep, add/divide, position shift, cache clear, cell reuse,
SWA wrap, and graph reuse update compressed and exact representations together.
Coverage reports the intrinsic/explicit request and any body-only degradation.

KVarN state version 12 serializes logical records, exact payloads, membership,
and structural/type/preset identity. It does not serialize transient workspace
as precision state, so `ub=128 -> ub=512` and the reverse restore without
changing logits. Tail length, exact type, KVarN preset, and component mismatch
fail closed. Version 11 is rejected because its physical stage/tail layout
cannot be reinterpreted safely under the logical-overlay contract.

## Validation manifest

- Validation source parent: `2226885615fdd336c85d71822465d6704e51958b`;
  the implementation is the commit containing this record.
- Hardware: NVIDIA RTX 3090, CUDA 13.1, architecture 86.
- Build: `powershell -File tmp/build-local-3090-cuda13.1.ps1 -Parallel 16`.
- Qwen target: `<QWEN_TARGET_GGUF>`.
- Gemma target: `<GEMMA_TARGET_GGUF>`.
- Corpus: `<KLD_CORPUS>`.
- Qwen base: `<QWEN_BF16_BASE>`.
- Gemma base: `<GEMMA_BF16_BASE>`.

The long Qwen matrix used context 65,536, `-b 2048`, `-ub 512`, seed 1,
BF16 reference logits, native CUDA FlashAttention, and symmetric `kvarn4`:

| Exact suffix | Mean KLD | Median KLD | 99.9% KLD | Same top |
|---:|---:|---:|---:|---:|
| 128 (request 0) | 0.003249 | 0.001119 | 0.108416 | 97.747% |
| 128 (explicit) | 0.003155 | 0.001107 | 0.107172 | 97.724% |
| 512 | 0.002973 | 0.001029 | 0.100770 | 97.823% |
| 1024 | 0.002845 | 0.000988 | 0.092756 | 97.866% |

The long Gemma matrix used context 16,384, `-b 2048`, `-ub 256`, seed 1,
BF16 exact storage/reference logits, native CUDA FlashAttention, and symmetric
`kvarn4`:

| Exact suffix | Mean KLD | Median KLD | Same top |
|---:|---:|---:|---:|
| 128 (request 0) | 0.593966 +/- 0.004359 | 0.059228 | 77.292% |
| 128 (explicit) | 0.591692 +/- 0.004344 | 0.058393 | 77.298% |
| 512 | 0.537648 +/- 0.004168 | 0.049017 | 78.478% |
| 1024 | 0.466760 +/- 0.003863 | 0.037311 | 80.182% |

Matched bounded Qwen runs at `ub=128/256/512` remained near Mean KLD
0.0019-0.0029 for requests 0, 128, and 512 after the body-metadata fix. Matched
Gemma runs at the same ubatches showed no segmentation discontinuity within
reported uncertainty. Full-window BF16 Gemma promotion was exact at every
ubatch (`abs(Mean KLD) <= 0.000001`, 100% same-top), and the model-backed state
test matched a direct BF16 cache with logits NMSE below `1e-10`.

Representative long-run command (change only the tail value for the matrix):

```powershell
$GemmaModel = '<GEMMA_TARGET_GGUF>'
$Corpus = '<KLD_CORPUS>'
$GemmaBase = '<GEMMA_BF16_BASE>'

build-local-rtx3090-cuda-13.1/bin/llama-perplexity.exe `
  -m $GemmaModel `
  -ngl all -c 16384 -b 2048 -ub 256 `
  --cache-type-k kvarn4 --cache-type-v kvarn4 `
  --kv-tail-type bf16 --kv-tail-tokens 512 --flash-attn on `
  --seed 1 --no-mmap --mlock --no-host --kv-unified `
  -f $Corpus `
  --kl-divergence-base $GemmaBase `
  --kl-divergence
```

Performance results are valid only from an otherwise idle GPU. Record median
prefill, decode, and peak VRAM for tails 128/512/1024 and F16/BF16, with exact
model, batch, ubatch, repetition count, and commit alongside the result.
