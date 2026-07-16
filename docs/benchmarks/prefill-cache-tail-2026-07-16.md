# 16K KV-cache prefill benchmark — 2026-07-16

This benchmark investigated 16K prompt processing with BF16, Q4_0, and
KVarN4 KV caches, including exact-tail requests. It identified and fixed a
KVarN eager-record store regression without changing persistent cache layout,
decode routing, or quantization.

## Configuration

- GPU: NVIDIA GeForce RTX 3090, driver 596.21, 297.5 W power limit
- CUDA build: CUDA 13.1, `sm_86`, Release, FlashAttention enabled
- Source base: `827bfda66e8a47f36cc9f31c86922d61ab33a9cb`
- Baseline binary identity: `8e9fd591c320bbf137cfc6250a415a3ebc253437ffbf0ac479c281587f20ab17`
- Candidate binary identity: `6cc2ad0512742ae50f8dabab937e67c4a7467597bb8441c010e5b213b9dbdd58`
- Candidate CUDA DLL SHA-256: `467795c1e23f6ac04e65bbbf9bac76a5375d518ca88cd478c2b5187c4a540878`
- Qwen model: `Qwen3.6-27B-Q5_K_S.gguf`, SHA-256
  `a514ac5864d1d35841bd0bed4fdcb8e81360b5224586c3634ecca0e5de886579`
- Gemma model: `gemma-4-31B-it-Q5_K_S.gguf`, SHA-256
  `f04706f56eac91d2f32d20fad79e4799451373b9c9d553131ad846100b39fe28`
- Synthetic `llama-bench` prompt: 16,384 tokens, no generated tokens
- Batch/ubatch: 2,048/512
- Three measured repetitions per row; medians are reported below
- BF16 tail: 0; Q4_0 and KVarN4 tails: 0, 1,024, and 2,048
- Exact-tail types: BF16 for Q4_0 and F16 for KVarN4

The command shape was:

```text
llama-bench -m <model.gguf> -ngl 999 -p 16384 -n 0 -d 0 \
  -b 2048 -ub 512 -r 3 -ctk <cache> -ctv <cache> \
  --kv-tail-tokens <tails> --kv-tail-type <type> \
  -fa on -mmp 0 --no-host 1 -o jsonl
```

## Final results

| Model | Cache | Tail | Median tok/s | SD | vs BF16 | vs pre-fix |
|---|---|---:|---:|---:|---:|---:|
| Qwen | BF16 | 0 | 1,178.02 | 2.22 | — | -0.21% |
| Qwen | Q4_0 | 0 | 1,166.30 | 1.93 | -0.99% | -0.15% |
| Qwen | Q4_0 | 1,024 | 1,138.22 | 1.00 | -3.38% | -0.34% |
| Qwen | Q4_0 | 2,048 | 1,124.40 | 37.88 | -4.55% | -0.08% |
| Qwen | KVarN4 | 0 | 1,119.94 | 0.74 | -4.93% | **+13.84%** |
| Qwen | KVarN4 | 1,024 | 1,109.75 | 0.41 | -5.80% | **+13.83%** |
| Qwen | KVarN4 | 2,048 | 1,097.44 | 0.12 | -6.84% | **+13.69%** |
| Gemma | BF16 | 0 | 992.47 | 3.31 | — | -0.22% |
| Gemma | Q4_0 | 0 | 977.22 | 1.64 | -1.54% | -0.49% |
| Gemma | Q4_0 | 1,024 | 945.07 | 0.62 | -4.78% | +0.01% |
| Gemma | Q4_0 | 2,048 | 928.87 | 0.47 | -6.41% | +0.06% |
| Gemma | KVarN4 | 0 | 822.74 | 0.70 | -17.10% | **+43.42%** |
| Gemma | KVarN4 | 1,024 | 941.47 | 0.23 | -5.14% | **+18.17%** |
| Gemma | KVarN4 | 2,048 | 925.33 | 0.53 | -6.76% | **+17.82%** |

BF16 and Q4_0 reproduce the pre-fix run within 0.5%, which makes the KVarN
gains attributable to the store change rather than a global clock shift. Q4_0
long tails cost 3–6%, matching the expected extra exact-tail work. KVarN4 is
now close to BF16 on Qwen and on Gemma with a long tail. Gemma's intrinsic
128-token tail remains the outlier because its SWA layers still quantize and
attend the compressed KVarN body; a 1,024-token tail promotes those small
windows to native exact storage.

## Profiler diagnosis

Nsight Systems 2025.6.3 profiles used the same 16K/2,048/512 shape with one
measured repetition and no warmup.

Before the fix, Gemma KVarN4 tail 0 spent 12.298 s of 28.21 s total GPU time
(43.6%) in 3,840 `kvarn_store_kernel_headwide` launches. The average launch
was 3.203 ms. Attention dequantization and tail merge kernels were individually
near 1% or below, ruling them out as the primary regression.

The exact-tail integration requires every completed non-sink record to be
committed eagerly. The CUDA dispatch consequently excluded all eager stores
from the pre-existing staged workspace route. That restored the old serialized
per-token/head-wide behavior, most visibly across Gemma's many SWA layers.

The corrected staged route transforms the ubatch once, quantizes every completed
group from that workspace immediately, then commits the newest stage slots. It
retains delayed flushing for non-eager callers and preserves the SWA ring's
last-writer rule when more than one absolute group aliases a record.

After the fix, the matched Gemma workspace flush launches take 2.300 s total,
0.599 ms on average: a 5.35x reduction in the isolated store kernel and about
10.0 s less GPU store time. A controlled Qwen profile likewise reduced store
time from 3.412 s for 1,024 head-wide launches to 0.277 s for the staged flush.

The remaining Gemma tail-0 work is genuine KVarN quantization and SWA body
attention. Removing it would require a different precision/storage policy;
that was rejected because the task forbids trading precision or VRAM for
prefill speed.

## Non-regression evidence

### Cache precision

`test-kvarn` compares the eager 512-token workspace route with the 256-token
fallback byte-for-byte after 4,096 tokens. It covers non-SWA and SWA ring
storage, K and V orientations, and two-slice heads. The full KVarN test passed.

A matched real-model `b=ub=512`, context-512, four-chunk spot against fresh
BF16 bases reported:

| Model | Mean KLD | Same-top probability |
|---|---:|---:|
| Qwen KVarN4 | 0.004282 ± 0.000907 | 98.137% ± 0.424% |
| Gemma KVarN4 | 0.309197 ± 0.028164 | 82.059% ± 1.202% |

These are consistent with the established bounded KVarN quality ranges. The
byte-equivalence oracle is the decisive check that the new scheduling does not
change cache contents.

### Decode

The canonical decode iteration covered both models, 1K/16K contexts, all three
KVarN tails, 128 timed decode tokens, and three repetitions. All 12 median
deltas versus the existing baseline were between -0.20% and -0.85%, with
identical attention-route counts. The changed workspace path requires at least
384 tokens, so single-token decode cannot enter it.

### VRAM

The canonical 18-row VRAM run covered Qwen at 16K/64K and Gemma at 16K.
For all nine KVarN rows, the following deterministic fields were byte-identical
to baseline:

- K/V and exact-tail resident bytes;
- staging, metadata, and allocator-padding bytes;
- transient high-water and KV-related peak bytes;
- CUDA compute-buffer bytes.

Raw driver peak measurements varied with CUDA VMM reservation order and teardown
state, while the component-owned totals did not change. The new workspace is a
short-lived allocation from the existing CUDA pool and does not change the
persistent KV-cache layout.

### Tests

- Prescribed CUDA 13.1 `sm_86` Release build: passed.
- `test-kvarn` and `test-kvarn-eager-workspace-static`: passed.
- Full CTest: 64/65 passed. The exhaustive `test-backend-ops` failed twice in
  two different unrelated numerical cases: first BF16 in-place RoPE at
  `1.25e-7` versus `1.00e-7`, then standard Q4 tail FlashAttention at
  `6.74e-4` versus `5.00e-4`. No KVarN case failed, and no tolerance was changed.
