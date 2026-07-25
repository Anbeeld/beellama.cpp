# Compact SWA KV-tail completion record

This record describes the correctness and memory completion of compact
standard and KVarN precision tails on top of baseline `cc71513e`. It is an
implementation audit, not a new performance claim.

## Architecture

Each cache group owns a per-layer `llama_kv_tail_layer_route`. The descriptor
is finalized only after ordinary body tensors have real backend buffers. It
contains:

- body and exact K/V types;
- body, bodyless, and graph-local current presence;
- the logical owner device and explicit-bias contract;
- the graph-local padded body execution extent;
- the selected native or bounded generic route.

Allocation, graph identity, attention construction, final backend validation,
diagnostics, and memory telemetry consume that descriptor. The native contract
is checked on the final `FLASH_ATTN_EXT` node with its packed body map,
history/current sources, masks, query plan, and KVarN descriptors attached.
Failure is explicit and occurs before scheduler execution.

Full and SWA cache groups own independent query-order tensors. They may have
the same apparent shape while requiring different physical source ordering;
sharing one mutable graph input is therefore invalid.

## Memory invariants

For resolved exact length `N`, SWA visibility window `W`, rollback horizon `R`,
physical ubatch `U`, and exact stream count `S`:

- partial compact overlay: `(N + R) * S` persistent exact rows;
- full-window bodyless SWA: `(W + R) * S` persistent exact rows;
- current K/V: graph-local only;
- body execution padding: graph-local only;
- `U`: descriptor/current/workspace sizing only.

No persistent tensor is rounded to the FlashAttention tile size. CUDA kernels
receive the logical history boundary separately from any graph-local execution
padding.

At Gemma 4 31B, context 16384, tail 1024:

- 50 SWA layers store `50 * 1024 * 16,384 = 838,860,800` exact bytes;
- 10 global layers store `10 * 1024 * 8,192 = 83,886,080` exact bytes;
- one rollback row per layer stores 901,120 bytes;
- total exact history plus reserve is 923,648,000 bytes.

The q5_0 full-SWA conversion removes 432,537,600 bytes of quantized SWA body,
so its persistent increase over tail zero is exactly
`923,648,000 - 432,537,600 = 491,110,400` bytes (468.36 MiB).

## Correctness defects and fixes

### Bodyless planner-mask dependency

The bodyless SWA graph previously replaced the ordinary body mask with a new
constant tensor. That removed the authoritative mask input from the reachable
graph, so allocation could prune it while the tail planner still attempted to
populate it. The bodyless anchor now depends on a one-row view of the original
mask and optional bias inputs. This preserves planner dataflow without adding
persistent storage.

### Full/SWA query-plan alias

The iSWA graph previously reused one query-order input for both the full and SWA
groups. Equal tensor shapes concealed the alias even when the groups required
different row order. The graph inputs, setters, reuse identity, and attention
consumers now keep independent full and SWA query-order tensors.

### Persistent execution padding

Persistent exact history was temporarily rounded from 1025 to 1280 rows to
satisfy an execution-tile constraint. That added about 220 MiB at Gemma 31B.
Persistent tensors now remain exactly `N + R` or `W + R`; the logical history
boundary is passed to CPU/CUDA attention, and required padding is confined to
graph-local workspaces.

## Backend routing

| Backend | Compact history/current behavior | Verification boundary |
|---|---|---|
| CUDA | Native segmented attention; Q4/Q4 D512 mixed-body and bodyless paths | Runtime operator oracle and Gemma route logs |
| HIP/ROCm | Shared portable segmented implementation and capability path | Source and CUDA compile-path audit; no AMD hardware |
| CPU | Dense one-softmax segmented reference | Runtime backend tests |
| Vulkan | Native 12-source form rejected; planned device concat/gather fallback includes current rows | Source/static audit; no new Vulkan hardware run |
| Other accelerators | Generic operator route only where owner operations are supported; otherwise fail closed | Source/static audit |

The retained 12-source `FLASH_ATTN_EXT` contract is covered by backend graph
copy/debug handling. Native route validation prevents silent CPU scheduling.

## Correctness evidence

The requested Gemma KLD rows used the same 16K corpus, baseline artifact,
`-b 2048`, and `-ub 512` as the historical gain curves. One repetition was
requested. Corrected results:

| Cache / tail | Mean KLD | Median KLD | P99 | P99.9 | Maximum | Same top |
|---|---:|---:|---:|---:|---:|---:|
| q5_0 / 0 | 0.626347 | 0.061747 | 9.084101 | 18.731647 | 36.826859 | 76.802% |
| q5_0 / 1024 | 0.462000 | 0.038596 | 7.410694 | 16.760483 | 40.052086 | 80.185% |
| kvarn5 / 0 | 0.483046 | 0.041221 | 7.608277 | 16.575455 | 33.928799 | 79.592% |
| kvarn5 / 1024 | 0.445017 | 0.035761 | 7.212839 | 16.502676 | 36.113117 | 80.590% |

The historical tail-1024 means were 0.469998 for q5_0 and 0.444201 for
kvarn5. The corrected rows are within normal repeatability and restore the
expected approximately 80% same-top result. Before the two graph-input fixes,
the same configurations produced mean KLD 17.740021 and 18.236149.

## Exact KV telemetry

Fresh-process `llama-bench --kv-memory --no-warmup -d 16384 -n 1` results:

| Cache / tail | Resident | Exact | Staging | Padding | Transient | Peak | Native bodyless/mixed | CPU/fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| q5_0 / 0 | 901,120,000 | 0 | 0 | 0 | 0 | 901,120,000 | 0/0 | 0/0 |
| q5_0 / 1024 | 1,392,230,400 | 923,648,000 | 0 | 0 | 261,882,368 | 1,654,112,768 | 50/10 | 0/0 |
| kvarn5 / 0 | 1,227,571,200 | 116,244,480 | 230,686,720 | 0 | 125,110,528 | 1,352,681,728 | 0/0 | 0/0 |
| kvarn5 / 1024 | 1,402,552,320 | 923,648,000 | 20,971,520 | 0 | 132,188,416 | 1,534,740,736 | 50/10 | 0/0 |

The primary artifact namespace is
`tmp/swa-tail-acceptance/final-corrected/`. It contains immutable KLD logs and
CSV summaries, the failed pre-fix rows, exact telemetry JSONL, and
`vram-final-20260725T002149Z/kv-memory-summary.csv`.

## Optimization ledger

| Hypothesis | Evidence | Decision |
|---|---|---|
| Persistently pad exact history to 256-row tiles | Added about 220 MiB at Gemma tail 1024 | Rejected; keep exact persistent rows and graph-local padding |
| Use a large-prefill direct shortcut | Real-model KLD did not change, so it was not the correctness cause | Reverted |
| Treat both iSWA groups as one query plan when shapes match | KVarN mean KLD rose to 3.68 in the isolated equal-shape case | Rejected; independent inputs |
| Allow scheduler fallback to validate planning | Reproduced 10 CPU attention placements and 22 graph splits | Rejected; validate the final node and fail closed |
| Retain plan-input bytes in both CUDA pool and compute accounting | Double-counted scheduler-owned tensors | Rejected; report separately |
| Use front removal for the ordered victim set | Avoids a linear `min_element` scan without changing order | Retained |

Performance measurements taken while the GPU is occupied are not acceptance
evidence. Earlier isolated prefill/decode artifacts remain preserved, and the
Qwen decode matrix was explicitly skipped after its completed results were
reviewed as normal.
