# BeeLlama v0.4.0 features

BeeLlama v0.4.0 is rebased on upstream llama.cpp and deliberately keeps the
fork surface narrow. The upstream implementation is the source of truth for
model architectures, normal speculative decoding, server APIs, and backends.

## Maintained Bee extensions

| Area | v0.4.0 behavior |
|---|---|
| KVarN | Target-context KV compression with `kvarn2` through `kvarn8`, native CUDA/Vulkan storage, descriptor-native FlashAttention, and KLD tooling. |
| Low-bit standard KV | Standard q-cache support includes q2/q3/q6 variants retained by the fork. |
| CUDA FA policy | Default 103 standard vector pairs; `GGML_CUDA_FA_ALL_QUANTS=ON` selects all 169. KVarN defaults to 15 fast decode pairs and falls back to native MMA for every other valid pair. |
| DFlash | Upstream `draft-dflash` speculative implementation. Historical `dflash` is a warned command-line alias. |
| Adaptive depth | Profit-only server controller for upstream DFlash draft depth. |
| Reasoning protection | Server loop guard that can force-close or stop repeated hidden reasoning. |
| Realtime reasoning control | An opted-in chat completion can be moved from hidden reasoning to its final answer through `/v1/chat/completions/control`. |
| KLD workflow | `llama-perplexity` base-logit save/load support for reproducible KVarN quality checks. |

## Removed in v0.4.0

| Removed surface | Migration |
|---|---|
| TurboQuant and TCQ cache types | Use standard q cache types or KVarN. The historical cache spellings redirect with a warning; removed TQ GGUF weights must be re-quantized from source. |
| `GGML_CUDA_FA_HALF_QUANTS` | Use the 103-pair default build or `GGML_CUDA_FA_ALL_QUANTS=ON`. |
| Bee GPU ring/capture/tape DFlash paths | Use upstream `draft-dflash`; benchmark it as a separate implementation. |
| DDTree and CopySpec | Use upstream draft or n-gram speculative modes. |
| Fringe adaptive draft-max | Use `--spec-dm-controller profit` or `off`. |
| Fork DFlash arguments and environment variables | Use upstream `--spec-draft-*` options. There is no replacement for fork-only cross-context, tree-budget, or GPU-ring controls. |

## DFlash compatibility

New launches use:

```text
--spec-type draft-dflash --spec-draft-model DRAFT.gguf
```

To migrate an older Bee/buun `dflash-draft` GGUF, run:

```text
python scripts/convert-dflash-draft-to-upstream.py legacy.gguf upstream.gguf --verify
```

The conversion promotes the legacy mask id into the tokenizer metadata expected
by upstream, rewrites metadata/tensor names, and does not requantize. If both the
legacy and canonical mask fields exist, equal values are deduplicated and a
conflict is rejected. Validate a converted file with the target model and
`--spec-type draft-dflash` before a production rollout.

## KVarN support boundaries

KVarN applies to the target context, not a draft context. It requires a supported
attention shape and an offloaded native backend. A matching K/V KVarN type is the
intended configuration; when a requested fast CUDA decode pair is not compiled,
the dispatcher uses descriptor-native MMA instead of silently treating KVarN views
as F16 tensors.

KVarN quality and speed are workload dependent. Report model file, cache types,
context length, ubatch, prompt, sampling settings, hardware, and commit with every
benchmark result.
