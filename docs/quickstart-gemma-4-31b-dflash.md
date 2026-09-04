# Gemma 4 31B DFlash quickstart

Use a DFlash GGUF trained for the exact Gemma 4 target. DFlash owns its draft
KV cache; this differs from Gemma 4 MTP, whose assistant shares the target
cache.

```bash
llama-server -m target.gguf \
  --spec-type draft-dflash \
  --spec-draft-model dflash.gguf \
  --spec-draft-n-max 16 \
  --spec-draft-type-k kvarn4 \
  --spec-draft-type-v kvarn2 \
  --flash-attn on --port 8080
```

The target and draft cache selections are independent. DFlash accepts all six
KVarN widths (`kvarn2`, `kvarn3`, `kvarn4`, `kvarn5`, `kvarn6`, and `kvarn8`)
and asymmetric K/V pairs. The same pair applies to full-attention and SWA draft
layers.

There is no draft precision-tail option. KVarN supplies an intrinsic exact
suffix of up to 128 tokens while the explicit draft tail request remains zero.
Non-causal DFlash block attention uses materialized attention over compressed
persistent KVarN storage. DSpark uses the same owned draft-cache route.

CUDA is the runtime-qualified DFlash draft-KVarN backend for this release.
HIP/ROCm, Vulkan, real multi-GPU, and multimodal DFlash routes remain
unqualified.
