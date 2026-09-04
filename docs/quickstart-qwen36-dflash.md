# Qwen3.6 DFlash quickstart

Use an upstream-format DFlash GGUF trained for the exact target model. Keep the
target and draft model paths paired as published by the model provider.

```bash
llama-server -m target.gguf \
  --spec-type draft-dflash \
  --spec-draft-model dflash.gguf \
  --spec-draft-n-max 16 \
  --spec-draft-type-k kvarn4 \
  --spec-draft-type-v kvarn2 \
  --flash-attn on --port 8080
```

`--spec-draft-type-k/v` select only the owned DFlash draft cache. Target cache
settings remain independent. All six KVarN widths (`kvarn2`, `kvarn3`,
`kvarn4`, `kvarn5`, `kvarn6`, and `kvarn8`) are accepted, including asymmetric
pairs. One-sided selection promotes the other side to the same width.

There is no draft precision-tail argument. The explicit draft tail request stays
zero and KVarN keeps its intrinsic exact suffix of up to 128 tokens. The same
K/V pair applies to full-attention and SWA draft layers. Non-causal DFlash block
attention uses materialized attention over compressed persistent KVarN storage.
DSpark models can use the same owned draft KVarN configuration; their
non-causal attention also uses the materialized correctness route.

CUDA is the runtime-qualified DFlash draft-KVarN backend for this release.
HIP/ROCm, Vulkan, real multi-GPU, and multimodal DFlash routes remain
unqualified.
