# KVarN AMD decode remediation completion record

- Date: 2026-08-09
- Issue: `Anbeeld/beellama.cpp#108`
- Source baseline: `0c5034c1b`
- Upstream reference: `tmp/upstream-llama.cpp`
- Result: implementation and available-hardware validation complete; physical AMD runtime remains an external gate

## Latest report and root cause

The latest report used an RX 7900 XTX on Windows with build `83d4173d8`, a
Qwen3.6-27B target, FlashAttention, and KVarN6 for both K and V. Prompt
processing and the first generated token succeeded. Subsequent single-token
decode repeatedly trapped at `mma.cuh:796` because `load_ldmatrix` reached its
`NO_DEVICE_CODE` branch for AMD architecture 1300.

The reported model geometry is head dimension 256, 24 query heads, and four KV
heads (GQA 6). The HIP route policy advertised both split and vector decode.
For `nq=1`, the dispatcher selected split decode, whose implementation uses
the NVIDIA `ldmatrix` and m16n8 MMA contract. Those primitives have only the
Turing device implementation in `mma.cuh`; compiling their host-visible
fallback under HIP does not make them valid AMD device code.

The generic HIP MMA route was already shape-gated correctly: RDNA accepts head
dimensions through 128, while CDNA accepts through 256. The portable native
route directly consumes KVarN records and supports the reported RDNA D=256
shape. The failure was therefore an over-advertised specialized capability,
not a KVarN record-format or model-graph defect.

## Remediation

- HIP no longer advertises split or vector KVarN decode. It retains the
  shape-gated generic MMA route and portable native fallback.
- MUSA also fails the split-support predicate closed because the same
  implementation depends on the NVIDIA-only primitive contract.
- The split-support function has a backend-local HIP/MUSA guard in addition to
  the host route policy, preventing a future policy regression from entering
  the unsafe kernel.
- Route-policy tests cover RDNA and CDNA families and require split/vector
  counters to remain zero on every AMD route-boundary case.
- A structural regression ties the fail-closed guard to the split kernel's
  `load_ldmatrix` use so the capability cannot be broadened without an explicit
  implementation change.

## Adjacent defects found during validation

The unfiltered Vulkan backend suite exposed two pre-existing capability
contract defects after the AMD change itself was already passing:

1. A four-byte-offset, logically contiguous RHS view was advertised for
   `MUL_MAT` and `MUL_MAT_ID`, then rejected by the storage-buffer alignment
   assertion. Ordinary and expert matmul now stage such RHS views through the
   existing generic copy shader, bind the aligned base, and pass the element
   offset through its push constants.
2. Vulkan advertised BF16 RoPE even though no BF16 RoPE pipeline exists. The
   support predicate now mirrors the actual F32/F16 pipeline matrix, including
   the vision-mode exception, so unsupported types fall back instead of
   aborting.

These fixes close the Vulkan residual recorded in
`kvarn-tail-fit-remediation-completion-20260809.md` without changing normal
aligned matmul routing.

## Builds and tests

Hardware and software available locally:

- Windows 11 Pro 10.0.26200
- NVIDIA GeForce RTX 3090, 24,576 MiB, driver 596.21
- CUDA 13.1, compute capability 8.6
- Vulkan SDK 1.4.350.0, Vulkan device `Vulkan0`

Validation results:

- The route-policy regression failed before the production change for RDNA,
  CDNA, and the missing fail-closed guard, then passed after the change.
- CUDA default-pair full CTest: 94/94 passed.
- Fresh CUDA default-pair `test-kvarn`: passed in normal mode, forced-portable
  attention mode, and forced-portable capability mode.
- CUDA expanded quant matrix (169 standard pairs and all 36 ordered KVarN
  pairs): built successfully; `test-kvarn` passed in 30.45 seconds.
- Vulkan full CTest: 94/94 passed in 487.48 seconds.
- Vulkan device-specific backend matrix: 19,123 selected cases passed in
  315.8 seconds, including the ordinary and expert offset-view regressions.
- KVarN route, rollback, HIP capability, HIP runtime-gate, eager-workspace,
  and standard-tail static/integration guards passed.

## Reporter-shaped real-model validation

Model and command geometry:

- Model: `D:\models\Qwen3.6-27B-GGUF\Qwen3.6-27B-Q5_K_S.gguf`
- Model size: 17.65 GiB, 26.90B parameters
- Backend: CUDA on RTX 3090, all layers offloaded
- Context/batching: 4,096 / 512 / 256
- Cache: KVarN6 K and KVarN6 V, FlashAttention enabled
- Portable route forced to reproduce the route that RDNA D=256 now selects

The single-turn smoke exited zero. All 64 logged KVarN attention dispatches
used `portable-native`; 32 were `nq=1` decode calls. Logged geometry was D=256,
K=6, V=6, GQA=6, and `nkv=256`. Prompt processing measured 32.2 tokens/s and
generation measured 23.5 tokens/s.

Controlled `llama-bench` generation results used `-p 0 -n 32 -r 3`, batch 512,
ubatch 256, KVarN6/KVarN6, and FlashAttention:

| Route | Throughput |
|---|---:|
| Forced portable native | 28.99 +/- 1.27 tokens/s |
| Native CUDA dispatcher | 28.94 +/- 1.17 tokens/s |

The portable route showed no measurable decode penalty on this geometry within
run-to-run variance.

The earlier long-context memory report is addressed by baseline `0c5034c1b`.
Its separate completion record contains a 52-row 50k/100k allocation matrix
and real-model runtime evidence: Qwen KVarN4 compute memory fell from 6,357 MiB
to 566 MiB at context 50,000 and from 11,281 MiB to 506 MiB at context 100,000,
with no context-sized materialization.

## Remaining hardware gate

No AMD GPU or ROCm/HIP toolchain is installed on this host. The repository's
HIP runtime validator requires a real RDNA or CDNA attestation and therefore
cannot be honestly satisfied here. Static policy coverage proves that RDNA
D=256 can no longer enter split/vector decode, and the reporter-shaped forced
portable runtime proves the selected replacement route for the same tensor
geometry. A physical RX 7900 XTX/RDNA run remains required to claim AMD runtime
attestation.
