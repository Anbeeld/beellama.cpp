# External physical two-GPU acceptance

- Build the tested commit with CUDA FlashAttention and tests, recording whether NCCL was found.
- Record both physical GPU IDs, names, memory, driver, CUDA runtime, executable/DLL composite hash, model hash, and config hash.
- Run the 16K control, q4_0/BF16-tail-1024, and kvarn4/F16-tail-1024 coordinates under layer and tensor placement.
- Run tensor splits `1,1` and `3,1`; repeat with NCCL enabled and disabled where supported.
- Run two server slots with unified and non-unified KV, including same-prefix divergence, distinct prompts, cancellation, reset, checkpoint save/restore, and teardown.
- Inject one initialization failure after each persistent allocation stage and confirm owner-correct rollback with no cumulative device-memory growth.
- Run the two tail-1024 64K KLD coordinates only when the verified BF16 baseline is available.
- Capture one bounded Nsight Systems trace for each cache family and confirm there is no host roundtrip, duplicated full cache, duplicate reduction, or per-token device-wide synchronization.
- Label the result physical multi-GPU only when two distinct GPU IDs executed the shards; same-GPU logical shards do not satisfy this checklist.
