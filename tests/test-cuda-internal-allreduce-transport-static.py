#!/usr/bin/env python3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEADER = (ROOT / "ggml/include/ggml-backend.h").read_text(encoding="utf-8")
META = (ROOT / "ggml/src/ggml-backend-meta.cpp").read_text(encoding="utf-8")
CUDA = (ROOT / "ggml/src/ggml-cuda/ggml-cuda.cu").read_text(encoding="utf-8")
NATIVE = (ROOT / "ggml/src/ggml-cuda/allreduce.cu").read_text(encoding="utf-8")
WDDM = (ROOT / "ggml/src/ggml-cuda/allreduce-wddm.cu").read_text(encoding="utf-8")


def require(text: str, needle: str, reason: str) -> None:
    if needle not in text:
        raise AssertionError(f"{reason}: missing {needle!r}")


require(HEADER, "int64_t graph_batch_size", "communication must retain whole-graph batch metadata")
require(META, "comm_graph_batch_size", "the meta scheduler must classify the complete projected graph")
require(CUDA, "ggml_cuda_ar_requires_wddm_host_transport", "the route needs the exact topology probe")
for needle, reason in (
    ("cudaDevAttrTccDriver", "the topology probe must distinguish WDDM from TCC"),
    ("cudaDeviceCanAccessPeer", "peer-capable devices must retain native AllReduce"),
    ("first == second", "virtual ranks on one physical GPU must remain unaffected"),
    ("ggml_cuda_ar_wddm_pipeline_init", "the topology route must own a separate pipeline"),
):
    require(CUDA, needle, reason)

dispatch = CUDA[
    CUDA.index("static bool ggml_backend_cuda_comm_allreduce_internal("):
    CUDA.index("static bool ggml_backend_cuda_comm_try_allreduce_internal(")
]
require(
    dispatch,
    "comm_ctx->ar_pipeline != nullptr || comm_ctx->ar_wddm_pipeline != nullptr",
    "the internal dispatcher must accept either mutually exclusive pipeline",
)
require(dispatch, "ggml_cuda_ar_wddm_allreduce(", "affected WDDM calls must use the one-way publisher")
require(dispatch, "ggml_cuda_ar_allreduce(comm_ctx->ar_pipeline", "unaffected topologies must retain native AllReduce")

init = CUDA[
    CUDA.index("static void ggml_backend_cuda_comm_init_internal("):
    CUDA.index("// Clear sticky CUDA error from the failed init.")
]
wddm_probe = init.index("ggml_cuda_ar_requires_wddm_host_transport")
wddm_init = init.index("ggml_cuda_ar_wddm_pipeline_init", wddm_probe)
native_init = init.index("ggml_cuda_ar_pipeline_init")
if not wddm_probe < wddm_init < native_init:
    raise AssertionError("the WDDM route must not allocate the unused native pipeline and its device scratch")

for needle, reason in (
    ("cudaHostAllocPortable", "each rank needs portable pinned publication staging"),
    ("device_local", "wire conversion must remain device-local before publication"),
    ("ggml_cuda_ar_wddm_pack_kernel", "each GPU must pack only its local contribution"),
    ("cudaMemcpyDeviceToHost", "publication must use the CUDA copy engine instead of mapped host stores"),
    ("cudaStreamWaitEvent(p->copy_stream[i], p->packed", "D2H publication must wait for local packing"),
    ("cudaEventSynchronize(p->published", "the host must observe each publisher"),
    ("cudaMemcpyHostToDevice", "the peer payload must move through the copy engine"),
    ("ggml_cuda_ar_wddm_add_kernel", "the final sum must be device-local"),
    ("n_blocks = std::min(n_blocks, 1024)", "the local add must use the recovered bounded parallel grid"),
    ("cudaEventRecord(p->complete", "the result stream must publish slot completion"),
    ("cudaEventSynchronize(p->complete", "slot reuse must retire the prior local add"),
):
    require(WDDM, needle, reason)

if "GGML_CUDA_AR_WDDM_COPY_THRESHOLD" in WDDM:
    raise AssertionError("affected WDDM reductions must not fall back to cross-device event waits by size")
pack = WDDM.index("LAUNCH_WDDM_PACK")
ready = WDDM.index("CUDA_CHECK(cudaEventRecord(p->packed", pack)
d2h_wait = WDDM.index("CUDA_CHECK(cudaStreamWaitEvent(p->copy_stream[i], p->packed", ready)
publish_copy = WDDM.index("cudaMemcpyDeviceToHost", d2h_wait)
publish = WDDM.index("CUDA_CHECK(cudaEventRecord(p->published", publish_copy)
observe = WDDM.index("CUDA_CHECK(cudaEventSynchronize(p->published")
h2d = WDDM.index("CUDA_CHECK(cudaMemcpyAsync(", observe)
h2d_complete = WDDM.index("CUDA_CHECK(cudaEventRecord(p->h2d", h2d)
compute_acquire = WDDM.index("CUDA_CHECK(cudaStreamWaitEvent(compute_streams[i], p->h2d", h2d_complete)
add = WDDM.index("LAUNCH_WDDM_ADD", compute_acquire)
complete = WDDM.index("CUDA_CHECK(cudaEventRecord(p->complete", add)
if not pack < ready < d2h_wait < publish_copy < publish < observe < h2d < h2d_complete < compute_acquire < add < complete:
    raise AssertionError("one-way pack, D2H publication, observation, H2D, add, and completion edges are out of order")

# Copy streams retain D2H/H2D overlap. The H2D event is the explicit ownership
# handoff that prevents the compute-stream add from consuming peer staging
# before the copy completes.
if WDDM.count("wire_bytes,\n                cudaMemcpyDeviceToHost,\n                p->copy_stream[i]") != 1:
    raise AssertionError("D2H publication must remain on the rank copy stream")
if WDDM.count("wire_bytes,\n                cudaMemcpyHostToDevice,\n                p->copy_stream[i]") != 1:
    raise AssertionError("peer H2D must remain on the rank copy stream")

for forbidden in (
    "cudaHostAllocMapped",
    "cudaHostGetDevicePointer",
    "__threadfence_system",
    "std::thread reduce_tail",
    "atomicAdd",
    "__nanosleep",
    "GGML_CUDA_INTERNAL_AR_TRANSPORT",
):
    if forbidden in WDDM:
        raise AssertionError(f"one-way WDDM production path must not contain {forbidden}")

for forbidden in (
    "GGML_CUDA_DIAG_SYNC_",
    "GGML_CUDA_DIAG_TRACE_ALLREDUCE",
):
    if forbidden in WDDM or forbidden in CUDA:
        raise AssertionError(f"production routing must not contain diagnostic switch {forbidden}")

if "wddm" in NATIVE.lower():
    raise AssertionError("the upstream native AllReduce implementation must remain topology-agnostic")

print("CUDA internal AllReduce scoped one-way WDDM routing checks passed")
