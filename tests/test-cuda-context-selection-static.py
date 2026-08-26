#!/usr/bin/env python3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CUDA = (ROOT / "ggml/src/ggml-cuda/ggml-cuda.cu").read_text(encoding="utf-8")


def function_body(signature: str, next_signature: str) -> str:
    start = CUDA.index(signature)
    return CUDA[start:CUDA.index(next_signature, start + len(signature))]


def require_before(text: str, first: str, second: str, reason: str) -> None:
    if first not in text or second not in text or text.index(first) > text.index(second):
        raise AssertionError(f"{reason}: expected {first!r} before {second!r}")


copy_async = function_body(
    "static bool ggml_backend_cuda_cpy_tensor_async",
    "static void ggml_backend_cuda_synchronize",
)
require_before(
    copy_async,
    "ggml_cuda_set_device(cuda_ctx_src->device);",
    "cudaMemcpyAsync",
    "CUDA tensor copies must select the source stream's device before submission",
)

buffer_context_destructor = function_body(
    "~ggml_backend_cuda_buffer_context()",
    "};",
)
require_before(
    buffer_context_destructor,
    "ggml_cuda_set_device(device);",
    "cudaFree(dev_ptr)",
    "CUDA buffers must select their owning device before release",
)

buffer_copy = function_body(
    "static bool ggml_backend_cuda_buffer_cpy_tensor",
    "static void ggml_backend_cuda_buffer_clear",
)
require_before(
    buffer_copy,
    "ggml_cuda_set_device(src_ctx->device);",
    "cudaMemcpyAsync",
    "synchronous CUDA buffer copies must select the source per-thread stream's device",
)

vmm_destructor = function_body(
    "~ggml_cuda_pool_vmm()",
    "void * alloc",
)
require_before(
    vmm_destructor,
    "ggml_cuda_set_device(device);",
    "cuMemUnmap",
    "VMM pool teardown must select the pool's CUDA context",
)

for signature, next_signature, operation in (
    (
        "static void ggml_backend_cuda_device_event_free",
        "static void ggml_backend_cuda_device_event_synchronize",
        "cudaEventDestroy",
    ),
    (
        "static void ggml_backend_cuda_device_event_synchronize",
        "static const ggml_backend_device_i",
        "cudaEventSynchronize",
    ),
):
    event_operation = function_body(signature, next_signature)
    require_before(
        event_operation,
        "ggml_cuda_set_device(dev_ctx->device);",
        operation,
        f"{signature} must select the event's owning CUDA device",
    )
require_before(
    copy_async,
    "ggml_cuda_set_device(cuda_ctx_dst->device);",
    "cudaStreamWaitEvent(cuda_ctx_dst->stream()",
    "cross-backend copies must select the destination device before queuing its wait",
)

synchronize = function_body(
    "static void ggml_backend_cuda_synchronize",
    "#ifndef NDEBUG",
)
require_before(
    synchronize,
    "ggml_cuda_set_device(cuda_ctx->device);",
    "cudaStreamSynchronize(cuda_ctx->stream())",
    "backend synchronization must select the stream's CUDA device",
)

event_record = function_body(
    "static void ggml_backend_cuda_event_record",
    "static void ggml_backend_cuda_event_wait",
)
require_before(
    event_record,
    "ggml_cuda_set_device(cuda_ctx->device);",
    "cudaEventRecord",
    "backend event recording must select the stream's CUDA device",
)

event_wait = function_body(
    "static void ggml_backend_cuda_event_wait",
    "static void ggml_backend_cuda_graph_optimize",
)
require_before(
    event_wait,
    "ggml_cuda_set_device(cuda_ctx->device);",
    "cudaStreamWaitEvent",
    "backend event waits must select the stream's CUDA device",
)

for signature, next_signature, operation in (
    (
        "static void ggml_backend_cuda_set_tensor_async",
        "static void ggml_backend_cuda_get_tensor_async",
        "cudaMemcpyAsync",
    ),
    (
        "static void ggml_backend_cuda_get_tensor_async",
        "static void ggml_backend_cuda_set_tensor_2d_async",
        "cudaMemcpyAsync",
    ),
    (
        "static void ggml_backend_cuda_set_tensor_2d_async",
        "static void ggml_backend_cuda_get_tensor_2d_async",
        "cudaMemcpy2DAsync",
    ),
    (
        "static void ggml_backend_cuda_get_tensor_2d_async",
        "static bool ggml_backend_cuda_cpy_tensor_async",
        "cudaMemcpy2DAsync",
    ),
):
    tensor_io = function_body(signature, next_signature)
    require_before(
        tensor_io,
        "ggml_cuda_set_device(cuda_ctx->device);",
        operation,
        f"{signature} must select the CUDA context that owns its stream",
    )

nccl_allreduce = function_body(
    "static bool ggml_backend_cuda_comm_allreduce_nccl",
    "static bool ggml_backend_cuda_comm_try_allreduce_nccl",
)
require_before(
    nccl_allreduce,
    "ggml_cuda_set_device(cuda_ctx->device);",
    "tmp[i].alloc(ne);",
    "communication temporaries must be allocated in the owning CUDA context",
)

print("CUDA context-selection contract checks passed")
