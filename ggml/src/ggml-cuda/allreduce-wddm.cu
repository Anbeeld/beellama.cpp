#include "allreduce.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#include "convert.cuh"
#include "ggml-impl.h"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <mutex>

// WDDM does not guarantee forward progress for a kernel that waits for work
// on another device. This topology-scoped transport is intentionally one-way:
// each GPU packs only its own contribution into device-local memory, ordinary
// D2H copies publish it to pinned host memory, the host observes both local
// completion events, and ordinary H2D copies move the peer payload into
// device-local scratch before a fixed-grid local add. No GPU spins on, polls,
// or dereferences host or peer-owned mapped state.

static constexpr int    GGML_CUDA_AR_WDDM_POOL_SIZE       = 2;
static constexpr int    GGML_CUDA_AR_WDDM_KERNEL_BLOCKS   = 8;
static constexpr size_t GGML_CUDA_AR_WDDM_STAGING_BYTES   = 32 * 1024 * 1024;

template <typename T_dst, typename T_wire>
static __global__ void ggml_cuda_ar_wddm_pack_kernel(
        const T_dst *              src,
        T_wire       * __restrict__ device_dst,
        int                         count) {
    constexpr int elems_per_vec = ggml_cuda_get_max_cpy_bytes() / sizeof(T_wire);
    const int tid = threadIdx.x;
    const int gtid = blockIdx.x * blockDim.x + tid;
    const int gnt = gridDim.x * blockDim.x;
    const int count_vec = count / elems_per_vec;
    const int tail = count_vec * elems_per_vec;

    for (int i = gtid; i < count_vec; i += gnt) {
        const int offset = i * elems_per_vec;
        T_wire wire[elems_per_vec];
#pragma unroll
        for (int k = 0; k < elems_per_vec; ++k) {
            wire[k] = ggml_cuda_cast<T_wire>(src[offset + k]);
        }
        ggml_cuda_memcpy_1<sizeof(wire)>(&device_dst[offset], wire);
    }
    if (blockIdx.x == 0 && tid < count - tail) {
        device_dst[tail + tid] = ggml_cuda_cast<T_wire>(src[tail + tid]);
    }
}

template <typename T_dst, typename T_wire>
static __global__ void ggml_cuda_ar_wddm_add_kernel(
        T_dst             * __restrict__ dst,
        const T_wire      * __restrict__ peer,
        int                              count) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int nt = gridDim.x * blockDim.x;
    for (int i = tid; i < count; i += nt) {
        const T_wire local = ggml_cuda_cast<T_wire>(dst[i]);
        dst[i] = ggml_cuda_cast<T_dst>(
            ggml_cuda_cast<float>(local) + ggml_cuda_cast<float>(peer[i]));
    }
}

struct ggml_cuda_ar_wddm_host_mapping {
    uint8_t * host = nullptr;

    cudaError_t alloc(size_t bytes, int device) {
        ggml_cuda_set_device(device);
        cudaError_t rc = cudaHostAlloc(
            reinterpret_cast<void **>(&host), bytes,
            cudaHostAllocPortable);
        if (rc != cudaSuccess) {
            host = nullptr;
        }
        return rc;
    }

    void free(int device) {
        if (host != nullptr) {
            ggml_cuda_set_device(device);
            (void) cudaFreeHost(host);
            host = nullptr;
        }
    }
};

struct ggml_cuda_ar_wddm_pipeline {
    int      devices[2] = {};
    size_t   staging_bytes = GGML_CUDA_AR_WDDM_STAGING_BYTES;
    size_t   bf16_threshold = 1;
    uint64_t call_count = 0;

    ggml_cuda_ar_wddm_host_mapping input[2];
    uint8_t * device_local[2] = {};
    uint8_t * device_peer[2] = {};
    cudaStream_t copy_stream[2] = {};
    cudaEvent_t packed[2][GGML_CUDA_AR_WDDM_POOL_SIZE] = {};
    cudaEvent_t published[2][GGML_CUDA_AR_WDDM_POOL_SIZE] = {};
    cudaEvent_t h2d[2][GGML_CUDA_AR_WDDM_POOL_SIZE] = {};
    cudaEvent_t complete[2][GGML_CUDA_AR_WDDM_POOL_SIZE] = {};
    bool slot_valid[GGML_CUDA_AR_WDDM_POOL_SIZE] = {};
    std::mutex call_mutex;
};

static uint64_t ggml_cuda_ar_wddm_env_u64(const char * name, uint64_t default_value) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }
    char * end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    return end != value ? static_cast<uint64_t>(parsed) : default_value;
}

ggml_cuda_ar_wddm_pipeline * ggml_cuda_ar_wddm_pipeline_init(
        const int * devices, size_t n_devices) {
    if (n_devices != 2 || devices[0] == devices[1]) {
        return nullptr;
    }

    auto * p = new ggml_cuda_ar_wddm_pipeline{};
    p->devices[0] = devices[0];
    p->devices[1] = devices[1];
    p->bf16_threshold = ggml_cuda_ar_wddm_env_u64("GGML_CUDA_AR_BF16_THRESHOLD", 1);
    const size_t slot_bytes = GGML_CUDA_AR_WDDM_POOL_SIZE * p->staging_bytes;

    for (int i = 0; i < 2; ++i) {
        if (p->input[i].alloc(slot_bytes, p->devices[i]) != cudaSuccess) {
            GGML_LOG_ERROR("%s: mapped publication allocation failed on device %d\n", __func__, p->devices[i]);
            ggml_cuda_ar_wddm_pipeline_free(p);
            return nullptr;
        }
        ggml_cuda_set_device(p->devices[i]);
        if (cudaMalloc(reinterpret_cast<void **>(&p->device_local[i]), slot_bytes) != cudaSuccess ||
                cudaMalloc(reinterpret_cast<void **>(&p->device_peer[i]), slot_bytes) != cudaSuccess ||
                cudaStreamCreateWithFlags(&p->copy_stream[i], cudaStreamNonBlocking) != cudaSuccess) {
            GGML_LOG_ERROR("%s: local peer staging allocation failed on device %d\n", __func__, p->devices[i]);
            ggml_cuda_ar_wddm_pipeline_free(p);
            return nullptr;
        }
        for (int slot = 0; slot < GGML_CUDA_AR_WDDM_POOL_SIZE; ++slot) {
            if (cudaEventCreateWithFlags(&p->packed[i][slot], cudaEventDisableTiming) != cudaSuccess ||
                    cudaEventCreateWithFlags(&p->published[i][slot], cudaEventDisableTiming) != cudaSuccess ||
                    cudaEventCreateWithFlags(&p->h2d[i][slot], cudaEventDisableTiming) != cudaSuccess ||
                    cudaEventCreateWithFlags(&p->complete[i][slot], cudaEventDisableTiming) != cudaSuccess) {
                GGML_LOG_ERROR("%s: event allocation failed on device %d slot %d\n",
                               __func__, p->devices[i], slot);
                ggml_cuda_ar_wddm_pipeline_free(p);
                return nullptr;
            }
        }
    }

    GGML_LOG_INFO(
        "%s: initialized one-way mapped-publish WDDM transport for devices %d,%d\n",
        __func__, p->devices[0], p->devices[1]);
    return p;
}

void ggml_cuda_ar_wddm_pipeline_free(ggml_cuda_ar_wddm_pipeline * p) {
    if (p == nullptr) {
        return;
    }
    for (int i = 0; i < 2; ++i) {
        ggml_cuda_set_device(p->devices[i]);
        (void) cudaDeviceSynchronize();
        for (int slot = 0; slot < GGML_CUDA_AR_WDDM_POOL_SIZE; ++slot) {
            if (p->packed[i][slot] != nullptr) { (void) cudaEventDestroy(p->packed[i][slot]); }
            if (p->published[i][slot] != nullptr) { (void) cudaEventDestroy(p->published[i][slot]); }
            if (p->h2d[i][slot] != nullptr) { (void) cudaEventDestroy(p->h2d[i][slot]); }
            if (p->complete[i][slot] != nullptr) { (void) cudaEventDestroy(p->complete[i][slot]); }
        }
        if (p->copy_stream[i] != nullptr) { (void) cudaStreamDestroy(p->copy_stream[i]); }
        if (p->device_local[i] != nullptr) { (void) cudaFree(p->device_local[i]); }
        if (p->device_peer[i] != nullptr) { (void) cudaFree(p->device_peer[i]); }
        p->input[i].free(p->devices[i]);
    }
    delete p;
}

bool ggml_cuda_ar_wddm_allreduce(
        ggml_cuda_ar_wddm_pipeline * p,
        ggml_backend_t             * backends,
        ggml_tensor                ** tensors) {
    GGML_ASSERT(p != nullptr);
    GGML_ASSERT(ggml_nelements(tensors[0]) == ggml_nelements(tensors[1]));
    GGML_ASSERT(tensors[0]->type == tensors[1]->type);

    const ggml_type type = tensors[0]->type;
    const int64_t ne = ggml_nelements(tensors[0]);
    const size_t type_size = ggml_type_size(type);
    const bool use_bf16 = type == GGML_TYPE_F32 &&
        p->bf16_threshold > 0 && ggml_nbytes(tensors[0]) >= p->bf16_threshold;
    const size_t wire_size = use_bf16 ? sizeof(nv_bfloat16) : type_size;
    const int64_t chunk_capacity = static_cast<int64_t>(p->staging_bytes / wire_size);
    GGML_ASSERT(chunk_capacity > 0);

    const bool compute[2] = {
        (tensors[0]->flags & GGML_TENSOR_FLAG_COMPUTE) != 0,
        (tensors[1]->flags & GGML_TENSOR_FLAG_COMPUTE) != 0,
    };

    std::lock_guard<std::mutex> lock(p->call_mutex);
    for (int64_t start = 0; start < ne; start += chunk_capacity) {
        const int64_t count64 = std::min(chunk_capacity, ne - start);
        GGML_ASSERT(count64 <= std::numeric_limits<int>::max());
        const int count = static_cast<int>(count64);
        const size_t dst_bytes = static_cast<size_t>(count) * type_size;
        const size_t wire_bytes = static_cast<size_t>(count) * wire_size;
        const uint64_t sequence = p->call_count;
        const int slot = static_cast<int>(sequence % GGML_CUDA_AR_WDDM_POOL_SIZE);
        p->call_count++;

        if (p->slot_valid[slot]) {
            for (int i = 0; i < 2; ++i) {
                ggml_cuda_set_device(p->devices[i]);
                CUDA_CHECK(cudaEventSynchronize(p->complete[i][slot]));
            }
        }

        ggml_backend_cuda_context * cuda_ctx[2] = {};
        cudaStream_t compute_streams[2] = {};
        uint8_t * data[2] = {};
        for (int i = 0; i < 2; ++i) {
            ggml_cuda_set_device(p->devices[i]);
            cuda_ctx[i] = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
            GGML_ASSERT(cuda_ctx[i]->device == p->devices[i]);
            compute_streams[i] = cuda_ctx[i]->stream();
            data[i] = static_cast<uint8_t *>(tensors[i]->data) + start * static_cast<int64_t>(type_size);
            if (!compute[i]) {
                CUDA_CHECK(cudaMemsetAsync(data[i], 0, dst_bytes, compute_streams[i]));
            }

#define LAUNCH_WDDM_PACK(T_dst, T_wire) \
            ggml_cuda_ar_wddm_pack_kernel<T_dst, T_wire><<< \
                dim3(GGML_CUDA_AR_WDDM_KERNEL_BLOCKS), dim3(256), 0, compute_streams[i]>>>( \
                    reinterpret_cast<const T_dst *>(data[i]), \
                    reinterpret_cast<T_wire *>(p->device_local[i] + static_cast<size_t>(slot) * p->staging_bytes), \
                    count)

            if (use_bf16) {
                LAUNCH_WDDM_PACK(float, nv_bfloat16);
            } else {
                switch (type) {
                    case GGML_TYPE_F32:  LAUNCH_WDDM_PACK(float,       float);       break;
                    case GGML_TYPE_F16:  LAUNCH_WDDM_PACK(half,        half);        break;
                    case GGML_TYPE_BF16: LAUNCH_WDDM_PACK(nv_bfloat16, nv_bfloat16); break;
                    default: GGML_ASSERT(false);
                }
            }
#undef LAUNCH_WDDM_PACK
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(p->packed[i][slot], compute_streams[i]));
        }

        // Copy engines publish device-local wire buffers into ordinary pinned
        // host memory. This avoids long-running kernels issuing PCIe stores,
        // whose forward progress is not guaranteed by WDDM under sustained
        // multi-GPU pressure.
        for (int i = 0; i < 2; ++i) {
            ggml_cuda_set_device(p->devices[i]);
            CUDA_CHECK(cudaStreamWaitEvent(p->copy_stream[i], p->packed[i][slot]));
            CUDA_CHECK(cudaMemcpyAsync(
                p->input[i].host + static_cast<size_t>(slot) * p->staging_bytes,
                p->device_local[i] + static_cast<size_t>(slot) * p->staging_bytes,
                wire_bytes,
                cudaMemcpyDeviceToHost,
                p->copy_stream[i]));
            CUDA_CHECK(cudaEventRecord(p->published[i][slot], p->copy_stream[i]));
        }

        // Both local publishers are submitted before the host blocks. D2H
        // completion makes the ordinary host pointers safe H2D sources.
        for (int i = 0; i < 2; ++i) {
            ggml_cuda_set_device(p->devices[i]);
            CUDA_CHECK(cudaEventSynchronize(p->published[i][slot]));
        }

        for (int i = 0; i < 2; ++i) {
            const int peer = 1 - i;
            ggml_cuda_set_device(p->devices[i]);
            uint8_t * const peer_device =
                p->device_peer[i] + static_cast<size_t>(slot) * p->staging_bytes;
            CUDA_CHECK(cudaMemcpyAsync(
                peer_device,
                p->input[peer].host + static_cast<size_t>(slot) * p->staging_bytes,
                wire_bytes,
                cudaMemcpyHostToDevice,
                p->copy_stream[i]));
            CUDA_CHECK(cudaEventRecord(p->h2d[i][slot], p->copy_stream[i]));
        }

        for (int i = 0; i < 2; ++i) {
            ggml_cuda_set_device(p->devices[i]);
            uint8_t * const peer_device =
                p->device_peer[i] + static_cast<size_t>(slot) * p->staging_bytes;
            CUDA_CHECK(cudaStreamWaitEvent(compute_streams[i], p->h2d[i][slot]));

            const int block_size = 256;
            int n_blocks = (count + block_size - 1) / block_size;
            n_blocks = std::min(n_blocks, 1024);

#define LAUNCH_WDDM_ADD(T_dst, T_wire) \
            ggml_cuda_ar_wddm_add_kernel<T_dst, T_wire><<<dim3(n_blocks), dim3(block_size), 0, compute_streams[i]>>>( \
                reinterpret_cast<T_dst *>(data[i]), \
                reinterpret_cast<const T_wire *>(peer_device), \
                count)

            if (use_bf16) {
                LAUNCH_WDDM_ADD(float, nv_bfloat16);
            } else {
                switch (type) {
                    case GGML_TYPE_F32:  LAUNCH_WDDM_ADD(float,       float);       break;
                    case GGML_TYPE_F16:  LAUNCH_WDDM_ADD(half,        half);        break;
                    case GGML_TYPE_BF16: LAUNCH_WDDM_ADD(nv_bfloat16, nv_bfloat16); break;
                    default: GGML_ASSERT(false);
                }
            }
#undef LAUNCH_WDDM_ADD
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(p->complete[i][slot], compute_streams[i]));
        }
        p->slot_valid[slot] = true;
    }

    return true;
}

#else

ggml_cuda_ar_wddm_pipeline * ggml_cuda_ar_wddm_pipeline_init(const int *, size_t) {
    return nullptr;
}
void ggml_cuda_ar_wddm_pipeline_free(ggml_cuda_ar_wddm_pipeline *) {
}
bool ggml_cuda_ar_wddm_allreduce(ggml_cuda_ar_wddm_pipeline *, ggml_backend_t *, ggml_tensor **) {
    return false;
}

#endif
