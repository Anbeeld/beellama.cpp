#if defined(GGML_USE_HIP)
#include "vendors/hip.h"
#else
#include <cuda_runtime.h>
#endif
#include <cstdlib>
#include <cstdio>
#include <cstring>

// GPU cross-attention ring buffer for DFlash speculative decoding.
// Keeps per-layer ring buffers on GPU and interleaves them into the layout
// expected by the drafter's target_hidden tensor, avoiding the CPU round-trip.

struct dflash_cross_ring_gpu {
    int device;               // CUDA device where ring buffers are allocated
    int n_layers;
    int n_embd;
    int ring_size;

    float ** d_layer_rings;   // device: array of n_layers device pointers
    float *  d_staging;       // device: interleaved output [ring_size * n_layers * n_embd]
    float ** h_layer_ptrs;    // host: copy of per-layer device pointers
};

// Interleave kernel: reads per-layer circular ring, writes interleaved output.
// Grid: (cross_len, n_layers), Block: 256
// Each thread block copies one (token, layer) slice of n_embd floats.
__global__ static void k_cross_ring_interleave(
        const float * const * __restrict__ d_rings,
        float * __restrict__ d_out,
        const int ring_size,
        const int read_start,
        const int cross_len,
        const int n_layers,
        const int n_embd) {
    const int t = blockIdx.x; // token index [0, cross_len)
    const int l = blockIdx.y; // layer index [0, n_layers)

    if (t >= cross_len || l >= n_layers) return;

    const int slot = (read_start + t) % ring_size;
    const float * src = d_rings[l] + (size_t)slot * n_embd;
    float * dst = d_out + (size_t)t * n_layers * n_embd + (size_t)l * n_embd;

    for (int i = threadIdx.x; i < n_embd; i += blockDim.x) {
        dst[i] = src[i];
    }
}

extern "C" void * dflash_cross_ring_gpu_alloc(int n_layers, int n_embd, int ring_size) {
    // env var override
    const char * env = getenv("GGML_DFLASH_GPU_RING");
    if (env && atoi(env) == 0) {
        return nullptr;
    }

    auto * ring = new dflash_cross_ring_gpu();
    cudaGetDevice(&ring->device);
    ring->n_layers  = n_layers;
    ring->n_embd    = n_embd;
    ring->ring_size = ring_size;
    // per-layer ring buffers on device
    ring->h_layer_ptrs = new float*[n_layers];
    for (int l = 0; l < n_layers; l++) {
        cudaError_t err = cudaMalloc(&ring->h_layer_ptrs[l], (size_t)ring_size * n_embd * sizeof(float));
        if (err != cudaSuccess) {
            fprintf(stderr, "dflash gpu ring: cudaMalloc failed for layer %d: %s\n", l, cudaGetErrorString(err));
            for (int j = 0; j < l; j++) cudaFree(ring->h_layer_ptrs[j]);
            delete[] ring->h_layer_ptrs;
            delete ring;
            return nullptr;
        }
        cudaMemset(ring->h_layer_ptrs[l], 0, (size_t)ring_size * n_embd * sizeof(float));
    }

    // device array of layer pointers
    cudaError_t err = cudaMalloc(&ring->d_layer_rings, n_layers * sizeof(float *));
    if (err != cudaSuccess) {
        for (int l = 0; l < n_layers; l++) cudaFree(ring->h_layer_ptrs[l]);
        delete[] ring->h_layer_ptrs;
        delete ring;
        return nullptr;
    }
    cudaMemcpy(ring->d_layer_rings, ring->h_layer_ptrs, n_layers * sizeof(float *), cudaMemcpyHostToDevice);

    // staging buffer for interleaved output
    err = cudaMalloc(&ring->d_staging, (size_t)ring_size * n_layers * n_embd * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(ring->d_layer_rings);
        for (int l = 0; l < n_layers; l++) cudaFree(ring->h_layer_ptrs[l]);
        delete[] ring->h_layer_ptrs;
        delete ring;
        return nullptr;
    }

    size_t total_mb = ((size_t)ring_size * n_embd * sizeof(float) * n_layers +
                       (size_t)ring_size * n_layers * n_embd * sizeof(float)) / (1024 * 1024);
    fprintf(stderr, "dflash gpu ring: allocated %d layers x %d slots x %d embd + staging (~%zu MB)\n",
            n_layers, ring_size, n_embd, total_mb);

    return ring;
}

extern "C" void dflash_cross_ring_gpu_free(void * handle) {
    if (!handle) return;
    auto * ring = (dflash_cross_ring_gpu *)handle;

    cudaFree(ring->d_staging);
    cudaFree(ring->d_layer_rings);
    for (int l = 0; l < ring->n_layers; l++) {
        cudaFree(ring->h_layer_ptrs[l]);
    }
    delete[] ring->h_layer_ptrs;
    delete ring;
}

// Upload host data to a specific position in the GPU ring for one layer.
// Handles wrap-around: if ring_pos + n_tokens > ring_size, splits into two copies.
extern "C" void dflash_cross_ring_gpu_write(
        void * handle, int layer, int ring_pos,
        const float * host_data, int n_tokens, int n_embd) {
    if (!handle) return;
    auto * ring = (dflash_cross_ring_gpu *)handle;

    if (layer < 0 || layer >= ring->n_layers) return;
    if (n_tokens <= 0) return;
    if (n_embd != ring->n_embd) return;
    if (ring->ring_size <= 0) return;

    // Ensure cudaStreamPerThread belongs to the ring's device regardless of
    // which GPU the caller (target model decode) last set as current.
    (void)cudaSetDevice(ring->device);

    float * dst = ring->h_layer_ptrs[layer];
    const size_t stride = (size_t)n_embd * sizeof(float);

    int pos = ring_pos % ring->ring_size;
    if (pos < 0) {
        pos += ring->ring_size;
    }
    if (n_tokens > ring->ring_size) {
        const int skip = n_tokens - ring->ring_size;
        host_data += (size_t)skip * n_embd;
        pos = (pos + skip) % ring->ring_size;
        n_tokens = ring->ring_size;
    }

    int first = ring->ring_size - pos;
    if (first >= n_tokens) {
        // no wrap
        cudaMemcpyAsync(dst + (size_t)pos * n_embd, host_data,
                         (size_t)n_tokens * stride, cudaMemcpyHostToDevice, cudaStreamPerThread);
    } else {
        // wrap: two copies
        cudaMemcpyAsync(dst + (size_t)pos * n_embd, host_data,
                         (size_t)first * stride, cudaMemcpyHostToDevice, cudaStreamPerThread);
        cudaMemcpyAsync(dst, host_data + (size_t)first * n_embd,
                         (size_t)(n_tokens - first) * stride, cudaMemcpyHostToDevice, cudaStreamPerThread);
    }
}

extern "C" bool dflash_cross_ring_gpu_write_d2d(
        void * handle, int layer, int ring_pos,
        const void * device_data, int n_tokens, int n_embd) {
    if (!handle || !device_data) return false;
    auto * ring = (dflash_cross_ring_gpu *)handle;

    if (layer < 0 || layer >= ring->n_layers) return false;
    if (n_tokens <= 0) return false;
    if (n_embd != ring->n_embd) return false;
    if (ring->ring_size <= 0) return false;

    (void)cudaSetDevice(ring->device);

    cudaPointerAttributes attr;
    cudaError_t attr_err = cudaPointerGetAttributes(&attr, device_data);
    if (attr_err != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
#if CUDART_VERSION >= 10000
    if (attr.type != cudaMemoryTypeDevice || attr.device != ring->device) {
        return false;
    }
#else
    if (attr.memoryType != cudaMemoryTypeDevice || attr.device != ring->device) {
        return false;
    }
#endif

    float * dst = ring->h_layer_ptrs[layer];
    const char * src = (const char *)device_data;
    const size_t stride = (size_t)n_embd * sizeof(float);

    int pos = ring_pos % ring->ring_size;
    if (pos < 0) {
        pos += ring->ring_size;
    }
    if (n_tokens > ring->ring_size) {
        const int skip = n_tokens - ring->ring_size;
        src += (size_t)skip * stride;
        pos = (pos + skip) % ring->ring_size;
        n_tokens = ring->ring_size;
    }

    int first = ring->ring_size - pos;
    if (first >= n_tokens) {
        cudaMemcpyAsync(dst + (size_t)pos * n_embd, src,
                        (size_t)n_tokens * stride, cudaMemcpyDeviceToDevice, cudaStreamPerThread);
    } else {
        cudaMemcpyAsync(dst + (size_t)pos * n_embd, src,
                        (size_t)first * stride, cudaMemcpyDeviceToDevice, cudaStreamPerThread);
        cudaMemcpyAsync(dst, src + (size_t)first * stride,
                        (size_t)(n_tokens - first) * stride, cudaMemcpyDeviceToDevice, cudaStreamPerThread);
    }

    return cudaGetLastError() == cudaSuccess;
}

__global__ static void k_dflash_rebuild_conv_state(
        float * __restrict__ r_state,
        const float * __restrict__ qkv,
        const int n_accepted,
        const int conv_ch,
        const int conv_window) {
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= conv_ch) return;

    float * dst = r_state + (size_t) ch * conv_window;
    for (int w = 0; w < conv_window; ++w) {
        const int src_pos = n_accepted + w;
        dst[w] = src_pos < conv_window
            ? dst[src_pos]
            : qkv[(size_t)(src_pos - conv_window) * conv_ch + ch];
    }
}

extern "C" bool dflash_rebuild_conv_state(
        void * r_state, const void * qkv,
        int n_accepted, int conv_ch, int conv_window) {
    if (!r_state || !qkv) return false;
    if (n_accepted <= 0 || conv_ch <= 0 || conv_window <= 0) return false;

    cudaPointerAttributes r_attr;
    cudaError_t r_err = cudaPointerGetAttributes(&r_attr, r_state);
    if (r_err != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    cudaPointerAttributes qkv_attr;
    cudaError_t qkv_err = cudaPointerGetAttributes(&qkv_attr, qkv);
    if (qkv_err != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
#if CUDART_VERSION >= 10000
    if (r_attr.type != cudaMemoryTypeDevice || qkv_attr.type != cudaMemoryTypeDevice ||
            r_attr.device != qkv_attr.device) {
        return false;
    }
#else
    if (r_attr.memoryType != cudaMemoryTypeDevice || qkv_attr.memoryType != cudaMemoryTypeDevice ||
            r_attr.device != qkv_attr.device) {
        return false;
    }
#endif

    (void)cudaSetDevice(r_attr.device);
    const int block = 256;
    const int grid = (conv_ch + block - 1) / block;
    k_dflash_rebuild_conv_state<<<grid, block, 0, cudaStreamPerThread>>>(
        (float *) r_state, (const float *) qkv, n_accepted, conv_ch, conv_window);
    return cudaGetLastError() == cudaSuccess;
}

extern "C" bool dflash_cuda_copy_d2d(void * dst, const void * src, size_t size) {
    if (!dst || !src || size == 0) return false;

    cudaPointerAttributes dst_attr;
    cudaError_t dst_err = cudaPointerGetAttributes(&dst_attr, dst);
    if (dst_err != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    cudaPointerAttributes src_attr;
    cudaError_t src_err = cudaPointerGetAttributes(&src_attr, src);
    if (src_err != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
#if CUDART_VERSION >= 10000
    if (dst_attr.type != cudaMemoryTypeDevice || src_attr.type != cudaMemoryTypeDevice ||
            dst_attr.device != src_attr.device) {
        return false;
    }
    (void)cudaSetDevice(dst_attr.device);
#else
    if (dst_attr.memoryType != cudaMemoryTypeDevice || src_attr.memoryType != cudaMemoryTypeDevice ||
            dst_attr.device != src_attr.device) {
        return false;
    }
    (void)cudaSetDevice(dst_attr.device);
#endif

    cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, cudaStreamPerThread);
    return cudaGetLastError() == cudaSuccess;
}

extern "C" bool dflash_cuda_prepare_ptr(const void * ptr) {
    if (!ptr) return false;

    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
    if (err != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
#if CUDART_VERSION >= 10000
    if (attr.type != cudaMemoryTypeDevice) {
        return false;
    }
    (void)cudaSetDevice(attr.device);
#else
    if (attr.memoryType != cudaMemoryTypeDevice) {
        return false;
    }
    (void)cudaSetDevice(attr.device);
#endif

    return true;
}

extern "C" bool dflash_cuda_copy_d2d_no_check(void * dst, const void * src, size_t size) {
    if (!dst || !src || size == 0) return false;
    cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, cudaStreamPerThread);
    return cudaGetLastError() == cudaSuccess;
}

extern "C" bool dflash_cuda_synchronize_ptr(const void * ptr) {
    if (!ptr) return false;

    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
    if (err != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
#if CUDART_VERSION >= 10000
    if (attr.type != cudaMemoryTypeDevice) {
        return false;
    }
    (void)cudaSetDevice(attr.device);
#else
    if (attr.memoryType != cudaMemoryTypeDevice) {
        return false;
    }
    (void)cudaSetDevice(attr.device);
#endif

    return cudaStreamSynchronize(cudaStreamPerThread) == cudaSuccess;
}

extern "C" void dflash_cross_ring_gpu_synchronize(void * handle) {
    if (!handle) return;
    auto * ring = (dflash_cross_ring_gpu *)handle;
    (void)cudaSetDevice(ring->device);
    cudaStreamSynchronize(cudaStreamPerThread);
}

// Launch interleave kernel. Returns device pointer to interleaved staging buffer.
extern "C" const float * dflash_cross_ring_gpu_interleave(
        void * handle, int write_pos, int filled, int ctx_window) {
    if (!handle) return nullptr;
    auto * ring = (dflash_cross_ring_gpu *)handle;

    int cross_len = filled < ctx_window ? filled : ctx_window;
    if (cross_len <= 0) return nullptr;

    (void)cudaSetDevice(ring->device);

    int read_start = ((write_pos - cross_len) % ring->ring_size + ring->ring_size) % ring->ring_size;

    dim3 grid(cross_len, ring->n_layers);
    dim3 block(256);

    k_cross_ring_interleave<<<grid, block, 0, cudaStreamPerThread>>>(
        (const float * const *)ring->d_layer_rings,
        ring->d_staging,
        ring->ring_size,
        read_start,
        cross_len,
        ring->n_layers,
        ring->n_embd);

    // Ordered with the following set_tensor copies on cudaStreamPerThread.
    // set_tensor synchronizes before the GGML backend graph reads cross data.

    return ring->d_staging;
}

// D2D copy: from device source to device destination (raw pointers).
// Synchronize because subsequent GGML backend graph reads run on backend streams.
extern "C" void dflash_cross_ring_gpu_set_tensor(
        void * d_dst, const void * d_src, size_t offset, size_t size) {
    if (!d_dst || !d_src || size == 0) return;
    cudaMemcpyAsync((char *)d_dst + offset, d_src, size,
                     cudaMemcpyDeviceToDevice, cudaStreamPerThread);
    cudaStreamSynchronize(cudaStreamPerThread);
}

extern "C" bool dflash_kv_cache_write_d2d(
        void * d_ring, const void * d_src,
        int ring_size, int ring_pos, int n_tokens, int n_elem) {
    if (!d_ring || !d_src) return false;
    if (ring_size <= 0 || n_tokens <= 0 || n_elem <= 0) return false;

    cudaPointerAttributes ring_attr;
    cudaError_t ring_err = cudaPointerGetAttributes(&ring_attr, d_ring);
    if (ring_err != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    cudaPointerAttributes src_attr;
    cudaError_t src_err = cudaPointerGetAttributes(&src_attr, d_src);
    if (src_err != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
#if CUDART_VERSION >= 10000
    if (ring_attr.type != cudaMemoryTypeDevice || src_attr.type != cudaMemoryTypeDevice ||
            ring_attr.device != src_attr.device) {
        return false;
    }
    (void)cudaSetDevice(ring_attr.device);
#else
    if (ring_attr.memoryType != cudaMemoryTypeDevice || src_attr.memoryType != cudaMemoryTypeDevice ||
            ring_attr.device != src_attr.device) {
        return false;
    }
    (void)cudaSetDevice(ring_attr.device);
#endif

    const char * src = (const char *) d_src;
    char * dst = (char *) d_ring;
    const size_t stride = (size_t) n_elem * sizeof(float);

    int pos = ring_pos % ring_size;
    if (pos < 0) {
        pos += ring_size;
    }
    if (n_tokens > ring_size) {
        const int skip = n_tokens - ring_size;
        src += (size_t) skip * stride;
        pos = (pos + skip) % ring_size;
        n_tokens = ring_size;
    }

    const int first = ring_size - pos;
    if (first >= n_tokens) {
        cudaMemcpyAsync(dst + (size_t) pos * stride, src,
                        (size_t) n_tokens * stride, cudaMemcpyDeviceToDevice, cudaStreamPerThread);
    } else {
        cudaMemcpyAsync(dst + (size_t) pos * stride, src,
                        (size_t) first * stride, cudaMemcpyDeviceToDevice, cudaStreamPerThread);
        cudaMemcpyAsync(dst, src + (size_t) first * stride,
                        (size_t) (n_tokens - first) * stride, cudaMemcpyDeviceToDevice, cudaStreamPerThread);
    }

    return cudaGetLastError() == cudaSuccess && cudaStreamSynchronize(cudaStreamPerThread) == cudaSuccess;
}

__global__ static void k_dflash_kv_cache_shift_left(
        float * __restrict__ cache,
        const int keep_tokens,
        const int drop_tokens,
        const int n_elem) {
    const int64_t n = (int64_t) keep_tokens * n_elem;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) blockDim.x * gridDim.x) {
        cache[i] = cache[i + (int64_t) drop_tokens * n_elem];
    }
}

extern "C" bool dflash_kv_cache_append_d2d(
        void * d_cache, const void * d_src,
        int cache_size, int filled, int n_tokens, int n_elem) {
    if (!d_cache || !d_src) return false;
    if (cache_size <= 0 || n_tokens <= 0 || n_elem <= 0) return false;

    cudaPointerAttributes cache_attr;
    cudaError_t cache_err = cudaPointerGetAttributes(&cache_attr, d_cache);
    if (cache_err != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    cudaPointerAttributes src_attr;
    cudaError_t src_err = cudaPointerGetAttributes(&src_attr, d_src);
    if (src_err != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
#if CUDART_VERSION >= 10000
    if (cache_attr.type != cudaMemoryTypeDevice || src_attr.type != cudaMemoryTypeDevice ||
            cache_attr.device != src_attr.device) {
        return false;
    }
    (void)cudaSetDevice(cache_attr.device);
#else
    if (cache_attr.memoryType != cudaMemoryTypeDevice || src_attr.memoryType != cudaMemoryTypeDevice ||
            cache_attr.device != src_attr.device) {
        return false;
    }
    (void)cudaSetDevice(cache_attr.device);
#endif

    filled = filled < 0 ? 0 : (filled > cache_size ? cache_size : filled);

    const char * src = (const char *) d_src;
    float * cache = (float *) d_cache;
    const size_t stride = (size_t) n_elem * sizeof(float);

    if (n_tokens >= cache_size) {
        src += (size_t) (n_tokens - cache_size) * stride;
        cudaMemcpyAsync(cache, src, (size_t) cache_size * stride,
                        cudaMemcpyDeviceToDevice, cudaStreamPerThread);
        return cudaGetLastError() == cudaSuccess && cudaStreamSynchronize(cudaStreamPerThread) == cudaSuccess;
    }

    const int total = filled + n_tokens;
    int keep = filled;
    if (total > cache_size) {
        const int drop = total - cache_size;
        keep = filled - drop;
        if (keep > 0) {
            const int64_t n = (int64_t) keep * n_elem;
            const int block = 256;
            const int grid = (int) ((n + block - 1) / block);
            k_dflash_kv_cache_shift_left<<<grid, block, 0, cudaStreamPerThread>>>(
                cache, keep, drop, n_elem);
        }
    }

    cudaMemcpyAsync(cache + (size_t) keep * n_elem, src, (size_t) n_tokens * stride,
                    cudaMemcpyDeviceToDevice, cudaStreamPerThread);
    return cudaGetLastError() == cudaSuccess && cudaStreamSynchronize(cudaStreamPerThread) == cudaSuccess;
}

extern "C" bool dflash_kv_cache_append_d2d_no_check(
        void * d_cache, const void * d_src,
        int cache_size, int filled, int n_tokens, int n_elem) {
    if (!d_cache || !d_src) return false;
    if (cache_size <= 0 || n_tokens <= 0 || n_elem <= 0) return false;

    filled = filled < 0 ? 0 : (filled > cache_size ? cache_size : filled);

    const char * src = (const char *) d_src;
    float * cache = (float *) d_cache;
    const size_t stride = (size_t) n_elem * sizeof(float);

    if (n_tokens >= cache_size) {
        src += (size_t) (n_tokens - cache_size) * stride;
        cudaMemcpyAsync(cache, src, (size_t) cache_size * stride,
                        cudaMemcpyDeviceToDevice, cudaStreamPerThread);
        return cudaGetLastError() == cudaSuccess;
    }

    const int total = filled + n_tokens;
    int keep = filled;
    if (total > cache_size) {
        const int drop = total - cache_size;
        keep = filled - drop;
        if (keep > 0) {
            const int64_t n = (int64_t) keep * n_elem;
            const int block = 256;
            const int grid = (int) ((n + block - 1) / block);
            k_dflash_kv_cache_shift_left<<<grid, block, 0, cudaStreamPerThread>>>(
                cache, keep, drop, n_elem);
        }
    }

    cudaMemcpyAsync(cache + (size_t) keep * n_elem, src, (size_t) n_tokens * stride,
                    cudaMemcpyDeviceToDevice, cudaStreamPerThread);
    return cudaGetLastError() == cudaSuccess;
}

__global__ static void k_dflash_kv_cache_interleave(
        const float * __restrict__ ring,
        float * __restrict__ stage,
        const int ring_size,
        const int read_start,
        const int cross_len,
        const int n_elem) {
    const int t = blockIdx.x;
    if (t >= cross_len) return;

    const int slot = (read_start + t) % ring_size;
    const float * src = ring + (size_t) slot * n_elem;
    float * dst = stage + (size_t) t * n_elem;

    for (int i = threadIdx.x; i < n_elem; i += blockDim.x) {
        dst[i] = src[i];
    }
}

extern "C" bool dflash_kv_cache_interleave(
        const void * d_ring, void * d_stage,
        int ring_size, int write_pos, int filled, int ctx_window, int n_elem) {
    if (!d_ring || !d_stage) return false;
    if (ring_size <= 0 || ctx_window <= 0 || n_elem <= 0) return false;

    cudaPointerAttributes ring_attr;
    cudaError_t ring_err = cudaPointerGetAttributes(&ring_attr, d_ring);
    if (ring_err != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    cudaPointerAttributes stage_attr;
    cudaError_t stage_err = cudaPointerGetAttributes(&stage_attr, d_stage);
    if (stage_err != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
#if CUDART_VERSION >= 10000
    if (ring_attr.type != cudaMemoryTypeDevice || stage_attr.type != cudaMemoryTypeDevice ||
            ring_attr.device != stage_attr.device) {
        return false;
    }
    (void)cudaSetDevice(ring_attr.device);
#else
    if (ring_attr.memoryType != cudaMemoryTypeDevice || stage_attr.memoryType != cudaMemoryTypeDevice ||
            ring_attr.device != stage_attr.device) {
        return false;
    }
    (void)cudaSetDevice(ring_attr.device);
#endif

    const int cross_len = filled < ctx_window ? filled : ctx_window;
    if (cross_len <= 0) {
        cudaMemsetAsync(d_stage, 0, (size_t) ctx_window * (size_t) n_elem * sizeof(float), cudaStreamPerThread);
        return cudaGetLastError() == cudaSuccess;
    }

    const int read_start = ((write_pos - cross_len) % ring_size + ring_size) % ring_size;
    k_dflash_kv_cache_interleave<<<cross_len, 256, 0, cudaStreamPerThread>>>(
        (const float *) d_ring, (float *) d_stage,
        ring_size, read_start, cross_len, n_elem);

    if (cross_len < ctx_window) {
        cudaMemsetAsync((float *) d_stage + (size_t) cross_len * n_elem, 0,
                        (size_t) (ctx_window - cross_len) * (size_t) n_elem * sizeof(float),
                        cudaStreamPerThread);
    }

    return cudaGetLastError() == cudaSuccess;
}
