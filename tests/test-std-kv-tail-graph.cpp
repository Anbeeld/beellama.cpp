#include "ggml.h"
#include "ggml-backend.h"
#include "llama-kv-cache-tail.h"

#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void fail(const char * message) {
    std::fprintf(stderr, "%s\n", message);
    std::exit(1);
}

static void test_representation_topology() {
    const auto make_plan = [](uint32_t requested, uint32_t effective, uint32_t window,
                              bool native_capable, bool already_exact, uint64_t promotion,
                              uint64_t overlay) {
        return llama_kv_tail_storage_plan_for({
            GGML_TYPE_Q4_0, GGML_TYPE_Q5_0, GGML_TYPE_F16,
            requested, effective, 1, 32, window, 4096,
            96, promotion, overlay, native_capable, already_exact,
            true, false, true, true,
        });
    };

    const auto disabled = make_plan(0, 0, 4096, true, false, 64, 64);
    const auto overlay = make_plan(256, 256, 4096, true, false, 64, 64);
    const auto native = make_plan(4096, 4096, 4096, true, false, 64, 64);
    if (disabled.kind != LLAMA_KV_TAIL_STORAGE_DISABLED || disabled.shadow_k || disabled.shadow_v) {
        fail("disabled representation unexpectedly requires tail graph inputs");
    }
    if (overlay.kind != LLAMA_KV_TAIL_STORAGE_OVERLAY || !overlay.shadow_k || !overlay.shadow_v ||
            overlay.layout.total_slots == 0) {
        fail("overlay representation lacks shadow graph topology");
    }
    if (native.kind != LLAMA_KV_TAIL_STORAGE_NATIVE_EXACT || native.shadow_k || native.shadow_v ||
            native.layout.total_slots == 0) {
        fail("native-exact representation unexpectedly requires a shadow merge graph");
    }
}

static ggml_backend_meta_split_state test_standard_meta_split(
        const ggml_tensor * tensor, void *) {
    if (std::strcmp(tensor->name, "meta_cache") == 0 ||
            std::strcmp(tensor->name, "meta_current") == 0) {
        ggml_backend_meta_split_state result = {
            GGML_BACKEND_SPLIT_AXIS_0, { 0 }, { 1 }, 1
        };
        // Two complete four-element heads over three logical devices.  This
        // is the zero-head-shard case exercised by upstream tensor mode when
        // there are more participating devices than KV heads.
        result.ne[0] = 0;
        result.ne[1] = 4;
        result.ne[2] = 4;
        return result;
    }
    return { GGML_BACKEND_SPLIT_AXIS_MIRRORED, { 0 }, { 1 }, 1 };
}

static void test_shadow_roundtrip(ggml_backend_t backend) {
    constexpr int64_t width = 8;
    constexpr int64_t slots = 7;
    constexpr int64_t writes = 3;
    constexpr int64_t tail = 3;
    constexpr int64_t queries = 3;
    ggml_init_params params = { 1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * storage = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, width, slots);
    ggml_tensor * source = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, writes);
    ggml_tensor * write_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, writes);
    ggml_tensor * read_idxs = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, tail, queries);
    ggml_tensor * write = ggml_set_rows(ctx, storage, source, write_idxs);
    ggml_tensor * rows = ggml_get_rows_as(
            ctx, write, ggml_reshape_1d(ctx, read_idxs, tail*queries), GGML_TYPE_F16);
    rows = ggml_reshape_4d(ctx, rows, width, 1, tail, queries);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, write);
    ggml_build_forward_expand(graph, rows);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    const std::vector<float> source_data = {
        1,2,3,4,5,6,7,8, 11,12,13,14,15,16,17,18, 21,22,23,24,25,26,27,28,
    };
    const int64_t write_data[] = { 2, 5, 1 };
    const int32_t read_data[] = { 2,5,1, 1,2,5, 5,1,2 };
    ggml_backend_tensor_set(source, source_data.data(), 0, source_data.size()*sizeof(float));
    ggml_backend_tensor_set(write_idxs, write_data, 0, sizeof(write_data));
    ggml_backend_tensor_set(read_idxs, read_data, 0, sizeof(read_data));
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        fail("shadow roundtrip compute failed");
    }
    std::vector<float> got(ggml_nelements(rows));
    std::vector<ggml_fp16_t> got_f16(got.size());
    ggml_backend_tensor_get(rows, got_f16.data(), 0, got_f16.size()*sizeof(ggml_fp16_t));
    ggml_fp16_to_fp32_row(got_f16.data(), got.data(), got.size());
    for (int64_t iq = 0; iq < queries; ++iq) {
        for (int64_t it = 0; it < tail; ++it) {
            int source_row = -1;
            for (int iw = 0; iw < writes; ++iw) {
                if (write_data[iw] == read_data[it + tail*iq]) {
                    source_row = iw;
                }
            }
            for (int64_t i = 0; i < width; ++i) {
                const float expected = source_data[i + width*source_row];
                const float actual = got[i + width*(it + tail*iq)];
                if (actual != expected) {
                    std::vector<ggml_fp16_t> written_f16(ggml_nelements(write));
                    std::vector<float> written(written_f16.size());
                    ggml_backend_tensor_get(write, written_f16.data(), 0, written_f16.size()*sizeof(ggml_fp16_t));
                    ggml_fp16_to_fp32_row(written_f16.data(), written.data(), written.size());
                    std::fprintf(stderr, "shadow roundtrip mismatch q=%lld tail=%lld i=%lld: got %.3f expected %.3f\n",
                            (long long) iq, (long long) it, (long long) i, actual, expected);
                    std::fprintf(stderr, "write row 2 first=%.3f rows_type=%s graph_nodes=%d\n",
                            written[2*width], ggml_type_name(rows->type), ggml_graph_n_nodes(graph));
                    for (int inode = 0; inode < ggml_graph_n_nodes(graph); ++inode) {
                        ggml_tensor * node = ggml_graph_node(graph, inode);
                        std::fprintf(stderr, "node %d op=%s type=%s\n", inode,
                                ggml_op_name(node->op), ggml_type_name(node->type));
                    }
                    std::exit(1);
                }
            }
        }
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

static void test_meta_shadow_roundtrip(ggml_backend_t backend) {
    constexpr int64_t width = 8;
    constexpr int64_t slots = 7;
    constexpr int64_t writes = 3;
    constexpr int64_t tail = 3;
    constexpr int64_t queries = 3;
    ggml_init_params params = { 1024*1024, nullptr, true };
    ggml_context * static_ctx = ggml_init(params);
    ggml_tensor * storage = ggml_new_tensor_2d(static_ctx, GGML_TYPE_F16, width, slots);
    ggml_tensor * source = ggml_new_tensor_2d(static_ctx, GGML_TYPE_F32, width, writes);
    ggml_tensor * write_idxs = ggml_new_tensor_1d(static_ctx, GGML_TYPE_I64, writes);
    ggml_tensor * read_idxs = ggml_new_tensor_2d(static_ctx, GGML_TYPE_I32, tail, queries);
    ggml_set_name(storage, "meta_cache");
    ggml_set_name(source, "meta_current");
    ggml_backend_buffer_t static_buffer = ggml_backend_alloc_ctx_tensors(
            static_ctx, backend);
    if (!static_buffer) {
        fail("failed to allocate static meta exact-tail tensors");
    }

    ggml_context * compute_ctx = ggml_init(params);
    ggml_tensor * write = ggml_set_rows(
            compute_ctx, storage, source, write_idxs);
    ggml_tensor * rows = ggml_get_rows_as(
            compute_ctx, write,
            ggml_reshape_1d(compute_ctx, read_idxs, tail*queries),
            GGML_TYPE_F16);
    rows = ggml_reshape_4d(compute_ctx, rows, width, 1, tail, queries);
    ggml_cgraph * graph = ggml_new_graph(compute_ctx);
    ggml_build_forward_expand(graph, write);
    ggml_build_forward_expand(graph, rows);
    ggml_backend_t cpu_backend = ggml_backend_init_by_type(
            GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!cpu_backend) {
        fail("failed to initialize CPU scheduler fallback");
    }
    ggml_backend_t backends[] = { backend, cpu_backend };
    ggml_backend_sched_t sched = ggml_backend_sched_new(
            backends, nullptr, 2, 32, false, true);
    if (!sched || !ggml_backend_sched_alloc_graph(sched, graph)) {
        fail("failed to allocate scheduled meta exact-tail graph");
    }

    const std::vector<float> source_data = {
        1,2,3,4,5,6,7,8, 11,12,13,14,15,16,17,18, 21,22,23,24,25,26,27,28,
    };
    const int64_t write_data[] = { 2, 5, 1 };
    const int32_t read_data[] = { 2,5,1, 1,2,5, 5,1,2 };
    ggml_backend_tensor_set(source, source_data.data(), 0, ggml_nbytes(source));
    ggml_backend_tensor_set(write_idxs, write_data, 0, sizeof(write_data));
    ggml_backend_tensor_set(read_idxs, read_data, 0, sizeof(read_data));
    if (ggml_backend_sched_graph_compute(sched, graph) != GGML_STATUS_SUCCESS) {
        fail("scheduled meta exact-tail compute failed");
    }
    std::vector<ggml_fp16_t> got_f16(ggml_nelements(rows));
    std::vector<float> got(got_f16.size());
    ggml_backend_tensor_get(rows, got_f16.data(), 0,
            got_f16.size()*sizeof(ggml_fp16_t));
    ggml_fp16_to_fp32_row(got_f16.data(), got.data(), got.size());
    for (int64_t iq = 0; iq < queries; ++iq) {
        for (int64_t it = 0; it < tail; ++it) {
            int source_row = -1;
            for (int iw = 0; iw < writes; ++iw) {
                if (write_data[iw] == read_data[it + tail*iq]) {
                    source_row = iw;
                }
            }
            for (int64_t i = 0; i < width; ++i) {
                const float expected = source_data[i + width*source_row];
                const float actual = got[i + width*(it + tail*iq)];
                if (actual != expected) {
                    fail("scheduled meta exact-tail roundtrip mismatch");
                }
            }
        }
    }

    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu_backend);
    ggml_free(compute_ctx);
    ggml_backend_buffer_free(static_buffer);
    ggml_free(static_ctx);
}

static void test_fully_masked_quant_body(ggml_backend_t backend, ggml_type body_type) {
    constexpr int64_t d = 32;
    constexpr int64_t n_kv_head = 2;
    constexpr int64_t n_head = 4;
    constexpr int64_t n_body = 6;
    constexpr int64_t n_tail = 5;
    constexpr int64_t n_query = 3;
    ggml_init_params params = { 8*1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, n_head, n_query);
    ggml_tensor * kb = ggml_new_tensor_4d(ctx, body_type, d, n_body, n_kv_head, 1);
    ggml_tensor * vb = ggml_new_tensor_4d(ctx, body_type, d, n_body, n_kv_head, 1);
    ggml_tensor * kt_input = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, n_kv_head, n_tail, n_query);
    ggml_tensor * vt_input = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, n_kv_head, n_tail, n_query);
    ggml_tensor * mb = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n_body, n_query, 1, 1);
    ggml_tensor * mt = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n_tail, n_query, 1, 1);

    ggml_tensor * qb = ggml_permute(ctx,
            ggml_reshape_4d(ctx, q, d, n_head, n_query, 1), 0, 2, 1, 3);
    ggml_tensor * qt = ggml_permute(ctx,
            ggml_reshape_4d(ctx, q, d, n_head, 1, n_query), 0, 2, 1, 3);
    ggml_tensor * kt = ggml_permute(ctx, kt_input, 0, 2, 1, 3);
    ggml_tensor * vt = ggml_permute(ctx, vt_input, 0, 2, 1, 3);
    ggml_tensor * sb = ggml_mul_mat(ctx, kb, qb);
    ggml_tensor * st = ggml_mul_mat(ctx, kt, qt);
    st = ggml_reshape_4d(ctx, st, n_tail, n_head, n_query, 1);
    st = ggml_permute(ctx, st, 0, 2, 1, 3);
    ggml_tensor * scores = ggml_concat(ctx, sb, st, 0);
    ggml_tensor * mask = ggml_concat(ctx, mb, mt, 0);
    scores = ggml_soft_max_ext(ctx, scores, mask, 1.0f, 0.0f);

    ggml_tensor * wb = ggml_view_4d(ctx, scores,
            n_body, scores->ne[1], scores->ne[2], scores->ne[3],
            scores->nb[1], scores->nb[2], scores->nb[3], 0);
    ggml_tensor * wt = ggml_view_4d(ctx, scores,
            n_tail, scores->ne[1], scores->ne[2], scores->ne[3],
            scores->nb[1], scores->nb[2], scores->nb[3], n_body*scores->nb[0]);
    ggml_tensor * body_out = ggml_out_prod(ctx, vb, ggml_transpose(ctx, wb));
    wt = ggml_cont(ctx, ggml_permute(ctx, wt, 0, 2, 1, 3));
    wt = ggml_reshape_4d(ctx, wt, n_tail, 1, n_head, n_query);
    vt = ggml_cont(ctx, ggml_transpose(ctx, vt));
    ggml_tensor * tail_out = ggml_mul_mat(ctx, vt, wt);
    tail_out = ggml_reshape_4d(ctx, tail_out, d, n_head, n_query, 1);
    tail_out = ggml_permute(ctx, tail_out, 0, 2, 1, 3);
    ggml_tensor * out = ggml_cont(ctx, ggml_add(ctx, body_out, tail_out));

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    std::vector<float> q_data(ggml_nelements(q));
    std::vector<float> kt_data_f32(ggml_nelements(kt_input));
    std::vector<float> vt_data_f32(ggml_nelements(vt_input));
    for (int64_t iq = 0; iq < n_query; ++iq) {
        for (int64_t ih = 0; ih < n_head; ++ih) {
            for (int64_t i = 0; i < d; ++i) {
                q_data[i + d*(ih + n_head*iq)] = 0.002f*float(1 + i + 3*ih + 5*iq);
            }
        }
        for (int64_t it = 0; it < n_tail; ++it) {
            for (int64_t ih = 0; ih < n_kv_head; ++ih) {
                for (int64_t i = 0; i < d; ++i) {
                    kt_data_f32[i + d*(ih + n_kv_head*(it + n_tail*iq))] =
                            0.003f*float(1 + i + 7*ih + 11*it + 13*iq);
                    vt_data_f32[i + d*(ih + n_kv_head*(it + n_tail*iq))] =
                            0.004f*float(1 + 2*i + 5*ih + 17*it + 19*iq);
                }
            }
        }
    }
    std::vector<ggml_fp16_t> kt_data(kt_data_f32.size());
    std::vector<ggml_fp16_t> vt_data(vt_data_f32.size());
    ggml_fp32_to_fp16_row(kt_data_f32.data(), kt_data.data(), kt_data.size());
    ggml_fp32_to_fp16_row(vt_data_f32.data(), vt_data.data(), vt_data.size());
    std::vector<float> body_zeros(ggml_nelements(kb), 0.0f);
    std::vector<uint8_t> kb_data(ggml_nbytes(kb));
    std::vector<uint8_t> vb_data(ggml_nbytes(vb));
    ggml_quantize_chunk(body_type, body_zeros.data(), kb_data.data(), 0,
            n_body*n_kv_head, d, nullptr);
    ggml_quantize_chunk(body_type, body_zeros.data(), vb_data.data(), 0,
            n_body*n_kv_head, d, nullptr);
    std::vector<float> mb_data(ggml_nelements(mb), -INFINITY);
    std::vector<float> mt_data(ggml_nelements(mt), 0.0f);
    ggml_backend_tensor_set(q, q_data.data(), 0, q_data.size()*sizeof(float));
    ggml_backend_tensor_set(kb, kb_data.data(), 0, kb_data.size());
    ggml_backend_tensor_set(vb, vb_data.data(), 0, vb_data.size());
    ggml_backend_tensor_set(kt_input, kt_data.data(), 0, kt_data.size()*sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(vt_input, vt_data.data(), 0, vt_data.size()*sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(mb, mb_data.data(), 0, mb_data.size()*sizeof(float));
    ggml_backend_tensor_set(mt, mt_data.data(), 0, mt_data.size()*sizeof(float));
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        fail("masked quantized-body graph compute failed");
    }
    std::vector<float> got(ggml_nelements(out));
    ggml_backend_tensor_get(out, got.data(), 0, got.size()*sizeof(float));
    for (int64_t iq = 0; iq < n_query; ++iq) {
        for (int64_t ih = 0; ih < n_head; ++ih) {
            const int64_t ikh = ih/(n_head/n_kv_head);
            std::vector<float> logits(n_tail);
            float maximum = -INFINITY;
            for (int64_t it = 0; it < n_tail; ++it) {
                float value = 0.0f;
                for (int64_t i = 0; i < d; ++i) {
                    value += q_data[i + d*(ih + n_head*iq)]*
                            ggml_fp16_to_fp32(kt_data[i + d*(ikh + n_kv_head*(it + n_tail*iq))]);
                }
                logits[it] = value;
                maximum = std::max(maximum, value);
            }
            float norm = 0.0f;
            for (float & value : logits) {
                value = std::exp(value - maximum);
                norm += value;
            }
            for (int64_t i = 0; i < d; ++i) {
                float expected = 0.0f;
                for (int64_t it = 0; it < n_tail; ++it) {
                    expected += logits[it]/norm*ggml_fp16_to_fp32(
                            vt_data[i + d*(ikh + n_kv_head*(it + n_tail*iq))]);
                }
                const size_t index = size_t(i + d*(iq + n_query*ih));
                if (std::fabs(got[index] - expected) > 1e-3f) {
                    std::fprintf(stderr, "masked %s body mismatch q=%lld h=%lld d=%lld: got %.8f expected %.8f\n",
                            ggml_type_name(body_type), (long long) iq, (long long) ih,
                            (long long) i, got[index], expected);
                    std::exit(1);
                }
            }
        }
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

static void test_attention_graph(ggml_backend_t backend) {
    constexpr int64_t d = 4;
    constexpr int64_t dv = 3;
    constexpr int64_t n_kv_head = 2;
    constexpr int64_t n_head = 4;
    constexpr int64_t n_tail = 5;
    constexpr int64_t n_query = 3;

    ggml_init_params params = {
        /* .mem_size   = */ 4*1024*1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fail("failed to initialize ggml context");
    }

    ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, n_head, n_query);
    ggml_tensor * k_input = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, n_kv_head, n_tail, n_query);
    ggml_tensor * v_input = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dv, n_kv_head, n_tail, n_query);
    ggml_tensor * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n_tail, n_query, 1, 1);

    ggml_tensor * q_batched = ggml_reshape_4d(ctx, q, d, n_head, 1, n_query);
    q_batched = ggml_permute(ctx, q_batched, 0, 2, 1, 3);
    ggml_tensor * k = ggml_permute(ctx, k_input, 0, 2, 1, 3);
    ggml_tensor * v = ggml_permute(ctx, v_input, 0, 2, 1, 3);

    ggml_tensor * scores = ggml_mul_mat(ctx, k, q_batched);
    scores = ggml_reshape_4d(ctx, scores, n_tail, n_head, n_query, 1);
    scores = ggml_cont(ctx, ggml_permute(ctx, scores, 0, 2, 1, 3));
    scores = ggml_soft_max_ext(ctx, scores, mask, 1.0f, 0.0f);

    ggml_tensor * weights = ggml_cont(ctx, ggml_permute(ctx, scores, 0, 2, 1, 3));
    weights = ggml_reshape_4d(ctx, weights, n_tail, 1, n_head, n_query);
    v = ggml_cont(ctx, ggml_transpose(ctx, v));
    ggml_tensor * out = ggml_mul_mat(ctx, v, weights);
    out = ggml_reshape_4d(ctx, out, dv, n_head, n_query, 1);
    out = ggml_permute(ctx, out, 0, 2, 1, 3);
    out = ggml_cont(ctx, out);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        fail("failed to allocate graph tensors");
    }

    std::vector<float> q_data(ggml_nelements(q));
    std::vector<float> k_data(ggml_nelements(k_input));
    std::vector<float> v_data(ggml_nelements(v_input));
    std::vector<float> mask_data(ggml_nelements(mask), 0.0f);
    for (int64_t iq = 0; iq < n_query; ++iq) {
        for (int64_t ih = 0; ih < n_head; ++ih) {
            for (int64_t i = 0; i < d; ++i) {
                q_data[i + d*(ih + n_head*iq)] = 0.03f*float(1 + i + 3*ih + 7*iq);
            }
        }
        for (int64_t it = 0; it < n_tail; ++it) {
            for (int64_t ih = 0; ih < n_kv_head; ++ih) {
                for (int64_t i = 0; i < d; ++i) {
                    k_data[i + d*(ih + n_kv_head*(it + n_tail*iq))] =
                            0.02f*float(1 + 2*i + 5*ih + 11*it + 17*iq);
                }
                for (int64_t i = 0; i < dv; ++i) {
                    v_data[i + dv*(ih + n_kv_head*(it + n_tail*iq))] =
                            0.05f*float(1 + 3*i + 7*ih + 13*it + 19*iq);
                }
            }
        }
    }

    ggml_backend_tensor_set(q, q_data.data(), 0, q_data.size()*sizeof(float));
    ggml_backend_tensor_set(k_input, k_data.data(), 0, k_data.size()*sizeof(float));
    ggml_backend_tensor_set(v_input, v_data.data(), 0, v_data.size()*sizeof(float));
    ggml_backend_tensor_set(mask, mask_data.data(), 0, mask_data.size()*sizeof(float));
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        fail("tail graph compute failed");
    }

    std::vector<float> got(ggml_nelements(out));
    ggml_backend_tensor_get(out, got.data(), 0, got.size()*sizeof(float));
    for (int64_t iq = 0; iq < n_query; ++iq) {
        for (int64_t ih = 0; ih < n_head; ++ih) {
            const int64_t ikh = ih/(n_head/n_kv_head);
            std::vector<float> logits(n_tail);
            float max_logit = -INFINITY;
            for (int64_t it = 0; it < n_tail; ++it) {
                float logit = 0.0f;
                for (int64_t i = 0; i < d; ++i) {
                    logit += q_data[i + d*(ih + n_head*iq)] *
                            k_data[i + d*(ikh + n_kv_head*(it + n_tail*iq))];
                }
                logits[it] = logit;
                max_logit = std::max(max_logit, logit);
            }
            float norm = 0.0f;
            for (float & logit : logits) {
                logit = std::exp(logit - max_logit);
                norm += logit;
            }
            for (int64_t i = 0; i < dv; ++i) {
                float expected = 0.0f;
                for (int64_t it = 0; it < n_tail; ++it) {
                    expected += logits[it]/norm *
                            v_data[i + dv*(ikh + n_kv_head*(it + n_tail*iq))];
                }
                const size_t index = size_t(i + dv*(iq + n_query*ih));
                if (std::fabs(got[index] - expected) > 1e-5f) {
                    std::fprintf(stderr,
                            "tail graph mismatch q=%lld h=%lld d=%lld: got %.8f expected %.8f\n",
                            (long long) iq, (long long) ih, (long long) i, got[index], expected);
                    std::exit(1);
                }
            }
        }
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

static void set_matrix_data(
        ggml_tensor * tensor,
        const std::vector<float> & values,
        int64_t rows,
        int64_t columns) {
    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(tensor, values.data(), 0, values.size()*sizeof(float));
        return;
    }
    if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> converted(values.size());
        ggml_fp32_to_fp16_row(values.data(), converted.data(), converted.size());
        ggml_backend_tensor_set(tensor, converted.data(), 0, converted.size()*sizeof(ggml_fp16_t));
        return;
    }
    if (tensor->type == GGML_TYPE_BF16) {
        std::vector<ggml_bf16_t> converted(values.size());
        ggml_fp32_to_bf16_row(values.data(), converted.data(), converted.size());
        ggml_backend_tensor_set(tensor, converted.data(), 0, converted.size()*sizeof(ggml_bf16_t));
        return;
    }
    if (ggml_is_quantized(tensor->type)) {
        std::vector<uint8_t> converted(ggml_nbytes(tensor));
        ggml_quantize_chunk(tensor->type, values.data(), converted.data(), 0, rows, columns, nullptr);
        ggml_backend_tensor_set(tensor, converted.data(), 0, converted.size());
        return;
    }
    fail("unsupported mixed-side test type");
}

static std::vector<float> run_attached_tail_attention(
        ggml_backend_t backend,
        int64_t n_query,
        ggml_type body_type,
        ggml_type tail_type,
        bool segmented = false) {
    constexpr int64_t d = 128;
    constexpr int64_t n_q_head = 2;
    constexpr int64_t n_kv_head = 1;
    constexpr int64_t n_tail = 8;
    constexpr int64_t n_current = 4;
    const int64_t n_history = segmented ? n_tail - n_current : n_tail;
    const int64_t n_kv = std::max<int64_t>(64, n_query);
    ggml_init_params params = { 64*1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, n_query, n_q_head, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, body_type, d, n_kv, n_kv_head, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, body_type, d, n_kv, n_kv_head, 1);
    ggml_tensor * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv, n_query, 1, 1);
    ggml_tensor * kt_storage = ggml_new_tensor_4d(ctx, tail_type, d, n_kv_head, n_history, 1);
    ggml_tensor * vt_storage = ggml_new_tensor_4d(ctx, tail_type, d, n_kv_head, n_history, 1);
    ggml_tensor * kt = ggml_permute(ctx, kt_storage, 0, 2, 1, 3);
    ggml_tensor * vt = ggml_permute(ctx, vt_storage, 0, 2, 1, 3);
    ggml_tensor * kt_current_storage = segmented ?
            ggml_new_tensor_4d(ctx, tail_type, d, n_kv_head, n_current, 1) : nullptr;
    ggml_tensor * vt_current_storage = segmented ?
            ggml_new_tensor_4d(ctx, tail_type, d, n_kv_head, n_current, 1) : nullptr;
    ggml_tensor * kt_current = segmented ?
            ggml_permute(ctx, kt_current_storage, 0, 2, 1, 3) : nullptr;
    ggml_tensor * vt_current = segmented ?
            ggml_permute(ctx, vt_current_storage, 0, 2, 1, 3) : nullptr;
    ggml_tensor * tail_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_tail, n_query, 1, 1);
    ggml_tensor * query_order = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_query, 1);
    ggml_tensor * run_desc = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 6 + n_tail, 1);
    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f/std::sqrt(float(d)), 0.0f, 0.0f);
    if (segmented) {
        out = ggml_kv_tail_attention_merge_segmented(
                ctx, out, kt, vt, kt_current, vt_current,
                tail_mask, query_order, run_desc);
        ggml_flash_attn_ext_set_kv_tail_history_slots(out, int32_t(n_history));
    } else {
        out = ggml_kv_tail_attention_merge(
                ctx, out, kt, vt, tail_mask, query_order, run_desc);
    }
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);

    if (!ggml_backend_supports_op(backend, out)) {
        fail("backend rejected bounded standard quantized body-plus-tail attention");
    }
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) fail("attached-tail graph allocation failed");

    std::vector<float> q_data(ggml_nelements(q));
    std::vector<float> k_data(ggml_nelements(k));
    std::vector<float> v_data(ggml_nelements(v));
    std::vector<float> kt_data(ggml_nelements(kt_storage));
    std::vector<float> vt_data(ggml_nelements(vt_storage));
    std::vector<float> kt_current_data(segmented ? ggml_nelements(kt_current_storage) : 0);
    std::vector<float> vt_current_data(segmented ? ggml_nelements(vt_current_storage) : 0);
    for (size_t i = 0; i < q_data.size(); ++i) q_data[i] = 0.002f*float(1 + i%19);
    for (size_t i = 0; i < k_data.size(); ++i) k_data[i] = 0.003f*float(int(i%17) - 8);
    for (size_t i = 0; i < v_data.size(); ++i) v_data[i] = 0.004f*float(int(i%23) - 11);
    for (size_t i = 0; i < kt_data.size(); ++i) kt_data[i] = 0.005f*float(int(i%13) - 6);
    for (size_t i = 0; i < vt_data.size(); ++i) vt_data[i] = 0.006f*float(int(i%11) - 5);
    for (size_t i = 0; i < kt_current_data.size(); ++i) kt_current_data[i] = 0.007f*float(int(i%17) - 8);
    for (size_t i = 0; i < vt_current_data.size(); ++i) vt_current_data[i] = 0.008f*float(int(i%19) - 9);
    set_matrix_data(q, q_data, n_query*n_q_head, d);
    set_matrix_data(k, k_data, n_kv*n_kv_head, d);
    set_matrix_data(v, v_data, n_kv*n_kv_head, d);
    set_matrix_data(kt_storage, kt_data, n_history*n_kv_head, d);
    set_matrix_data(vt_storage, vt_data, n_history*n_kv_head, d);
    if (segmented) {
        set_matrix_data(kt_current_storage, kt_current_data, n_current*n_kv_head, d);
        set_matrix_data(vt_current_storage, vt_current_data, n_current*n_kv_head, d);
    }
    std::vector<ggml_fp16_t> mask_data(ggml_nelements(mask), ggml_fp32_to_fp16(0.0f));
    std::vector<ggml_fp16_t> tail_mask_data(ggml_nelements(tail_mask), ggml_fp32_to_fp16(0.0f));
    std::vector<int32_t> query_order_data(n_query);
    for (int32_t i = 0; i < n_query; ++i) query_order_data[size_t(i)] = i;
    std::vector<int32_t> run_data(6 + n_tail, -1);
    run_data[4] = int32_t(n_tail);
    for (int32_t i = 0; i < n_tail; ++i) run_data[size_t(6 + i)] = i;
    ggml_backend_tensor_set(mask, mask_data.data(), 0, mask_data.size()*sizeof(mask_data[0]));
    ggml_backend_tensor_set(tail_mask, tail_mask_data.data(), 0, tail_mask_data.size()*sizeof(tail_mask_data[0]));
    ggml_backend_tensor_set(query_order, query_order_data.data(), 0, query_order_data.size()*sizeof(int32_t));
    ggml_backend_tensor_set(run_desc, run_data.data(), 0, run_data.size()*sizeof(int32_t));
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        fail("attached-tail graph compute failed");
    }
    std::vector<float> result(ggml_nelements(out));
    ggml_backend_tensor_get(out, result.data(), 0, result.size()*sizeof(float));
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return result;
}

static void test_bounded_attached_tail_attention(ggml_backend_t gpu, ggml_backend_t cpu) {
    for (int64_t n_query : { 1, 16, 17, 512 }) {
        for (ggml_type tail_type : { GGML_TYPE_F16, GGML_TYPE_BF16 }) {
            if (n_query == 512 && tail_type == GGML_TYPE_BF16) continue;
            const auto expected = run_attached_tail_attention(cpu, n_query, GGML_TYPE_Q4_0, tail_type);
            const auto actual = run_attached_tail_attention(gpu, n_query, GGML_TYPE_Q4_0, tail_type);
            if (actual.size() != expected.size()) fail("attached-tail result size mismatch");
            double mse = 0.0;
            for (size_t i = 0; i < actual.size(); ++i) {
                const double diff = double(actual[i]) - expected[i];
                mse += diff*diff;
            }
            const double rmse = std::sqrt(mse/std::max<size_t>(actual.size(), 1));
            if (rmse > 2e-3) {
                std::fprintf(stderr, "attached-tail nq=%lld type=%s RMSE %.8f\n",
                        (long long) n_query, ggml_type_name(tail_type), rmse);
                std::exit(1);
            }
        }
    }
    for (int64_t n_query : { 17, 512 }) {
        const auto expected = run_attached_tail_attention(
                cpu, n_query, GGML_TYPE_Q4_0, GGML_TYPE_F16, true);
        const auto actual = run_attached_tail_attention(
                gpu, n_query, GGML_TYPE_Q4_0, GGML_TYPE_F16, true);
        double mse = 0.0;
        for (size_t i = 0; i < actual.size(); ++i) {
            const double diff = double(actual[i]) - expected[i];
            mse += diff*diff;
        }
        if (std::sqrt(mse/std::max<size_t>(actual.size(), 1)) > 2e-3) {
            fail("segmented current standard tail differs from CPU oracle");
        }
    }
}

static void test_mixed_side_generic_attention(
        ggml_backend_t backend,
        ggml_type body_k_type,
        ggml_type body_v_type,
        ggml_type tail_k_type,
        ggml_type tail_v_type,
        bool transposed_v) {
    constexpr int64_t d = 32;
    constexpr int64_t n_body = 5;
    constexpr int64_t n_tail = 3;
    constexpr int64_t n_query = 2;
    ggml_init_params params = { 8*1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * q = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, n_query);
    ggml_tensor * kb = ggml_new_tensor_2d(ctx, body_k_type, d, n_body);
    ggml_tensor * kt = ggml_new_tensor_2d(ctx, tail_k_type, d, n_tail);
    ggml_tensor * vb = transposed_v ?
            ggml_new_tensor_2d(ctx, body_v_type, n_body, d) :
            ggml_new_tensor_2d(ctx, body_v_type, d, n_body);
    ggml_tensor * vt = ggml_new_tensor_2d(ctx, tail_v_type, d, n_tail);
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_body + n_tail, n_query);

    ggml_tensor * scores_body = ggml_mul_mat(ctx, kb, q);
    ggml_tensor * scores_tail = ggml_mul_mat(ctx, kt, q);
    ggml_tensor * scores = ggml_concat(ctx, scores_body, scores_tail, 0);
    scores = ggml_soft_max_ext(ctx, scores, mask, 1.0f, 0.0f);
    ggml_tensor * weights_body = ggml_view_2d(
            ctx, scores, n_body, n_query, scores->nb[1], 0);
    ggml_tensor * weights_tail = ggml_view_2d(
            ctx, scores, n_tail, n_query, scores->nb[1], n_body*scores->nb[0]);
    // Match the production tail merge: Vulkan matmul descriptors require the
    // sliced tail weights to be copied to an aligned contiguous tensor.
    weights_tail = ggml_cont(ctx, weights_tail);
    ggml_tensor * body_out = transposed_v ?
            ggml_mul_mat(ctx, vb, weights_body) :
            ggml_out_prod(ctx, vb, ggml_transpose(ctx, weights_body));
    ggml_tensor * tail_out = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, vt)), weights_tail);
    ggml_tensor * out = ggml_add(ctx, body_out, tail_out);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        fail("mixed-side graph allocation failed");
    }

    std::vector<float> q_data(d*n_query);
    std::vector<float> kb_data(d*n_body);
    std::vector<float> kt_data(d*n_tail);
    std::vector<float> vb_rows(d*n_body);
    std::vector<float> vt_data(d*n_tail);
    for (int64_t iq = 0; iq < n_query; ++iq) {
        for (int64_t i = 0; i < d; ++i) {
            q_data[i + d*iq] = 0.01f*float(1 + (i%7) + 3*iq);
        }
    }
    for (int64_t row = 0; row < n_body; ++row) {
        for (int64_t i = 0; i < d; ++i) {
            kb_data[i + d*row] = 0.015f*float(1 + (i%5) + 2*row);
            vb_rows[i + d*row] = 0.02f*float(1 + (i%9) + 3*row);
        }
    }
    for (int64_t row = 0; row < n_tail; ++row) {
        for (int64_t i = 0; i < d; ++i) {
            kt_data[i + d*row] = 0.017f*float(1 + (i%6) + 4*row);
            vt_data[i + d*row] = 0.023f*float(1 + (i%8) + 5*row);
        }
    }
    std::vector<float> vb_data(vb_rows.size());
    if (transposed_v) {
        for (int64_t row = 0; row < n_body; ++row) {
            for (int64_t i = 0; i < d; ++i) {
                vb_data[row + n_body*i] = vb_rows[i + d*row];
            }
        }
    } else {
        vb_data = vb_rows;
    }
    std::vector<float> mask_data((n_body + n_tail)*n_query, 0.0f);
    set_matrix_data(q, q_data, n_query, d);
    set_matrix_data(kb, kb_data, n_body, d);
    set_matrix_data(kt, kt_data, n_tail, d);
    set_matrix_data(vb, vb_data, transposed_v ? d : n_body, transposed_v ? n_body : d);
    set_matrix_data(vt, vt_data, n_tail, d);
    ggml_backend_tensor_set(mask, mask_data.data(), 0, mask_data.size()*sizeof(float));

    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        fail("mixed-side generic graph compute failed");
    }
    std::vector<float> got(d*n_query);
    ggml_backend_tensor_get(out, got.data(), 0, got.size()*sizeof(float));
    for (int64_t iq = 0; iq < n_query; ++iq) {
        std::vector<float> logits(n_body + n_tail);
        float maximum = -INFINITY;
        for (int64_t row = 0; row < n_body + n_tail; ++row) {
            float value = 0.0f;
            const float * key = row < n_body ? kb_data.data() + d*row :
                    kt_data.data() + d*(row - n_body);
            for (int64_t i = 0; i < d; ++i) {
                value += q_data[i + d*iq]*key[i];
            }
            logits[row] = value;
            maximum = std::max(maximum, value);
        }
        float norm = 0.0f;
        for (float & value : logits) {
            value = std::exp(value - maximum);
            norm += value;
        }
        for (int64_t i = 0; i < d; ++i) {
            float expected = 0.0f;
            for (int64_t row = 0; row < n_body + n_tail; ++row) {
                const float value = row < n_body ? vb_rows[i + d*row] :
                        vt_data[i + d*(row - n_body)];
                expected += logits[row]/norm*value;
            }
            const float error = std::fabs(got[i + d*iq] - expected);
            if (error > 0.025f) {
                std::fprintf(stderr,
                        "mixed-side mismatch K=%s V=%s tK=%s tV=%s trans=%d q=%lld d=%lld got=%.6f expected=%.6f\n",
                        ggml_type_name(body_k_type), ggml_type_name(body_v_type),
                        ggml_type_name(tail_k_type), ggml_type_name(tail_v_type), int(transposed_v),
                        (long long) iq, (long long) i, got[i + d*iq], expected);
                std::exit(1);
            }
        }
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

static void test_sparse_swa_packed_oracle(ggml_backend_t backend, llama_swa_type swa_type) {
    constexpr int64_t d = 32;
    constexpr int64_t n_kv = 12;
    constexpr int64_t n_tail = 2;
    constexpr int64_t n_query = 4;
    constexpr uint32_t n_swa = 4;
    const llama_pos query_pos[n_query] = { 5, 7, 9, 11 };
    const uint32_t tail_cells[n_tail] = { 7, 10 };

    std::vector<float> body_mask(n_kv*n_query, -INFINITY);
    std::vector<float> tail_mask(n_tail*n_query, -INFINITY);
    for (int64_t iq = 0; iq < n_query; ++iq) {
        for (int64_t cell = 0; cell < n_kv; ++cell) {
            const bool visible = cell <= query_pos[iq] &&
                    !llama_hparams::is_masked_swa(n_swa, swa_type, llama_pos(cell), query_pos[iq]);
            bool exact = false;
            for (int64_t it = 0; it < n_tail; ++it) {
                if (tail_cells[it] == uint32_t(cell)) {
                    exact = true;
                    tail_mask[it + n_tail*iq] = visible ? 0.0f : -INFINITY;
                }
            }
            if (visible && !exact) {
                body_mask[cell + n_kv*iq] = 0.0f;
            }
        }
    }

    std::vector<llama_kv_tail_body_row> packed_rows;
    std::vector<llama_kv_tail_query_window> queries;
    for (uint32_t cell = 0; cell < n_kv; ++cell) {
        packed_rows.push_back({ llama_pos(cell), cell, int32_t(cell) });
    }
    for (uint32_t iq = 0; iq < n_query; ++iq) {
        queries.push_back({ iq, query_pos[iq] });
    }
    std::vector<uint8_t> scratch;
    llama_kv_tail_union_swa_rows(
            packed_rows, queries, n_swa, swa_type, true,
            [&](uint32_t iq, uint32_t cell) {
                return std::isfinite(body_mask[cell + n_kv*iq]);
            }, scratch);
    if (packed_rows.size() >= n_kv) {
        fail("SWA packed oracle did not reduce the physical body");
    }

    ggml_init_params params = { 16*1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * q = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, n_query);
    ggml_tensor * k_body = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, n_kv);
    ggml_tensor * v_body = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, n_kv);
    ggml_tensor * k_tail = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, n_tail);
    ggml_tensor * v_tail = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, n_tail);
    ggml_tensor * packed_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, packed_rows.size());
    ggml_tensor * mask_full = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_kv + n_tail, n_query);
    ggml_tensor * mask_packed = ggml_new_tensor_2d(
            ctx, GGML_TYPE_F32, packed_rows.size() + n_tail, n_query);
    ggml_tensor * k_packed = ggml_get_rows(ctx, k_body, packed_idxs);
    ggml_tensor * v_packed = ggml_get_rows(ctx, v_body, packed_idxs);

    const auto build_generic = [&](ggml_tensor * kb, ggml_tensor * vb, ggml_tensor * mask) {
        const int64_t n_body = kb->ne[1];
        ggml_tensor * scores = ggml_concat(
                ctx, ggml_mul_mat(ctx, kb, q), ggml_mul_mat(ctx, k_tail, q), 0);
        scores = ggml_soft_max_ext(ctx, scores, mask, 1.0f, 0.0f);
        ggml_tensor * wb = ggml_view_2d(ctx, scores, n_body, n_query, scores->nb[1], 0);
        ggml_tensor * wt = ggml_view_2d(
                ctx, scores, n_tail, n_query, scores->nb[1], n_body*scores->nb[0]);
        wt = ggml_cont(ctx, wt);
        ggml_tensor * body_out = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, vb)), wb);
        ggml_tensor * tail_out = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, v_tail)), wt);
        return ggml_add(ctx, body_out, tail_out);
    };
    ggml_tensor * ordinary = build_generic(k_body, v_body, mask_full);
    ggml_tensor * packed = build_generic(k_packed, v_packed, mask_packed);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ordinary);
    ggml_build_forward_expand(graph, packed);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        fail("SWA packed oracle allocation failed");
    }

    std::vector<float> q_data(d*n_query);
    std::vector<float> k_data(d*n_kv);
    std::vector<float> v_data(d*n_kv);
    for (int64_t iq = 0; iq < n_query; ++iq) {
        for (int64_t i = 0; i < d; ++i) {
            q_data[i + d*iq] = 0.006f*float(1 + i%7 + 2*iq);
        }
    }
    for (int64_t cell = 0; cell < n_kv; ++cell) {
        for (int64_t i = 0; i < d; ++i) {
            k_data[i + d*cell] = 0.009f*float(1 + i%5 + 3*cell);
            v_data[i + d*cell] = 0.013f*float(1 + i%11 + 2*cell);
        }
    }
    std::vector<float> kt_data(d*n_tail);
    std::vector<float> vt_data(d*n_tail);
    for (int64_t it = 0; it < n_tail; ++it) {
        std::copy_n(k_data.data() + d*tail_cells[it], d, kt_data.data() + d*it);
        std::copy_n(v_data.data() + d*tail_cells[it], d, vt_data.data() + d*it);
    }
    std::vector<int32_t> idx_data;
    std::vector<float> packed_mask((packed_rows.size() + n_tail)*n_query, -INFINITY);
    idx_data.reserve(packed_rows.size());
    for (const auto & row : packed_rows) {
        idx_data.push_back(int32_t(row.cell));
    }
    std::vector<float> full_mask((n_kv + n_tail)*n_query, -INFINITY);
    for (int64_t iq = 0; iq < n_query; ++iq) {
        for (int64_t cell = 0; cell < n_kv; ++cell) {
            full_mask[cell + (n_kv + n_tail)*iq] = body_mask[cell + n_kv*iq];
        }
        for (int64_t it = 0; it < n_tail; ++it) {
            full_mask[n_kv + it + (n_kv + n_tail)*iq] = tail_mask[it + n_tail*iq];
            packed_mask[packed_rows.size() + it + (packed_rows.size() + n_tail)*iq] =
                    tail_mask[it + n_tail*iq];
        }
        for (size_t row = 0; row < packed_rows.size(); ++row) {
            packed_mask[row + (packed_rows.size() + n_tail)*iq] =
                    body_mask[packed_rows[row].cell + n_kv*iq];
        }
    }
    ggml_backend_tensor_set(q, q_data.data(), 0, q_data.size()*sizeof(float));
    ggml_backend_tensor_set(k_body, k_data.data(), 0, k_data.size()*sizeof(float));
    ggml_backend_tensor_set(v_body, v_data.data(), 0, v_data.size()*sizeof(float));
    ggml_backend_tensor_set(k_tail, kt_data.data(), 0, kt_data.size()*sizeof(float));
    ggml_backend_tensor_set(v_tail, vt_data.data(), 0, vt_data.size()*sizeof(float));
    ggml_backend_tensor_set(packed_idxs, idx_data.data(), 0, idx_data.size()*sizeof(int32_t));
    ggml_backend_tensor_set(mask_full, full_mask.data(), 0, full_mask.size()*sizeof(float));
    ggml_backend_tensor_set(mask_packed, packed_mask.data(), 0, packed_mask.size()*sizeof(float));
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        fail("SWA packed oracle compute failed");
    }
    std::vector<float> expected(d*n_query);
    std::vector<float> actual(d*n_query);
    ggml_backend_tensor_get(ordinary, expected.data(), 0, expected.size()*sizeof(float));
    ggml_backend_tensor_get(packed, actual.data(), 0, actual.size()*sizeof(float));
    for (size_t i = 0; i < actual.size(); ++i) {
        if (std::fabs(actual[i] - expected[i]) > 1e-5f) {
            std::fprintf(stderr, "packed SWA %d mismatch at %zu: got %.8f expected %.8f\n",
                    int(swa_type), i, actual[i], expected[i]);
            std::exit(1);
        }
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

int main() {
    test_representation_topology();
    ggml_backend_load_all();
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend) {
        fail("failed to initialize CPU backend");
    }
    test_attention_graph(backend);
    test_shadow_roundtrip(backend);
    ggml_backend_free(backend);

    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu) {
        fail("failed to find CPU device for meta exact-tail test");
    }
    ggml_backend_dev_t logical_devices[] = { cpu, cpu, cpu };
    ggml_backend_dev_t meta_dev = ggml_backend_meta_device(
            logical_devices, 3, test_standard_meta_split, nullptr);
    ggml_backend_t meta_backend = ggml_backend_dev_init(meta_dev, nullptr);
    if (!meta_backend) {
        fail("failed to initialize meta backend for exact-tail test");
    }
    test_meta_shadow_roundtrip(meta_backend);
    ggml_backend_free(meta_backend);

    backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (backend) {
        ggml_backend_t reference = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        test_bounded_attached_tail_attention(backend, reference);
        ggml_backend_free(reference);
        test_attention_graph(backend);
        test_shadow_roundtrip(backend);
        test_fully_masked_quant_body(backend, GGML_TYPE_Q4_0);
        test_fully_masked_quant_body(backend, GGML_TYPE_Q8_0);
        // F32 exact sides always use ordinary SET_ROWS plus the generic merge,
        // even when FlashAttention was requested for the surrounding graph.
        test_mixed_side_generic_attention(
                backend, GGML_TYPE_Q4_0, GGML_TYPE_F32,
                GGML_TYPE_F16, GGML_TYPE_F32, true);
        test_mixed_side_generic_attention(
                backend, GGML_TYPE_F32, GGML_TYPE_Q4_0,
                GGML_TYPE_F32, GGML_TYPE_BF16, false);
        // Existing aligned half/bfloat mixed sides remain numerically valid.
        test_mixed_side_generic_attention(
                backend, GGML_TYPE_F16, GGML_TYPE_Q4_0,
                GGML_TYPE_F16, GGML_TYPE_BF16, false);
        test_sparse_swa_packed_oracle(backend, LLAMA_SWA_TYPE_STANDARD);
        test_sparse_swa_packed_oracle(backend, LLAMA_SWA_TYPE_CHUNKED);
        ggml_backend_free(backend);
    }
    return 0;
}
