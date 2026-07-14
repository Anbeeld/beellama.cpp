#include "ggml.h"
#include "ggml-backend.h"
#include "llama-kv-cache-tail.h"

#include <cmath>
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
    backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (backend) {
        test_attention_graph(backend);
        test_shadow_roundtrip(backend);
        test_fully_masked_quant_body(backend, GGML_TYPE_Q4_0);
        test_fully_masked_quant_body(backend, GGML_TYPE_Q8_0);
        ggml_backend_free(backend);
    }
    return 0;
}
