#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-impl.h"
#include "llama-sinq.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

static void check(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}

static void run_case(ggml_backend_t backend, const char * name, ggml_type type,
                     int k, int width, int tokens, int batches, bool four_dimensional,
                     bool with_bias, bool identity, bool materialize_product = true) {
    const int n = 19;
    std::vector<float> source(n*k), divided(n*k), input(k*tokens*batches), product(input.size());
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            source[i*k+j] = float(std::sin(i*3.17+j*.73)*(1+(j%17)*.19)*(1+i*.11)+.031);
        }
    }
    llama_sinq_options options;
    options.strip_width = width;
    options.gate = false;
    options.identity = identity;
    llama_sinq_statistics statistics;
    std::vector<ggml_fp16_t> scales;
    check(llama_sinq_compute_scales(source.data(), GGML_TYPE_F32, n, k, options, 2, scales, statistics),
          "optimizer failed");
    check(scales.size() == size_t(k) && statistics.identity == identity, "optimizer scale metadata");
    bool nonidentity = false;
    for (int j = 0; j < k; ++j) {
        const float t = ggml_fp16_to_fp32(scales[j]);
        check(std::isfinite(t) && t > 0, "invalid serialized scale");
        nonidentity = nonidentity || t != 1.f;
        if (identity) { check(t == 1.f, "identity scale changed"); }
        for (int i = 0; i < n; ++i) { divided[i*k+j] = source[i*k+j]/t; }
    }
    check(identity || nonidentity, "fixture did not exercise nonidentity scaling");
    std::vector<uint8_t> packed(n*ggml_row_size(type, k));
    const std::vector<float> importance(k,1.f);
    const float * weights=ggml_quantize_requires_imatrix(type) ? importance.data() : nullptr;
    check(ggml_quantize_chunk(type, divided.data(), packed.data(), 0, n, k, weights) == packed.size(),
          "ordinary QK quantization failed");
    if (identity) {
        std::vector<uint8_t> original(packed.size());
        check(ggml_quantize_chunk(type, source.data(), original.data(), 0, n, k, weights) == original.size(),
              "original ordinary QK quantization failed");
        check(original == packed, "identity t changed original ordinary QK bytes");
    }

    auto * ctx = ggml_init({64*ggml_tensor_overhead()+2*ggml_graph_overhead(), nullptr, true});
    check(ctx != nullptr, "context allocation");
    auto * w = ggml_new_tensor_2d(ctx, type, k, n);
    auto * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, k);
    // Both a 1D companion and a 2D weight broadcast across these batch axes.
    const int dim2 = four_dimensional ? std::min(2, batches) : batches;
    const int dim3 = batches/dim2;
    auto * x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, k, tokens, dim2, dim3);
    auto * precomputed = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, k, tokens, dim2, dim3);
    auto * scaled = ggml_mul(ctx, x, ggml_cast(ctx, t, GGML_TYPE_F32));
    // Observable intermediates must not be elided by backend fusion.
    if (materialize_product) { ggml_set_output(scaled); }
    auto * actual = ggml_mul_mat(ctx, w, scaled);
    // Independent control: no graph cast, multiply, or scale-graph helper.
    // The very same ordinary QK tensor and backend kernel consume host t*x.
    auto * expected = ggml_mul_mat(ctx, w, precomputed);
    check(ggml_backend_supports_op(backend, actual) && ggml_backend_supports_op(backend, expected),
          "backend rejects ordinary QK matmul");
    std::vector<float> bias_values(n);
    ggml_tensor * bias = nullptr;
    if (with_bias) {
        bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        for (int i = 0; i < n; ++i) { bias_values[i] = float(i%7)*.125f; }
        actual = ggml_add(ctx, actual, bias);
        expected = ggml_add(ctx, expected, bias);
    }
    auto * graph = ggml_new_graph(ctx);
    auto * control = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, actual);
    ggml_build_forward_expand(control, expected);
    // Direct backend calls bypass scheduler UID assignment. Keep both IDs
    // stable through warmup, CUDA capture, and replay, including changed data.
    graph->uid = ggml_graph_next_uid();
    control->uid = ggml_graph_next_uid();
    const auto graph_uid = graph->uid;
    const auto control_uid = control->uid;
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    check(buffer != nullptr, "backend buffer allocation");
    ggml_backend_tensor_set(w, packed.data(), 0, packed.size());
    ggml_backend_tensor_set(t, scales.data(), 0, scales.size()*sizeof(ggml_fp16_t));
    if (bias) { ggml_backend_tensor_set(bias, bias_values.data(), 0, bias_values.size()*sizeof(float)); }
    std::vector<float> observed(n*tokens*batches), reference(observed.size()), backend_product(product.size());
    std::vector<uint8_t> readback(packed.size());
    for (int repeat = 0; repeat < 3; ++repeat) {
        for (size_t i = 0; i < input.size(); ++i) {
            input[i] = float(std::cos(.13*i+.47*repeat))*float(1+i%7);
            // Serialized F16 t, promoted to F32, followed by one F32 multiply.
            product[i] = input[i]*ggml_fp16_to_fp32(scales[i%k]);
        }
        if (identity) { check(input == product, "identity changed host activations"); }
        ggml_backend_tensor_set(x, input.data(), 0, input.size()*sizeof(float));
        ggml_backend_tensor_set(precomputed, product.data(), 0, product.size()*sizeof(float));
        check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "scale graph compute");
        ggml_backend_tensor_get(actual, observed.data(), 0, observed.size()*sizeof(float));
        if (materialize_product) {
        ggml_backend_tensor_get(scaled, backend_product.data(), 0, backend_product.size()*sizeof(float));
        // With identical stock-kernel inputs, activation quantization/conversion
        // error is shared, not grounds for widening a float-dot tolerance.
        check(backend_product == product, "backend activation product differs from independent host product");
        for (int j=0;j<k;++j) {
            double original_moment=0,recaptured_moment=0;
            for (size_t i=size_t(j);i<input.size();i+=size_t(k)) {
                original_moment+=double(input[i])*input[i];
                recaptured_moment+=double(backend_product[i])*backend_product[i];
            }
            const double factor=ggml_fp16_to_fp32(scales[j]);
            const double transformed_moment=factor*factor*original_moment;
            check(std::abs(recaptured_moment-transformed_moment)<=5e-7*std::max(1.,transformed_moment),
                  "backend recaptured moments differ from t^2 I channel orientation");
        }
        }
        check(ggml_backend_graph_compute(backend, control) == GGML_STATUS_SUCCESS, "control graph compute");
        ggml_backend_tensor_get(expected, reference.data(), 0, reference.size()*sizeof(float));
        float max_error = 0;
        for (size_t i = 0; i < observed.size(); ++i) {
            const float error = std::abs(observed[i]-reference[i]);
            max_error = std::max(max_error, error);
            // Same QK bytes, shape, backend, and F32 activations: allow only
            // eight F32 epsilons, not error against an unquantized scalar dot.
            const float tolerance = 8*std::numeric_limits<float>::epsilon()*std::max(1.f, std::abs(reference[i]));
            if (!std::isfinite(observed[i]) || !std::isfinite(reference[i]) || error > tolerance) {
                std::fprintf(stderr, "index=%zu actual=%.9g expected=%.9g tolerance=%g\n",
                             i, observed[i], reference[i], tolerance);
                check(false, "same-stock-QK backend numeric mismatch");
            }
        }
        check(graph->uid == graph_uid && control->uid == control_uid, "graph UID changed during replay");
        ggml_backend_tensor_get(w, readback.data(), 0, readback.size());
        check(readback == packed, "backend changed ordinary QK weight bytes");
        std::printf("%s %s K=%d strip=%d tokens=%d batches=%d dims=%dx%d bias=%d identity=%d replay=%d max_abs=%g\n",
                    name, ggml_type_name(type), k, width, tokens, batches, dim2, dim3,
                    with_bias, identity, repeat, max_error);
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

static void run_lookup_and_head(ggml_backend_t backend, ggml_type type, int k) {
    constexpr int rows = 33, count = 4;
    std::vector<float> original(rows*k), decoded(rows*k), input(k*count,0.f);
    for (size_t i = 0; i < original.size(); ++i) { original[i] = float(std::sin(.19*i)*2+.1); }
    std::vector<uint8_t> packed(rows*ggml_row_size(type,k));
    const std::vector<float> importance(k,1.f);
    check(ggml_quantize_chunk(type,original.data(),packed.data(),0,rows,k,ggml_quantize_requires_imatrix(type) ? importance.data() : nullptr)==packed.size(),"head/embedding encoding");
    ggml_get_type_traits(type)->to_float(packed.data(),decoded.data(),rows*k);
    const int columns[count] = {0,k-1,k/2,17};
    for (int j = 0; j < count; ++j) { input[j*k+columns[j]] = 1.f; }
    auto * ctx = ggml_init({32*ggml_tensor_overhead()+ggml_graph_overhead(),nullptr,true});
    check(ctx!=nullptr,"head/embedding context");
    auto * weight = ggml_new_tensor_2d(ctx,type,k,rows);
    auto * ids = ggml_new_tensor_1d(ctx,GGML_TYPE_I32,count);
    auto * x = ggml_new_tensor_2d(ctx,GGML_TYPE_F32,k,count);
    auto * lookup = ggml_get_rows(ctx,weight,ids);
    auto * head = ggml_mul_mat(ctx,weight,x);
    check(weight->type==type && lookup->src[0]==weight && head->src[0]==weight,"native head/embedding source promoted");
    check(ggml_backend_supports_op(backend,lookup) && ggml_backend_supports_op(backend,head),"native head/embedding unsupported");
    auto * graph = ggml_new_graph(ctx); graph->uid = ggml_graph_next_uid();
    ggml_build_forward_expand(graph,lookup); ggml_build_forward_expand(graph,head);
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx,backend); check(buffer!=nullptr,"head/embedding buffer");
    ggml_backend_tensor_set(weight,packed.data(),0,packed.size());
    ggml_backend_tensor_set(x,input.data(),0,input.size()*sizeof(float));
    std::vector<float> selected(k*count),logits(rows*count);
    for (int repeat = 0; repeat < 3; ++repeat) {
        const int32_t indices[count] = {repeat,rows-1,repeat,17};
        ggml_backend_tensor_set(ids,indices,0,sizeof(indices));
        check(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"head/embedding graph");
        ggml_backend_tensor_get(lookup,selected.data(),0,selected.size()*sizeof(float));
        ggml_backend_tensor_get(head,logits.data(),0,logits.size()*sizeof(float));
        for (int j = 0; j < count; ++j) {
            for (int i = 0; i < k; ++i) {
                const float expected = decoded[indices[j]*k+i];
                check(std::isfinite(selected[j*k+i]) && std::abs(selected[j*k+i]-expected)<=8*std::numeric_limits<float>::epsilon()*std::max(1.f,std::abs(expected)),"embedding lookup differs from independent native row decode");
            }
            for (int i = 0; i < rows; ++i) {
                const float expected = decoded[i*k+columns[j]];
                // One-hot inputs isolate an encoded weight coordinate. Native
                // CUDA Q8_1 activation scales round to F16, unlike CPU Q8_K.
                check(std::isfinite(logits[j*rows+i]) && std::abs(logits[j*rows+i]-expected)<=.001f*std::max(1.f,std::abs(expected)),"head differs from decoded one-hot reference");
            }
        }
    }
    std::vector<uint8_t> readback(packed.size()); ggml_backend_tensor_get(weight,readback.data(),0,readback.size());
    check(readback==packed,"head/embedding changed quantized weight bytes");
    ggml_backend_buffer_free(buffer); ggml_free(ctx);
}

static void run_scaled_glu(ggml_backend_t backend, ggml_type type, bool observe_intermediate, bool custom_oai = false) {
    constexpr int k = 512, n = 128;
    std::vector<float> weights(k*n), x(k), a(k), b(k), xa(k), xb(k), importance(k,1.f);
    for (int j=0;j<k;++j) {
        x[j] = std::sin(.17f*j)*3;
        a[j] = .25f + (j%13)*.125f;
        b[j] = .5f + (j%17)*.0625f;
        xa[j] = x[j]*a[j]; xb[j] = x[j]*b[j];
    }
    for (int j=0;j<k*n;++j) { weights[j] = std::cos(.07f*j); }
    std::vector<uint8_t> packed(n*ggml_row_size(type,k));
    check(ggml_quantize_chunk(type,weights.data(),packed.data(),0,n,k,
          ggml_quantize_requires_imatrix(type) ? importance.data() : nullptr)==packed.size(),"GLU encoding");
    auto * ctx = ggml_init({32*ggml_tensor_overhead()+2*ggml_graph_overhead(),nullptr,true});
    auto * w = ggml_new_tensor_2d(ctx,type,k,n);
    auto * v = ggml_new_tensor_2d(ctx,type,k,n);
    auto * input = ggml_new_tensor_2d(ctx,GGML_TYPE_F32,k,1);
    auto * sa = ggml_new_tensor_1d(ctx,GGML_TYPE_F32,k);
    auto * sb = ggml_new_tensor_1d(ctx,GGML_TYPE_F32,k);
    auto * pa = ggml_new_tensor_2d(ctx,GGML_TYPE_F32,k,1);
    auto * pb = ggml_new_tensor_2d(ctx,GGML_TYPE_F32,k,1);
    auto * scaled_a = ggml_mul(ctx,input,sa);
    auto * scaled_b = ggml_mul(ctx,input,sb);
    if (observe_intermediate) { ggml_set_output(scaled_a); }
    auto * gate = ggml_mul_mat(ctx,w,scaled_a);
    auto * up = ggml_mul_mat(ctx,v,scaled_b);
    auto * gate_control = ggml_mul_mat(ctx,w,pa);
    auto * up_control = ggml_mul_mat(ctx,v,pb);
    auto * actual = custom_oai ? ggml_swiglu_oai(ctx,gate,up,1.f,2.f) : ggml_swiglu_split(ctx,gate,up);
    auto * expected = custom_oai ? ggml_swiglu_oai(ctx,gate_control,up_control,1.f,2.f) : ggml_swiglu_split(ctx,gate_control,up_control);
    auto * graph = ggml_new_graph(ctx); graph->uid = ggml_graph_next_uid();
    auto * control = ggml_new_graph(ctx); control->uid = ggml_graph_next_uid();
    ggml_build_forward_expand(graph,actual); ggml_build_forward_expand(control,expected);
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx,backend); check(buffer!=nullptr,"GLU buffer");
    ggml_backend_tensor_set(w,packed.data(),0,packed.size());
    ggml_backend_tensor_set(v,packed.data(),0,packed.size());
    ggml_backend_tensor_set(sa,a.data(),0,k*sizeof(float));
    ggml_backend_tensor_set(sb,b.data(),0,k*sizeof(float));
    std::vector<float> got(n),ref(n),readback(k);
    for (int repeat=0;repeat<3;++repeat) {
        for (int j=0;j<k;++j) { x[j]=std::sin(.17f*j+.31f*repeat)*3; xa[j]=x[j]*a[j]; xb[j]=x[j]*b[j]; }
        ggml_backend_tensor_set(input,x.data(),0,k*sizeof(float));
        ggml_backend_tensor_set(pa,xa.data(),0,k*sizeof(float));
        ggml_backend_tensor_set(pb,xb.data(),0,k*sizeof(float));
        check(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"scaled GLU graph");
        check(ggml_backend_graph_compute(backend,control)==GGML_STATUS_SUCCESS,"scaled GLU control");
        ggml_backend_tensor_get(actual,got.data(),0,n*sizeof(float));
        ggml_backend_tensor_get(expected,ref.data(),0,n*sizeof(float));
        for (int j=0;j<n;++j) { check(std::isfinite(got[j]) && std::abs(got[j]-ref[j])<=8*std::numeric_limits<float>::epsilon()*std::max(1.f,std::abs(ref[j])),"scaled GLU differs from host-product oracle"); }
        if (observe_intermediate) {
            ggml_backend_tensor_get(scaled_a,readback.data(),0,k*sizeof(float));
            check(readback==xa,"observed GLU intermediate elided");
        }
    }
    std::printf("scaled GLU %s observable=%d replay passed\n",ggml_type_name(type),observe_intermediate);
    ggml_backend_buffer_free(buffer); ggml_free(ctx);
}

static void run_scale_layout(ggml_backend_t backend, bool column_scale, bool offset_x, bool offset_scale) {
    constexpr int k=256, tokens=256, n=128;
    auto * ctx=ggml_init({32*ggml_tensor_overhead()+2*ggml_graph_overhead(),nullptr,true});
    auto * w=ggml_new_tensor_2d(ctx,GGML_TYPE_Q4_K,k,n);
    auto * xp=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,k*tokens+1);
    auto * sp=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,k+1);
    auto * x=ggml_reshape_2d(ctx,ggml_view_1d(ctx,xp,k*tokens,offset_x ? sizeof(float) : 0),k,tokens);
    auto * scale=ggml_view_1d(ctx,sp,k,offset_scale ? sizeof(float) : 0);
    if (column_scale) { scale=ggml_reshape_2d(ctx,scale,1,k); }
    auto * host_product=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,k,tokens);
    auto * actual=ggml_mul_mat(ctx,w,ggml_mul(ctx,x,scale));
    auto * expected=ggml_mul_mat(ctx,w,host_product);
    auto * graph=ggml_new_graph(ctx);graph->uid=ggml_graph_next_uid();
    auto * control=ggml_new_graph(ctx);control->uid=ggml_graph_next_uid();
    ggml_build_forward_expand(graph,actual);ggml_build_forward_expand(control,expected);
    auto * buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);check(buffer!=nullptr,"layout buffer");
    std::vector<float> weights(k*n),input(k*tokens),s(k),product(k*tokens),got(n*tokens),ref(n*tokens);
    for (int j=0;j<k*n;++j) { weights[j]=std::sin(.017f*j); }
    for (int j=0;j<k;++j) { s[j]=.25f+(j%29)*.0625f; }
    for (int j=0;j<k*tokens;++j) { input[j]=std::cos(.19f*j);product[j]=input[j]*s[column_scale ? j/k : j%k]; }
    std::vector<uint8_t> packed(n*ggml_row_size(GGML_TYPE_Q4_K,k));
    check(ggml_quantize_chunk(GGML_TYPE_Q4_K,weights.data(),packed.data(),0,n,k,nullptr)==packed.size(),"layout encoding");
    ggml_backend_tensor_set(w,packed.data(),0,packed.size());
    ggml_backend_tensor_set(x,input.data(),0,input.size()*sizeof(float));
    ggml_backend_tensor_set(scale,s.data(),0,s.size()*sizeof(float));
    ggml_backend_tensor_set(host_product,product.data(),0,product.size()*sizeof(float));
    check(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"layout graph");
    check(ggml_backend_graph_compute(backend,control)==GGML_STATUS_SUCCESS,"layout control");
    ggml_backend_tensor_get(actual,got.data(),0,got.size()*sizeof(float));
    ggml_backend_tensor_get(expected,ref.data(),0,ref.size()*sizeof(float));
    for (size_t j=0;j<got.size();++j) { check(std::isfinite(got[j]) && std::abs(got[j]-ref[j])<=8*std::numeric_limits<float>::epsilon()*std::max(1.f,std::abs(ref[j])),"scale broadcast/alignment differs from host-product oracle"); }
    std::printf("scale layout column=%d offset_x=%d offset_scale=%d passed\n",column_scale,offset_x,offset_scale);
    ggml_backend_buffer_free(buffer);ggml_free(ctx);
}

int main(int argc, char ** argv) try {
    check(argc <= 2, "usage: test-qk-sinq-backend [CPU|CUDA0]");
    const char * name = argc > 1 ? argv[1] : "CPU";
    check(std::strcmp(name, "CPU") == 0 || std::strcmp(name, "CUDA0") == 0, "expected CPU or CUDA0");
    // Initialize CPU FP16 conversion tables before running the optimizer.
    auto * init = ggml_init({1024, nullptr, true});
    check(init != nullptr, "conversion table initialization");
    ggml_free(init);
    ggml_backend_load_all();
    auto * dev = ggml_backend_dev_by_name(name);
    check(dev != nullptr, "requested backend unavailable");
    auto * backend = ggml_backend_dev_init(dev, nullptr);
    check(backend != nullptr, "backend initialization");
    for (auto type : { GGML_TYPE_Q3_K, GGML_TYPE_Q4_K }) {
      for (int k : { 256, 4096 }) { run_lookup_and_head(backend,type,k); }
      for (int k : { 256, 512 }) {
       for (int width : { 64, 128 }) {
        for (int tokens : { 1, 4, 17, 128 }) {
         for (int batches : { 1, 2, 4 }) {
          for (bool four_dimensional : { false, true }) {
           for (bool with_bias : { false, true }) {
            for (bool identity : { false, true }) {
                run_case(backend, name, type, k, width, tokens, batches, four_dimensional, with_bias, identity);
            }
           }
          }
         }
        }
       }
      }
    }
    for (auto type : {GGML_TYPE_IQ1_S,GGML_TYPE_IQ1_M,GGML_TYPE_IQ2_XXS,GGML_TYPE_IQ2_XS,
                      GGML_TYPE_IQ2_S,GGML_TYPE_IQ3_XXS,GGML_TYPE_IQ3_S,GGML_TYPE_IQ4_XS,
                      GGML_TYPE_IQ4_NL,GGML_TYPE_Q5_0,GGML_TYPE_Q5_1,GGML_TYPE_Q6_K,GGML_TYPE_Q8_0}) {
        for (int k : {256,4096}) { run_lookup_and_head(backend,type,k); }
        for (int k : {4096,12288}) { run_case(backend,name,type,k,64,4,2,false,true,false); }
        for (bool identity : {false,true}) {
            for (int tokens : {1,4,128}) { run_case(backend,name,type,256,64,tokens,2,false,true,identity); }
        }
    }
    // Characterize fused input scaling against independently precomputed F32
    // activations, including tail padding, batched shapes and CUDA graph replay.
    for (auto type : {GGML_TYPE_Q3_K,GGML_TYPE_Q4_K,GGML_TYPE_Q6_K,GGML_TYPE_Q8_0,
                      GGML_TYPE_IQ1_S,GGML_TYPE_IQ2_XXS,GGML_TYPE_IQ4_XS}) {
        for (int tokens : {1,4,17,128}) {
            for (int batches : {1,4}) {
                run_case(backend,name,type,256,64,tokens,batches,true,false,false,false);
            }
        }
    }
    for (auto type : {GGML_TYPE_Q3_K,GGML_TYPE_Q4_K,GGML_TYPE_Q6_K,GGML_TYPE_IQ1_S,GGML_TYPE_IQ2_XXS,GGML_TYPE_IQ4_XS}) {
        run_scaled_glu(backend,type,false);
        run_scaled_glu(backend,type,true);
    }
    run_scale_layout(backend,true,false,false);
    run_scale_layout(backend,false,false,true);
    run_scale_layout(backend,false,true,false);
    run_scaled_glu(backend,GGML_TYPE_Q4_K,false,true);
    ggml_backend_free(backend);
    ggml_quantize_free();
    std::puts("QK SINQ backend tests passed");
    return 0;
} catch (const std::exception & e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
}
