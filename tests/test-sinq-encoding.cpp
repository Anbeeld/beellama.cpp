#include "sinq/sinq-native-codec.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); std::exit(1); } } while (0)
template<class F> static void rejects(F f) {
    bool caught = false;
    try { f(); } catch (const std::runtime_error &) { caught = true; }
    CHECK(caught);
}

static void importance_tests() {
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    float importance[] = {0, .37f, 1.13f, 9.17f};
    ggml_fp16_t scales[] = {ggml_fp32_to_fp16(.5f), ggml_fp32_to_fp16(1.3f),
                           ggml_fp32_to_fp16(2.7f), ggml_fp32_to_fp16(.7f)};
    float implicit[4], explicit_p2[4], p1[4], p15[4];
    sinq::transform_importance(importance, implicit, 4, scales);
    sinq::transform_importance(importance, explicit_p2, 4, scales, 2.0);
    sinq::transform_importance(importance, p1, 4, scales, 1.0);
    sinq::transform_importance(importance, p15, 4, scales, 1.5);
    bool different1 = false, different15 = false;
    for (int j = 0; j < 4; ++j) {
        const float u = ggml_fp16_to_fp32(scales[j]);
        const float old = importance[j]*u*u;
        CHECK(implicit[j] == old && explicit_p2[j] == old);
        CHECK(p1[j] == float(double(importance[j])*std::pow(double(u), 1.0)));
        CHECK(p15[j] == float(double(importance[j])*std::pow(double(u), 1.5)));
        different1 |= p1[j] != old;
        different15 |= p15[j] != old;
    }
    CHECK(different1 && different15);
    for (double p : {0.0, -1.0, -2.5, 1.73, 2.12}) {
        sinq::transform_importance(importance, implicit, 4, scales, p);
        for (int j = 0; j < 4; ++j) {
            CHECK(implicit[j] == float(double(importance[j])*std::pow(double(ggml_fp16_to_fp32(scales[j])), p)));
        }
    }
    rejects([&] { sinq::transform_importance(nullptr, implicit, 4, scales); });
    rejects([&] { sinq::transform_importance(importance, nullptr, 4, scales); });
    rejects([&] { sinq::transform_importance(importance, implicit, 4, nullptr); });
    rejects([&] { sinq::transform_importance(importance, implicit, 0, scales); });
    for (double p : {double(inf), double(nan)}) {
        rejects([&] { sinq::transform_importance(importance, implicit, 4, scales, p); });
    }
    for (float bad : {-1.f, nan, inf}) {
        importance[0] = bad;
        rejects([&] { sinq::transform_importance(importance, implicit, 4, scales); });
    }
    importance[0] = std::numeric_limits<float>::max();
    scales[0] = ggml_fp32_to_fp16(2);
    for (double p : {2.0, 1.5}) {
        rejects([&] { sinq::transform_importance(importance, implicit, 4, scales, p); });
    }
    importance[0] = 1;
    for (float bad : {0.f, -1.f, nan, inf}) {
        scales[0] = ggml_fp32_to_fp16(bad);
        rejects([&] { sinq::transform_importance(importance, implicit, 4, scales); });
    }
}

static void native_fixture(ggml_type codec) {
    const int64_t columns = 256, rows = 7;
    const size_t bytes = ggml_row_size(codec, columns);
    std::vector<float> source(rows*columns), importance(columns, 1), decoded(rows*columns);
    for (int64_t i = 0; i < rows*columns; ++i) {
        source[i] = float(std::sin(double(i)*.173) + .31*std::cos(double(i)*.037));
    }
    std::vector<uint8_t> payload(rows*bytes);
    std::vector<std::thread> workers;
    CHECK(sinq::native_encode(codec, source.data(), payload.data(), columns, 0, rows,
          rows, columns, importance.data(), workers, 1) == payload.size());
    const auto fixed_payload = payload;
    const auto * traits = ggml_get_type_traits(codec);
    for (int64_t r = 0; r < rows; ++r) {
        traits->to_float(payload.data() + r*bytes, decoded.data() + r*columns, columns);
    }
    // Independent column-wise LS oracle over a fixed native payload. The original
    // weights deliberately differ from the native decoder's input.
    std::vector<float> original(rows*columns);
    std::vector<double> oracle(columns);
    for (int64_t j = 0; j < columns; ++j) {
        double numerator = 0, denominator = 0;
        for (int64_t r = 0; r < rows; ++r) {
            const int64_t i = r*columns + j;
            original[i] = float(double(decoded[i])*(1.4 + double(j%13)*.03));
            numerator += double(original[i])*double(decoded[i]);
            denominator += double(decoded[i])*double(decoded[i]);
        }
        CHECK(denominator > 0);
        oracle[j] = numerator/denominator;
    }
    sinq::scale_refinement whole(columns), split(columns);
    whole.accumulate(original.data(), payload.data(), codec, rows);
    split.accumulate(original.data(), payload.data(), codec, 2);
    split.accumulate(original.data() + 2*columns, payload.data() + 2*bytes, codec, 1);
    split.accumulate(original.data() + 3*columns, payload.data() + 3*bytes, codec, rows - 3);
    std::vector<ggml_fp16_t> initial(columns);
    for (int64_t j = 0; j < columns; ++j) { initial[j] = ggml_fp32_to_fp16(.7f + float(j%9)*.025f); }
    for (double lambda : {0.0, .17, .25, .5, .63, 1.0}) {
        auto actual = initial, slab = initial;
        const auto stats = whole.apply(actual.data(), lambda);
        const auto split_stats = split.apply(slab.data(), lambda);
        CHECK(stats.invalid_ls_count == 0 && stats.invalid_blend_count == 0);
        CHECK(split_stats.invalid_ls_count == 0 && split_stats.invalid_blend_count == 0);
        CHECK(actual == slab);
        bool changed = false;
        for (int64_t j = 0; j < columns; ++j) {
            const double u0 = ggml_fp16_to_fp32(initial[j]);
            CHECK(actual[j] == ggml_fp32_to_fp16(float((1 - lambda)*u0 + lambda*oracle[j])));
            changed |= actual[j] != initial[j];
        }
        CHECK(lambda == 0 ? actual == initial : changed);
    }
    CHECK(payload == fixed_payload);
}

static void refinement_edges() {
    // F16 trait decoding makes edge-case LS values exact and independent of a
    // quantizer's zero representation or approximation error.
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    ggml_fp16_t b[] = {ggml_fp32_to_fp16(0), ggml_fp32_to_fp16(1), ggml_fp32_to_fp16(1),
                       ggml_fp32_to_fp16(1), ggml_fp32_to_fp16(1), ggml_fp32_to_fp16(1)};
    float w[] = {1, 0, -1, 70000, 1.e-20f, 2};
    sinq::scale_refinement edge(6);
    edge.accumulate(w, b, GGML_TYPE_F16, 1);
    std::vector<ggml_fp16_t> initial(6, ggml_fp32_to_fp16(1));
    auto scales = initial;
    auto stats = edge.apply(scales.data(), 0);
    CHECK(scales == initial && stats.invalid_ls_count == 0 && stats.invalid_blend_count == 0);
    stats = edge.apply(scales.data(), 1);
    CHECK(stats.invalid_ls_count == 5 && stats.invalid_blend_count == 0);
    for (int j = 0; j < 5; ++j) { CHECK(scales[j] == initial[j]); }
    CHECK(scales[5] == ggml_fp32_to_fp16(2));
    sinq::scale_refinement empty(6);
    scales = initial;
    CHECK(empty.apply(scales.data(), .5).invalid_ls_count == 6);
    CHECK(scales == initial);
    for (double lambda : {-1.0, 1.01, double(nan), double(inf)}) {
        rejects([&] { edge.apply(scales.data(), lambda); });
    }
    rejects([&] { edge.apply(nullptr, .5); });
    for (float bad : {0.f, -1.f, nan, inf}) {
        scales = initial;
        scales[0] = ggml_fp32_to_fp16(bad);
        rejects([&] { edge.apply(scales.data(), .5); });
        rejects([&] { edge.apply(scales.data(), 0); });
    }
    // Values beyond the F16 finite range retain u0 even if rounding would
    // happen to land back on the largest finite half.
    sinq::scale_refinement boundary(1);
    const float too_large=65510.f;
    const ggml_fp16_t one=ggml_fp32_to_fp16(1.f);
    boundary.accumulate(&too_large,&one,GGML_TYPE_F16,1);
    auto retained=one;
    CHECK(boundary.apply(&retained,.25).invalid_ls_count==1 && retained==one);
    rejects([] { sinq::scale_refinement bad(0); });
    rejects([] { sinq::scale_refinement bad(-1); });
    rejects([] { sinq::scale_refinement bad(std::numeric_limits<int64_t>::max()); });
    rejects([&] { edge.accumulate(nullptr, b, GGML_TYPE_F16, 1); });
    rejects([&] { edge.accumulate(w, nullptr, GGML_TYPE_F16, 1); });
    rejects([&] { edge.accumulate(w, b, GGML_TYPE_F16, 0); });
    rejects([&] { edge.accumulate(w, b, GGML_TYPE_F16, std::numeric_limits<int64_t>::max()); });
    rejects([&] { edge.accumulate(w, b, GGML_TYPE_COUNT, 1); });
    rejects([&] { edge.accumulate(w, b, GGML_TYPE_Q4_K, 1); });
    for (float bad : {nan, inf}) {
        w[0] = bad;
        rejects([&] { sinq::scale_refinement(6).accumulate(w, b, GGML_TYPE_F16, 1); });
        w[0] = 1;
        b[0] = ggml_fp32_to_fp16(bad);
        rejects([&] { sinq::scale_refinement(6).accumulate(w, b, GGML_TYPE_F16, 1); });
        b[0] = ggml_fp32_to_fp16(0);
    }
}

int main() {
    ggml_context * ctx = ggml_init({1024, nullptr, true});
    CHECK(ctx);
    ggml_free(ctx);
    importance_tests();
    for (ggml_type codec : {GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_S,
                           GGML_TYPE_Q4_K, GGML_TYPE_Q6_K}) {
        native_fixture(codec);
    }
    refinement_edges();
    std::puts("SINQ importance and fixed-payload scale refinement tests passed");
    return 0;
}
