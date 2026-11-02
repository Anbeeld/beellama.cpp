// SINQ numerical adaptation from huawei-csl/SINQ at 42ca83edf8fe1afc48c40fca6acc973c5d2bb59f.
// Copyright 2025 Huawei Technologies Switzerland AG, Apache-2.0.
// See licenses/LICENSE-SINQ and licenses/NOTICE-SINQ for attribution and changes.
#include "llama-sinq.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <thread>
#include <vector>

namespace {
bool valid_iterations(int iterations) { return iterations == 4 || iterations == 8 || iterations == 16; }
constexpr uint64_t workspace_limit = 128ull*1024*1024;
bool valid_shape(int64_t rows, int64_t columns, int width) {
    return rows >= 2 && llama_sinq_valid_strip_width(width) && columns > 0 && columns % width == 0 &&
           uint64_t(rows) <= std::numeric_limits<size_t>::max()/sizeof(float)/uint64_t(columns);
}
float clamp(float x, float lo, float hi) { return std::min(hi,std::max(lo,x)); }
float source_at(const void * source, ggml_type type, int64_t i) {
    if (type == GGML_TYPE_F16) { return ggml_fp16_to_fp32(static_cast<const ggml_fp16_t *>(source)[i]); }
    const float x = type == GGML_TYPE_F32 ? static_cast<const float *>(source)[i] :
        ggml_bf16_to_fp32(static_cast<const ggml_bf16_t *>(source)[i]);
    return ggml_fp16_to_fp32(ggml_fp32_to_fp16(x));
}
struct moments {
    float mean = 0, m2 = 0, count = 0;
    void add(float x) {
        count += 1; const float delta = x-mean;
        mean += delta/count;
        m2 = std::fma(delta,x-mean,m2);
    }
};
moments combine(moments a, moments b) {
    if (a.count == 0) { return b; }
    if (b.count == 0) { return a; }
    const float delta = b.mean-a.mean, count = a.count+b.count, ratio = b.count/count;
    return {std::fma(delta,ratio,a.mean), std::fma(delta*delta*a.count,ratio,a.m2+b.m2),count};
}
void std_axes(const std::vector<float> & a, int64_t n, int g, std::vector<float> & rows, std::vector<float> & cols) {
    // FP32 Welford, paired accumulators and CUDA-reference reduction ordering.
    int height = 1; while (height*2 <= n && height < 16) { height *= 2; }
    const int width = std::min(g,512/height);
    std::vector<moments> lanes(width);
    for (int64_t i = 0; i < n; ++i) {
        for (int lane = 0; lane < width; ++lane) {
            moments x, y;
            for (int j = lane, step = 0; j < g; j += width, ++step) { (step%2 ? y : x).add(a[i*g+j]); }
            lanes[lane] = combine(x,y);
        }
        int dim = width;
        while (dim > 32) {
            for (int lane = 0; lane < dim/2; ++lane) { lanes[lane] = combine(lanes[lane],lanes[lane+dim/2]); }
            dim /= 2;
        }
        for (int offset = 1; offset < dim; offset *= 2) {
            for (int lane = 0; lane+offset < dim; ++lane) { lanes[lane] = combine(lanes[lane],lanes[lane+offset]); }
        }
        rows[i] = std::sqrt(lanes[0].m2/float(g-1));
    }
    int col_height = 1;
    const int col_width = g/4;
    while (col_height*2 <= n && col_height*2 <= 128/col_width) { col_height *= 2; }
    const int col_lanes = n >= std::min(col_height*16,256) ? col_height : 1;
    std::vector<moments> col_moments(col_lanes);
    for (int j = 0; j < g; ++j) {
        for (int lane = 0; lane < col_lanes; ++lane) {
            moments x, y;
            for (int64_t i = lane, step = 0; i < n; i += col_lanes, ++step) { (step%2 ? y : x).add(a[i*g+j]); }
            col_moments[lane] = combine(x,y);
        }
        for (int offset = col_lanes/2; offset > 0; offset /= 2) {
            for (int lane = 0; lane < offset; ++lane) { col_moments[lane] = combine(col_moments[lane],col_moments[lane+offset]); }
        }
        cols[j] = std::sqrt(col_moments[0].m2/float(n-1));
    }
}
float imbalance(const std::vector<float> & rows, const std::vector<float> & cols) {
    float lo = std::numeric_limits<float>::infinity(), hi = 0;
    for (float x : rows) { lo = std::min(lo,x); hi = std::max(hi,x); }
    for (float x : cols) { lo = std::min(lo,x); hi = std::max(hi,x); }
    return hi/std::max(1.e-12f,lo);
}
} // namespace

bool llama_sinq_workspace_size(int64_t rows, int width, size_t & bytes) {
    if (!valid_shape(rows,width,width)) { return false; }
    // Includes caller input tile, balance_tile's current tile, all row/column
    // vectors and Welford scratch. Divide before multiplying to avoid overflow.
    const uint64_t fixed = uint64_t(width)*32+8192;
    const uint64_t per_row = uint64_t(width)*8+24;
    if (uint64_t(rows) > (workspace_limit-fixed)/per_row) { return false; }
    bytes = size_t(uint64_t(rows)*per_row+fixed);
    return true;
}

bool llama_sinq_balance_tile(const float * weights, int64_t n, int g, int iterations,
                            float * columns, float * row_factors, llama_sinq_tile_statistics * statistics) {
    size_t workspace;
    if (!weights || !columns || !row_factors || !valid_shape(n,g,g) || !valid_iterations(iterations) ||
        !llama_sinq_workspace_size(n,g,workspace)) { return false; }
    for (int64_t i = 0; i < n*g; ++i) { if (!std::isfinite(weights[i])) { return false; } }
    try {
        std::vector<float> cur(weights,weights+n*g), rows(n), cols(g), lr(n,0), lc(g,0), br(n,1), bc(g,1), mr(n), mc(g);
        std_axes(cur,n,g,rows,cols);
        for (float x : rows) { if (!std::isfinite(x)) { return false; } }
        for (float x : cols) { if (!std::isfinite(x)) { return false; } }
        float target = 1.e3f;
        for (float x : rows) { target = std::min(target,clamp(x,1.e-3f,1.e3f)); }
        for (float x : cols) { target = std::min(target,clamp(x,1.e-3f,1.e3f)); }
        target += 1.e-6f;
        float best = imbalance(rows,cols); bool stopped = false;
        llama_sinq_tile_statistics diagnostic;
        diagnostic.initial_imbalance = best;
        // Candidates are recorded before each update, including k=0. The last
        // update is never a candidate, matching pinned sinkhorn_log(matrix,16).
        for (int it = 0; it < iterations; ++it) {
            for (int64_t i = 0; i < n; ++i) { mr[i] = std::exp(lr[i]); }
            for (int j = 0; j < g; ++j) { mc[j] = std::exp(lc[j]); }
            for (int64_t i = 0; i < n; ++i) {
                for (int j = 0; j < g; ++j) { cur[i*g+j] = weights[i*g+j]/mc[j]/mr[i]; }
            }
            std_axes(cur,n,g,rows,cols);
            const float ib = imbalance(rows,cols);
            if (ib <= best) { best = ib; br = mr; bc = mc; diagnostic.chosen_iteration = it; }
            if (ib > best) { stopped = true; }
            if (!stopped) {
                for (int64_t i = 0; i < n; ++i) { lr[i] = clamp(lr[i]+std::log(clamp(clamp(rows[i],1.e-3f,1.e3f)/target,.7f,2.f)),-.3f,10.f); }
                for (int j = 0; j < g; ++j) { lc[j] = clamp(lc[j]+std::log(clamp(clamp(cols[j],1.e-3f,1.e3f)/target,.7f,2.f)),-.3f,10.f); }
            }
        }
        if (statistics) {
            diagnostic.best_imbalance = best;
            // The terminal update is observable but remains an unscored state.
            for (int64_t i = 0; i < n; ++i) { mr[i] = std::exp(lr[i]); }
            for (int j = 0; j < g; ++j) { mc[j] = std::exp(lc[j]); }
            for (int64_t i = 0; i < n; ++i) {
                for (int j = 0; j < g; ++j) { cur[i*g+j] = weights[i*g+j]/mc[j]/mr[i]; }
            }
            std_axes(cur,n,g,rows,cols);
            diagnostic.terminal_imbalance = imbalance(rows,cols);
            *statistics = diagnostic;
        }
        std::copy(bc.begin(),bc.end(),columns); std::copy(br.begin(),br.end(),row_factors);
        return true;
    } catch (const std::exception &) { return false; }
}

bool llama_sinq_compute_scales(const void * source, ggml_type source_type,
                              int64_t n, int64_t k, const llama_sinq_options & options,
                              int threads, std::vector<ggml_fp16_t> & scales,
                              llama_sinq_statistics & statistics, std::vector<float> * unrounded) {
    if (!source || !valid_shape(n,k,options.strip_width) || !valid_iterations(options.iterations) || threads <= 0 ||
        !std::isfinite(options.strength) || options.strength < 0.f || options.strength > 1.f ||
        (source_type != GGML_TYPE_F32 && source_type != GGML_TYPE_F16 && source_type != GGML_TYPE_BF16)) { return false; }
    size_t workspace = 0;
    if (!(options.identity || options.strength == 0.f) && !llama_sinq_workspace_size(n,options.strip_width,workspace)) { return false; }
    // Explicit identity must not bypass finite validation of the same
    // FP16-rounded source convention used by normalization.
    for (int64_t i = 0; i < n*k; ++i) {
        if (!std::isfinite(source_at(source,source_type,i))) { return false; }
    }
    llama_sinq_statistics result;
    result.identity = options.identity || options.strength == 0.f;
    try {
        std::vector<ggml_fp16_t> output(size_t(k),ggml_fp32_to_fp16(1.f));
        std::vector<float> raw_output(unrounded ? size_t(k) : 0,1.f);
        if (!result.identity) {
            const int g = options.strip_width;
            const int64_t groups = k/g;
            result.strips.resize(size_t(groups));
            std::atomic<int64_t> next{0}; std::atomic<bool> ok{true};
            auto work = [&]() {
                try {
                    std::vector<float> tile(n*g), t(g), r(n);
                    for (int64_t group; ok.load() && (group = next.fetch_add(1)) < groups;) {
                        for (int64_t i = 0; i < n; ++i) {
                            for (int j = 0; j < g; ++j) { tile[i*g+j] = source_at(source,source_type,i*k+group*g+j); }
                        }
                        if (!llama_sinq_balance_tile(tile.data(),n,g,options.iterations,t.data(),r.data(),&result.strips[size_t(group)])) { ok = false; break; }
                        for (int j = 0; j < g; ++j) {
                            // Preserve the exact legacy rounding path at strength1.
                            const float value = options.strength == 1.f ? t[j] : std::exp(options.strength*std::log(t[j]));
                            const ggml_fp16_t h = ggml_fp32_to_fp16(value);
                            const float f = ggml_fp16_to_fp32(h);
                            if (!(f > 0) || !std::isfinite(f)) { ok = false; break; }
                            output[group*g+j] = h;
                            if (unrounded) { raw_output[group*g+j] = value; }
                        }
                    }
                } catch (const std::exception &) { ok = false; }
            };
            // A full-height strip is indivisible. Its checked workspace fits
            // the hard limit; cap workers so their aggregate fits too.
            const int workers = int(std::min<int64_t>(groups,std::min<int64_t>(threads,
                workspace_limit/workspace)));
            std::vector<std::thread> pool;
            try { for (int i = 1; i < workers; ++i) { pool.emplace_back(work); } }
            catch (const std::exception &) { ok = false; }
            if (ok.load()) { work(); }
            for (auto & worker : pool) { worker.join(); }
            if (!ok.load()) { return false; }
        }
        result.min_scale = std::numeric_limits<float>::infinity(); result.max_scale = 0;
        for (ggml_fp16_t h : output) {
            const float f = ggml_fp16_to_fp32(h);
            result.min_scale = std::min(result.min_scale,f); result.max_scale = std::max(result.max_scale,f);
        }
        std::vector<float> ordered; ordered.reserve(output.size());
        for (ggml_fp16_t h : output) { ordered.push_back(ggml_fp16_to_fp32(h)); }
        std::sort(ordered.begin(),ordered.end());
        result.median_scale = ordered.size()%2 ? ordered[ordered.size()/2] :
            (ordered[ordered.size()/2-1]+ordered[ordered.size()/2])*.5f;
        for (const auto & strip : result.strips) { result.selected_identity_count += strip.chosen_iteration == 0; }
        if (unrounded) { unrounded->swap(raw_output); }
        scales.swap(output); statistics = std::move(result);
        return true;
    } catch (const std::exception &) { return false; }
}
