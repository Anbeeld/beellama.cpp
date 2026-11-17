#include "../ggml/src/ggml-cuda/fattn-kvarn-geometry-policy.h"
#include <cstdio>
#include <cstdlib>
#include <initializer_list>

static void require(bool ok, const char * message) {
    if (!ok) { std::fprintf(stderr, "%s\n", message); std::exit(1); }
}

int main() {
    for (int cc : { 860, 870, 890, 1100, 1200, 1210 }) {
        require(ggml_cuda_kvarn_min_blocks(cc, 256, 2) == 4,
                "1536-thread SMs preserve the four-block register target");
        require(ggml_cuda_kvarn_min_blocks(cc, 512, 2) == 3,
                "512-thread bound must fit the 1536-thread ceiling");
        require(ggml_cuda_kvarn_min_blocks(cc, 1024, 2) == 1,
                "1024-thread bound must fit the 1536-thread ceiling");
    }
    for (int cc : { 750, 800, 880, 900, 1000, 1030, 0, 1300, 1000000 }) {
        require(ggml_cuda_kvarn_min_blocks(cc, 256, 2) == 2,
                "other and unknown architectures retain the upstream target");
        require(ggml_cuda_kvarn_min_blocks(cc, 512, 1) == 1,
                "fallback preserves the head-specific upstream target");
    }
    require(ggml_cuda_kvarn_geometry_work(1, 64, 256, 8, 1) <
            ggml_cuda_kvarn_geometry_work(1, 128, 256, 8, 1), "equal waves favor less work");
    require(ggml_cuda_kvarn_geometry_work(8, 64, 256, 8, 1) >
            ggml_cuda_kvarn_geometry_work(4, 128, 256, 8, 1), "enough parallelism amortizes CTA setup");
    require(ggml_cuda_kvarn_geometry_work(4, 64, 256, 8, 1) >
            ggml_cuda_kvarn_geometry_work(4, 64, 256, 16, 1), "warp geometry changes tile work");
    require(ggml_cuda_kvarn_geometry_better(80, 77, 256, 512, 78, 90),
            "upstream's 95-percent efficiency band should favor half the partial traffic");
    require(!ggml_cuda_kvarn_geometry_better(84, 77, 256, 512, 78, 90),
            "traffic preference must not hide substantially greater tile work");
    require(ggml_cuda_kvarn_geometry_better(70, 80, 512, 256, 90, 78),
            "substantially less work overrides partial traffic");
    require(!ggml_cuda_kvarn_geometry_better(40, 44, 128, 256, 78, 78, false, true),
            "retain enough independent work before amortizing CTA setup");
    require(ggml_cuda_kvarn_geometry_better(80, 77, 256, 512, 78, 90, true, true),
            "once populated, lower partial traffic may win the efficiency band");
    return 0;
}
