#include "../ggml/src/ggml-vulkan/fattn-kvarn-route-policy.h"

#include <cstdio>

static int failures = 0;
#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); ++failures; \
} } while (0)

int main() {
    CHECK(ggml_vk_kvarn_shader_core_count(24, true, false) == 48);
    CHECK(ggml_vk_kvarn_shader_core_count(24, true, true) == 24);
    CHECK(ggml_vk_kvarn_shader_core_count(24, false, false) == 24);
    CHECK(ggml_vk_kvarn_shader_core_count(0, true, false) == 16);
    CHECK(ggml_vk_kvarn_shader_core_count(0, false, false) == 16);

    // The monolithic store launches one x workgroup per logical head.
    // Slices and tokens from all streams are serial work inside that CTA.
    CHECK(!ggml_vk_kvarn_store_parallel_preferred(128, 8, 1, 16));
    CHECK(ggml_vk_kvarn_store_parallel_preferred(129, 8, 1, 16));
    CHECK(ggml_vk_kvarn_store_parallel_preferred(256, 8, 1, 16));
    CHECK(ggml_vk_kvarn_store_parallel_preferred(384, 8, 1, 16));
    CHECK(!ggml_vk_kvarn_store_parallel_preferred(384, 32, 1, 16));
    CHECK(ggml_vk_kvarn_store_parallel_preferred(384, 31, 1, 16));
    CHECK(ggml_vk_kvarn_store_parallel_preferred(384, 8, 4, 16));
    CHECK(ggml_vk_kvarn_store_parallel_preferred(384, 8, 3, 16));
    CHECK(!ggml_vk_kvarn_store_parallel_preferred(384, 64, 1, 16));
    CHECK(!ggml_vk_kvarn_store_parallel_preferred(384, 8, 0, 16));

    CHECK(ggml_vk_kvarn_wht_full_head_preferred(true, 512, 32));
    CHECK(ggml_vk_kvarn_wht_full_head_preferred(true, 256, 0));
    CHECK(!ggml_vk_kvarn_wht_full_head_preferred(false, 256, 32));
    CHECK(!ggml_vk_kvarn_wht_full_head_preferred(false, 512, 64));
    CHECK(!ggml_vk_kvarn_wht_full_head_preferred(false, 256, 0));
    CHECK(ggml_vk_kvarn_wht_full_head_preferred(false, 256, 256));
    CHECK(ggml_vk_kvarn_wht_full_head_preferred(false, 128, 32));

    const auto intel = ggml_vk_fattn_kvarn_plan({4096, 1, 8, 8, 1,
        ggml_vk_kvarn_shader_core_count(24, true, false)});
    const auto xe2 = ggml_vk_fattn_kvarn_plan({4096, 1, 8, 8, 1,
        ggml_vk_kvarn_shader_core_count(24, true, true)});
    const auto unknown = ggml_vk_fattn_kvarn_plan({4096, 1, 8, 8, 1,
        ggml_vk_kvarn_shader_core_count(0, false, false)});
    CHECK(intel.split_k == 12);
    CHECK(xe2.split_k == 6);
    CHECK(unknown.split_k == 4);
    CHECK(ggml_vk_fattn_kvarn_plan({128, 1, 8, 8, 1, 48}).split_k == 1);
    CHECK(ggml_vk_fattn_kvarn_plan({129, 1, 8, 8, 1, 48}).split_k == 2);

    if (failures == 0) {
        std::puts("Vulkan KVarN host policy tests passed");
    }
    return failures != 0;
}
