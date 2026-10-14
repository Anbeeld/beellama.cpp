// Exercise the Prism quantization ABI from C, including non-MSVC compilers.
#include "ggml-quants.h"

#include <stddef.h>
#include <stdio.h>

static_assert(QK_PQ2_0 == 128, "PQ2_0 group width changed");
static_assert(QK_PTQ1_0 == 128, "PTQ1_0 group width changed");
static_assert(sizeof(block_pq2_0) == 34, "PQ2_0 GGUF block layout changed");
static_assert(sizeof(block_ptq1_0) == 28, "PTQ1_0 GGUF block layout changed");
static_assert(offsetof(block_pq2_0, qs) == 2, "PQ2_0 code offset changed");
static_assert(offsetof(block_ptq1_0, d) == 26, "PTQ1_0 scale offset changed");

int main(void) {
    float input[2 * QK_PQ2_0];
    float output[2 * QK_PQ2_0];
    block_pq2_0 pq[2];
    block_ptq1_0 ptq[2];

    // Two independently scaled ternary blocks must round-trip exactly.
    for (int i = 0; i < 2 * QK_PQ2_0; ++i) {
        input[i] = (float) (i % 3 - 1) * (i < QK_PQ2_0 ? 1.0f : 2.0f);
    }

    quantize_row_pq2_0_ref(input, pq, 2 * QK_PQ2_0);
    dequantize_row_pq2_0(pq, output, 2 * QK_PQ2_0);
    for (int i = 0; i < 2 * QK_PQ2_0; ++i) {
        if (output[i] != input[i]) {
            fprintf(stderr, "PQ2_0 round-trip failed at %d\n", i);
            return 1;
        }
    }

    quantize_row_ptq1_0_ref(input, ptq, 2 * QK_PTQ1_0);
    dequantize_row_ptq1_0(ptq, output, 2 * QK_PTQ1_0);
    for (int i = 0; i < 2 * QK_PTQ1_0; ++i) {
        if (output[i] != input[i]) {
            fprintf(stderr, "PTQ1_0 round-trip failed at %d\n", i);
            return 1;
        }
    }

    return 0;
}
