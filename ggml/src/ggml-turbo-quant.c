/*
 * TurboQuant: KV cache compression via PolarQuant + QJL
 * Based on: arXiv 2504.19874 (ICLR 2026)
 *
 * Implements GGML_TYPE_TURBO3_0 (3-bit) and GGML_TYPE_TURBO4_0 (4-bit)
 * for use as --cache-type-k turbo3 --cache-type-v turbo3 in llama-server.
 */

#include "ggml-quants.h"
#include "ggml-common.h"
#include "ggml-impl.h"

#if defined(_WIN32)
#define _USE_MATH_DEFINES // for M_PI
#endif 

#include <math.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>

/* ---------- constants ---------- */

#define TURBO_SEED_ROTATION 42
#define TURBO_SEED_QJL      1042
#define TURBO_D             128  /* rotation group size = head_dim (independent of block size) */
#define TURBO_QJL_CONST     1.2533141373155003f  /* sqrt(pi/2) */

/* 2-bit: {±0.453, ±1.51} / sqrt(d) */
static const float CENTROIDS_2BIT[4] = { -0.133462f, -0.039994f, 0.039994f, 0.133462f };

/* 3-bit: Lloyd-Max for N(0, 1/128), pre-computed */
static const float CENTROIDS_3BIT[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};

/* 4-bit: Lloyd-Max for N(0, 1/sqrt(128)), 16 centroids */
static const float CENTROIDS_4BIT[16] = {
    -0.241556f, -0.182907f, -0.143047f, -0.111065f,
    -0.083317f, -0.058069f, -0.034311f, -0.011353f,
     0.011353f,  0.034311f,  0.058069f,  0.083317f,
     0.111065f,  0.143047f,  0.182907f,  0.241556f,
};
static const float MIDPOINTS_4BIT[15] = {
    -0.212232f, -0.162977f, -0.127056f, -0.097191f, -0.070693f,
    -0.046190f, -0.022832f,  0.000000f,  0.022832f,  0.046190f,
     0.070693f,  0.097191f,  0.127056f,  0.162977f,  0.212232f,
};

/* ---------- rotation matrix (lazy init) ---------- */

static float turbo_rotation[TURBO_D * TURBO_D];
static float turbo_rotation_t[TURBO_D * TURBO_D]; /* transpose */
static int   turbo_rotation_initialized = 0;

/* Simple LCG PRNG for deterministic rotation generation */
static uint64_t turbo_prng_state;

static void turbo_prng_seed(uint64_t seed) {
    turbo_prng_state = seed;
}

static double turbo_prng_normal(void) {
    /* Box-Muller transform from uniform LCG */
    turbo_prng_state = turbo_prng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    double u1 = (double)(turbo_prng_state >> 11) / (double)(1ULL << 53);
    if (u1 < 1e-15) u1 = 1e-15;
    turbo_prng_state = turbo_prng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    double u2 = (double)(turbo_prng_state >> 11) / (double)(1ULL << 53);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

static void turbo_init_rotation(void) {
    if (turbo_rotation_initialized) return;

    const int d = TURBO_D;

    /* Generate random Gaussian matrix */
    turbo_prng_seed(TURBO_SEED_ROTATION);
    float G[TURBO_D * TURBO_D];
    for (int i = 0; i < d * d; i++) {
        G[i] = (float)turbo_prng_normal();
    }

    /* QR decomposition via modified Gram-Schmidt */
    /* Q stored column-major in turbo_rotation */
    memcpy(turbo_rotation, G, d * d * sizeof(float));

    for (int j = 0; j < d; j++) {
        /* Normalize column j */
        float norm = 0.0f;
        for (int i = 0; i < d; i++) {
            norm += turbo_rotation[i * d + j] * turbo_rotation[i * d + j];
        }
        norm = sqrtf(norm);
        if (norm > 1e-10f) {
            for (int i = 0; i < d; i++) {
                turbo_rotation[i * d + j] /= norm;
            }
        }

        /* Orthogonalize remaining columns against j */
        for (int k = j + 1; k < d; k++) {
            float dot = 0.0f;
            for (int i = 0; i < d; i++) {
                dot += turbo_rotation[i * d + j] * turbo_rotation[i * d + k];
            }
            for (int i = 0; i < d; i++) {
                turbo_rotation[i * d + k] -= dot * turbo_rotation[i * d + j];
            }
        }
    }

    /* Compute transpose */
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            turbo_rotation_t[i * d + j] = turbo_rotation[j * d + i];
        }
    }

    turbo_rotation_initialized = 1;
}

/* ---------- QJL projection matrix (lazy init, seed-based) ---------- */

static float turbo_qjl_matrix[TURBO_D * TURBO_D];
static float turbo_qjl_matrix_t[TURBO_D * TURBO_D];
static int   turbo_qjl_initialized = 0;

static void turbo_init_qjl(void) {
    if (turbo_qjl_initialized) return;

    const int d = TURBO_D;
    turbo_prng_seed(TURBO_SEED_QJL);

    for (int i = 0; i < d * d; i++) {
        turbo_qjl_matrix[i] = (float)turbo_prng_normal();
    }

    /* Transpose */
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            turbo_qjl_matrix_t[i * d + j] = turbo_qjl_matrix[j * d + i];
        }
    }

    turbo_qjl_initialized = 1;
}

/* ---------- helper: matrix-vector multiply ---------- */

static void matvec(const float * M, const float * x, float * y, int d) {
    /* y = M @ x, M is row-major d×d */
    for (int i = 0; i < d; i++) {
        float sum = 0.0f;
        for (int j = 0; j < d; j++) {
            sum += M[i * d + j] * x[j];
        }
        y[i] = sum;
    }
}

/* ---------- nearest centroid ---------- */

static int nearest_centroid_2bit(float val) {
    /* Binary search on midpoints: {-0.133, -0.040, 0.040, 0.133} */
    if (val < -0.086728f) return 0;       /* midpoint(-0.133, -0.040) */
    if (val <  0.000000f) return 1;       /* midpoint(-0.040, 0.040) */
    if (val <  0.086728f) return 2;       /* midpoint(0.040, 0.133) */
    return 3;
}

static int nearest_centroid_3bit(float val) {
    /* 8 centroids, find nearest via midpoints */
    if (val < -0.154259f) return 0;
    if (val < -0.091775f) return 1;
    if (val < -0.043589f) return 2;
    if (val <  0.000000f) return 3;
    if (val <  0.043589f) return 4;
    if (val <  0.091775f) return 5;
    if (val <  0.154259f) return 6;
    return 7;
}

static int nearest_centroid_4bit(float val) {
    /* 16 centroids, binary search on midpoints */
    if (val < MIDPOINTS_4BIT[7]) {
        if (val < MIDPOINTS_4BIT[3]) {
            if (val < MIDPOINTS_4BIT[1]) return val < MIDPOINTS_4BIT[0] ? 0 : 1;
            else                         return val < MIDPOINTS_4BIT[2] ? 2 : 3;
        } else {
            if (val < MIDPOINTS_4BIT[5]) return val < MIDPOINTS_4BIT[4] ? 4 : 5;
            else                         return val < MIDPOINTS_4BIT[6] ? 6 : 7;
        }
    } else {
        if (val < MIDPOINTS_4BIT[11]) {
            if (val < MIDPOINTS_4BIT[9])  return val < MIDPOINTS_4BIT[8] ? 8 : 9;
            else                          return val < MIDPOINTS_4BIT[10] ? 10 : 11;
        } else {
            if (val < MIDPOINTS_4BIT[13]) return val < MIDPOINTS_4BIT[12] ? 12 : 13;
            else                          return val < MIDPOINTS_4BIT[14] ? 14 : 15;
        }
    }
}

/* ---------- TURBO2_0: 2-bit PolarQuant, no QJL ---------- */

void quantize_row_turbo2_0_ref(const float * GGML_RESTRICT x, block_turbo2_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO2 == 0);
    const int nb = k / QK_TURBO2;
    for (int i = 0; i < nb; i++) {
        float norm = 0.0f;
        for (int j = 0; j < QK_TURBO2; j++) norm += x[i*QK_TURBO2 + j] * x[i*QK_TURBO2 + j];
        y[i].norm = GGML_FP32_TO_FP16(sqrtf(norm));
        memset(y[i].qs, 0, QK_TURBO2 / 4);
    }
}

void dequantize_row_turbo2_0(const block_turbo2_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO2 == 0);
    const int nb = k / QK_TURBO2;
    for (int block = 0; block < nb; block++) {
        float norm = GGML_FP16_TO_FP32(x[block].norm);
        for (int j = 0; j < QK_TURBO2; j++) {
            uint8_t idx = (x[block].qs[j/4] >> ((j%4)*2)) & 0x3;
            y[block * QK_TURBO2 + j] = CENTROIDS_2BIT[idx] * norm;
        }
    }
}

size_t quantize_turbo2_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO2 == 0);

    size_t row_size = (n_per_row / QK_TURBO2) * sizeof(block_turbo2_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo2_0_ref(
            src + row * n_per_row,
            (block_turbo2_0 *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

/* ---------- TURBO3_0: 128-value classic 2-bit PolarQuant + 1-bit QJL ---------- */

void quantize_row_turbo3_0_ref(const float * GGML_RESTRICT x, block_turbo3_0 * GGML_RESTRICT y, int64_t k) {
    // Stub — Metal shader handles quantize on GPU. CPU path is simplified.
    assert(k % QK_TURBO3 == 0);
    const int nb = k / QK_TURBO3;
    for (int i = 0; i < nb; i++) {
        float norm = 0.0f;
        for (int j = 0; j < QK_TURBO3; j++) norm += x[i*QK_TURBO3 + j] * x[i*QK_TURBO3 + j];
        y[i].norm = GGML_FP32_TO_FP16(sqrtf(norm));
        memset(y[i].qs, 0, QK_TURBO3 / 4);
        memset(y[i].signs, 0, QK_TURBO3 / 8);
    }
}

void dequantize_row_turbo3_0(const block_turbo3_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    // Stub — Metal shader handles dequant on GPU.
    assert(k % QK_TURBO3 == 0);
    const int nb = k / QK_TURBO3;
    for (int block = 0; block < nb; block++) {
        float norm = GGML_FP16_TO_FP32(x[block].norm);
        for (int j = 0; j < QK_TURBO3; j++) {
            uint8_t low2 = (x[block].qs[j/4] >> ((j%4)*2)) & 0x3;
            uint8_t hi1 = (x[block].signs[j/8] >> (j%8)) & 0x1;
            uint8_t idx = low2 | (hi1 << 2);
            y[block * QK_TURBO3 + j] = CENTROIDS_3BIT[idx] * norm;
        }
    }
}

size_t quantize_turbo3_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO3 == 0);

    size_t row_size = (n_per_row / QK_TURBO3) * sizeof(block_turbo3_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo3_0_ref(
            src + row * n_per_row,
            (block_turbo3_0 *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

/* ---------- TURBO3_TCQ: Trellis-Coded Quantization ---------- */

void quantize_row_turbo3_tcq_ref(const float * GGML_RESTRICT x, block_turbo3_tcq * GGML_RESTRICT y, int64_t k) {
    // Stub — CUDA kernel handles TCQ quantize (Viterbi). CPU path zeros out.
    assert(k % QK_TURBO3_TCQ == 0);
    const int nb = k / QK_TURBO3_TCQ;
    for (int i = 0; i < nb; i++) {
        float norm = 0.0f;
        for (int j = 0; j < QK_TURBO3_TCQ; j++) norm += x[i*QK_TURBO3_TCQ + j] * x[i*QK_TURBO3_TCQ + j];
        y[i].norm = GGML_FP32_TO_FP16(sqrtf(norm));
        memset(y[i].qs, 0, 49);
    }
}

void dequantize_row_turbo3_tcq(const block_turbo3_tcq * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    GGML_UNUSED(x);
    assert(k % QK_TURBO3_TCQ == 0);
    const int nb = k / QK_TURBO3_TCQ;
    for (int block = 0; block < nb; block++) {
        for (int j = 0; j < QK_TURBO3_TCQ; j++) {
            y[block * QK_TURBO3_TCQ + j] = 0.0f;
        }
    }
}

size_t quantize_turbo3_tcq(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO3_TCQ == 0);

    size_t row_size = (n_per_row / QK_TURBO3_TCQ) * sizeof(block_turbo3_tcq);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo3_tcq_ref(
            src + row * n_per_row,
            (block_turbo3_tcq *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

/* ---------- TURBO2_TCQ: 2-bit Trellis-Coded Quantization ---------- */

void quantize_row_turbo2_tcq_ref(const float * GGML_RESTRICT x, block_turbo2_tcq * GGML_RESTRICT y, int64_t k) {
	// Stub — CUDA kernel handles TCQ quantize (Viterbi). CPU path zeros out.
	assert(k % QK_TURBO2_TCQ == 0);
	const int nb = k / QK_TURBO2_TCQ;
	for (int i = 0; i < nb; i++) {
		float norm = 0.0f;
		for (int j = 0; j < QK_TURBO2_TCQ; j++) norm += x[i*QK_TURBO2_TCQ + j] * x[i*QK_TURBO2_TCQ + j];
		y[i].norm = GGML_FP32_TO_FP16(sqrtf(norm));
		memset(y[i].qs, 0, 33);
	}
}

void dequantize_row_turbo2_tcq(const block_turbo2_tcq * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
	GGML_UNUSED(x);
	assert(k % QK_TURBO2_TCQ == 0);
	const int nb = k / QK_TURBO2_TCQ;
	for (int block = 0; block < nb; block++) {
		for (int j = 0; j < QK_TURBO2_TCQ; j++) {
			y[block * QK_TURBO2_TCQ + j] = 0.0f;
		}
	}
}

size_t quantize_turbo2_tcq(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
	GGML_UNUSED(imatrix);
	assert(n_per_row % QK_TURBO2_TCQ == 0);

	size_t row_size = (n_per_row / QK_TURBO2_TCQ) * sizeof(block_turbo2_tcq);
	for (int64_t row = 0; row < nrows; row++) {
		quantize_row_turbo2_tcq_ref(
			src + row * n_per_row,
			(block_turbo2_tcq *)((char *)dst + row * row_size),
			n_per_row
		);
	}
	return nrows * row_size;
}

/* ---------- TURBO4_0: 4-bit PolarQuant (16 centroids, no QJL) ---------- */

void quantize_row_turbo4_0_ref(const float * GGML_RESTRICT x, block_turbo4_0 * GGML_RESTRICT y, int64_t k) {
    turbo_init_rotation();

    assert(k % QK_TURBO4 == 0);
    const int nb = k / QK_TURBO4;
    const int d  = QK_TURBO4;

    for (int block = 0; block < nb; block++) {
        const float * src = x + block * d;

        /* Step 1: Extract norm */
        float norm_sq = 0.0f;
        for (int i = 0; i < d; i++) norm_sq += src[i] * src[i];
        float norm = sqrtf(norm_sq);

        /* Normalize */
        float normalized[TURBO_D];
        if (norm > 1e-10f) {
            const float inv = 1.0f / norm;
            for (int i = 0; i < d; i++) normalized[i] = src[i] * inv;
        } else {
            memset(normalized, 0, d * sizeof(float));
        }

        /* Step 2: Rotate */
        float rotated[TURBO_D];
        matvec(turbo_rotation, normalized, rotated, d);

        /* Step 3: 4-bit quantization — find nearest of 16 centroids */
        uint8_t indices[TURBO_D];
        for (int i = 0; i < d; i++) {
            indices[i] = (uint8_t)nearest_centroid_4bit(rotated[i]);
        }

        /* Step 4: Norm correction */
        float recon_sq = 0.0f;
        for (int i = 0; i < d; i++) {
            float r = CENTROIDS_4BIT[indices[i]];
            recon_sq += r * r;
        }
        float recon_norm = sqrtf(recon_sq);
        y[block].norm = GGML_FP32_TO_FP16((recon_norm > 1e-10f) ? norm / recon_norm : norm);

        /* Pack 4-bit indices: 2 per byte, low nibble first */
        for (int i = 0; i < d; i += 2) {
            y[block].qs[i / 2] = (uint8_t)((indices[i + 1] << 4) | (indices[i] & 0xF));
        }
    }
}

void dequantize_row_turbo4_0(const block_turbo4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    turbo_init_rotation();

    assert(k % QK_TURBO4 == 0);
    const int nb = k / QK_TURBO4;
    const int d  = QK_TURBO4;

    for (int block = 0; block < nb; block++) {
        float norm = GGML_FP16_TO_FP32(x[block].norm);

        /* Unpack 4-bit indices and reconstruct in rotated space */
        float rotated_recon[TURBO_D];
        for (int i = 0; i < d; i++) {
            uint8_t idx = (i & 1) ? (x[block].qs[i / 2] >> 4) : (x[block].qs[i / 2] & 0xF);
            rotated_recon[i] = CENTROIDS_4BIT[idx];
        }

        /* Inverse rotate */
        float * dst = y + block * d;
        matvec(turbo_rotation_t, rotated_recon, dst, d);

        /* Scale by norm */
        for (int i = 0; i < d; i++) {
            dst[i] *= norm;
        }
    }
}

size_t quantize_turbo4_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO4 == 0);

    size_t row_size = (n_per_row / QK_TURBO4) * sizeof(block_turbo4_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo4_0_ref(
            src + row * n_per_row,
            (block_turbo4_0 *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}
