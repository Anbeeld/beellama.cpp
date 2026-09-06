#pragma once

#include "fattn-mma-kvarn-decode-decl.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-kvarn-load.cuh"
#include "mma.cuh"

using namespace ggml_cuda_mma;

static constexpr int GGML_CUDA_FATTN_KVARN_DECODE_THREADS = 256;

// Сколько блоков на мультипроцессор требовать от компилятора. Ядро упирается в
// регистры, а не в shared и не в нити: замер на sm_86 (cuobjdump) для нашей
// формы D=256, MAX_GQA=6, NWARPS=8, k5/v5 даёт REG:80 SHARED:12864 при 256
// нитях, то есть 65536/(80*256) = 3 блока против 7 по shared и 6 по нитям.
// Три блока по восемь варпов — 24 варпа из 48, ровно половина занятости.
//
// Авторы оригинальной KVarN пришли к тому же диагнозу (упор в пропускную
// способность L1/TEX при занятости, ограниченной регистрами) и лечили его не
// конвейеризацией, а наоборот — потолком на регистры ради большего числа
// резидентных блоков: у них это дало -27% времени ядра и +7..8% сквозной
// скорости. Здесь тот же приём: 4 блока требуют уложиться в 64 регистра.
//
// На численный результат не влияет: распределение регистров не меняет ни
// набор операций, ни их порядок. Возможные спиллы в локальную память точны.
#ifndef GGML_CUDA_FATTN_KVARN_DECODE_MIN_BLOCKS
#define GGML_CUDA_FATTN_KVARN_DECODE_MIN_BLOCKS 4
#endif
static constexpr int GGML_CUDA_FATTN_KVARN_DECODE_CHUNK   = 16;

static __device__ __forceinline__ bool ggml_cuda_fattn_kvarn_decode_group_from_record(
        const ggml_cuda_fattn_kvarn_desc & desc,
        const int group) {
    if (desc.swa) {
        return false;
    }
    return ggml_cuda_fattn_kvarn_group_from_record(desc, group);
}

// True when an absolute SWA group sits in the record store (not the live/stage tail and
// still inside the ring window). Mirrors the SWA branch of ggml_cuda_fattn_kvarn_load_rotated.
static __device__ __forceinline__ bool ggml_cuda_fattn_kvarn_decode_swa_record_backed(
        const ggml_cuda_fattn_kvarn_desc & desc,
        const int group_global) {
    if (group_global < 0) {
        return false;
    }
    return ggml_cuda_fattn_kvarn_group_from_record(desc, group_global);
}

// Per-split tile plan: whether the cooperative direct-record fast path is usable, the in-group
// position of the first tile token, and the ring record-group index for that first group.
//
// Non-SWA: token == in-stream position, so a 64-token split always lands inside one record group
// (pos_begin is 0 or 64) — identical to the original linear mapping.
//
// SWA: the ring presents window cells in piecewise-contiguous absolute-position order (one wrap
// discontinuity per decode). We take the fast path when the split's valid tokens are contiguous in
// absolute position (endpoint check) and the first absolute group is record-backed. Group-straddle,
// stage (live group), ring-wrap, and non-contiguous tiles fall back per element to load_rotated,
// which independently resolves group/pos for any cell. Empty/out-of-window cells carry idx < 0 and
// are -inf-masked downstream, so reads for them are harmless.
struct ggml_cuda_fattn_kvarn_decode_tile {
    bool fast;
    int  pos_begin;
    int  record_group;
};

static __device__ __forceinline__ ggml_cuda_fattn_kvarn_decode_tile
ggml_cuda_fattn_kvarn_decode_plan_tile(
        const ggml_cuda_fattn_kvarn_desc & desc,
        const int token_begin,
        const int token_end,
        const int group,
        const int group_pos_begin) {
    ggml_cuda_fattn_kvarn_decode_tile tile;
    if (desc.swa || desc.read_indirect) {
        const int64_t e0 = desc.indices[token_begin];
        const int64_t e1 = desc.indices[token_end - 1];
        bool stage0;
        bool stage1;
        const int64_t a0 = ggml_cuda_fattn_kvarn_read_cell(desc, e0, stage0);
        const int64_t a1 = ggml_cuda_fattn_kvarn_read_cell(desc, e1, stage1);
        const int span = (token_end - 1) - token_begin;
        const int g0 = e0 != -1 ? (int) (a0 / GGML_CUDA_FATTN_KVARN_DIM) : -1;
        const int g1 = e1 != -1 ? (int) (a1 / GGML_CUDA_FATTN_KVARN_DIM) : -2;
        const bool contiguous = e0 != -1 && e1 != -1 && stage0 == stage1 &&
            (int) (a1 - a0) == span && g0 == g1;
        const bool rec0 = desc.swa ?
            ggml_cuda_fattn_kvarn_decode_swa_record_backed(desc, g0) :
            (!stage0 && (desc.read_indirect || ggml_cuda_fattn_kvarn_decode_group_from_record(desc, g0)));
        tile.fast = contiguous && rec0;
        tile.pos_begin = tile.fast ? (int) (a0 % GGML_CUDA_FATTN_KVARN_DIM) : 0;
        tile.record_group = tile.fast ? (desc.swa ?
            (g0 % desc.groups_per_stream) :
            desc.stream * desc.groups_per_stream + g0) : 0;
    } else {
        tile.fast = ggml_cuda_fattn_kvarn_decode_group_from_record(desc, group);
        tile.pos_begin = group_pos_begin;
        tile.record_group = desc.stream * desc.groups_per_stream + group;
    }
    return tile;
}

template<int BITS>
static __device__ __forceinline__ int ggml_cuda_fattn_kvarn_decode_unpack(
        const uint8_t * raw,
        const int index,
        const int bits) {
    if constexpr (BITS == 8) {
        return raw[index];
    } else if constexpr (BITS == 4) {
        return (raw[index >> 1] >> (4 * (index & 1))) & 0x0f;
    } else if constexpr (BITS == 2) {
        return (raw[index >> 2] >> (2 * (index & 3))) & 0x03;
    } else if constexpr (BITS > 0) {
        // Читаем 32-битными словами вместо пары однобайтовых загрузок со сдвигом.
        // Извлекаются ровно те же биты, то есть результат побитово прежний, но
        // инструкций загрузки на элемент становится ~1.13 вместо 2, а трафик L1
        // падает примерно на 40%. Авторы оригинальной KVarN отдельно измерили
        // этот приём на осях квантования: вдвое меньше транзакций L1 при
        // побитово том же значении, и отметили, что горячее ядро упирается
        // именно в пропускную способность L1, а не в память.
        //
        // Выравнивание: строка записи занимает 16*BITS байт (кратно 4), а начало
        // записи выровнено на 16, поэтому raw всегда кратен четырём. Второе слово
        // читается только когда элемент через него переходит, а тогда оно заведомо
        // внутри строки — последний бит элемента лежит в пределах строки по
        // построению формата.
        const uint32_t * words = (const uint32_t *) raw;
        const int bit_offset = index * BITS;
        const int word_offset = bit_offset >> 5;
        const int shift = bit_offset & 31;
        // ЗАМЕРЕНО И ОТКАЧЕНО: __ldg на этих чтениях плюс __restrict__ на
        // указателях ядра дали регрессию 27.81 -> 27.15 tok/s при побитово том же
        // выводе. Стековый кадр при этом вырос с 32 до 80 байт — компилятор стал
        // больше держать в локальной памяти, а её трафик идёт через тот же кэш,
        // который правка и пыталась разгрузить.
        uint64_t packed = (uint64_t) words[word_offset];
        if (shift + BITS > 32) {
            packed |= (uint64_t) words[word_offset + 1] << 32;
        }
        return (int) ((packed >> shift) & (uint64_t) ((1u << BITS) - 1u));
    }

    if (bits == 8) {
        return raw[index];
    }
    if (bits == 4) {
        return (raw[index >> 1] >> (4 * (index & 1))) & 0x0f;
    }
    if (bits == 2) {
        return (raw[index >> 2] >> (2 * (index & 3))) & 0x03;
    }
    const int bit_offset = index * bits;
    const int byte_offset = bit_offset >> 3;
    const int shift = bit_offset & 7;
    uint16_t packed = (uint16_t) raw[byte_offset];
    if (shift + bits > 8) {
        packed |= (uint16_t) raw[byte_offset + 1] << 8;
    }
    return (packed >> shift) & ((1 << bits) - 1);
}

// Перестановка строк матричного фрагмента.
//
// Раскладка операнда A у mma.m16n8k16 закреплена железом: нить держит строки
// t/4 и t/4+8. В записи KVarN эти две строки — элементы, разнесённые на восемь
// позиций, то есть на 8*BITS бит, и каждый требует своего 32-битного слова.
// Отсюда 1.125 инструкции загрузки на пятибитный элемент: 4.4 полезных бита из
// 32 прочитанных.
//
// Номер строки фрагмента — это просто метка: mma считает скалярное произведение
// по индексу свёртки, а по индексу строки ничего не смешивает. Поэтому строку r
// можно объявить токеном 2*(r%8) + r/8. Тогда пара строк одной нити становится
// парой СОСЕДНИХ токенов, и оба элемента приходят одной загрузкой.
//
// Перестановка обратима на месте записи результата: фрагмент C имеет тот же
// набор строк {t/4, t/4+8}, и там применяется ровно то же отображение. Ни один
// порядок суммирования не меняется, поэтому результат побитово прежний.
#define KVARN_FRAG_ROW(r) (2 * ((r) & 7) + ((r) >> 3))

// Два СОСЕДНИХ элемента одной строки записи за одну загрузку.
// Пара укладывается в 2*BITS <= 16 бит, поэтому второе слово требуется только
// когда пара пересекает границу слова — для пятибитной укладки это четверть
// случаев против восьмой у поэлементного чтения, но покрывает вдвое больше
// элементов.
template<int BITS>
static __device__ __forceinline__ void ggml_cuda_fattn_kvarn_decode_unpack2(
        const uint8_t * raw,
        const int index,
        int & a,
        int & b) {
    const uint32_t * words = (const uint32_t *) raw;
    const int bit_offset = index * BITS;
    const int word_offset = bit_offset >> 5;
    const int shift = bit_offset & 31;
    uint64_t packed = (uint64_t) words[word_offset];
    if (shift + 2 * BITS > 32) {
        packed |= (uint64_t) words[word_offset + 1] << 32;
    }
    const uint32_t mask = (1u << BITS) - 1u;
    a = (int) ((packed >> shift) & mask);
    b = (int) ((packed >> (shift + BITS)) & mask);
}

// Q_TILE — сколько строк запроса обслуживает один блок. Раньше их всегда была
// одна, и при n_q > 1 несколько блоков независимо распаковывали одни и те же
// записи кэша. Внутри блока распакованный фрагмент лежит в регистрах, и
// выполнить по нему Q_TILE матричных инструкций с разными строками запроса
// стоит только лишнего аккумулятора. Порядок суммирования внутри каждой
// строки не меняется, поэтому результат побитово тот же.
template<int D, int MAX_GQA, int SPLIT_TOKENS, int NWARPS, int K_BITS, int V_BITS, int Q_TILE = 1>
// Потолок нитей на мультипроцессор у sm_86 равен 1536, поэтому требовать четыре
// блока можно только пока NWARPS*32*4 в него укладывается. При NWARPS=16 это 2048,
// и ptxas молча игнорирует указание целиком — берём максимум допустимого.
__launch_bounds__(NWARPS * 32,
    (1536 / (NWARPS * 32)) < GGML_CUDA_FATTN_KVARN_DECODE_MIN_BLOCKS ?
        (1536 / (NWARPS * 32)) : GGML_CUDA_FATTN_KVARN_DECODE_MIN_BLOCKS)
static __global__ void ggml_cuda_fattn_kvarn_decode_mma_kernel(
        const char * Q,
        const ggml_cuda_fattn_kvarn_desc * k_descs,
        const ggml_cuda_fattn_kvarn_desc * v_descs,
        const char * mask,
        float * partial,
        float2 * partial_meta,
        float scale,
        float logit_softcap,
        int64_t nb01,
        int64_t nb02,
        int64_t nb03,
        int64_t nb30,
        int64_t nb31,
        int64_t nb33,
        int ne33,
        int n_kv,
        int n_q,
        int n_q_heads,
        int n_kv_heads,
        int gqa_ratio,
        int n_gqa_blocks,
        int n_splits) {
    const int split = blockIdx.x;
    const int n_q_tiles = (n_q + Q_TILE - 1) / Q_TILE;
    const int q_base = (blockIdx.y % n_q_tiles) * Q_TILE;
    const int q_count = min(Q_TILE, n_q - q_base);
    const int gqa_block = (blockIdx.y / n_q_tiles) % n_gqa_blocks;
    const int kv_head = blockIdx.y / (n_q_tiles * n_gqa_blocks);
    const int stream = blockIdx.z;
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    constexpr int PHYSICAL_WAVE_SIZE = ggml_cuda_get_physical_warp_size();
    const int tid = warp * PHYSICAL_WAVE_SIZE + lane;
    constexpr int SLICES = D / GGML_CUDA_FATTN_KVARN_DIM;
    constexpr int TOKENS_PER_CHUNK = GGML_CUDA_FATTN_KVARN_DECODE_CHUNK;
    constexpr int TOKEN_CHUNKS = SPLIT_TOKENS / TOKENS_PER_CHUNK;
    constexpr int WARPS_PER_CHUNK = D / GGML_CUDA_FATTN_KVARN_DIM;
    constexpr int CHUNKS_PER_PASS = NWARPS / WARPS_PER_CHUNK;
    constexpr int Q_STRIDE2 = D / 2 + 4;
    // load_ldmatrix ниже читает из q_sh фиксированные 8 строк, поэтому при
    // MAX_GQA == 6 массив на шесть строк давал чтение за его границей (попадало
    // в score_partial_sh). Результат лишних строк отбрасывается по h < MAX_GQA,
    // но само чтение было некорректным.
    constexpr int Q_ROWS = MAX_GQA < 8 ? 8 : MAX_GQA;
    constexpr int P_ROWS = MAX_GQA < 8 ? 8 : MAX_GQA;
    constexpr int P_STRIDE2 = SPLIT_TOKENS / 2 + 4;

    using T_A = tile<16, 8, half2>;
    using T_B = tile<8, 8, half2>;
    using T_C = tile<16, 8, float>;

    static_assert(D == 128 || D == 256 || D == 512, "KVarN decode MMA supports 128/256/512-wide heads");
    static_assert(MAX_GQA > 0 && MAX_GQA <= 8, "KVarN decode MMA expects at most eight GQA heads");
    static_assert(SPLIT_TOKENS == 64 || SPLIT_TOKENS == 128,
        "KVarN decode MMA production splits use 64 or 128 KV tokens");
    static_assert(Q_TILE >= 1 && Q_TILE <= 4,
        "KVarN decode MMA serves between one and four query rows per block");
    static_assert(NWARPS % WARPS_PER_CHUNK == 0 && NWARPS >= WARPS_PER_CHUNK &&
        NWARPS <= TOKEN_CHUNKS * WARPS_PER_CHUNK && NWARPS <= 16,
        "KVarN decode MMA needs whole 128-dim slice groups and at most sixteen warps");
    static_assert(K_BITS == 2 || K_BITS == 3 || K_BITS == 4 ||
        K_BITS == 5 || K_BITS == 6 || K_BITS == 8, "invalid compile-time KVarN K bits");
    static_assert(V_BITS == 2 || V_BITS == 3 || V_BITS == 4 ||
        V_BITS == 5 || V_BITS == 6 || V_BITS == 8, "invalid compile-time KVarN V bits");
    static_assert(PHYSICAL_WAVE_SIZE == 32 || PHYSICAL_WAVE_SIZE == 64,
        "KVarN decode MMA requires a physical wave size of 32 or 64");

    __shared__ __align__(16) half2 q_sh[Q_TILE][Q_ROWS][Q_STRIDE2];
    // score_partial_sh намеренно БЕЗ измерения Q_TILE: свёртка по варпам идёт по
    // строкам запроса по очереди, а сами частичные суммы к этому моменту лежат в
    // регистрах. Так правка стоит на 4 КиБ разделяемой памяти меньше, и четыре
    // блока на мультипроцессор сохраняются.
    __shared__ __align__(16) float score_partial_sh[NWARPS][MAX_GQA * TOKENS_PER_CHUNK];
    __shared__ __align__(16) float score_sh[Q_TILE][MAX_GQA][SPLIT_TOKENS];
    __shared__ __align__(16) half2 p_sh[Q_TILE][P_ROWS][P_STRIDE2];
    // Оси квантования лежат в записи как half и раньше раскладывались в shared
    // как float. Точности это не добавляло ни одного бита — значение уже прошло
    // округление до half при записи, — зато стоило вдвое больше байт shared и
    // вдвое больше трафика LDS в горячем цикле. Держим их half и переводим во
    // float в точке использования: результат побитово тот же.
    //
    // Три оси лежат в записи подряд и выровнены на 16 байт (payload_bytes всегда
    // кратен 16), поэтому одна область на слайс копируется векторными uint4.
    constexpr int AXES_HALVES = 3 * GGML_CUDA_FATTN_KVARN_DIM;
    constexpr int AXES_VEC    = (AXES_HALVES * (int) sizeof(half)) / (int) sizeof(uint4);
    __shared__ __align__(16) half axes_sh[SLICES][AXES_HALVES];
#define KVARN_AXIS_SCALE(sl, i) __half2float(axes_sh[sl][(i)])
#define KVARN_AXIS_ZP(sl, i)    __half2float(axes_sh[sl][GGML_CUDA_FATTN_KVARN_DIM + (i)])
#define KVARN_AXIS_OTHER(sl, i) __half2float(axes_sh[sl][2 * GGML_CUDA_FATTN_KVARN_DIM + (i)])
    __shared__ float zq_sh[Q_TILE][SLICES][MAX_GQA];
    __shared__ float m_sh[Q_TILE][MAX_GQA];
    // NVCC 13.1/sm_86 trims eight bytes from this kernel when the TU-local
    // combine instantiation is removed. Preserve the verified baseline resource
    // footprint only for that compiler target; other backends/toolkits retain
    // the original declaration until separately proven to need compensation.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 860 && \
    defined(__CUDACC_VER_MAJOR__) && __CUDACC_VER_MAJOR__ == 13 && \
    defined(__CUDACC_VER_MINOR__) && __CUDACC_VER_MINOR__ == 1 && \
    !defined(__HIPCC__) && !defined(GGML_USE_MUSA)
    static constexpr int GGML_CUDA_FATTN_KVARN_DECODE_RESOURCE_PAD =
        D == 128 && MAX_GQA == 6 && SPLIT_TOKENS == 64 && NWARPS == 4 && Q_TILE == 1 ? 2 : 0;
#else
    static constexpr int GGML_CUDA_FATTN_KVARN_DECODE_RESOURCE_PAD = 0;
#endif
    __shared__ float denom_sh[Q_TILE][MAX_GQA + GGML_CUDA_FATTN_KVARN_DECODE_RESOURCE_PAD];

    const ggml_cuda_fattn_kvarn_desc & k_desc = k_descs[stream * n_kv_heads + kv_head];
    const ggml_cuda_fattn_kvarn_desc & v_desc = v_descs[stream * n_kv_heads + kv_head];
    const int q_head0 = kv_head * gqa_ratio + gqa_block * MAX_GQA;
    const int gqa_head_count = min(MAX_GQA, gqa_ratio - gqa_block * MAX_GQA);
    const int token_begin = split * SPLIT_TOKENS;
    const int token_end = min(n_kv, token_begin + SPLIT_TOKENS);

    // Пропуск полностью замаскированного сплита.
    //
    // При объединённом кэше (--kv-unified) ячейки чужой последовательности
    // замаскированы целиком. Ядро всё равно распаковывало для них полный тайл,
    // а объединение потом отбрасывало результат: при маске -inf все веса
    // софтмакса равны нулю, знаменатель выходит нулевым, и partial тоже нулевой.
    // Проверка маски заранее убирает эту работу, не меняя результата.
    //
    // Замер батчинга показывает ровно эту трату: два слота против одного дают
    // у q8_0 совокупный прирост 1.24x, а у KVarN 0.99x. Апстрим лечит то же
    // самое отдельным проходом flash_attn_mask_to_KV_max — тот находит
    // последнюю живую позицию и обрезает хвост. Такая граница спасает только ту
    // последовательность, что лежит в кэше первой; проверка по сплиту работает
    // для обеих, потому что каждый блок и так независим.
    //
    // Скипаем только полный сплит: у последнего, неполного, позиции за token_end
    // несут -FLT_MAX/2 без маски, и знаменатель у него бывает ненулевым.
    if (mask != nullptr && token_end - token_begin == SPLIT_TOKENS) {
        // Сплит можно пропустить только если он полностью замаскирован для ВСЕХ
        // строк запроса, которые ведёт этот блок: они делят распакованные записи,
        // но не маску.
        int live = 0;
        for (int qt = 0; qt < q_count; ++qt) {
            const char * mask_base = mask + nb33 * (stream % ne33) + nb31 * (q_base + qt);
            for (int token = token_begin + tid; token < token_end;
                    token += NWARPS * PHYSICAL_WAVE_SIZE) {
                const half mv = *(const half *) (mask_base + nb30 * token);
                live |= !__hisinf(mv);
            }
        }
        if (__syncthreads_or(live) == 0) {
            if (tid < gqa_head_count && q_head0 + tid < n_q_heads) {
                for (int qt = 0; qt < q_count; ++qt) {
                    const size_t base =
                        (((size_t) stream * n_q + (q_base + qt)) * n_q_heads + (q_head0 + tid)) *
                        (size_t) n_splits + split;
                    partial_meta[base] = make_float2(-FLT_MAX / 2.0f, 0.0f);
                }
            }
            return;
        }
    }
    const int group = token_begin / GGML_CUDA_FATTN_KVARN_DIM;
    const int group_pos_begin = token_begin - group * GGML_CUDA_FATTN_KVARN_DIM;
    const ggml_cuda_fattn_kvarn_decode_tile k_tile =
        ggml_cuda_fattn_kvarn_decode_plan_tile(k_desc, token_begin, token_end, group, group_pos_begin);
    const ggml_cuda_fattn_kvarn_decode_tile v_tile =
        ggml_cuda_fattn_kvarn_decode_plan_tile(v_desc, token_begin, token_end, group, group_pos_begin);
    const bool k_from_record = k_tile.fast;
    const bool v_from_record = v_tile.fast;
    const int k_payload_bytes = GGML_CUDA_FATTN_KVARN_DIM * GGML_CUDA_FATTN_KVARN_DIM * K_BITS / 8;
    const int v_payload_bytes = GGML_CUDA_FATTN_KVARN_DIM * GGML_CUDA_FATTN_KVARN_DIM * V_BITS / 8;
    const int k_row_bytes = GGML_CUDA_FATTN_KVARN_DIM * K_BITS / 8;
    const int v_row_bytes = GGML_CUDA_FATTN_KVARN_DIM * V_BITS / 8;
    const int record_group_k = k_tile.record_group;
    const int record_group_v = v_tile.record_group;

    const bool k_split_in_group =
        k_from_record && (k_tile.pos_begin + SPLIT_TOKENS) <= GGML_CUDA_FATTN_KVARN_DIM;

    const uint8_t * k_records[SLICES];
    const uint8_t * v_records[SLICES];
#pragma unroll
    for (int slice = 0; slice < SLICES; ++slice) {
        k_records[slice] = k_desc.records +
            ((int64_t) record_group_k * k_desc.n_record_heads + k_desc.head_base + slice) * k_desc.record_bytes;
        v_records[slice] = v_desc.records +
            ((int64_t) record_group_v * v_desc.n_record_heads + v_desc.head_base + slice) * v_desc.record_bytes;
    }

    for (int qt = 0; qt < Q_TILE; ++qt) {
        half * q_h = (half *) q_sh[qt];
        for (int i = tid; i < Q_ROWS * D; i += NWARPS * PHYSICAL_WAVE_SIZE) {
            const int h = i / D;
            const int dim = i % D;
            float value = 0.0f;
            if (qt < q_count && h < gqa_head_count && q_head0 + h < n_q_heads) {
                const float * q = (const float *)
                    (Q + nb03 * stream + nb02 * (q_head0 + h) + nb01 * (q_base + qt));
                value = q[dim] * scale;
            }
            q_h[h * (2 * Q_STRIDE2) + dim] = __float2half(value);
        }
    }

    if (k_split_in_group) {
        for (int i = tid; i < SLICES * AXES_VEC; i += NWARPS * PHYSICAL_WAVE_SIZE) {
            const int slice = i / AXES_VEC;
            const int vec   = i % AXES_VEC;
            const uint4 * src = (const uint4 *) (k_records[slice] + k_payload_bytes);
            ((uint4 *) axes_sh[slice])[vec] = src[vec];
        }
    }
    __syncthreads();

    if (k_split_in_group) {
        // Замер на 3090 (парный прогон, один слот, глубина 65000): распараллеливание
        // этого цикла по варпам с древовидной свёрткой zq дало 27.70 против 27.71
        // tok/s, то есть ровно ничего, при том что меняло порядок суммирования zq
        // и, значит, численный результат. Оставляем последовательную форму: она
        // побитово совпадает с эталоном и не требует отдельной проверки точности.
        const int n_targets = Q_TILE * SLICES * gqa_head_count;
        for (int target = tid; target < n_targets; target += NWARPS * PHYSICAL_WAVE_SIZE) {
            const int qt = target / (SLICES * gqa_head_count);
            const int rest = target % (SLICES * gqa_head_count);
            const int slice = rest / gqa_head_count;
            const int h = rest % gqa_head_count;
            const half * q_row = (const half *) q_sh[qt] + h * (2 * Q_STRIDE2) + slice * GGML_CUDA_FATTN_KVARN_DIM;
            float zq = 0.0f;
            for (int dim = 0; dim < GGML_CUDA_FATTN_KVARN_DIM; ++dim) {
                const float q_val = __half2float(q_row[dim]);
                zq += KVARN_AXIS_ZP(slice, dim) * q_val;
                const float q_prime = KVARN_AXIS_SCALE(slice, dim) * q_val;
                ((half *) q_sh[qt])[h * (2 * Q_STRIDE2) + slice * GGML_CUDA_FATTN_KVARN_DIM + dim] =
                    __float2half(q_prime);
            }
            zq_sh[qt][slice][h] = zq;
        }
    }

    __syncthreads();

    const int local_chunk = warp / WARPS_PER_CHUNK;
    const int warp_in_chunk = warp % WARPS_PER_CHUNK;
    const half * mask_h = mask != nullptr ? (const half *) (mask + nb33 * (stream % ne33)) : nullptr;

#pragma unroll 1
    for (int chunk_base = 0; chunk_base < TOKEN_CHUNKS; chunk_base += CHUNKS_PER_PASS) {
        const int chunk = chunk_base + local_chunk;
        const bool chunk_active = chunk < TOKEN_CHUNKS;
        const int token0 = token_begin + chunk * TOKENS_PER_CHUNK;
        T_C scores[Q_TILE];
#pragma unroll
        for (int qt = 0; qt < Q_TILE; ++qt) {
#pragma unroll
            for (int l = 0; l < T_C::ne; ++l) {
                scores[qt].x[l] = 0.0f;
            }
        }

        if (chunk_active) {
            // ЗАМЕРЕНО И ОТКАЧЕНО: развёртка вдвое здесь и в цикле V дала
            // 27.54 против 27.87 tok/s на одном слоте и 12.59 против 13.88 на двух,
            // то есть регрессию, при том что регистровое давление даже упало
            // (61 против 64) и вытеснения не появилось. Задержка глобальных
            // обращений не прячется числом висящих загрузок: она встроена в шаблон
            // доступа — раскладка матричного фрагмента требует от нити элементы из
            // четырёх разных строк записи, и одна инструкция неизбежно задевает
            // четыре сектора, забирая из каждого по нескольку байт.
#pragma unroll 1
            for (int dim0 = warp_in_chunk * GGML_CUDA_FATTN_KVARN_DIM;
                    dim0 < (warp_in_chunk + 1) * GGML_CUDA_FATTN_KVARN_DIM;
                    dim0 += 2 * T_A::J) {
                T_A k_a;
                // Строки фрагмента идут парами: l и l+1 отличаются только
                // строкой (get_j совпадает), а после перестановки KVARN_FRAG_ROW
                // это соседние токены одной строки записи — одна загрузка на пару.
#pragma unroll
                for (int lp = 0; lp < T_A::ne; lp += 2) {
                    const int dim = dim0 + 2 * T_A::get_j(lp);
                    const int slice = dim / GGML_CUDA_FATTN_KVARN_DIM;
                    const int local_dim = dim % GGML_CUDA_FATTN_KVARN_DIM;
                    const int token_local = KVARN_FRAG_ROW(T_A::get_i(lp));
                    const int pos = k_tile.pos_begin + chunk * TOKENS_PER_CHUNK + token_local;
                    float x00, x01, x10, x11;
                    if (k_split_in_group) {
                        const uint8_t * row0 = k_records[slice] + (local_dim + 0) * k_row_bytes;
                        const uint8_t * row1 = k_records[slice] + (local_dim + 1) * k_row_bytes;
                        int q00, q01, q10, q11;
                        ggml_cuda_fattn_kvarn_decode_unpack2<K_BITS>(row0, pos, q00, q01);
                        ggml_cuda_fattn_kvarn_decode_unpack2<K_BITS>(row1, pos, q10, q11);
                        x00 = (float) q00; x01 = (float) q01;
                        x10 = (float) q10; x11 = (float) q11;
                    } else {
                        const int token = token0 + token_local;
                        x00 = token + 0 < token_end ?
                            ggml_cuda_fattn_kvarn_load_rotated(k_desc, token + 0, slice, local_dim + 0) : 0.0f;
                        x10 = token + 0 < token_end ?
                            ggml_cuda_fattn_kvarn_load_rotated(k_desc, token + 0, slice, local_dim + 1) : 0.0f;
                        x01 = token + 1 < token_end ?
                            ggml_cuda_fattn_kvarn_load_rotated(k_desc, token + 1, slice, local_dim + 0) : 0.0f;
                        x11 = token + 1 < token_end ?
                            ggml_cuda_fattn_kvarn_load_rotated(k_desc, token + 1, slice, local_dim + 1) : 0.0f;
                    }
                    k_a.x[lp + 0] = make_half2(x00, x10);
                    k_a.x[lp + 1] = make_half2(x01, x11);
                }
                // Распакованный фрагмент K лежит в регистрах. Прогоняем по нему
                // все строки запроса тайла: лишняя строка стоит одной загрузки
                // ldmatrix и одной инструкции mma, а не повторной распаковки.
#pragma unroll
                for (int qt = 0; qt < Q_TILE; ++qt) {
                    // q_count одинаков для всего блока, поэтому ветвление не
                    // расходится по варпу. При нечётном n_q последний тайл ведёт
                    // одну строку, и вторая пара ldmatrix+mma здесь лишняя.
                    if (qt >= q_count) {
                        break;
                    }
                    T_B q_b;
                    load_ldmatrix(q_b, q_sh[qt][0] + dim0 / 2, Q_STRIDE2);
                    mma(scores[qt], k_a, q_b);
                }
            }
        }

        // Свёртка по варпам и запись в score_sh идут по строкам запроса по
        // очереди: score_partial_sh на все строки один, что и экономит
        // разделяемую память.
#pragma unroll
        for (int qt = 0; qt < Q_TILE; ++qt) {
            if (chunk_active) {
#pragma unroll
                for (int l = 0; l < T_C::ne; ++l) {
                    const int j = KVARN_FRAG_ROW(T_C::get_i(l));
                    const int h = T_C::get_j(l);
                    if (h < MAX_GQA) {
                        float v = scores[qt].x[l];
                        if (k_split_in_group && h < gqa_head_count) {
                            const int pos = k_tile.pos_begin + chunk * TOKENS_PER_CHUNK + j;
                            v = KVARN_AXIS_OTHER(warp_in_chunk, pos) * (v + zq_sh[qt][warp_in_chunk][h]);
                        }
                        score_partial_sh[warp][h * TOKENS_PER_CHUNK + j] = v;
                    }
                }
            }
            __syncthreads();

#pragma unroll
            for (int stride = WARPS_PER_CHUNK / 2; stride > 0; stride >>= 1) {
                if (chunk_active && warp_in_chunk < stride) {
#pragma unroll
                    for (int l = 0; l < T_C::ne; ++l) {
                        const int j = KVARN_FRAG_ROW(T_C::get_i(l));
                        const int h = T_C::get_j(l);
                        if (h < MAX_GQA) {
                            score_partial_sh[warp][h * TOKENS_PER_CHUNK + j] +=
                                score_partial_sh[warp + stride][h * TOKENS_PER_CHUNK + j];
                        }
                    }
                }
                __syncthreads();
            }

            if (chunk_active && warp_in_chunk == 0) {
#pragma unroll
                for (int l = 0; l < T_C::ne; ++l) {
                    const int j = KVARN_FRAG_ROW(T_C::get_i(l));
                    const int h = T_C::get_j(l);
                    const int token = token0 + j;
                    float score = -FLT_MAX / 2.0f;
                    if (qt < q_count && h < gqa_head_count && q_head0 + h < n_q_heads && token < token_end) {
                        score = score_partial_sh[warp][h * TOKENS_PER_CHUNK + j];
                        if (logit_softcap != 0.0f) {
                            score = logit_softcap * tanhf(score);
                        }
                        if (mask_h != nullptr) {
                            score += __half2float(*(const half *) ((const char *) mask_h +
                                nb30 * token + nb31 * (q_base + qt)));
                        }
                    }
                    if (h < MAX_GQA) {
                        score_sh[qt][h][chunk * TOKENS_PER_CHUNK + j] = score;
                    }
                }
            }
            __syncthreads();
        }
    }

#pragma unroll
    for (int qt = 0; qt < Q_TILE; ++qt) {
        half * p_h = (half *) p_sh[qt];
        // load_ldmatrix below reads a physical eight-row probability tile. When
        // MAX_GQA is six, initialize the two rows that are not produced by
        // softmax so every Q_TILE row has defined input for the V mma.
        for (int i = tid; i < (P_ROWS - MAX_GQA) * SPLIT_TOKENS;
                i += NWARPS * PHYSICAL_WAVE_SIZE) {
            const int h = MAX_GQA + i / SPLIT_TOKENS;
            const int token = i % SPLIT_TOKENS;
            p_h[h * (2 * P_STRIDE2) + token] = __float2half(0.0f);
        }
#if defined(GGML_USE_HIP) && defined(CDNA)
        if (tid < MAX_GQA) {
            const int h = tid;
            float m = -FLT_MAX / 2.0f;
            for (int token = 0; token < SPLIT_TOKENS; ++token) {
                m = fmaxf(m, score_sh[qt][h][token] + FATTN_KQ_MAX_OFFSET);
            }
            float denom = 0.0f;
            for (int token = 0; token < SPLIT_TOKENS; ++token) {
                const float diff = score_sh[qt][h][token] - m;
                const float weight = diff >= SOFTMAX_FTZ_THRESHOLD ? expf(diff) : 0.0f;
                denom += weight;
                p_h[h * (2 * P_STRIDE2) + token] = __float2half(weight);
            }
            m_sh[qt][h] = m;
            denom_sh[qt][h] = denom;
        }
#else
        const int h = tid / 16;
        const int lane_h = tid % 16;
        float m = -FLT_MAX / 2.0f;
        if (h < MAX_GQA) {
            for (int token = lane_h; token < SPLIT_TOKENS; token += 16) {
                m = fmaxf(m, score_sh[qt][h][token] + FATTN_KQ_MAX_OFFSET);
            }
        }
#pragma unroll
        for (int offset = 8; offset > 0; offset >>= 1) {
            m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, offset, 16));
        }

        float denom = 0.0f;
        if (h < MAX_GQA) {
            for (int token = lane_h; token < SPLIT_TOKENS; token += 16) {
                const float diff = score_sh[qt][h][token] - m;
                const float weight = diff >= SOFTMAX_FTZ_THRESHOLD ? expf(diff) : 0.0f;
                denom += weight;
                p_h[h * (2 * P_STRIDE2) + token] = __float2half(weight);
            }
        }
#pragma unroll
        for (int offset = 8; offset > 0; offset >>= 1) {
            denom += __shfl_xor_sync(0xFFFFFFFFu, denom, offset, 16);
        }
        if (h < MAX_GQA && lane_h == 0) {
            m_sh[qt][h] = m;
            denom_sh[qt][h] = denom;
        }
#endif
    }
    __syncthreads();

    if (v_from_record) {
        for (int i = tid; i < SLICES * AXES_VEC; i += NWARPS * PHYSICAL_WAVE_SIZE) {
            const int slice = i / AXES_VEC;
            const int vec   = i % AXES_VEC;
            const uint4 * src = (const uint4 *) (v_records[slice] + v_payload_bytes);
            ((uint4 *) axes_sh[slice])[vec] = src[vec];
        }
    }
    __syncthreads();

    // ЗАМЕРЕНО И ОТКАЧЕНО: свёртка стороны V по образцу стороны K (по-токенный
    // масштаб в веса софтмакса, сумма p*zp в скаляр, по-канальный множитель на
    // выходе). Реализована, корректна — потокенная сверка показала ожидаемое
    // расхождение округления на 16-м и 82-м токене, — но скорости не дала ни на
    // одном режиме: 39.79 против 39.75 и 34.60 против 34.53 tok/s.
    //
    // Причина: профиль ncu показывает простой варпа 6.1 такта из 15.2 в ожидании
    // L1TEX, но в терминах ncu L1TEX это локальная, глобальная, поверхностная и
    // текстурная память — разделяемая учитывается отдельно. Ждём мы глобальных
    // загрузок при распаковке полезной нагрузки, а не чтений масштабов из
    // разделяемой памяти, которые свёртка и убирала. Число LDG в машинном коде
    // при свёртке не изменилось: 1007 против 1007.
    //
    // Настоящая цель — раскладка матричного фрагмента: одна инструкция загрузки
    // задевает четыре строки записи на расстоянии 160 байт, забирая из каждой по
    // несколько байт. Лечится коллективной коалесцированной загрузкой полезной
    // нагрузки в разделяемую память с последующей раздачей по фрагментам либо
    // сменой раскладки записи.
#pragma unroll 1
    for (int dim0 = warp * TOKENS_PER_CHUNK; dim0 < D; dim0 += NWARPS * TOKENS_PER_CHUNK) {
        const int slice = dim0 / GGML_CUDA_FATTN_KVARN_DIM;
        const int local_dim0 = dim0 % GGML_CUDA_FATTN_KVARN_DIM;
        T_C out[Q_TILE];
#pragma unroll
        for (int qt = 0; qt < Q_TILE; ++qt) {
#pragma unroll
            for (int l = 0; l < T_C::ne; ++l) {
                out[qt].x[l] = 0.0f;
            }
        }

#pragma unroll 1
        for (int chunk = 0; chunk < TOKEN_CHUNKS; ++chunk) {
            const int token0 = token_begin + chunk * TOKENS_PER_CHUNK;
            T_A v_a;
            // Здесь строка фрагмента — это измерение, а не токен, но приём тот же:
            // после перестановки lp и lp+1 берут соседние измерения одной строки
            // записи V (строка = токен), то есть снова одна загрузка на пару.
#pragma unroll
            for (int lp = 0; lp < T_A::ne; lp += 2) {
                const int local_dim = KVARN_FRAG_ROW(T_A::get_i(lp));
                const int token_local = 2 * T_A::get_j(lp);
                const int dim_a = local_dim0 + local_dim + 0;
                const int dim_b = local_dim0 + local_dim + 1;
                float x00, x01, x10, x11;
                const int pos0 = v_tile.pos_begin + chunk * TOKENS_PER_CHUNK + token_local + 0;
                const int pos1 = v_tile.pos_begin + chunk * TOKENS_PER_CHUNK + token_local + 1;
                if (v_from_record && pos1 < GGML_CUDA_FATTN_KVARN_DIM) {
                    const uint8_t * row0 = v_records[slice] + pos0 * v_row_bytes;
                    const uint8_t * row1 = v_records[slice] + pos1 * v_row_bytes;
                    int q0a, q0b, q1a, q1b;
                    ggml_cuda_fattn_kvarn_decode_unpack2<V_BITS>(row0, dim_a, q0a, q0b);
                    ggml_cuda_fattn_kvarn_decode_unpack2<V_BITS>(row1, dim_a, q1a, q1b);
                    const float other_a = KVARN_AXIS_OTHER(slice, dim_a);
                    const float other_b = KVARN_AXIS_OTHER(slice, dim_b);
                    const float s0 = KVARN_AXIS_SCALE(slice, pos0);
                    const float z0 = KVARN_AXIS_ZP(slice, pos0);
                    const float s1 = KVARN_AXIS_SCALE(slice, pos1);
                    const float z1 = KVARN_AXIS_ZP(slice, pos1);
                    x00 = (float(q0a) * s0 + z0) * other_a;
                    x10 = (float(q1a) * s1 + z1) * other_a;
                    x01 = (float(q0b) * s0 + z0) * other_b;
                    x11 = (float(q1b) * s1 + z1) * other_b;
                } else {
                    const int token0_pair = token0 + token_local;
                    x00 = token0_pair + 0 < token_end ?
                        ggml_cuda_fattn_kvarn_load_rotated(v_desc, token0_pair + 0, slice, dim_a) : 0.0f;
                    x10 = token0_pair + 1 < token_end ?
                        ggml_cuda_fattn_kvarn_load_rotated(v_desc, token0_pair + 1, slice, dim_a) : 0.0f;
                    x01 = token0_pair + 0 < token_end ?
                        ggml_cuda_fattn_kvarn_load_rotated(v_desc, token0_pair + 0, slice, dim_b) : 0.0f;
                    x11 = token0_pair + 1 < token_end ?
                        ggml_cuda_fattn_kvarn_load_rotated(v_desc, token0_pair + 1, slice, dim_b) : 0.0f;
                }
                v_a.x[lp + 0] = make_half2(x00, x10);
                v_a.x[lp + 1] = make_half2(x01, x11);
            }
            // Распакованный фрагмент V — тоже в регистрах, и по нему так же
            // проходят все строки запроса тайла.
#pragma unroll
            for (int qt = 0; qt < Q_TILE; ++qt) {
                if (qt >= q_count) {
                    break;
                }
                T_B p_b;
                load_ldmatrix(p_b, p_sh[qt][0] + chunk * (TOKENS_PER_CHUNK / 2), P_STRIDE2);
                mma(out[qt], v_a, p_b);
            }
        }

#pragma unroll
        for (int qt = 0; qt < Q_TILE; ++qt) {
            if (qt >= q_count) {
                continue;
            }
#pragma unroll
            for (int l = 0; l < T_C::ne; ++l) {
                const int dim = dim0 + KVARN_FRAG_ROW(T_C::get_i(l));
                const int head = T_C::get_j(l);
                const int q_head = q_head0 + head;
                if (head < gqa_head_count && q_head < n_q_heads) {
                    const size_t base =
                        (((size_t) stream * n_q + (q_base + qt)) * n_q_heads + q_head) * n_splits + split;
                    partial[base * D + dim] = out[qt].x[l];
                }
            }
        }
        __syncwarp();
    }
    __syncthreads();

    if (tid < gqa_head_count && q_head0 + tid < n_q_heads) {
        const int q_head = q_head0 + tid;
#pragma unroll
        for (int qt = 0; qt < Q_TILE; ++qt) {
            if (qt >= q_count) {
                continue;
            }
            const size_t base =
                (((size_t) stream * n_q + (q_base + qt)) * n_q_heads + q_head) * n_splits + split;
            partial_meta[base] = make_float2(m_sh[qt][tid], denom_sh[qt][tid]);
        }
    }
}

static inline int ggml_cuda_fattn_kvarn_decode_div_up_i64(const int64_t x, const int y) {
    return (int) ((x + y - 1) / y);
}

// The combine reduction uses n_splits floats of dynamic shared memory. Keep
// the original per-specialization preparation in the caller while caching the
// canonical kernel pointer, so steady-state launches add no cross-TU host call.
template<int D>
static void ggml_cuda_fattn_kvarn_decode_combine_prepare(
        ggml_cuda_fattn_kvarn_decode_combine_kernel_t kernel,
        const int nbytes_shared_combine) {
    const int device = ggml_cuda_get_device();
    GGML_ASSERT(device >= 0 && device < GGML_CUDA_MAX_DEVICES);
    constexpr int nbytes_static_combine = GGML_CUDA_FATTN_KVARN_DECODE_THREADS * (int) sizeof(float);
    const int smem_max = (int) ggml_cuda_info().devices[device].smpbo - nbytes_static_combine;
    GGML_ASSERT(nbytes_shared_combine <= smem_max &&
        "KVarN decode combine shared-mem demand exceeds device opt-in ceiling");
#if !defined(GGML_USE_MUSA)
    static bool raised[GGML_CUDA_MAX_DEVICES] = {};
    if (!raised[device]) {
        CUDA_CHECK(cudaFuncSetAttribute(
            reinterpret_cast<const void *>(kernel),
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max));
        raised[device] = true;
    }
#else
    GGML_UNUSED(kernel);
#endif
}

static inline int ggml_cuda_fattn_kvarn_decode_wave_efficiency_percent(
        const int64_t blocks_total,
        const int blocks_per_wave,
        int * n_waves) {
    if (blocks_total <= 0 || blocks_per_wave <= 0) {
        *n_waves = 0;
        return 0;
    }
    *n_waves = ggml_cuda_fattn_kvarn_decode_div_up_i64(blocks_total, blocks_per_wave);
    return (int) (100 * blocks_total / ((int64_t) *n_waves * blocks_per_wave));
}

template<int D, int MAX_GQA, int SPLIT_TOKENS, int NWARPS, int K_BITS, int V_BITS, int Q_TILE = 1>
static int ggml_cuda_fattn_kvarn_decode_active_blocks_per_sm() {
    // Occupancy depends only on the compiled kernel and the device arch, not the workload,
    // so cache it per device: the selector queries several candidates on every decode op.
    static int cache[GGML_CUDA_MAX_DEVICES] = {};
    const int device = ggml_cuda_get_device();
    if (device >= 0 && device < GGML_CUDA_MAX_DEVICES && cache[device] > 0) {
        return cache[device];
    }
    const int wave_size = ggml_cuda_info().devices[device].warp_size;
    const dim3 block_dim(wave_size, NWARPS, 1);
    int max_blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm,
        ggml_cuda_fattn_kvarn_decode_mma_kernel<D, MAX_GQA, SPLIT_TOKENS, NWARPS, K_BITS, V_BITS, Q_TILE>,
        block_dim.x * block_dim.y * block_dim.z,
        0));
    if (device >= 0 && device < GGML_CUDA_MAX_DEVICES) {
        cache[device] = max_blocks_per_sm;
    }
    return max_blocks_per_sm;
}

template<int D, int MAX_GQA, int SPLIT_TOKENS, int NWARPS, int K_BITS, int V_BITS, int Q_TILE = 1>
static void ggml_cuda_fattn_kvarn_decode_consider(
        ggml_cuda_fattn_kvarn_decode_geometry & best,
        int64_t & best_score,
        const int nsm,
        const int n_kv,
        const int n_q,
        const int n_q_heads,
        const int n_kv_heads,
        const int n_stream,
        const int only_split = 0,
        const int only_nwarps = 0,
        const int only_gqa = 0) {
    if (only_split > 0 && (SPLIT_TOKENS != only_split || NWARPS != only_nwarps ||
            MAX_GQA != only_gqa)) {
        return;
    }
    // On RTX 3090, split-64 is 1.8% faster at 16K for a single query, while
    // split-128 is 0.6% faster at 32K and 4.5% faster at 64K. Multi-query
    // verification keeps split-128 at shorter depths because it enables query
    // tiling and avoids duplicated partial-output traffic.
    if constexpr (SPLIT_TOKENS == 128) {
        if (n_q == 1 && n_kv < 32768) {
            return;
        }
    }
    const int gqa_ratio = n_q_heads / n_kv_heads;
    const int n_splits = ggml_cuda_fattn_kvarn_decode_div_up_i64(n_kv, SPLIT_TOKENS);
    const int n_gqa_blocks = ggml_cuda_fattn_kvarn_decode_div_up_i64(gqa_ratio, MAX_GQA);
    const int max_blocks_per_sm =
        ggml_cuda_fattn_kvarn_decode_active_blocks_per_sm<D, MAX_GQA, SPLIT_TOKENS, NWARPS, K_BITS, V_BITS, Q_TILE>();
    if (max_blocks_per_sm <= 0 || n_splits <= 0 || n_gqa_blocks <= 0) {
        return;
    }
    // Тайл шире одной строки запроса имеет смысл только когда строк несколько:
    // при n_q == 1 он ничего не экономит, а разделяемой памяти просит больше.
    if (Q_TILE > 1 && n_q < Q_TILE) {
        return;
    }
    ++best.candidate_count;

    const int n_q_tiles = ggml_cuda_fattn_kvarn_decode_div_up_i64(n_q, Q_TILE);
    const int blocks_per_wave = nsm * max_blocks_per_sm;
    const int64_t blocks_total =
        (int64_t) n_splits * n_kv_heads * n_gqa_blocks * n_q_tiles * n_stream;
    int n_waves = 0;
    const int wave_efficiency_percent =
        ggml_cuda_fattn_kvarn_decode_wave_efficiency_percent(blocks_total, blocks_per_wave, &n_waves);

    // Upstream queries actual occupancy, then scores wave efficiency inside one kernel family.
    // KVarN is choosing between different CTA shapes; when split/combine work is equal, the
    // occupancy result is the architectural signal, and wave efficiency is secondary.
    // Вес эффективности волны зависит от того, сколько волн вообще будет.
    //
    // Прежняя постоянная 10000 делала четыре процентных пункта эффективности
    // дороже, чем двукратное сокращение числа сплитов, и геометрия на 64 токена
    // выигрывала всегда:
    //
    //   n_kv=65024: split=64  волн=25 эффективность=99% оценка=4987984
    //               split=128 волн=13 эффективность=95% оценка=4948492
    //
    // Замер показывает обратное: принудительный сплит 128 даёт 29.70 против
    // 28.36 tok/s на одном слоте, то есть +4.7% при побитово том же выводе.
    // Причина в том, что у блока есть постоянная цена — загрузка Q, осей,
    // последовательный расчёт zq, барьеры подготовки, — и она не зависит от
    // длины сплита. Вдвое меньше блоков означает вдвое меньше этой цены.
    //
    // Потеря на хвосте последней волны стоит своего веса только когда волн
    // мало: при десятке волн недогруз последней размазан по всем и почти ничего
    // не стоит. Порог в четыре волны отделяет «параллелизма с запасом» от «его в
    // обрез».
    //
    // Вес подобран по ЗАМЕРЕННЫМ значениям, а не по предполагаемым. Отладка
    // маршрутов на реальной нагрузке (n_kv=65024, один слот, D=256, k5/v5):
    //
    //   сплит  64: сплитов 1016, блоков на SM 4, эффективность 95%, волн 13
    //   сплит 128: сплитов  508, блоков на SM 4, эффективность 88%, волн  7
    //
    // Разрыв по эффективности семь пунктов, поэтому при весе 100 выигрывал
    // сплит 64 с перевесом 192 — а замер показывает обратное: 32.51 против
    // 31.00 tok/s в пользу 128. То есть эффективность волны была переоценена
    // как минимум в этом отношении. При весе 10 сплит 128 побеждает на всех
    // глубинах от 40 тысяч до 130 тысяч и при одном и при двух слотах, с
    // перевесом не меньше 312.
    //
    // Осторожно: 100 давало верный ответ почти везде и ошибалось лишь в узком
    // окне около n_kv=65024 при одном слоте. Проверять такие пороги надо
    // отладкой маршрутов на реальной нагрузке, а не подстановкой в формулу:
    // именно подстановка предполагаемого числа блоков (8 вместо истинных 4)
    // и увела оценку.
    // ОЦЕНКА КАНДИДАТА. Прежняя ставила первым слагаемым число резидентных
    // блоков с весом миллион, а эффективность волны и число сплитов — мелкими
    // поправками. Это оказалось неверной моделью, и вот на каких замерах она
    // сломалась (глубина 60000, D=256, kvarn5, MTP n_max 2):
    //
    //   тайл 1, сплит 128: блоков на SM 4, волн 18 -> 46.90 tok/s
    //   тайл 2, сплит 128: блоков на SM 4, волн 12 -> 49.84
    //   тайл 3, сплит 128: блоков на SM 3, волн  8 -> 53.42
    //
    // Тайл на три строки ЛУЧШИЙ, хотя резидентных блоков у него меньше. Прежняя
    // оценка отвергла бы его: штраф за блок (миллион) перевешивал любую премию.
    //
    // Верная величина — число волн. Оно уже содержит в себе и занятость, и
    // число блоков: waves = blocks_total / (nsm * blocks_per_sm), а blocks_total
    // падает вместе с шириной тайла. Время ядра пропорционально волнам, и это
    // подтверждается: 18 / 12 / 8 волн против 46.90 / 49.84 / 53.42 tok/s при
    // доле внимания около четверти шага.
    //
    // Одних волн, однако, мало. Буфер частичных сумм имеет размер
    // n_splits * n_q * n_q_heads * D, то есть пропорционален числу сплитов, и
    // его запись плюс чтение объединяющим ядром — настоящий трафик памяти.
    // Замер при РАВНЫХ волнах (12 против 12, n_q = 2): сплит 128 даёт 46.14,
    // сплит 64 при том же числе волн — 27.58. Разница целиком в этом буфере.
    // Поэтому число сплитов входит отдельным штрафом.
    //
    // Вес штрафа выбран осторожно: 4000 за сплит означает, что удвоение числа
    // сплитов при n_kv=60000 стоит примерно двух волн. Замер говорит, что стоит
    // больше, но направление важнее величины, а заниженный вес безопаснее.
    //
    // Проверка формулы на всех имеющихся замерах:
    //   n_q=1: сплит 64 (волн 13, сплитов 1016) против 128 (волн 7, сплитов 508)
    //          -> 128 выигрывает; замер: 32.51 против 31.00 в пользу 128. Верно.
    //   n_q=2: тайл1/128 (волн 12, сплитов 470) против тайл2/64 (волн 12, 938)
    //          -> 128 выигрывает; замер: 46.14 против 27.58. Верно.
    //   n_q=3: тайл3 (волн 8) > тайл2 (волн 12) > тайл1 (волн 18). Верно.
    const int64_t score =
        - (int64_t) n_waves * 1000000
        - (int64_t) n_splits * 4000
        + (int64_t) wave_efficiency_percent * 100
        - (int64_t) n_gqa_blocks * 10;

    if (score > best_score) {
        best_score = score;
        best.use_split = true;
        best.split_tokens = SPLIT_TOKENS;
        best.nwarps = NWARPS;
        best.gqa_per_block = MAX_GQA;
        best.n_splits = n_splits;
        best.n_gqa_blocks = n_gqa_blocks;
        best.max_blocks_per_sm = max_blocks_per_sm;
        best.wave_efficiency_percent = wave_efficiency_percent;
        best.n_waves = n_waves;
        best.q_tile = Q_TILE;
    }
}

template<int D, int K_BITS, int V_BITS>
ggml_cuda_fattn_kvarn_decode_geometry ggml_cuda_fattn_kvarn_decode_select(
        int device,
        int n_kv,
        int n_q,
        int n_q_heads,
        int n_kv_heads,
        int n_stream) {
    ggml_cuda_fattn_kvarn_decode_geometry best = {};
    if (n_kv <= 0 || n_q <= 0 || n_q_heads <= 0 || n_kv_heads <= 0 ||
            n_stream <= 0 || n_q_heads % n_kv_heads != 0) {
        return best;
    }

    const int nsm = ggml_cuda_info().devices[device].nsm;
    int64_t best_score = INT64_MIN;
    // Select the KV partition using single-query geometry so generation and
    // speculative verification use the same online-softmax summation order.
    // A second pass below may still select a wider query tile without changing
    // that partition.
    const int n_q_geometry = 1;
    if constexpr (D == 512) {
        ggml_cuda_fattn_kvarn_decode_consider<D, 6, 64,  8, K_BITS, V_BITS>(
            best, best_score, nsm, n_kv, n_q_geometry, n_q_heads, n_kv_heads, n_stream);
        ggml_cuda_fattn_kvarn_decode_consider<D, 8, 64,  8, K_BITS, V_BITS>(
            best, best_score, nsm, n_kv, n_q_geometry, n_q_heads, n_kv_heads, n_stream);
        ggml_cuda_fattn_kvarn_decode_consider<D, 6, 64, 16, K_BITS, V_BITS>(
            best, best_score, nsm, n_kv, n_q_geometry, n_q_heads, n_kv_heads, n_stream);
        ggml_cuda_fattn_kvarn_decode_consider<D, 8, 64, 16, K_BITS, V_BITS>(
            best, best_score, nsm, n_kv, n_q_geometry, n_q_heads, n_kv_heads, n_stream);
    } else if constexpr (D == 256) {
        ggml_cuda_fattn_kvarn_decode_consider<D, 6, 64, 8, K_BITS, V_BITS>(
            best, best_score, nsm, n_kv, n_q_geometry, n_q_heads, n_kv_heads, n_stream);
        ggml_cuda_fattn_kvarn_decode_consider<D, 8, 64, 8, K_BITS, V_BITS>(
            best, best_score, nsm, n_kv, n_q_geometry, n_q_heads, n_kv_heads, n_stream);
        // Сплит на 128 токенов совпадает с группой записи: постоянная цена
        // блока (загрузка Q, осей, последовательный расчёт zq, барьеры)
        // раскладывается на вдвое большее число токенов, а число сплитов —
        // и вместе с ним буфер частичных сумм и работа объединения — вдвое
        // падает. Разделяемой памяти нужно на ~5 КиБ больше, четыре блока на
        // мультипроцессор при этом сохраняются.
        ggml_cuda_fattn_kvarn_decode_consider<D, 6, 128, 8, K_BITS, V_BITS>(
            best, best_score, nsm, n_kv, n_q_geometry, n_q_heads, n_kv_heads, n_stream);
        ggml_cuda_fattn_kvarn_decode_consider<D, 8, 128, 8, K_BITS, V_BITS>(
            best, best_score, nsm, n_kv, n_q_geometry, n_q_heads, n_kv_heads, n_stream);
        // Тайл на две строки запроса — ТОЛЬКО со сплитом 128, и вот почему.
        //
        // Буфер частичных сумм имеет размер n_splits * n_q * n_q_heads * D: он
        // пропорционален числу сплитов. Сплит 64 удваивает число сплитов, а
        // значит и запись этого буфера, и его чтение объединяющим ядром. При
        // n_kv=60000, n_q=2, 24 головах и D=256 это 46 МБ против 23 на каждый
        // вызов внимания.
        //
        // ЗАМЕРЕНО. Первая версия правки сажала тайл на сплит 64, потому что там
        // хватало разделяемой памяти. При РАВНОМ числе блоков и равном числе
        // волн (12 против 12, n_q=2, глубина 60000) она дала 27.58 против 46.14
        // tok/s: лишний трафик буфера съел всю экономию на распаковке и добавил
        // сверху. Экономия на распаковке реальна, но она вдвое меньше того, что
        // стоит удвоение частичных сумм.
        //
        // Бюджет разделяемой памяти при сплите 128 (числа из cuobjdump):
        //   MAX_GQA 6, тайл 1: 13632, вторая строка добавляет 9024 -> 22656, влезает
        //   MAX_GQA 8, тайл 1: 16256, вторая строка добавляет 10624 -> 26880, нет
        // Порог четырёх блоков на мультипроцессор — 25344 байта. Поэтому широкий
        // тайл существует только в варианте на шесть голов GQA; при gqa_ratio 6
        // этого ровно достаточно, чтобы обойтись одним блоком по головам.
        ggml_cuda_fattn_kvarn_decode_consider<D, 6, 128, 8, K_BITS, V_BITS, 2>(
            best, best_score, nsm, n_kv, n_q_geometry, n_q_heads, n_kv_heads, n_stream);
        // Тайл на три строки укладывает типичный шаг MTP (n_q = 3) в один блок
        // вместо двух, но просит 31680 байт разделяемой памяти — это три блока
        // на мультипроцессор вместо четырёх. Размен проверяется замером.
        ggml_cuda_fattn_kvarn_decode_consider<D, 6, 128, 8, K_BITS, V_BITS, 3>(
            best, best_score, nsm, n_kv, n_q_geometry, n_q_heads, n_kv_heads, n_stream);
    } else if constexpr (D == 128) {
        ggml_cuda_fattn_kvarn_decode_consider<D, 6, 64, 4, K_BITS, V_BITS>(
            best, best_score, nsm, n_kv, n_q_geometry, n_q_heads, n_kv_heads, n_stream);
        ggml_cuda_fattn_kvarn_decode_consider<D, 8, 64, 4, K_BITS, V_BITS>(
            best, best_score, nsm, n_kv, n_q_geometry, n_q_heads, n_kv_heads, n_stream);
    }

    if (!best.use_split || best.n_splits <= 1) {
        best.use_split = false;
        return best;
    }

    if constexpr (D == 256) {
        if (n_q > 1) {
            const int only_split = best.split_tokens;
            const int only_nwarps = best.nwarps;
            const int only_gqa = best.gqa_per_block;
            ggml_cuda_fattn_kvarn_decode_geometry tiled = {};
            int64_t tiled_score = INT64_MIN;
            ggml_cuda_fattn_kvarn_decode_consider<D, 6, 64, 8, K_BITS, V_BITS>(
                tiled, tiled_score, nsm, n_kv, n_q, n_q_heads, n_kv_heads, n_stream,
                only_split, only_nwarps, only_gqa);
            ggml_cuda_fattn_kvarn_decode_consider<D, 8, 64, 8, K_BITS, V_BITS>(
                tiled, tiled_score, nsm, n_kv, n_q, n_q_heads, n_kv_heads, n_stream,
                only_split, only_nwarps, only_gqa);
            ggml_cuda_fattn_kvarn_decode_consider<D, 6, 128, 8, K_BITS, V_BITS>(
                tiled, tiled_score, nsm, n_kv, n_q, n_q_heads, n_kv_heads, n_stream,
                only_split, only_nwarps, only_gqa);
            ggml_cuda_fattn_kvarn_decode_consider<D, 8, 128, 8, K_BITS, V_BITS>(
                tiled, tiled_score, nsm, n_kv, n_q, n_q_heads, n_kv_heads, n_stream,
                only_split, only_nwarps, only_gqa);
            ggml_cuda_fattn_kvarn_decode_consider<D, 6, 128, 8, K_BITS, V_BITS, 2>(
                tiled, tiled_score, nsm, n_kv, n_q, n_q_heads, n_kv_heads, n_stream,
                only_split, only_nwarps, only_gqa);
            ggml_cuda_fattn_kvarn_decode_consider<D, 6, 128, 8, K_BITS, V_BITS, 3>(
                tiled, tiled_score, nsm, n_kv, n_q, n_q_heads, n_kv_heads, n_stream,
                only_split, only_nwarps, only_gqa);
            if (tiled.use_split) {
                const int candidates = best.candidate_count + tiled.candidate_count;
                best = tiled;
                best.candidate_count = candidates;
            }
        }
    }

    const int direct_blocks_per_wave = nsm * best.max_blocks_per_sm;
    const int64_t direct_blocks =
        (int64_t) n_kv_heads * best.n_gqa_blocks * n_q * n_stream;
    int direct_waves = 0;
    const int direct_efficiency_percent =
        ggml_cuda_fattn_kvarn_decode_wave_efficiency_percent(
            direct_blocks, direct_blocks_per_wave, &direct_waves);

    // Mirror upstream's "don't add parallel work when the base grid is already efficient"
    // idea. Once the unsplit decode would occupy at least two efficient waves, the split
    // combine pass is usually worse than falling through to the generic native MMA path.
    if (direct_waves >= 2 && direct_efficiency_percent >= 75) {
        best.use_split = false;
    }
    return best;
}

template<int D, int MAX_GQA, int SPLIT_TOKENS, int NWARPS, int K_BITS, int V_BITS, int Q_TILE = 1>
static void ggml_cuda_fattn_kvarn_decode_launch_geometry(
        const ggml_cuda_fattn_kvarn_decode_args & args,
        const dim3 blocks_split) {
    ggml_cuda_fattn_kvarn_decode_mma_kernel<D, MAX_GQA, SPLIT_TOKENS, NWARPS, K_BITS, V_BITS, Q_TILE>
        <<<blocks_split, dim3(args.wave_size, NWARPS, 1), 0, args.stream>>>(
            args.Q, args.k_descs, args.v_descs, args.mask, args.partial, args.partial_meta,
            args.scale, args.logit_softcap, args.nb01, args.nb02, args.nb03,
            args.nb30, args.nb31, args.nb33, args.ne33, args.n_kv, args.n_q,
            args.n_q_heads, args.n_kv_heads, args.gqa_ratio, args.n_gqa_blocks, args.n_splits);
}

template<int D, int MAX_GQA, int K_BITS, int V_BITS>
static void ggml_cuda_fattn_kvarn_decode_launch_gqa(
        const ggml_cuda_fattn_kvarn_decode_args & args,
        const dim3 blocks_split) {
    if constexpr (D == 512) {
        if (args.split_tokens == 64 && args.nwarps == 8) {
            ggml_cuda_fattn_kvarn_decode_launch_geometry<D, MAX_GQA, 64, 8, K_BITS, V_BITS>(args, blocks_split);
            return;
        }
        if (args.split_tokens == 64 && args.nwarps == 16) {
            ggml_cuda_fattn_kvarn_decode_launch_geometry<D, MAX_GQA, 64, 16, K_BITS, V_BITS>(args, blocks_split);
            return;
        }
    } else if constexpr (D == 256) {
        if (args.split_tokens == 64 && args.nwarps == 8) {
            ggml_cuda_fattn_kvarn_decode_launch_geometry<D, MAX_GQA, 64, 8, K_BITS, V_BITS>(args, blocks_split);
            return;
        }
        // Проверка широкого тайла обязана идти ПЕРЕД общей веткой сплита 128,
        // иначе та перехватит запуск и тайл никогда не выполнится.
        if (args.split_tokens == 128 && args.nwarps == 8 && args.q_tile == 3 && MAX_GQA == 6) {
            ggml_cuda_fattn_kvarn_decode_launch_geometry<D, MAX_GQA, 128, 8, K_BITS, V_BITS, 3>(args, blocks_split);
            return;
        }
        if (args.split_tokens == 128 && args.nwarps == 8 && args.q_tile == 2 && MAX_GQA == 6) {
            ggml_cuda_fattn_kvarn_decode_launch_geometry<D, MAX_GQA, 128, 8, K_BITS, V_BITS, 2>(args, blocks_split);
            return;
        }
        if (args.split_tokens == 128 && args.nwarps == 8) {
            ggml_cuda_fattn_kvarn_decode_launch_geometry<D, MAX_GQA, 128, 8, K_BITS, V_BITS>(args, blocks_split);
            return;
        }
    } else if constexpr (D == 128) {
        if (args.split_tokens == 64 && args.nwarps == 4) {
            ggml_cuda_fattn_kvarn_decode_launch_geometry<D, MAX_GQA, 64, 4, K_BITS, V_BITS>(args, blocks_split);
            return;
        }
    }
    GGML_ABORT("unsupported KVarN decode geometry D=%d split=%d nwarps=%d max_gqa=%d",
        D, args.split_tokens, args.nwarps, MAX_GQA);
}

template<int D, int K_BITS, int V_BITS>
void ggml_cuda_fattn_kvarn_decode_launch(const ggml_cuda_fattn_kvarn_decode_args & args) {
    // Строки запроса делятся на тайлы: один блок ведёт q_tile строк сразу, и
    // именно на столько же падает число блоков, а вместе с ним — повторная
    // распаковка записей кэша.
    const int q_tile = args.q_tile > 0 ? args.q_tile : 1;
    const int n_q_tiles = (args.n_q + q_tile - 1) / q_tile;
    const dim3 blocks_split(
        (uint32_t) args.n_splits,
        (uint32_t) (args.n_kv_heads * args.n_gqa_blocks * n_q_tiles),
        (uint32_t) args.n_stream);

    if (args.gqa_per_block == 6) {
        ggml_cuda_fattn_kvarn_decode_launch_gqa<D, 6, K_BITS, V_BITS>(args, blocks_split);
    } else if (args.gqa_per_block == 8) {
        ggml_cuda_fattn_kvarn_decode_launch_gqa<D, 8, K_BITS, V_BITS>(args, blocks_split);
    } else {
        GGML_ABORT("unsupported KVarN decode GQA block size %d", args.gqa_per_block);
    }
    CUDA_CHECK(cudaGetLastError());

    static const ggml_cuda_fattn_kvarn_decode_combine_kernel_t combine_kernel =
        ggml_cuda_fattn_kvarn_decode_combine_get_kernel<D>();
    const dim3 blocks_combine((uint32_t) args.n_q_heads, (uint32_t) args.n_q, (uint32_t) args.n_stream);
    const int nbytes_shared_combine = args.n_splits * (int) sizeof(float);
    ggml_cuda_fattn_kvarn_decode_combine_prepare<D>(combine_kernel, nbytes_shared_combine);
    combine_kernel
        <<<blocks_combine, GGML_CUDA_FATTN_KVARN_DECODE_THREADS, nbytes_shared_combine, args.stream>>>(
            args.partial, args.partial_meta, args.dst, args.dst_meta,
            args.n_splits, args.n_q, args.n_q_heads);
    CUDA_CHECK(cudaGetLastError());
}
