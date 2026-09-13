#ifndef MINI_CPUATTN_SIMD_PARALLEL_H
#define MINI_CPUATTN_SIMD_PARALLEL_H

#include <stdint.h>

#include "isa.h"

static inline void cpuattn_qk_packed_tile(
    const float *restrict q,
    const float *restrict packed_k,
    float *restrict scores,
    int64_t d_size,
    int64_t packed_skv,
    int m_valid) {
    for (int nb = 0; nb < TILE_K; nb += QK_VECTORS * CPUATTN_SIMD_LANES) {
        cpuattn_simd_t total[TILE_Q][QK_VECTORS];
        for (int m = 0; m < m_valid; ++m) {
            for (int column = 0; column < QK_VECTORS; ++column)
                total[m][column] = cpuattn_simd_zero();
        }
        for (int64_t db = 0; db < d_size; db += TILE_D) {
            int64_t dend = db + TILE_D < d_size ? db + TILE_D : d_size;
            for (int64_t d = db; d < dend; ++d) {
                const float *panel = packed_k + d * packed_skv + nb;
                for (int column = 0; column < QK_VECTORS; ++column) {
                    cpuattn_simd_t keys = cpuattn_simd_load(
                        panel + column * CPUATTN_SIMD_LANES);
                    for (int m = 0; m < m_valid; ++m) {
                        float query = q[(int64_t)m * d_size + d];
                        total[m][column] = cpuattn_simd_fma(
                            query, keys, total[m][column]);
                    }
                }
            }
        }
        for (int m = 0; m < m_valid; ++m) {
            for (int column = 0; column < QK_VECTORS; ++column)
                cpuattn_simd_store(
                    scores + m * TILE_K + nb + column * CPUATTN_SIMD_LANES,
                    total[m][column]);
        }
    }
}

static inline void cpuattn_qk_direct_tile(
    const float *restrict q,
    const float *restrict k,
    float *restrict scores,
    int64_t d_size,
    int64_t key_begin,
    int n_valid,
    int m_valid) {
    for (int m = 0; m < m_valid; ++m) {
        const float *query = q + (int64_t)m * d_size;
        for (int n = 0; n < n_valid; ++n) {
            const float *key = k + (key_begin + n) * d_size;
            cpuattn_simd_t total_v = cpuattn_simd_zero();
            int64_t d = 0;
            for (; d + CPUATTN_SIMD_LANES <= d_size; d += CPUATTN_SIMD_LANES) {
                total_v = cpuattn_simd_fma_vec(
                    cpuattn_simd_load(query + d),
                    cpuattn_simd_load(key + d),
                    total_v);
            }
            float total = cpuattn_simd_reduce_add(total_v);
            for (; d < d_size; ++d) total += query[d] * key[d];
            scores[m * TILE_K + n] = total;
        }
    }
}

static inline void cpuattn_softmax_tile(
    float *scores,
    float *row_max,
    float *row_sum,
    float *output_scale,
    int n_valid,
    int m_valid) {
    for (int m = 0; m < m_valid; ++m) {
        float *row = scores + m * TILE_K;
        float next_max = fmaxf(row_max[m], cpuattn_reduce_max(row, n_valid));
        float scale = row_sum[m] > 0.0f ? expf(row_max[m] - next_max) : 0.0f;
        /* Masked lanes hold exactly -INFINITY, written by the caller's masked
           score transform. cpuattn_exp_ps clamps its input to [-87, 88], so
           exp(-INFINITY) would yield ~1.6e-38 rather than 0; the explicit
           select below is what zeroes masked lanes. */
        const cpuattn_simd_t offset = cpuattn_simd_set1(next_max);
        const cpuattn_simd_t negative_infinity = cpuattn_simd_set1(-INFINITY);
        cpuattn_simd_t total = cpuattn_simd_zero();
        float tail_sum = 0.0f;
        int n = 0;
        for (; n + CPUATTN_SIMD_LANES <= n_valid; n += CPUATTN_SIMD_LANES) {
            cpuattn_simd_t values = cpuattn_simd_load(row + n);
            cpuattn_mask_t masked = cpuattn_simd_eq(values, negative_infinity);
            cpuattn_simd_t result = cpuattn_simd_exp(
                cpuattn_simd_sub(values, offset));
            result = cpuattn_simd_select(masked, cpuattn_simd_zero(), result);
            cpuattn_simd_store(row + n, result);
            total = cpuattn_simd_add(total, result);
        }
        for (; n < n_valid; ++n) {
            float value = row[n];
            float result = value == -INFINITY ? 0.0f : expf(value - next_max);
            row[n] = result;
            tail_sum += result;
        }
        row_sum[m] = row_sum[m] * scale
            + cpuattn_simd_reduce_add(total) + tail_sum;
        row_max[m] = next_max;
        output_scale[m] = scale;
    }
}

static inline void cpuattn_pv_tile(
    const float *restrict weights,
    const float *restrict v,
    float *restrict output,
    int64_t dv_size,
    int n_valid,
    int m_valid) {
    for (int64_t dv_block = 0; dv_block < dv_size; dv_block += TILE_DV) {
        int64_t dv_end = dv_block + TILE_DV < dv_size ? dv_block + TILE_DV : dv_size;
        int64_t dv = dv_block;
        for (; dv + CPUATTN_SIMD_LANES <= dv_end; dv += CPUATTN_SIMD_LANES) {
            cpuattn_simd_t total[TILE_Q];
            for (int m = 0; m < m_valid; ++m)
                total[m] = cpuattn_simd_load(output + (int64_t)m * dv_size + dv);
            for (int n = 0; n < n_valid; ++n) {
                /* V is walked column-strided (dv_size between keys), so the
                   hardware prefetcher rarely sees the next line in time. The
                   hint is non-semantic: it changes no result. */
                if (n + 4 < n_valid)
                    __builtin_prefetch(
                        v + (int64_t)(n + 4) * dv_size + dv, 0, 3);
                cpuattn_simd_t values = cpuattn_simd_load(v + (int64_t)n * dv_size + dv);
                for (int m = 0; m < m_valid; ++m)
                    total[m] = cpuattn_simd_fma(
                        weights[m * TILE_K + n], values, total[m]);
            }
            for (int m = 0; m < m_valid; ++m)
                cpuattn_simd_store(output + (int64_t)m * dv_size + dv, total[m]);
        }
        for (; dv < dv_end; ++dv) {
            for (int m = 0; m < m_valid; ++m) {
                float total = output[(int64_t)m * dv_size + dv];
                for (int n = 0; n < n_valid; ++n)
                    total += weights[m * TILE_K + n] * v[(int64_t)n * dv_size + dv];
                output[(int64_t)m * dv_size + dv] = total;
            }
        }
    }
}

static inline void cpuattn_merge_inplace(
    float *restrict target,
    const float *restrict source,
    int64_t n,
    float target_scale,
    float source_scale) {
    int64_t i = 0;
    for (; i + CPUATTN_SIMD_LANES <= n; i += CPUATTN_SIMD_LANES) {
        cpuattn_simd_t total = cpuattn_simd_zero();
        total = cpuattn_simd_fma(
            target_scale, cpuattn_simd_load(target + i), total);
        total = cpuattn_simd_fma(
            source_scale, cpuattn_simd_load(source + i), total);
        cpuattn_simd_store(target + i, total);
    }
    for (; i < n; ++i)
        target[i] = target[i] * target_scale + source[i] * source_scale;
}

#endif
