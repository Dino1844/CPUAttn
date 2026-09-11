#ifndef MINI_CPUATTN_SIMD_LINEAR_H
#define MINI_CPUATTN_SIMD_LINEAR_H

#include <stdint.h>

#include "isa.h"

static inline void cpuattn_zero(float *target, int64_t n) {
    int64_t i = 0;
    for (; i + CPUATTN_SIMD_LANES <= n; i += CPUATTN_SIMD_LANES)
        cpuattn_simd_store(target + i, cpuattn_simd_zero());
    for (; i < n; ++i) target[i] = 0.0f;
}

static inline void cpuattn_axpy_inplace(
    float *restrict target,
    const float *restrict source,
    int64_t n,
    float scale) {
    int64_t i = 0;
    for (; i + CPUATTN_SIMD_LANES <= n; i += CPUATTN_SIMD_LANES) {
        cpuattn_simd_t total = cpuattn_simd_load(target + i);
        total = cpuattn_simd_fma(scale, cpuattn_simd_load(source + i), total);
        cpuattn_simd_store(target + i, total);
    }
    for (; i < n; ++i) target[i] += scale * source[i];
}

static inline void cpuattn_copy(
    float *restrict target, const float *restrict source, int64_t n) {
    int64_t i = 0;
    for (; i + CPUATTN_SIMD_LANES <= n; i += CPUATTN_SIMD_LANES)
        cpuattn_simd_store(target + i, cpuattn_simd_load(source + i));
    for (; i < n; ++i) target[i] = source[i];
}

static inline float cpuattn_dot(
    const float *restrict left, const float *restrict right, int64_t n) {
    cpuattn_simd_t total = cpuattn_simd_zero();
    int64_t i = 0;
    for (; i + CPUATTN_SIMD_LANES <= n; i += CPUATTN_SIMD_LANES)
        total = cpuattn_simd_fma_vec(
            cpuattn_simd_load(left + i), cpuattn_simd_load(right + i), total);
    float result = cpuattn_simd_reduce_add(total);
    for (; i < n; ++i) result += left[i] * right[i];
    return result;
}

static inline void cpuattn_linear_readout_block(
    const float *restrict q,
    const float *restrict state,
    float *restrict output,
    int64_t d_size,
    int64_t dv_size,
    int rows,
    const float *restrict row_scale) {
    for (int64_t dv = 0; dv < dv_size; dv += CPUATTN_SIMD_LANES) {
        int width = (int)(dv_size - dv < CPUATTN_SIMD_LANES
            ? dv_size - dv : CPUATTN_SIMD_LANES);
        cpuattn_simd_t total[LINEAR_BLOCK];
        for (int m = 0; m < rows; ++m) total[m] = cpuattn_simd_zero();
        for (int64_t d = 0; d < d_size; ++d) {
            cpuattn_simd_t values = cpuattn_simd_load_partial(
                state + d * dv_size + dv, width);
            for (int m = 0; m < rows; ++m)
                total[m] = cpuattn_simd_fma(
                    q[(int64_t)m * d_size + d] * row_scale[m], values, total[m]);
        }
        for (int m = 0; m < rows; ++m)
            cpuattn_simd_store_partial(
                output + (int64_t)m * dv_size + dv, total[m], width);
    }
}

static inline void cpuattn_linear_values_block(
    const float *restrict weights,
    const float *restrict values,
    float *restrict output,
    int64_t dv_size,
    int rows) {
    for (int64_t dv = 0; dv < dv_size; dv += CPUATTN_SIMD_LANES) {
        int width = (int)(dv_size - dv < CPUATTN_SIMD_LANES
            ? dv_size - dv : CPUATTN_SIMD_LANES);
        cpuattn_simd_t total[LINEAR_BLOCK];
        for (int m = 0; m < rows; ++m)
            total[m] = cpuattn_simd_load_partial(
                output + (int64_t)m * dv_size + dv, width);
        for (int j = 0; j < rows; ++j) {
            cpuattn_simd_t value = cpuattn_simd_load_partial(
                values + (int64_t)j * dv_size + dv, width);
            for (int m = 0; m < rows; ++m)
                total[m] = cpuattn_simd_fma(
                    weights[m * LINEAR_BLOCK + j], value, total[m]);
        }
        for (int m = 0; m < rows; ++m)
            cpuattn_simd_store_partial(
                output + (int64_t)m * dv_size + dv, total[m], width);
    }
}

static inline void cpuattn_state_update_block(
    float *restrict state,
    const float *restrict v_block,
    const float *restrict k_block,
    const float *restrict suffix,
    int64_t d_size,
    int64_t dv_size,
    int rows,
    float block_scale) {
    for (int64_t d = 0; d < d_size; ++d) {
        float *row = state + d * dv_size;
        for (int64_t dv = 0; dv < dv_size; dv += CPUATTN_SIMD_LANES) {
            int width = (int)(dv_size - dv < CPUATTN_SIMD_LANES
                ? dv_size - dv : CPUATTN_SIMD_LANES);
            cpuattn_simd_t acc = cpuattn_simd_mul(
                cpuattn_simd_load_partial(row + dv, width),
                cpuattn_simd_set1(block_scale));
            for (int j = 0; j < rows; ++j) {
                acc = cpuattn_simd_fma(
                    k_block[(int64_t)j * d_size + d] * suffix[j],
                    cpuattn_simd_load_partial(
                        v_block + (int64_t)j * dv_size + dv, width),
                    acc);
            }
            cpuattn_simd_store_partial(row + dv, acc, width);
        }
    }
}

#endif
