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
        /* A compile-time bound keeps the accumulators in registers; rows past
           the valid tail are computed but never stored. */
        cpuattn_simd_t total[LINEAR_BLOCK];
        for (int m = 0; m < LINEAR_BLOCK; ++m) total[m] = cpuattn_simd_zero();
        for (int64_t d = 0; d < d_size; ++d) {
            cpuattn_simd_t values = cpuattn_simd_load_partial(
                state + d * dv_size + dv, width);
            for (int m = 0; m < LINEAR_BLOCK; ++m) {
                /* Rows past the valid tail are computed but never stored; clamp
                   the read so an aliased input is never read out of range. */
                const float *row = q + (int64_t)(m < rows ? m : 0) * d_size;
                total[m] = cpuattn_simd_fma(
                    row[d] * row_scale[m], values, total[m]);
            }
        }
        for (int m = 0; m < LINEAR_BLOCK; ++m)
            if (m < rows)
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
        /* A compile-time row bound keeps total[] in registers; the reduction
           walks only the valid rows so stale weights never enter. */
        cpuattn_simd_t total[LINEAR_BLOCK];
        for (int m = 0; m < LINEAR_BLOCK; ++m)
            total[m] = cpuattn_simd_load_partial(
                output + (int64_t)m * dv_size + dv, width);
        for (int j = 0; j < rows; ++j) {
            cpuattn_simd_t value = cpuattn_simd_load_partial(
                values + (int64_t)j * dv_size + dv, width);
            for (int m = 0; m < LINEAR_BLOCK; ++m)
                total[m] = cpuattn_simd_fma(
                    weights[m * LINEAR_BLOCK + j], value, total[m]);
        }
        for (int m = 0; m < LINEAR_BLOCK; ++m)
            if (m < rows)
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

/* state += left (x) (state^T . right), computed with the reduction held in a
   register so the temporary is written once instead of once per row. */
static inline void cpuattn_linear_rank1(
    float *restrict state,
    const float *restrict right,
    const float *restrict left,
    float *restrict tmp,
    int64_t d_size,
    int64_t dv_size) {
    for (int64_t dv = 0; dv < dv_size; dv += CPUATTN_SIMD_LANES) {
        int width = (int)(dv_size - dv < CPUATTN_SIMD_LANES
            ? dv_size - dv : CPUATTN_SIMD_LANES);
        cpuattn_simd_t acc = cpuattn_simd_zero();
        for (int64_t d = 0; d < d_size; ++d) {
            acc = cpuattn_simd_fma(
                right[d],
                cpuattn_simd_load_partial(state + d * dv_size + dv, width),
                acc);
        }
        cpuattn_simd_store_partial(tmp + dv, acc, width);
    }
    for (int64_t d = 0; d < d_size; ++d) {
        for (int64_t dv = 0; dv < dv_size; dv += CPUATTN_SIMD_LANES) {
            int width = (int)(dv_size - dv < CPUATTN_SIMD_LANES
                ? dv_size - dv : CPUATTN_SIMD_LANES);
            cpuattn_simd_store_partial(
                state + d * dv_size + dv,
                cpuattn_simd_fma(
                    left[d],
                    cpuattn_simd_load_partial(tmp + dv, width),
                    cpuattn_simd_load_partial(state + d * dv_size + dv, width)),
                width);
        }
    }
}

/* output = query^T . state, with the reduction held in a register. */
static inline void cpuattn_linear_matvec(
    float *restrict output,
    const float *restrict query,
    const float *restrict state,
    int64_t d_size,
    int64_t dv_size) {
    for (int64_t dv = 0; dv < dv_size; dv += CPUATTN_SIMD_LANES) {
        int width = (int)(dv_size - dv < CPUATTN_SIMD_LANES
            ? dv_size - dv : CPUATTN_SIMD_LANES);
        cpuattn_simd_t acc = cpuattn_simd_zero();
        for (int64_t d = 0; d < d_size; ++d) {
            acc = cpuattn_simd_fma(
                query[d],
                cpuattn_simd_load_partial(state + d * dv_size + dv, width),
                acc);
        }
        cpuattn_simd_store_partial(output + dv, acc, width);
    }
}

/* Solve (I + L) W = W in place by forward substitution; L is rows x stride
   strictly lower triangular. */
static inline void cpuattn_delta_solve(
    float *restrict w,
    const float *restrict l,
    int rows,
    int stride,
    int64_t dv_size) {
    for (int m = 0; m < rows; ++m) {
        for (int j = 0; j < m; ++j)
            cpuattn_axpy_inplace(
                w + (int64_t)m * dv_size, w + (int64_t)j * dv_size,
                dv_size, -l[(int64_t)m * stride + j]);
    }
}

#endif
