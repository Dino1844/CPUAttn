#ifndef MINI_CPUATTN_SIMD_EXPR_H
#define MINI_CPUATTN_SIMD_EXPR_H

#include "isa.h"

static inline cpuattn_simd_t cpuattn_simd_load_strided(
    const float *pointer, int64_t stride) {
    if (stride == 1) return cpuattn_simd_load(pointer);
    float lanes[CPUATTN_SIMD_LANES];
    for (int i = 0; i < CPUATTN_SIMD_LANES; ++i) lanes[i] = pointer[(int64_t)i * stride];
    return cpuattn_simd_load(lanes);
}

static inline cpuattn_simd_t cpuattn_simd_iota(float first) {
    float lanes[CPUATTN_SIMD_LANES];
    for (int i = 0; i < CPUATTN_SIMD_LANES; ++i) lanes[i] = first + (float)i;
    return cpuattn_simd_load(lanes);
}

static inline cpuattn_simd_t cpuattn_simd_neg(cpuattn_simd_t value) {
    return cpuattn_simd_sub(cpuattn_simd_zero(), value);
}

static inline cpuattn_simd_t cpuattn_simd_sigmoid(cpuattn_simd_t value) {
    return cpuattn_simd_div(
        cpuattn_simd_set1(1.0f),
        cpuattn_simd_add(
            cpuattn_simd_set1(1.0f),
            cpuattn_simd_exp(cpuattn_simd_neg(value))));
}

static inline cpuattn_simd_t cpuattn_simd_tanh(cpuattn_simd_t value) {
    return cpuattn_simd_sub(
        cpuattn_simd_mul(
            cpuattn_simd_set1(2.0f),
            cpuattn_simd_sigmoid(
                cpuattn_simd_mul(cpuattn_simd_set1(2.0f), value))),
        cpuattn_simd_set1(1.0f));
}

static inline cpuattn_simd_t cpuattn_simd_relu(cpuattn_simd_t value) {
    return cpuattn_simd_max(value, cpuattn_simd_zero());
}

static inline cpuattn_mask_t cpuattn_mask_true(void) {
    return cpuattn_simd_eq(cpuattn_simd_zero(), cpuattn_simd_zero());
}

static inline cpuattn_mask_t cpuattn_mask_false(void) {
    return cpuattn_simd_ne(cpuattn_simd_zero(), cpuattn_simd_zero());
}

#endif
