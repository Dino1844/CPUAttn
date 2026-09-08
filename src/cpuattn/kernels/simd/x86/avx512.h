#ifndef MINI_CPUATTN_SIMD_X86_AVX512_H
#define MINI_CPUATTN_SIMD_X86_AVX512_H

#include <math.h>
#include <stdint.h>
#include <immintrin.h>

#define CPUATTN_SIMD_LANES 16
typedef __m512 cpuattn_simd_t;
typedef __mmask16 cpuattn_mask_t;

static inline cpuattn_simd_t cpuattn_simd_zero(void) {
    return _mm512_setzero_ps();
}

static inline cpuattn_simd_t cpuattn_simd_set1(float value) { return _mm512_set1_ps(value); }
static inline cpuattn_simd_t cpuattn_simd_add(cpuattn_simd_t a, cpuattn_simd_t b) { return _mm512_add_ps(a, b); }
static inline cpuattn_simd_t cpuattn_simd_sub(cpuattn_simd_t a, cpuattn_simd_t b) { return _mm512_sub_ps(a, b); }
static inline cpuattn_simd_t cpuattn_simd_mul(cpuattn_simd_t a, cpuattn_simd_t b) { return _mm512_mul_ps(a, b); }
static inline cpuattn_simd_t cpuattn_simd_div(cpuattn_simd_t a, cpuattn_simd_t b) { return _mm512_div_ps(a, b); }
static inline cpuattn_simd_t cpuattn_simd_min(cpuattn_simd_t a, cpuattn_simd_t b) { return _mm512_min_ps(a, b); }
static inline cpuattn_simd_t cpuattn_simd_max(cpuattn_simd_t a, cpuattn_simd_t b) { return _mm512_max_ps(a, b); }
static inline cpuattn_simd_t cpuattn_simd_abs(cpuattn_simd_t x) {
    return _mm512_abs_ps(x);
}
static inline cpuattn_mask_t cpuattn_simd_lt(cpuattn_simd_t a, cpuattn_simd_t b) { return _mm512_cmp_ps_mask(a, b, _CMP_LT_OQ); }
static inline cpuattn_mask_t cpuattn_simd_le(cpuattn_simd_t a, cpuattn_simd_t b) { return _mm512_cmp_ps_mask(a, b, _CMP_LE_OQ); }
static inline cpuattn_mask_t cpuattn_simd_gt(cpuattn_simd_t a, cpuattn_simd_t b) { return _mm512_cmp_ps_mask(a, b, _CMP_GT_OQ); }
static inline cpuattn_mask_t cpuattn_simd_ge(cpuattn_simd_t a, cpuattn_simd_t b) { return _mm512_cmp_ps_mask(a, b, _CMP_GE_OQ); }
static inline cpuattn_mask_t cpuattn_simd_eq(cpuattn_simd_t a, cpuattn_simd_t b) { return _mm512_cmp_ps_mask(a, b, _CMP_EQ_OQ); }
static inline cpuattn_mask_t cpuattn_simd_ne(cpuattn_simd_t a, cpuattn_simd_t b) { return _mm512_cmp_ps_mask(a, b, _CMP_NEQ_UQ); }
static inline cpuattn_mask_t cpuattn_mask_and(cpuattn_mask_t a, cpuattn_mask_t b) { return a & b; }
static inline cpuattn_mask_t cpuattn_mask_or(cpuattn_mask_t a, cpuattn_mask_t b) { return a | b; }
static inline cpuattn_mask_t cpuattn_mask_not(cpuattn_mask_t a) { return (cpuattn_mask_t)~a; }
static inline cpuattn_simd_t cpuattn_simd_select(cpuattn_mask_t mask, cpuattn_simd_t yes, cpuattn_simd_t no) {
    return _mm512_mask_blend_ps(mask, no, yes);
}
static inline void cpuattn_mask_store_u8(uint8_t *target, cpuattn_mask_t mask) {
    for (int i = 0; i < CPUATTN_SIMD_LANES; ++i) target[i] = (uint8_t)((mask >> i) & 1u);
}

static inline cpuattn_simd_t cpuattn_simd_load(const float *pointer) {
    return _mm512_loadu_ps(pointer);
}

static inline void cpuattn_simd_store(float *pointer, cpuattn_simd_t value) {
    _mm512_storeu_ps(pointer, value);
}

static inline __mmask16 cpuattn_tail_mask(int n) {
    return n >= CPUATTN_SIMD_LANES ? (__mmask16)0xffffu
        : (__mmask16)((1u << n) - 1u);
}

static inline cpuattn_simd_t cpuattn_simd_load_partial(const float *pointer, int n) {
    return _mm512_maskz_loadu_ps(cpuattn_tail_mask(n), pointer);
}

static inline void cpuattn_simd_store_partial(float *pointer, cpuattn_simd_t value, int n) {
    _mm512_mask_storeu_ps(pointer, cpuattn_tail_mask(n), value);
}

static inline cpuattn_simd_t cpuattn_simd_fma(
    float scalar, cpuattn_simd_t vector, cpuattn_simd_t total) {
    return _mm512_fmadd_ps(_mm512_set1_ps(scalar), vector, total);
}

static inline cpuattn_simd_t cpuattn_simd_fma_vec(
    cpuattn_simd_t left, cpuattn_simd_t right, cpuattn_simd_t total) {
    return _mm512_fmadd_ps(left, right, total);
}

static inline float cpuattn_simd_reduce_add(cpuattn_simd_t value) {
    return _mm512_reduce_add_ps(value);
}

static inline float cpuattn_reduce_max(const float *x, int n) {
    __m512 maximum = _mm512_set1_ps(-INFINITY);
    int i = 0;
    for (; i + CPUATTN_SIMD_LANES <= n; i += CPUATTN_SIMD_LANES)
        maximum = _mm512_max_ps(maximum, _mm512_loadu_ps(x + i));
    if (i < n) {
        __mmask16 mask = (__mmask16)((1u << (n - i)) - 1u);
        maximum = _mm512_max_ps(
            maximum,
            _mm512_mask_loadu_ps(_mm512_set1_ps(-INFINITY), mask, x + i));
    }
    return _mm512_reduce_max_ps(maximum);
}

static inline float cpuattn_reduce_sum(const float *x, int n) {
    __m512 total = _mm512_setzero_ps();
    int i = 0;
    for (; i + CPUATTN_SIMD_LANES <= n; i += CPUATTN_SIMD_LANES)
        total = _mm512_add_ps(total, _mm512_loadu_ps(x + i));
    if (i < n) {
        __mmask16 mask = (__mmask16)((1u << (n - i)) - 1u);
        total = _mm512_add_ps(total, _mm512_maskz_loadu_ps(mask, x + i));
    }
    return _mm512_reduce_add_ps(total);
}

static inline __m512 cpuattn_exp_ps(__m512 x) {
    x = _mm512_min_ps(
        _mm512_max_ps(x, _mm512_set1_ps(-87.0f)),
        _mm512_set1_ps(88.0f));
    const __m512 log2e = _mm512_set1_ps(1.44269504088896341f);
    const __m512 ln2 = _mm512_set1_ps(0.6931471805599453f);
    __m512 kf = _mm512_roundscale_ps(
        _mm512_mul_ps(x, log2e),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m512 y = _mm512_fnmadd_ps(kf, ln2, x);
    __m512i exponent = _mm512_cvtps_epi32(kf);
    exponent = _mm512_add_epi32(exponent, _mm512_set1_epi32(127));
    exponent = _mm512_slli_epi32(exponent, 23);
    __m512 polynomial = _mm512_fmadd_ps(
        _mm512_set1_ps(1.0f / 120.0f), y, _mm512_set1_ps(1.0f / 24.0f));
    polynomial = _mm512_fmadd_ps(y, polynomial, _mm512_set1_ps(1.0f / 6.0f));
    polynomial = _mm512_fmadd_ps(y, polynomial, _mm512_set1_ps(0.5f));
    polynomial = _mm512_fmadd_ps(y, polynomial, _mm512_set1_ps(1.0f));
    polynomial = _mm512_fmadd_ps(y, polynomial, _mm512_set1_ps(1.0f));
    return _mm512_mul_ps(_mm512_castsi512_ps(exponent), polynomial);
}

static inline cpuattn_simd_t cpuattn_simd_exp(cpuattn_simd_t x) { return cpuattn_exp_ps(x); }

static inline cpuattn_simd_t cpuattn_simd_log(cpuattn_simd_t x) {
    __m512i bits = _mm512_castps_si512(x);
    __m512i exponent = _mm512_sub_epi32(
        _mm512_and_si512(_mm512_srli_epi32(bits, 23), _mm512_set1_epi32(255)),
        _mm512_set1_epi32(127));
    __m512 mantissa = _mm512_castsi512_ps(_mm512_or_si512(
        _mm512_and_si512(bits, _mm512_set1_epi32(0x7fffff)),
        _mm512_set1_epi32(0x3f800000)));
    __m512 y = _mm512_div_ps(
        _mm512_sub_ps(mantissa, _mm512_set1_ps(1.0f)),
        _mm512_add_ps(mantissa, _mm512_set1_ps(1.0f)));
    __m512 y2 = _mm512_mul_ps(y, y);
    __m512 polynomial = _mm512_set1_ps(1.0f / 9.0f);
    polynomial = _mm512_fmadd_ps(polynomial, y2, _mm512_set1_ps(1.0f / 7.0f));
    polynomial = _mm512_fmadd_ps(polynomial, y2, _mm512_set1_ps(1.0f / 5.0f));
    polynomial = _mm512_fmadd_ps(polynomial, y2, _mm512_set1_ps(1.0f / 3.0f));
    polynomial = _mm512_fmadd_ps(polynomial, y2, _mm512_set1_ps(1.0f));
    __m512 result = _mm512_fmadd_ps(
        _mm512_cvtepi32_ps(exponent), _mm512_set1_ps(0.6931471805599453f),
        _mm512_mul_ps(_mm512_set1_ps(2.0f), _mm512_mul_ps(y, polynomial)));
    __m512 zero = _mm512_setzero_ps();
    result = _mm512_mask_mov_ps(result, _mm512_cmp_ps_mask(x, zero, _CMP_EQ_OQ), _mm512_set1_ps(-INFINITY));
    return _mm512_mask_mov_ps(result, _mm512_cmp_ps_mask(x, zero, _CMP_LT_OQ), _mm512_set1_ps(NAN));
}

static inline void cpuattn_exp_submax_inplace(float *x, float maximum, int n) {
    const __m512 offset = _mm512_set1_ps(maximum);
    const __m512 negative_infinity = _mm512_set1_ps(-INFINITY);
    int i = 0;
    for (; i + CPUATTN_SIMD_LANES <= n; i += CPUATTN_SIMD_LANES) {
        __m512 input = _mm512_loadu_ps(x + i);
        __mmask16 masked = _mm512_cmp_ps_mask(input, negative_infinity, _CMP_EQ_OQ);
        __m512 result = cpuattn_exp_ps(_mm512_sub_ps(input, offset));
        _mm512_storeu_ps(x + i, _mm512_mask_mov_ps(result, masked, _mm512_setzero_ps()));
    }
    if (i < n) {
        __mmask16 tail = (__mmask16)((1u << (n - i)) - 1u);
        __m512 input = _mm512_maskz_loadu_ps(tail, x + i);
        __mmask16 masked = _mm512_cmp_ps_mask(input, negative_infinity, _CMP_EQ_OQ);
        __m512 result = cpuattn_exp_ps(_mm512_sub_ps(input, offset));
        result = _mm512_mask_mov_ps(result, masked, _mm512_setzero_ps());
        _mm512_mask_storeu_ps(x + i, tail, result);
    }
}

static inline void cpuattn_scale_inplace(float *x, int64_t n, float scale) {
    const __m512 factor = _mm512_set1_ps(scale);
    int64_t i = 0;
    for (; i + CPUATTN_SIMD_LANES <= n; i += CPUATTN_SIMD_LANES)
        _mm512_storeu_ps(x + i, _mm512_mul_ps(_mm512_loadu_ps(x + i), factor));
    if (i < n) {
        __mmask16 mask = (__mmask16)((1u << (n - i)) - 1u);
        __m512 values = _mm512_maskz_loadu_ps(mask, x + i);
        _mm512_mask_storeu_ps(x + i, mask, _mm512_mul_ps(values, factor));
    }
}

#endif
