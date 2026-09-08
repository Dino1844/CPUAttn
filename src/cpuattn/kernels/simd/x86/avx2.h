#ifndef MINI_CPUATTN_SIMD_X86_AVX2_H
#define MINI_CPUATTN_SIMD_X86_AVX2_H

#include <math.h>
#include <stdint.h>
#include <immintrin.h>

#define CPUATTN_SIMD_LANES 8
typedef __m256 cpuattn_simd_t;
typedef __m256 cpuattn_mask_t;

static inline cpuattn_simd_t cpuattn_simd_zero(void) {
    return _mm256_setzero_ps();
}

static inline cpuattn_simd_t cpuattn_simd_set1(float value) { return _mm256_set1_ps(value); }
static inline cpuattn_simd_t cpuattn_simd_add(cpuattn_simd_t a, cpuattn_simd_t b) { return _mm256_add_ps(a, b); }
static inline cpuattn_simd_t cpuattn_simd_sub(cpuattn_simd_t a, cpuattn_simd_t b) { return _mm256_sub_ps(a, b); }
static inline cpuattn_simd_t cpuattn_simd_mul(cpuattn_simd_t a, cpuattn_simd_t b) { return _mm256_mul_ps(a, b); }
static inline cpuattn_simd_t cpuattn_simd_div(cpuattn_simd_t a, cpuattn_simd_t b) { return _mm256_div_ps(a, b); }
static inline cpuattn_simd_t cpuattn_simd_min(cpuattn_simd_t a, cpuattn_simd_t b) { return _mm256_min_ps(a, b); }
static inline cpuattn_simd_t cpuattn_simd_max(cpuattn_simd_t a, cpuattn_simd_t b) { return _mm256_max_ps(a, b); }
static inline cpuattn_simd_t cpuattn_simd_abs(cpuattn_simd_t x) {
    return _mm256_andnot_ps(_mm256_set1_ps(-0.0f), x);
}
static inline cpuattn_mask_t cpuattn_simd_lt(cpuattn_simd_t a, cpuattn_simd_t b) { return _mm256_cmp_ps(a, b, _CMP_LT_OQ); }
static inline cpuattn_mask_t cpuattn_simd_le(cpuattn_simd_t a, cpuattn_simd_t b) { return _mm256_cmp_ps(a, b, _CMP_LE_OQ); }
static inline cpuattn_mask_t cpuattn_simd_gt(cpuattn_simd_t a, cpuattn_simd_t b) { return _mm256_cmp_ps(a, b, _CMP_GT_OQ); }
static inline cpuattn_mask_t cpuattn_simd_ge(cpuattn_simd_t a, cpuattn_simd_t b) { return _mm256_cmp_ps(a, b, _CMP_GE_OQ); }
static inline cpuattn_mask_t cpuattn_simd_eq(cpuattn_simd_t a, cpuattn_simd_t b) { return _mm256_cmp_ps(a, b, _CMP_EQ_OQ); }
static inline cpuattn_mask_t cpuattn_simd_ne(cpuattn_simd_t a, cpuattn_simd_t b) { return _mm256_cmp_ps(a, b, _CMP_NEQ_UQ); }
static inline cpuattn_mask_t cpuattn_mask_and(cpuattn_mask_t a, cpuattn_mask_t b) { return _mm256_and_ps(a, b); }
static inline cpuattn_mask_t cpuattn_mask_or(cpuattn_mask_t a, cpuattn_mask_t b) { return _mm256_or_ps(a, b); }
static inline cpuattn_mask_t cpuattn_mask_not(cpuattn_mask_t a) {
    return _mm256_xor_ps(a, _mm256_castsi256_ps(_mm256_set1_epi32(-1)));
}
static inline cpuattn_simd_t cpuattn_simd_select(cpuattn_mask_t mask, cpuattn_simd_t yes, cpuattn_simd_t no) {
    return _mm256_blendv_ps(no, yes, mask);
}
static inline void cpuattn_mask_store_u8(uint8_t *target, cpuattn_mask_t mask) {
    unsigned bits = (unsigned)_mm256_movemask_ps(mask);
    for (int i = 0; i < CPUATTN_SIMD_LANES; ++i) target[i] = (uint8_t)((bits >> i) & 1u);
}

static inline cpuattn_simd_t cpuattn_simd_load(const float *pointer) {
    return _mm256_loadu_ps(pointer);
}

static inline void cpuattn_simd_store(float *pointer, cpuattn_simd_t value) {
    _mm256_storeu_ps(pointer, value);
}

static inline cpuattn_simd_t cpuattn_simd_load_partial(const float *pointer, int n) {
    int32_t mask_values[CPUATTN_SIMD_LANES];
    for (int i = 0; i < CPUATTN_SIMD_LANES; ++i) mask_values[i] = i < n ? -1 : 0;
    __m256i mask = _mm256_loadu_si256((const __m256i *)mask_values);
    return _mm256_maskload_ps(pointer, mask);
}

static inline void cpuattn_simd_store_partial(float *pointer, cpuattn_simd_t value, int n) {
    int32_t mask_values[CPUATTN_SIMD_LANES];
    for (int i = 0; i < CPUATTN_SIMD_LANES; ++i) mask_values[i] = i < n ? -1 : 0;
    __m256i mask = _mm256_loadu_si256((const __m256i *)mask_values);
    _mm256_maskstore_ps(pointer, mask, value);
}

static inline cpuattn_simd_t cpuattn_simd_fma(
    float scalar, cpuattn_simd_t vector, cpuattn_simd_t total) {
    return _mm256_fmadd_ps(_mm256_set1_ps(scalar), vector, total);
}

static inline cpuattn_simd_t cpuattn_simd_fma_vec(
    cpuattn_simd_t left, cpuattn_simd_t right, cpuattn_simd_t total) {
    return _mm256_fmadd_ps(left, right, total);
}

static inline float cpuattn_simd_reduce_add(cpuattn_simd_t value) {
    __m128 low = _mm256_castps256_ps128(value);
    __m128 high = _mm256_extractf128_ps(value, 1);
    __m128 sum = _mm_add_ps(low, high);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

static inline float cpuattn_reduce_max(const float *x, int n) {
    __m256 maximum = _mm256_set1_ps(-INFINITY);
    int i = 0;
    for (; i + CPUATTN_SIMD_LANES <= n; i += CPUATTN_SIMD_LANES)
        maximum = _mm256_max_ps(maximum, _mm256_loadu_ps(x + i));
    float lanes[CPUATTN_SIMD_LANES];
    _mm256_storeu_ps(lanes, maximum);
    float result = lanes[0];
    for (int lane = 1; lane < CPUATTN_SIMD_LANES; ++lane)
        result = fmaxf(result, lanes[lane]);
    for (; i < n; ++i) result = fmaxf(result, x[i]);
    return result;
}

static inline __m256 cpuattn_exp_ps(__m256 x) {
    x = _mm256_min_ps(
        _mm256_max_ps(x, _mm256_set1_ps(-87.0f)),
        _mm256_set1_ps(88.0f));
    const __m256 log2e = _mm256_set1_ps(1.44269504088896341f);
    const __m256 ln2 = _mm256_set1_ps(0.6931471805599453f);
    __m256 kf = _mm256_round_ps(
        _mm256_mul_ps(x, log2e),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 y = _mm256_fnmadd_ps(kf, ln2, x);
    __m256i exponent = _mm256_cvtps_epi32(kf);
    exponent = _mm256_add_epi32(exponent, _mm256_set1_epi32(127));
    exponent = _mm256_slli_epi32(exponent, 23);
    __m256 polynomial = _mm256_fmadd_ps(
        _mm256_set1_ps(1.0f / 120.0f), y, _mm256_set1_ps(1.0f / 24.0f));
    polynomial = _mm256_fmadd_ps(y, polynomial, _mm256_set1_ps(1.0f / 6.0f));
    polynomial = _mm256_fmadd_ps(y, polynomial, _mm256_set1_ps(0.5f));
    polynomial = _mm256_fmadd_ps(y, polynomial, _mm256_set1_ps(1.0f));
    polynomial = _mm256_fmadd_ps(y, polynomial, _mm256_set1_ps(1.0f));
    return _mm256_mul_ps(_mm256_castsi256_ps(exponent), polynomial);
}

static inline cpuattn_simd_t cpuattn_simd_exp(cpuattn_simd_t x) { return cpuattn_exp_ps(x); }

static inline cpuattn_simd_t cpuattn_simd_log(cpuattn_simd_t x) {
    __m256i bits = _mm256_castps_si256(x);
    __m256i exponent = _mm256_sub_epi32(
        _mm256_and_si256(_mm256_srli_epi32(bits, 23), _mm256_set1_epi32(255)),
        _mm256_set1_epi32(127));
    __m256 mantissa = _mm256_castsi256_ps(_mm256_or_si256(
        _mm256_and_si256(bits, _mm256_set1_epi32(0x7fffff)),
        _mm256_set1_epi32(0x3f800000)));
    __m256 y = _mm256_div_ps(
        _mm256_sub_ps(mantissa, _mm256_set1_ps(1.0f)),
        _mm256_add_ps(mantissa, _mm256_set1_ps(1.0f)));
    __m256 y2 = _mm256_mul_ps(y, y);
    __m256 polynomial = _mm256_set1_ps(1.0f / 9.0f);
    polynomial = _mm256_fmadd_ps(polynomial, y2, _mm256_set1_ps(1.0f / 7.0f));
    polynomial = _mm256_fmadd_ps(polynomial, y2, _mm256_set1_ps(1.0f / 5.0f));
    polynomial = _mm256_fmadd_ps(polynomial, y2, _mm256_set1_ps(1.0f / 3.0f));
    polynomial = _mm256_fmadd_ps(polynomial, y2, _mm256_set1_ps(1.0f));
    __m256 result = _mm256_fmadd_ps(
        _mm256_cvtepi32_ps(exponent), _mm256_set1_ps(0.6931471805599453f),
        _mm256_mul_ps(_mm256_set1_ps(2.0f), _mm256_mul_ps(y, polynomial)));
    __m256 zero = _mm256_setzero_ps();
    result = _mm256_blendv_ps(result, _mm256_set1_ps(-INFINITY), _mm256_cmp_ps(x, zero, _CMP_EQ_OQ));
    return _mm256_blendv_ps(result, _mm256_set1_ps(NAN), _mm256_cmp_ps(x, zero, _CMP_LT_OQ));
}

static inline void cpuattn_scale_inplace(float *x, int64_t n, float scale) {
    const __m256 factor = _mm256_set1_ps(scale);
    int64_t i = 0;
    for (; i + CPUATTN_SIMD_LANES <= n; i += CPUATTN_SIMD_LANES)
        _mm256_storeu_ps(x + i, _mm256_mul_ps(_mm256_loadu_ps(x + i), factor));
    for (; i < n; ++i) x[i] *= scale;
}

#endif
