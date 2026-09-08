#ifndef MINI_CPUATTN_SIMD_ARM64_SVE_H
#define MINI_CPUATTN_SIMD_ARM64_SVE_H

#include <math.h>
#include <stdint.h>
#include <arm_sve.h>

#define CPUATTN_SIMD_LANES 8
typedef svfloat32_t cpuattn_simd_t __attribute__((arm_sve_vector_bits(256)));
typedef svbool_t cpuattn_mask_t;

static inline cpuattn_simd_t cpuattn_simd_zero(void) {
    return svdup_n_f32(0.0f);
}

static inline cpuattn_simd_t cpuattn_simd_set1(float value) { return svdup_n_f32(value); }
static inline cpuattn_simd_t cpuattn_simd_add(cpuattn_simd_t a, cpuattn_simd_t b) { return svadd_f32_x(svptrue_b32(), a, b); }
static inline cpuattn_simd_t cpuattn_simd_sub(cpuattn_simd_t a, cpuattn_simd_t b) { return svsub_f32_x(svptrue_b32(), a, b); }
static inline cpuattn_simd_t cpuattn_simd_mul(cpuattn_simd_t a, cpuattn_simd_t b) { return svmul_f32_x(svptrue_b32(), a, b); }
static inline cpuattn_simd_t cpuattn_simd_div(cpuattn_simd_t a, cpuattn_simd_t b) { return svdiv_f32_x(svptrue_b32(), a, b); }
static inline cpuattn_simd_t cpuattn_simd_min(cpuattn_simd_t a, cpuattn_simd_t b) { return svmin_f32_x(svptrue_b32(), a, b); }
static inline cpuattn_simd_t cpuattn_simd_max(cpuattn_simd_t a, cpuattn_simd_t b) { return svmax_f32_x(svptrue_b32(), a, b); }
static inline cpuattn_simd_t cpuattn_simd_abs(cpuattn_simd_t x) { return svabs_f32_x(svptrue_b32(), x); }
static inline cpuattn_mask_t cpuattn_simd_lt(cpuattn_simd_t a, cpuattn_simd_t b) { return svcmplt_f32(svptrue_b32(), a, b); }
static inline cpuattn_mask_t cpuattn_simd_le(cpuattn_simd_t a, cpuattn_simd_t b) { return svcmple_f32(svptrue_b32(), a, b); }
static inline cpuattn_mask_t cpuattn_simd_gt(cpuattn_simd_t a, cpuattn_simd_t b) { return svcmpgt_f32(svptrue_b32(), a, b); }
static inline cpuattn_mask_t cpuattn_simd_ge(cpuattn_simd_t a, cpuattn_simd_t b) { return svcmpge_f32(svptrue_b32(), a, b); }
static inline cpuattn_mask_t cpuattn_simd_eq(cpuattn_simd_t a, cpuattn_simd_t b) { return svcmpeq_f32(svptrue_b32(), a, b); }
static inline cpuattn_mask_t cpuattn_simd_ne(cpuattn_simd_t a, cpuattn_simd_t b) { return svcmpne_f32(svptrue_b32(), a, b); }
static inline cpuattn_mask_t cpuattn_mask_and(cpuattn_mask_t a, cpuattn_mask_t b) { return svand_b_z(svptrue_b32(), a, b); }
static inline cpuattn_mask_t cpuattn_mask_or(cpuattn_mask_t a, cpuattn_mask_t b) { return svorr_b_z(svptrue_b32(), a, b); }
static inline cpuattn_mask_t cpuattn_mask_not(cpuattn_mask_t a) { return svnot_b_z(svptrue_b32(), a); }
static inline cpuattn_simd_t cpuattn_simd_select(cpuattn_mask_t mask, cpuattn_simd_t yes, cpuattn_simd_t no) {
    return svsel_f32(mask, yes, no);
}
static inline void cpuattn_mask_store_u8(uint8_t *target, cpuattn_mask_t mask) {
    uint32_t lanes[CPUATTN_SIMD_LANES];
    svuint32_t values = svsel_u32(mask, svdup_n_u32(1), svdup_n_u32(0));
    svst1_u32(svptrue_b32(), lanes, values);
    for (int i = 0; i < CPUATTN_SIMD_LANES; ++i) target[i] = (uint8_t)lanes[i];
}

static inline cpuattn_simd_t cpuattn_simd_load(const float *pointer) {
    return svld1_f32(svptrue_b32(), pointer);
}

static inline void cpuattn_simd_store(float *pointer, cpuattn_simd_t value) {
    svst1_f32(svptrue_b32(), pointer, value);
}

static inline cpuattn_simd_t cpuattn_simd_load_partial(const float *pointer, int n) {
    svbool_t active = svwhilelt_b32((uint64_t)0, (uint64_t)n);
    return svld1_f32(active, pointer);
}

static inline void cpuattn_simd_store_partial(float *pointer, cpuattn_simd_t value, int n) {
    svbool_t active = svwhilelt_b32((uint64_t)0, (uint64_t)n);
    svst1_f32(active, pointer, value);
}

static inline cpuattn_simd_t cpuattn_simd_fma(
    float scalar, cpuattn_simd_t vector, cpuattn_simd_t total) {
    return svmla_n_f32_x(svptrue_b32(), total, vector, scalar);
}

static inline cpuattn_simd_t cpuattn_simd_fma_vec(
    cpuattn_simd_t left, cpuattn_simd_t right, cpuattn_simd_t total) {
    return svmla_f32_x(svptrue_b32(), total, left, right);
}

static inline float cpuattn_simd_reduce_add(cpuattn_simd_t value) {
    return svaddv_f32(svptrue_b32(), value);
}

static inline float cpuattn_reduce_max(const float *x, int n) {
    svfloat32_t maximum = svdup_n_f32(-INFINITY);
    int i = 0;
    while (i < n) {
        svbool_t active = svwhilelt_b32((uint64_t)i, (uint64_t)n);
        maximum = svmax_f32_m(active, maximum, svld1_f32(active, x + i));
        i += CPUATTN_SIMD_LANES;
    }
    return svmaxv_f32(svptrue_b32(), maximum);
}

static inline svfloat32_t cpuattn_exp_ps(svbool_t active, svfloat32_t x) {
    x = svmax_n_f32_x(active, x, -87.0f);
    x = svmin_n_f32_x(active, x, 88.0f);
    svfloat32_t kf = svrintn_f32_x(
        active, svmul_n_f32_x(active, x, 1.44269504088896341f));
    svfloat32_t y = svmls_n_f32_x(active, x, kf, 0.6931471805599453f);
    svint32_t exponent = svcvt_s32_f32_x(active, kf);
    exponent = svadd_n_s32_x(active, exponent, 127);
    exponent = svlsl_n_s32_x(active, exponent, 23);
    svfloat32_t polynomial = svmla_n_f32_x(
        active, svdup_n_f32(1.0f / 24.0f), y, 1.0f / 120.0f);
    polynomial = svmla_f32_x(active, svdup_n_f32(1.0f / 6.0f), y, polynomial);
    polynomial = svmla_f32_x(active, svdup_n_f32(0.5f), y, polynomial);
    polynomial = svmla_f32_x(active, svdup_n_f32(1.0f), y, polynomial);
    polynomial = svmla_f32_x(active, svdup_n_f32(1.0f), y, polynomial);
    return svmul_f32_x(active, svreinterpret_f32_s32(exponent), polynomial);
}

static inline cpuattn_simd_t cpuattn_simd_exp(cpuattn_simd_t x) {
    return cpuattn_exp_ps(svptrue_b32(), x);
}

static inline cpuattn_simd_t cpuattn_simd_log(cpuattn_simd_t x) {
    svbool_t pg = svptrue_b32();
    svuint32_t bits = svreinterpret_u32_f32(x);
    svint32_t exponent = svsub_n_s32_x(
        pg, svreinterpret_s32_u32(svand_n_u32_x(pg, svlsr_n_u32_x(pg, bits, 23), 255)), 127);
    svfloat32_t mantissa = svreinterpret_f32_u32(
        svorr_n_u32_x(pg, svand_n_u32_x(pg, bits, 0x7fffff), 0x3f800000));
    svfloat32_t one = svdup_n_f32(1.0f);
    svfloat32_t y = svdiv_f32_x(
        pg, svsub_f32_x(pg, mantissa, one), svadd_f32_x(pg, mantissa, one));
    svfloat32_t y2 = svmul_f32_x(pg, y, y);
    svfloat32_t polynomial = svdup_n_f32(1.0f / 9.0f);
    polynomial = svmla_f32_x(pg, svdup_n_f32(1.0f / 7.0f), polynomial, y2);
    polynomial = svmla_f32_x(pg, svdup_n_f32(1.0f / 5.0f), polynomial, y2);
    polynomial = svmla_f32_x(pg, svdup_n_f32(1.0f / 3.0f), polynomial, y2);
    polynomial = svmla_f32_x(pg, one, polynomial, y2);
    svfloat32_t result = svmla_n_f32_x(
        pg, svmul_n_f32_x(pg, svmul_f32_x(pg, y, polynomial), 2.0f),
        svcvt_f32_s32_x(pg, exponent), 0.6931471805599453f);
    result = svsel_f32(svcmpeq_n_f32(pg, x, 0.0f), svdup_n_f32(-INFINITY), result);
    return svsel_f32(svcmplt_n_f32(pg, x, 0.0f), svdup_n_f32(NAN), result);
}

static inline void cpuattn_scale_inplace(float *x, int64_t n, float scale) {
    int64_t i = 0;
    while (i < n) {
        svbool_t active = svwhilelt_b32((uint64_t)i, (uint64_t)n);
        svfloat32_t values = svld1_f32(active, x + i);
        svst1_f32(active, x + i, svmul_n_f32_x(active, values, scale));
        i += CPUATTN_SIMD_LANES;
    }
}

#endif
