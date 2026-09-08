#ifndef MINI_CPUATTN_SIMD_ARM64_NEON_H
#define MINI_CPUATTN_SIMD_ARM64_NEON_H

#include <math.h>
#include <stdint.h>
#include <arm_neon.h>

#define CPUATTN_SIMD_LANES 4
typedef float32x4_t cpuattn_simd_t;
typedef uint32x4_t cpuattn_mask_t;

static inline cpuattn_simd_t cpuattn_simd_zero(void) {
    return vdupq_n_f32(0.0f);
}

static inline cpuattn_simd_t cpuattn_simd_set1(float value) { return vdupq_n_f32(value); }
static inline cpuattn_simd_t cpuattn_simd_add(cpuattn_simd_t a, cpuattn_simd_t b) { return vaddq_f32(a, b); }
static inline cpuattn_simd_t cpuattn_simd_sub(cpuattn_simd_t a, cpuattn_simd_t b) { return vsubq_f32(a, b); }
static inline cpuattn_simd_t cpuattn_simd_mul(cpuattn_simd_t a, cpuattn_simd_t b) { return vmulq_f32(a, b); }
static inline cpuattn_simd_t cpuattn_simd_div(cpuattn_simd_t a, cpuattn_simd_t b) { return vdivq_f32(a, b); }
static inline cpuattn_simd_t cpuattn_simd_min(cpuattn_simd_t a, cpuattn_simd_t b) { return vminq_f32(a, b); }
static inline cpuattn_simd_t cpuattn_simd_max(cpuattn_simd_t a, cpuattn_simd_t b) { return vmaxq_f32(a, b); }
static inline cpuattn_simd_t cpuattn_simd_abs(cpuattn_simd_t x) { return vabsq_f32(x); }
static inline cpuattn_mask_t cpuattn_simd_lt(cpuattn_simd_t a, cpuattn_simd_t b) { return vcltq_f32(a, b); }
static inline cpuattn_mask_t cpuattn_simd_le(cpuattn_simd_t a, cpuattn_simd_t b) { return vcleq_f32(a, b); }
static inline cpuattn_mask_t cpuattn_simd_gt(cpuattn_simd_t a, cpuattn_simd_t b) { return vcgtq_f32(a, b); }
static inline cpuattn_mask_t cpuattn_simd_ge(cpuattn_simd_t a, cpuattn_simd_t b) { return vcgeq_f32(a, b); }
static inline cpuattn_mask_t cpuattn_simd_eq(cpuattn_simd_t a, cpuattn_simd_t b) { return vceqq_f32(a, b); }
static inline cpuattn_mask_t cpuattn_simd_ne(cpuattn_simd_t a, cpuattn_simd_t b) { return vmvnq_u32(vceqq_f32(a, b)); }
static inline cpuattn_mask_t cpuattn_mask_and(cpuattn_mask_t a, cpuattn_mask_t b) { return vandq_u32(a, b); }
static inline cpuattn_mask_t cpuattn_mask_or(cpuattn_mask_t a, cpuattn_mask_t b) { return vorrq_u32(a, b); }
static inline cpuattn_mask_t cpuattn_mask_not(cpuattn_mask_t a) { return vmvnq_u32(a); }
static inline cpuattn_simd_t cpuattn_simd_select(cpuattn_mask_t mask, cpuattn_simd_t yes, cpuattn_simd_t no) {
    return vbslq_f32(mask, yes, no);
}
static inline void cpuattn_mask_store_u8(uint8_t *target, cpuattn_mask_t mask) {
    uint32_t lanes[CPUATTN_SIMD_LANES];
    vst1q_u32(lanes, mask);
    for (int i = 0; i < CPUATTN_SIMD_LANES; ++i) target[i] = (uint8_t)(lanes[i] != 0);
}

static inline cpuattn_simd_t cpuattn_simd_load(const float *pointer) {
    return vld1q_f32(pointer);
}

static inline void cpuattn_simd_store(float *pointer, cpuattn_simd_t value) {
    vst1q_f32(pointer, value);
}

static inline cpuattn_simd_t cpuattn_simd_load_partial(const float *pointer, int n) {
    float lanes[CPUATTN_SIMD_LANES] = {0.0f};
    for (int i = 0; i < n; ++i) lanes[i] = pointer[i];
    return vld1q_f32(lanes);
}

static inline void cpuattn_simd_store_partial(float *pointer, cpuattn_simd_t value, int n) {
    float lanes[CPUATTN_SIMD_LANES];
    vst1q_f32(lanes, value);
    for (int i = 0; i < n; ++i) pointer[i] = lanes[i];
}

static inline cpuattn_simd_t cpuattn_simd_fma(
    float scalar, cpuattn_simd_t vector, cpuattn_simd_t total) {
    return vfmaq_n_f32(total, vector, scalar);
}

static inline cpuattn_simd_t cpuattn_simd_fma_vec(
    cpuattn_simd_t left, cpuattn_simd_t right, cpuattn_simd_t total) {
    return vfmaq_f32(total, left, right);
}

static inline float cpuattn_simd_reduce_add(cpuattn_simd_t value) {
    return vaddvq_f32(value);
}

static inline float cpuattn_reduce_max(const float *x, int n) {
    float32x4_t maximum = vdupq_n_f32(-INFINITY);
    int i = 0;
    for (; i + CPUATTN_SIMD_LANES <= n; i += CPUATTN_SIMD_LANES)
        maximum = vmaxq_f32(maximum, vld1q_f32(x + i));
    float result = vmaxvq_f32(maximum);
    for (; i < n; ++i) result = fmaxf(result, x[i]);
    return result;
}

static inline float32x4_t cpuattn_exp_ps(float32x4_t x) {
    x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(-87.0f)), vdupq_n_f32(88.0f));
    float32x4_t kf = vrndnq_f32(vmulq_n_f32(x, 1.44269504088896341f));
    float32x4_t y = vsubq_f32(x, vmulq_n_f32(kf, 0.6931471805599453f));
    int32x4_t exponent = vcvtq_s32_f32(kf);
    exponent = vaddq_s32(exponent, vdupq_n_s32(127));
    exponent = vshlq_n_s32(exponent, 23);
    float32x4_t polynomial = vfmaq_n_f32(
        vdupq_n_f32(1.0f / 24.0f), y, 1.0f / 120.0f);
    polynomial = vfmaq_f32(vdupq_n_f32(1.0f / 6.0f), y, polynomial);
    polynomial = vfmaq_f32(vdupq_n_f32(0.5f), y, polynomial);
    polynomial = vfmaq_f32(vdupq_n_f32(1.0f), y, polynomial);
    polynomial = vfmaq_f32(vdupq_n_f32(1.0f), y, polynomial);
    return vmulq_f32(vreinterpretq_f32_s32(exponent), polynomial);
}

static inline cpuattn_simd_t cpuattn_simd_exp(cpuattn_simd_t x) { return cpuattn_exp_ps(x); }

static inline cpuattn_simd_t cpuattn_simd_log(cpuattn_simd_t x) {
    uint32x4_t bits = vreinterpretq_u32_f32(x);
    int32x4_t exponent = vsubq_s32(
        vreinterpretq_s32_u32(vandq_u32(vshrq_n_u32(bits, 23), vdupq_n_u32(255))),
        vdupq_n_s32(127));
    float32x4_t mantissa = vreinterpretq_f32_u32(vorrq_u32(
        vandq_u32(bits, vdupq_n_u32(0x7fffff)), vdupq_n_u32(0x3f800000)));
    float32x4_t y = vdivq_f32(
        vsubq_f32(mantissa, vdupq_n_f32(1.0f)),
        vaddq_f32(mantissa, vdupq_n_f32(1.0f)));
    float32x4_t y2 = vmulq_f32(y, y);
    float32x4_t polynomial = vdupq_n_f32(1.0f / 9.0f);
    polynomial = vfmaq_f32(vdupq_n_f32(1.0f / 7.0f), polynomial, y2);
    polynomial = vfmaq_f32(vdupq_n_f32(1.0f / 5.0f), polynomial, y2);
    polynomial = vfmaq_f32(vdupq_n_f32(1.0f / 3.0f), polynomial, y2);
    polynomial = vfmaq_f32(vdupq_n_f32(1.0f), polynomial, y2);
    float32x4_t result = vfmaq_n_f32(
        vmulq_n_f32(vmulq_f32(y, polynomial), 2.0f),
        vcvtq_f32_s32(exponent), 0.6931471805599453f);
    result = vbslq_f32(vceqq_f32(x, vdupq_n_f32(0.0f)), vdupq_n_f32(-INFINITY), result);
    return vbslq_f32(vcltq_f32(x, vdupq_n_f32(0.0f)), vdupq_n_f32(NAN), result);
}

static inline void cpuattn_scale_inplace(float *x, int64_t n, float scale) {
    int64_t i = 0;
    for (; i + CPUATTN_SIMD_LANES <= n; i += CPUATTN_SIMD_LANES)
        vst1q_f32(x + i, vmulq_n_f32(vld1q_f32(x + i), scale));
    for (; i < n; ++i) x[i] *= scale;
}

#endif
