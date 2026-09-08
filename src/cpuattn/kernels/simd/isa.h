#ifndef MINI_CPUATTN_SIMD_ISA_H
#define MINI_CPUATTN_SIMD_ISA_H

#if defined(__AVX512F__)
#include "x86/avx512.h"
#elif defined(__AVX2__)
#include "x86/avx2.h"
#elif defined(__ARM_FEATURE_SVE)
#include "arm64/sve.h"
#elif defined(__aarch64__)
#include "arm64/neon.h"
#else
#error "CPUAttn has no SIMD implementation for this target"
#endif

#endif
