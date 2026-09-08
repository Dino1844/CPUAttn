from .base import Backend


X86_AVX2 = Backend(
    "x86_avx2_fma", "x86_64", frozenset({"avx2", "fma"}), 32, "gcc",
    ("-mavx2", "-mfma"),
)

X86_AVX512 = Backend(
    "x86_avx512", "x86_64", frozenset({"avx512f", "fma"}), 64, "gcc",
    ("-mavx512f", "-mfma"),
)

__all__ = ["X86_AVX2", "X86_AVX512"]
