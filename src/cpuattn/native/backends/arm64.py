from .base import Backend


ARM_NEON = Backend(
    "arm64_neon", "aarch64", frozenset({"asimd"}), 16,
    "aarch64-linux-gnu-gcc", ("-march=armv8-a+simd",),
)

ARM_SVE = Backend(
    "arm64_sve", "aarch64", frozenset({"sve"}), 32,
    "aarch64-linux-gnu-gcc", ("-march=armv8.2-a+sve", "-msve-vector-bits=256"),
    sve_vector_bytes=32,
)

__all__ = ["ARM_NEON", "ARM_SVE"]
