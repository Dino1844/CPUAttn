from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AttentionWorkload:
    """One attention benchmark point; stream_steps>0 runs a growing-KV decode."""

    name: str
    b: int
    hq: int
    hkv: int
    sq: int
    skv: int
    d: int
    dv: int
    causal: bool = True
    kv_cache: bool = False
    start_skv: int = 0
    stream_steps: int = 0

    def shape(self) -> tuple[int, int, int, int, int, int, int]:
        return (self.b, self.hq, self.hkv, self.sq, self.skv, self.d, self.dv)


@dataclass(frozen=True, slots=True)
class LinearWorkload:
    """One linear-attention benchmark point; kind selects the recurrence."""

    name: str
    b: int
    groups: int
    heads: int
    sequence: int
    d: int
    dv: int
    kind: str = "standard"

    def shape(self) -> tuple[int, int, int, int, int, int]:
        return (self.b, self.groups, self.heads, self.sequence, self.d, self.dv)


@dataclass(frozen=True, slots=True)
class ThreadScalingWorkload:
    """Force every worker count and report the best plan per level."""

    name: str
    b: int
    hq: int
    hkv: int
    sq: int
    skv: int
    d: int
    dv: int
    causal: bool = True
    workers: tuple[int, ...] = (1, 2, 4, 8, 13, 26)


PREFILL: tuple[AttentionWorkload, ...] = (
    AttentionWorkload("prefill_gqa_256", 1, 8, 2, 256, 256, 64, 64),
    AttentionWorkload("prefill_gqa_1024", 1, 8, 2, 1024, 1024, 64, 64),
    AttentionWorkload("prefill_mqa_512", 1, 8, 1, 512, 512, 64, 64),
    AttentionWorkload("prefill_mha_512", 1, 8, 8, 512, 512, 64, 64),
    AttentionWorkload("prefill_batch2_gqa_512", 2, 8, 2, 512, 512, 64, 64),
    AttentionWorkload("prefill_long_2048", 1, 16, 16, 2048, 2048, 64, 64),
    AttentionWorkload("prefill_small_128", 1, 8, 2, 128, 128, 64, 64),
    AttentionWorkload("prefill_d128", 1, 4, 4, 512, 512, 128, 128),
    AttentionWorkload("prefill_narrow_dv", 1, 8, 2, 256, 256, 64, 32),
    AttentionWorkload("prefill_non_causal_256", 1, 8, 2, 256, 256, 64, 64, causal=False),
    AttentionWorkload("prefill_llama70b", 1, 64, 8, 512, 512, 128, 128),
    AttentionWorkload("prefill_chunked_4k", 1, 8, 2, 128, 4096, 64, 64),
    AttentionWorkload("prefill_d96", 1, 8, 2, 512, 512, 96, 96),
    AttentionWorkload("prefill_dv512", 1, 8, 2, 512, 512, 64, 512),
)

DECODE: tuple[AttentionWorkload, ...] = (
    AttentionWorkload("decode_gqa_512", 1, 8, 2, 1, 0, 64, 64, kv_cache=True, start_skv=449, stream_steps=64),
    AttentionWorkload("decode_gqa_1024", 1, 8, 2, 1, 0, 64, 64, kv_cache=True, start_skv=961, stream_steps=128),
    AttentionWorkload("decode_mqa_2048", 1, 8, 1, 1, 0, 64, 64, kv_cache=True, start_skv=1900, stream_steps=148),
    AttentionWorkload("decode_q32_4096", 1, 32, 8, 1, 0, 64, 64, kv_cache=True, start_skv=3000, stream_steps=96),
    AttentionWorkload("decode_d128", 1, 4, 4, 1, 0, 128, 128, kv_cache=True, start_skv=961, stream_steps=128),
    AttentionWorkload("decode_batch8", 8, 32, 8, 1, 0, 64, 64, kv_cache=True, start_skv=1000, stream_steps=64),
)

LINEAR: tuple[LinearWorkload, ...] = (
    LinearWorkload("linear_s128", 1, 4, 4, 128, 64, 64),
    LinearWorkload("linear_s1024", 1, 4, 4, 1024, 64, 64),
    LinearWorkload("linear_gqa_2_8", 1, 2, 8, 256, 64, 64),
    LinearWorkload("kda_s256", 1, 4, 4, 256, 64, 64, kind="kda"),
    LinearWorkload("kda_s512", 1, 4, 4, 512, 64, 64, kind="kda"),
    LinearWorkload("mamba2_s256", 1, 4, 4, 256, 64, 64, kind="mamba2"),
    LinearWorkload("mamba2_s1024", 1, 4, 4, 1024, 64, 64, kind="mamba2"),
)

THREADS: tuple[ThreadScalingWorkload, ...] = (
    ThreadScalingWorkload("threads_prefill_gqa_256", 1, 8, 2, 256, 256, 64, 64),
)

_SMOKE: tuple[str, ...] = (
    "prefill_gqa_256",
    "decode_gqa_512",
    "linear_s128",
    "kda_s256",
    "mamba2_s256",
    "threads_prefill_gqa_256",
)


def attention_matrix() -> tuple[AttentionWorkload, ...]:
    return PREFILL + DECODE


def full_matrix() -> tuple[AttentionWorkload | LinearWorkload | ThreadScalingWorkload, ...]:
    return attention_matrix() + LINEAR + THREADS


def smoke_matrix() -> tuple[AttentionWorkload | LinearWorkload | ThreadScalingWorkload, ...]:
    by_name = {workload.name: workload for workload in full_matrix()}
    return tuple(by_name[name] for name in _SMOKE)


def select(suite: str, pattern: str | None) -> tuple[AttentionWorkload | LinearWorkload | ThreadScalingWorkload, ...]:
    import re

    matrix = smoke_matrix() if suite == "smoke" else full_matrix()
    if pattern is None:
        return matrix
    matcher = re.compile(pattern)
    return tuple(item for item in matrix if matcher.search(item.name))
