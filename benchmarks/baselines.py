from __future__ import annotations

import numpy as np


def attention_numpy(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    causal: bool,
    scale: float,
    query_offset: int = 0,
) -> np.ndarray:
    """Batched float32 attention via BLAS; the reference baseline path."""
    b, hq, sq, d = q.shape
    _, hkv, skv, _ = k.shape
    dv = v.shape[3]
    expand = (hq // hkv) if hkv and hq % hkv == 0 and hq >= hkv else 1
    ke = np.repeat(k, expand, axis=1) if expand > 1 else k
    ve = np.repeat(v, expand, axis=1) if expand > 1 else v
    scores = np.einsum("bhqd,bhkd->bhqk", q, ke, optimize=True) * scale
    if causal:
        query_positions = query_offset + np.arange(sq)[:, None]
        key_positions = np.arange(skv)[None, :]
        scores = np.where(key_positions <= query_positions, scores, -np.inf)
    scores -= scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=-1, keepdims=True)
    return np.einsum("bhqk,bhkd->bhqd", weights, ve, optimize=True).astype(np.float32)


def linear_numpy(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    decay: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Standard linear attention (scale + outer(k,v), readout after) via BLAS.

    decay is a scalar or a per-batch, per-token array of shape (b, sequence).
    Grouped heads share K/Q through the same head mapping as the reference.
    """
    b, groups, sequence, d = q.shape
    heads = v.shape[1]
    dv = v.shape[3]
    head_map = np.arange(heads) * groups // heads
    k_head = k[:, head_map]
    q_head = q[:, head_map]
    output = np.zeros((b, heads, sequence, dv), dtype=np.float32)
    state = np.zeros((b, heads, d, dv), dtype=np.float32)
    for token in range(sequence):
        if np.isscalar(decay):
            state *= decay
        else:
            assert not isinstance(decay, float)
            state *= decay[:, token][:, None, None, None]
        state += k_head[:, :, token, :, None] * v[:, :, token, None, :]
        output[:, :, token] = np.einsum(
            "bhd,bhde->bhe", q_head[:, :, token], state, optimize=True
        )
    return output, state


def kda_numpy(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    gate: np.ndarray,
    beta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Kimi Delta Attention: gated delta rule with normalized keys via BLAS."""
    b, heads, sequence, d = k.shape
    dv = v.shape[3]
    output = np.zeros((b, heads, sequence, dv), dtype=np.float32)
    state = np.zeros((b, heads, d, dv), dtype=np.float32)
    for token in range(sequence):
        state *= gate[:, token][:, None, None, None]
        projected = np.einsum("bhd,bhde->bhe", k[:, :, token], state, optimize=True)
        scale = beta[:, token][:, None]
        state -= scale[:, :, None, None] * k[:, :, token, :, None] * projected[:, :, None, :]
        state += scale[:, :, None, None] * k[:, :, token, :, None] * v[:, :, token, None, :]
        output[:, :, token] = np.einsum("bhd,bhde->bhe", q[:, :, token], state, optimize=True)
    return output, state


try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - depends on environment
    torch = None
    F = None


def torch_available() -> bool:
    return torch is not None


def attention_torch(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    causal: bool,
    query_offset: int = 0,
    threads: int | None = None,
) -> np.ndarray:
    """PyTorch SDPA baseline; K/V are expanded for grouped-query shapes.

    Suffix decode (query_offset > 0) uses an explicit visibility mask because
    SDPA's is_causal is always top-left aligned.
    """
    assert torch is not None and F is not None
    if threads is not None:
        torch.set_num_threads(threads)
    hq, hkv = q.shape[1], k.shape[1]
    expand = hq // hkv if hq % hkv == 0 and hq >= hkv else 1
    tq = torch.from_numpy(q)
    tk = torch.from_numpy(np.repeat(k, expand, axis=1) if expand > 1 else k)
    tv = torch.from_numpy(np.repeat(v, expand, axis=1) if expand > 1 else v)
    mask = None
    if causal:
        sq, skv = q.shape[2], k.shape[2]
        query_positions = query_offset + np.arange(sq)[:, None]
        key_positions = np.arange(skv)[None, :]
        mask = torch.from_numpy(key_positions <= query_positions)
    with torch.inference_mode():
        if mask is None:
            result = F.scaled_dot_product_attention(tq, tk, tv)
        else:
            result = F.scaled_dot_product_attention(tq, tk, tv, attn_mask=mask)
    return result.numpy()
