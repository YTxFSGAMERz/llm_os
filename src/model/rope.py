import torch
import torch.nn as nn

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (RoPE)
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    # complex representation for easy rotation
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting along the batch and head dimensions.
    x shape: (B, T, n_heads, head_dim)
    freqs_cis shape: (T, head_dim/2)
    Output shape: (1, T, 1, head_dim/2)
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    shape[-1] = x.shape[-1]
    return freqs_cis.view(*shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
):
    """
    Apply rotary embeddings to queries and keys.
    xq: (B, T, n_heads, head_dim)
    xk: (B, T, n_kv_heads, head_dim)
    """
    # Reshape xq and xk to view as complex numbers
    # (B, T, n_heads, head_dim // 2, 2) -> complex64 (B, T, n_heads, head_dim // 2)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    
    # Rotate (multiply complex numbers)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)
