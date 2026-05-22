import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional, Tuple
from .config import ModelConfig
from .rope import apply_rotary_emb

class AttentionBackend(nn.Module):
    """Abstract base class for attention implementations."""
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = True) -> torch.Tensor:
        raise NotImplementedError

class NaiveAttentionBackend(AttentionBackend):
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.attn_dropout = nn.Dropout(dropout)
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = True) -> torch.Tensor:
        # q, k, v shape: (B, heads, T, head_dim)
        T_q = q.size(2)
        T_k = k.size(2)
        head_dim = q.size(-1)
        
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
        
        if is_causal and T_q > 1:
            # Create a causal mask for T_q x T_k
            mask = torch.ones(T_q, T_k, dtype=torch.bool, device=q.device)
            mask = torch.tril(mask, diagonal=(T_k - T_q))
            scores = scores.masked_fill(~mask.view(1, 1, T_q, T_k), float('-inf'))
            
        scores = F.softmax(scores, dim=-1)
        scores = self.attn_dropout(scores)
        
        return scores @ v

class FlashAttentionBackend(AttentionBackend):
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout_p = dropout
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = True) -> torch.Tensor:
        # F.scaled_dot_product_attention expects q, k, v in (B, heads, T, head_dim)
        # It automatically selects FlashAttention, MemoryEfficientAttention, or Math based on hardware.
        p = self.dropout_p if self.training else 0.0
        causal = is_causal if q.size(2) > 1 else False
        return F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=p, is_causal=causal)

class Attention(nn.Module):
    def __init__(self, config: ModelConfig, backend: str = "naive"):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads
        
        self.wq = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        
        self.wo = nn.Linear(config.d_model, config.d_model, bias=False)
        
        if backend == "flash":
            self.backend = FlashAttentionBackend(config.dropout)
        else:
            self.backend = NaiveAttentionBackend(config.dropout)
            
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, 
                use_cache: bool = False, 
                past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        if use_cache and past_kv is not None:
            return self.inference_forward(x, freqs_cis, past_kv)
        else:
            return self.training_forward(x, freqs_cis, use_cache=use_cache)

    def training_forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, use_cache: bool = False):
        B, T, C = x.shape
        
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)
        
        xq = xq.view(B, T, self.n_heads, self.head_dim)
        xk = xk.view(B, T, self.n_kv_heads, self.head_dim)
        xv = xv.view(B, T, self.n_kv_heads, self.head_dim)
        
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        
        # (B, T, heads, head_dim) -> (B, heads, T, head_dim)
        q = xq.transpose(1, 2)
        k = xk.transpose(1, 2)
        v = xv.transpose(1, 2)
        
        new_past_kv = (k, v) if use_cache else None
        
        if self.n_kv_heads != self.n_heads:
            num_queries_per_kv = self.n_heads // self.n_kv_heads
            k = torch.repeat_interleave(k, num_queries_per_kv, dim=1)
            v = torch.repeat_interleave(v, num_queries_per_kv, dim=1)
            
        output = self.backend(q, k, v, is_causal=True)
        
        output = output.transpose(1, 2).contiguous().view(B, T, -1)
        output = self.resid_dropout(self.wo(output))
        
        return output, new_past_kv
        
    def inference_forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, past_kv: Tuple[torch.Tensor, torch.Tensor]):
        B, T, C = x.shape
        
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)
        
        xq = xq.view(B, T, self.n_heads, self.head_dim)
        xk = xk.view(B, T, self.n_kv_heads, self.head_dim)
        xv = xv.view(B, T, self.n_kv_heads, self.head_dim)
        
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        
        q = xq.transpose(1, 2)
        k = xk.transpose(1, 2)
        v = xv.transpose(1, 2)
        
        past_k, past_v = past_kv
        
        # Append to KV cache along sequence dimension (dim=2 since shape is B, heads, T, head_dim)
        k = torch.cat([past_k, k], dim=2)
        v = torch.cat([past_v, v], dim=2)
        
        new_past_kv = (k, v)
        
        if self.n_kv_heads != self.n_heads:
            num_queries_per_kv = self.n_heads // self.n_kv_heads
            k = torch.repeat_interleave(k, num_queries_per_kv, dim=1)
            v = torch.repeat_interleave(v, num_queries_per_kv, dim=1)
            
        output = self.backend(q, k, v, is_causal=(T > 1))
        
        output = output.transpose(1, 2).contiguous().view(B, T, -1)
        output = self.resid_dropout(self.wo(output))
        
        return output, new_past_kv
