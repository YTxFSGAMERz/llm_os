import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from .config import ModelConfig
from .rope import apply_rotary_emb

class Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads
        
        # Q projection
        self.wq = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        # K and V projections (using n_kv_heads for future GQA compatibility)
        self.wk = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        
        self.wo = nn.Linear(config.d_model, config.d_model, bias=False)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor):
        B, T, C = x.shape
        
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)
        
        # Reshape to (B, T, heads, head_dim)
        xq = xq.view(B, T, self.n_heads, self.head_dim)
        xk = xk.view(B, T, self.n_kv_heads, self.head_dim)
        xv = xv.view(B, T, self.n_kv_heads, self.head_dim)
        
        # Apply RoPE to Q and K (NOT V!)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        
        # For naive attention, if n_kv_heads < n_heads (GQA), we need to repeat K and V.
        # In this first iteration n_kv_heads == n_heads, but let's prepare the code:
        if self.n_kv_heads != self.n_heads:
            num_queries_per_kv = self.n_heads // self.n_kv_heads
            # repeat_interleave is standard for MQA/GQA
            xk = torch.repeat_interleave(xk, num_queries_per_kv, dim=2)
            xv = torch.repeat_interleave(xv, num_queries_per_kv, dim=2)
            
        # Transpose for attention math: (B, heads, T, head_dim)
        q = xq.transpose(1, 2)
        k = xk.transpose(1, 2)
        v = xv.transpose(1, 2)
        
        # Naive attention
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Causal mask
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device)).view(1, 1, T, T)
        scores = scores.masked_fill(~mask, float('-inf'))
        
        scores = F.softmax(scores, dim=-1)
        scores = self.attn_dropout(scores)
        
        # (B, heads, T, T) @ (B, heads, T, head_dim) -> (B, heads, T, head_dim)
        output = scores @ v 
        
        # Reshape back to (B, T, C)
        output = output.transpose(1, 2).contiguous().view(B, T, -1)
        
        output = self.resid_dropout(self.wo(output))
        return output
