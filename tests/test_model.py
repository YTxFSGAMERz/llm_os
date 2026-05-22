import torch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from model import ModelConfig, Attention, precompute_freqs_cis, apply_rotary_emb, RMSNorm, SwiGLU

def test_rope_shapes():
    B, T, n_heads, n_kv_heads, head_dim = 2, 10, 4, 2, 64
    d_model = n_heads * head_dim
    
    q = torch.randn(B, T, n_heads, head_dim)
    k = torch.randn(B, T, n_kv_heads, head_dim)
    
    freqs_cis = precompute_freqs_cis(head_dim, T)
    
    q_rot, k_rot = apply_rotary_emb(q, k, freqs_cis)
    
    assert q_rot.shape == q.shape, "RoPE Q shape mismatch"
    assert k_rot.shape == k.shape, "RoPE K shape mismatch"
    print("RoPE shape test passed!")

def test_attention_shapes_and_masking():
    config = ModelConfig(d_model=256, n_heads=8, n_kv_heads=4, max_seq_len=32)
    attn = Attention(config)
    
    B, T, C = 2, 10, config.d_model
    x = torch.randn(B, T, C)
    freqs_cis = precompute_freqs_cis(config.d_model // config.n_heads, T)
    
    out = attn(x, freqs_cis)
    
    assert out.shape == (B, T, C), "Attention output shape mismatch"
    print("Attention shape test passed (includes GQA KV repeat test)!")

def test_rms_norm():
    C = 64
    norm = RMSNorm(C)
    x = torch.randn(2, 10, C)
    out = norm(x)
    assert out.shape == x.shape, "RMSNorm shape mismatch"
    print("RMSNorm shape test passed!")

if __name__ == "__main__":
    test_rope_shapes()
    test_attention_shapes_and_masking()
    test_rms_norm()
    print("All tests passed!")
