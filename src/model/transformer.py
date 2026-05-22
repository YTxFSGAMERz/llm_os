import torch
import torch.nn as nn
from torch.nn import functional as F
from .config import ModelConfig
from .attention import Attention
from .mlp import SwiGLU
from .norms import RMSNorm
from .rope import precompute_freqs_cis

class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn = Attention(config)
        self.mlp = SwiGLU(config)
        self.attn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.mlp_norm = RMSNorm(config.d_model, eps=config.norm_eps)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, 
                use_cache: bool = False, past_kv = None):
        attn_out, new_past_kv = self.attn(self.attn_norm(x), freqs_cis, use_cache=use_cache, past_kv=past_kv)
        h = x + attn_out
        out = h + self.mlp(self.mlp_norm(h))
        return out, new_past_kv

class TransformerModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.max_seq_len = config.max_seq_len
        
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model, eps=config.norm_eps)
        
        self.output = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Share weights between embedding and output projection
        self.tok_embeddings.weight = self.output.weight
        
        # Precompute RoPE frequencies
        freqs_cis = precompute_freqs_cis(
            config.d_model // config.n_heads,
            config.max_seq_len * 2,
            config.rope_theta
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor = None,
                use_cache: bool = False, past_kvs: list = None):
        B, T = tokens.shape
        
        h = self.tok_embeddings(tokens)
        h = self.drop(h)
        
        if use_cache and past_kvs is not None:
            # We are generating token by token. T is usually 1.
            # The absolute position is the cache length + current sequence length
            past_len = past_kvs[0][0].size(2)
            freqs_cis = self.freqs_cis[past_len:past_len+T]
        else:
            freqs_cis = self.freqs_cis[:T]
        
        new_past_kvs = [] if use_cache else None
        
        for i, layer in enumerate(self.layers):
            past_kv = past_kvs[i] if past_kvs is not None else None
            h, new_past_kv = layer(h, freqs_cis, use_cache=use_cache, past_kv=past_kv)
            if use_cache:
                new_past_kvs.append(new_past_kv)
            
        h = self.norm(h)
        logits = self.output(h)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        if use_cache:
            return logits, loss, new_past_kvs
        return logits, loss
