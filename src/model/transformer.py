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

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor):
        h = x + self.attn(self.attn_norm(x), freqs_cis)
        out = h + self.mlp(self.mlp_norm(h))
        return out

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

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor = None):
        B, T = tokens.shape
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds maximum {self.max_seq_len}"
        
        h = self.tok_embeddings(tokens)
        h = self.drop(h)
        
        freqs_cis = self.freqs_cis[:T]
        
        for layer in self.layers:
            h = layer(h, freqs_cis)
            
        h = self.norm(h)
        logits = self.output(h)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss
