import torch
import torch.nn as nn
from .config import ModelConfig

class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        # Standard transformer MLP expands by 4x. 
        # For SwiGLU, since we multiply two projections, we often adjust the hidden dim to maintain parameter count.
        # Commonly hidden_dim = int(8 * d_model / 3). For simplicity, we'll use 4 * d_model as the hidden space.
        hidden_dim = 4 * config.d_model
        
        self.w1 = nn.Linear(config.d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.d_model, bias=False)
        self.w3 = nn.Linear(config.d_model, hidden_dim, bias=False)
        
        self.silu = nn.SiLU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor):
        # SwiGLU(x) = SiLU(x * W1) * (x * W3)
        # Then projected by W2
        hidden = self.silu(self.w1(x)) * self.w3(x)
        out = self.dropout(self.w2(hidden))
        return out
