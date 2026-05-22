from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int = 50257
    d_model: int = 256
    n_heads: int = 8
    n_kv_heads: int = 8  # Set differently for GQA/MQA later
    n_layers: int = 4
    max_seq_len: int = 128
    dropout: float = 0.1
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
