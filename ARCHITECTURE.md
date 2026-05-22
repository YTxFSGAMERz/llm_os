# LLM OS Architecture

The LLM OS repository is built around a highly modular Transformer architecture, designed to make switching out components (like attention mechanisms or normalization layers) straightforward.

## Core Modules (`src/model/`)

* **`config.py`**: Defines the `ModelConfig` dataclass, which holds hyperparameters like `vocab_size`, `d_model`, `n_heads`, `n_layers`, and `max_seq_len`.
* **`transformer.py`**: Contains the top-level `TransformerModel` and `TransformerBlock`. It glues together the attention, normalization, and MLP components.
* **`attention.py`**: Implements Multi-Head Attention (MHA) and Grouped-Query Attention (GQA). Supports swappable backends (e.g., Naive Attention vs. FlashAttention).
* **`mlp.py`**: Implements the feed-forward networks, typically using GELU or SwiGLU activations.
* **`norms.py`**: Contains normalization layers, allowing easy toggling between `LayerNorm` and `RMSNorm`.
* **`rope.py`**: Implements Rotary Positional Embeddings (RoPE), replacing standard learned positional embeddings.
* **`generation.py`**: Contains the logic for auto-regressive token generation, including features like temperature scaling, top-k sampling, and KV Caching for fast inference.

## Evolution of the Model

The framework is built to iterate from a basic implementation to a more advanced, modern setup:

* **Phase 1A (Basic)**: Naive Self-Attention, standard Multi-Layer Perceptron with GELU, LayerNorm, and learned positional embeddings.
* **Phase 1B (Modernized)**: Integration of RoPE (Rotary Positional Embeddings), SwiGLU activations, RMSNorm, and FlashAttention to mimic the architecture of state-of-the-art models like LLaMA.
