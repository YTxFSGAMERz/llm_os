# LLM OS (Miniature)

A lightweight, PyTorch-based framework for training and benchmarking custom Large Language Models from scratch. It is designed to be educational, highly modular, and easily extensible, allowing experimentation with modern LLM architectures.

## Features

* **Modular Architecture**: The codebase separates attention mechanisms, normalization layers, positional encodings (like RoPE), and generation logic.
* **Modern LLM Components**:
  * FlashAttention backend support for faster inference and training.
  * KV Caching for accelerated token generation.
  * Rotary Positional Embeddings (RoPE).
  * SwiGLU / GELU MLPs.
  * RMSNorm / LayerNorm.
* **Built-in Benchmarking**: Scripts to measure forward pass speeds, generation speeds (with/without KV Cache), and compilation overhead using `torch.compile`.
* **Simple Training Loop**: An easy-to-understand training loop provided in `src/train.py`, initially set up for character/token-level training on the Tiny Shakespeare dataset.

## Installation

1. Clone the repository and navigate to the project root.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Data Preparation
To download and prepare the default Tiny Shakespeare dataset, run:
```bash
python prepare.py
```
This will create a `data/` directory and download `tinyshakespeare.txt`.

### 2. Training
To start training a model from scratch, execute:
```bash
python src/train.py
```
This script initializes a `TransformerModel`, loads the data using the provided `Tokenizer` and `DataLoader`, and begins the training loop. Once training concludes, a sample generation will be printed.

### 3. Benchmarking
To measure the performance of the model components (Forward Pass, KV Caching, FlashAttention, and `torch.compile`), run the benchmarking script:
```bash
python scripts/benchmark.py
```

## Directory Structure

* `data/`: Contains datasets (e.g., `tinyshakespeare.txt`).
* `scripts/`: Utility scripts like `benchmark.py` and data generation scripts.
* `src/`: Core source code.
  * `model/`: Modularized model components (`attention.py`, `transformer.py`, `rope.py`, `mlp.py`, etc.).
  * `data.py`: DataLoader implementations.
  * `tokenizer.py`: Tokenizer wrapper (using `tiktoken`).
  * `train.py`: The main training loop.
* `tests/`: Directory for unit tests.

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute.
