# Contributing to LLM OS

We welcome contributions to make LLM OS an even better educational resource and experimentation framework!

## Getting Started

1. **Fork the repository** and create your branch from `main`.
2. **Install dependencies**: `pip install -r requirements.txt`.
3. **Run tests** (if available) to ensure your environment is set up correctly.

## Development Workflow

* **Modular Design**: If you are adding a new architectural feature (e.g., a new attention mechanism or activation function), please place it in the appropriate file inside `src/model/` (e.g., `attention.py` or `mlp.py`).
* **Keep it Simple**: This project is meant to be educational and clean. Avoid overly complex abstractions unless they significantly improve performance or readability.
* **Benchmarking**: If you optimize a component, please run `python scripts/benchmark.py` and share the before/after results in your pull request.

## Submitting Changes

1. Ensure your code follows standard Python PEP-8 guidelines.
2. Write clear and descriptive commit messages.
3. Open a Pull Request detailing the changes you've made, the motivation behind them, and any benchmarking results if applicable.
