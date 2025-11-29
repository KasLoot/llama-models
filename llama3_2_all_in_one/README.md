# Llama 3.2 All-In-One Inference

This folder contains a self-contained, flattened implementation of the Llama 3.2 inference code. It is designed to be "reading-friendly" and easy to run without complex dependency chains from the original repository structure.

## Features

- **Self-Contained**: All necessary code (model, tokenizer, generation, multimodal support) is included in this directory.
- **Multimodal Support**: Includes support for Llama 3.2 Vision models (if using a vision checkpoint).
- **Simplified Structure**: Flattened imports and directory structure for easier navigation.

## Prerequisites

You need a Python environment with the following packages installed:

```bash
uv pip install torch fairscale fire termcolor tiktoken pillow
```

## Usage

The main entry point is `chat_completion.py`. You can run it directly:

```bash
python chat_completion.py
```

### Arguments

You can customize the execution using command-line arguments:

- `--ckpt_dir`: Path to the model checkpoint directory (containing `params.json`, `tokenizer.model`, and `.pth` files).
  - Default: `/media/yuxin/LinuxSD/models/Llama-3.2-1B-Instruct/original`
- `--temperature`: Sampling temperature (default: 0.8).
- `--top_p`: Nucleus sampling probability (default: 0.9).
- `--max_seq_len`: Maximum sequence length (default: 512).
- `--max_batch_size`: Maximum batch size (default: 4).

**Example:**

```bash
python chat_completion.py --ckpt_dir /path/to/your/llama-3.2-3b/ --temperature 0.6
```

## File Structure

- **`chat_completion.py`**: Main script to run inference.
- **`generation.py`**: Handles the generation loop (sampling, decoding).
- **`model.py`**: The core Transformer model architecture.
- **`tokenizer.py`**: Tokenizer implementation (using tiktoken).
- **`chat_format.py`**: Handles prompt formatting (special tokens, headers).
- **`multimodal/`**: Contains the Vision Encoder and image transformation logic.
- **`args.py`**, **`datatypes.py`**: Configuration classes and data structures.

## Checkpoints

This code expects the original Llama 3.2 checkpoint format (downloaded via `llama` CLI or from Hugging Face). The directory should contain:
- `params.json`
- `tokenizer.model`
- `consolidated.00.pth` (or multiple shards)
