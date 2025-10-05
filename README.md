# specdecode

A from-scratch PyTorch implementation of speculative decoding for accelerated LLM inference with evaluation tools for measuring latency and throughput tradeoffs.

## What is Speculative Decoding?

Speculative decoding is an inference acceleration technique that uses a small, fast "draft" model to propose multiple tokens at once, which are then verified in parallel by a larger "target" model. This approach can significantly reduce the wall-clock time for text generation while maintaining the same output quality as standard autoregressive decoding.

**Key benefits:**
- **Faster inference**: Reduces latency by accepting multiple tokens per iteration when draft proposals are correct
- **Quality preservation**: Maintains identical output distribution to the target model
- **No fine-tuning required**: Works with pre-trained models out of the box

## Features

This repository implements:

- **Greedy speculative decoding**: Deterministic token selection with greedy verification
- **Sampling speculative decoding**: Stochastic generation with probabilistic acceptance (coming soon)
- **Performance evaluation**: Metrics for both latency and throughput analysis
- **Configurable parameters**: YAML-based configuration for models, datasets, and decoding parameters
- **Batched inference**: Support for processing multiple sequences in parallel

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/specdecode.git
cd specdecode

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- **torch** (>=2.0.0): PyTorch for model execution
- **transformers** (>=4.40.0): HuggingFace transformers for model loading
- **datasets** (>=2.14.0): HuggingFace datasets for data loading
- **pyyaml** (>=6.0): YAML configuration parsing
- **bitsandbytes** (>=0.41.0): 4-bit quantization support (optional but recommended for lower memory usage)
- **accelerate** (>=0.20.0): Model parallelism and optimization utilities

**Note**: For Flash Attention 2 support, install `flash-attn`:
```bash
pip install flash-attn --no-build-isolation
```

## Usage

### Basic Usage

Run speculative decoding with a configuration file:

```bash
python launch.py --config configs/gemma-3-4b-it.yaml
```

### Configuration

Create a YAML configuration file (see `configs/gemma-3-4b-it.yaml` for an example):

```yaml
# Model configuration
draft_model: "google/gemma-3-270m-it"
target_model: "google/gemma-3-4b-it"

# Decoding parameters
k: 5
max_new_tokens: 256
max_length: 512

# Dataset configuration
num_samples: 100
dataset:
  path: "openai/gsm8k"
  name: "main"

# Performance settings
batch_size: 4
device: "cuda"
quantized: true
summary_file_name: "my_experiment"
```

### Key Parameters

- **draft_model**: HuggingFace model ID or local path for the smaller, faster draft model
- **target_model**: HuggingFace model ID or local path for the larger, higher-quality target model
- **k**: Number of tokens the draft model proposes per iteration
- **max_new_tokens**: Maximum tokens to generate per sequence
- **max_length**: Maximum input sequence length (for tokenization)
- **batch_size**: Number of sequences to process in parallel
- **quantized**: Enable 4-bit quantization (requires bitsandbytes)
- **device**: CUDA device to use (e.g., "cuda", "cuda:0", "cuda:1")

## Evaluation

The repository evaluates the tradeoffs between latency and throughput:

- **Latency**: Time to generate a single sequence (affected by batch size and acceptance rate)
- **Throughput**: Total tokens generated per second across all sequences
- **Block efficiency**: Average number of tokens accepted per speculation round (k = draft tokens proposed)
- **Baseline comparison**: Measures speedup vs. standard target-only generation
- **Acceptance rate**: Percentage of draft tokens accepted by the target model

Metrics are automatically computed and displayed after each run:

```
################################
Baseline Run (Target)
################################
Time: 15.32s

################################
Speculative Run (Target + Draft)
################################
Time: 8.45s
Speedup: 1.81x
Acceptance Rate: 76.3%
Block Efficiency: 3.82 tokens/round
```

## Project Structure

```
specdecode/
├── configs/           # YAML configuration files
├── decode/            # Decoding implementations
│   └── greedy.py     # Greedy speculative decoding
├── eval/              # Evaluation metrics
│   └── metrics.py    # Performance tracking
├── utils/             # Utility functions
│   ├── model_loader.py
│   └── timing.py
├── launch.py          # Main entry point
└── README.md
```

## Implementation Details

- **Cache management**: Uses KV-cache trimming for efficient memory usage
- **Device handling**: Supports CUDA acceleration with mixed precision (bfloat16)
- **Flash Attention 2**: Automatic detection and usage when available
- **Batch processing**: Handles variable-length sequences with padding

## License

MIT License - see LICENSE file for details

## References

- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) (Leviathan et al., 2022)
- [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) (Chen et al., 2023)
