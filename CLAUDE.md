# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**pico-qwen** is an experimental feature-enriched version of qwen3-rs targeting very low-resource systems. It's a Rust workspace implementing Qwen3 LLM inference with educational goals, focusing on transparency and minimal dependencies.

## Workspace Architecture

The project uses a Cargo workspace with three main crates:

- **qwen3-cli**: Command-line interface (main binary)
- **qwen3-export**: Model export from HuggingFace to custom binary format
- **qwen3-inference**: Core LLM inference library

## Build Commands

```bash
# Build entire workspace
cargo build --release

# Build specific crate
cargo build --release -p qwen3-cli
cargo build --release -p qwen3-export
cargo build --release -p qwen3-inference

# Run CLI with help
cargo run -p qwen3-cli -- --help
```

## Testing Commands

```bash
# Run all tests
cargo test

# Run tests for specific crate
cargo test -p qwen3-export
cargo test -p qwen3-inference

# Run specific test module
cargo test -p qwen3-export model_exporter_test
```

## Model Workflow

### 1. Export HuggingFace Model

```bash
# Export Qwen3 model to binary format
cargo run --release -p qwen3-cli -- export /path/to/huggingface/model /path/to/output.bin --group-size 64

# Available group sizes: powers of 2 (32, 64, 128, etc.)
```

### 2. Run Inference

```bash
# Interactive chat mode
cargo run --release -p qwen3-cli -- inference /path/to/output.bin -m chat

# Generate mode with prompt
cargo run --release -p qwen3-cli -- inference /path/to/output.bin -m generate -i "Your prompt here"

# With custom parameters
cargo run --release -p qwen3-cli -- inference /path/to/output.bin -t 0.7 -p 0.8 -s 42
```

## Key Architecture Components

### Model Export (qwen3-export)
- **BinaryModelExporter**: Quantizes weights to INT8 with configurable group size
- **TokenizerExporter**: Exports BPE tokenizer with vocabulary and merge rules
- **ChatTemplateExporter**: Exports prompt templates for chat formatting
- **TensorReader**: Reads HuggingFace safetensors format

### Inference Engine (qwen3-inference)
- **Transformer**: Main model with decoder-only architecture
- **Tokenizer**: Byte-level BPE tokenization
- **Sampler**: Temperature and top-p sampling
- **QuantizedTensor**: INT8 quantization with group-wise scaling

### CLI Interface (qwen3-cli)
- **Export command**: Model conversion from HuggingFace
- **Inference command**: Interactive chat and text generation

## File Structure

```
pico-qwen/
├── Cargo.toml                    # Workspace configuration
├── qwen3-cli/                   # Command-line interface
│   ├── src/main.rs              # CLI entry point
├── qwen3-export/                # Model export utilities
│   ├── src/
│   │   ├── model_exporter.rs    # Binary format export
│   │   ├── tokenizer_exporter.rs
│   │   └── tensor_reader.rs
├── qwen3-inference/             # Core inference library
│   ├── src/
│   │   ├── transformer.rs       # Main model architecture
│   │   ├── tokenizer.rs         # BPE tokenizer
│   │   ├── sampler.rs           # Sampling strategies
│   │   └── tensor.rs            # Tensor operations
└── docs/                        # Technical documentation
```

## Development Notes

### Model Requirements
- HuggingFace format: config.json, tokenizer.json, *.safetensors files
- Supported models: Qwen3-0.6B, Qwen3-4B, Qwen3-8B, and DeepSeek-R1-0528-Qwen3-8B (more performant) variants
- Output format: Custom binary (.bin) + tokenizer files

### Key Dependencies
- **anyhow**: Error handling
- **safetensors**: HuggingFace tensor format
- **rayon**: Parallel processing
- **memmap2**: Memory-mapped file access
- **byteorder**: Binary format handling

### Performance Features
- INT8 quantization for memory efficiency
- Memory-mapped model loading
- Parallel attention computation
- KV cache for autoregressive generation
- Grouped Query Attention (GQA) optimization

## Common Development Tasks

```bash
# Quick build and test cycle
cargo check && cargo test

# Run single test
cargo test -p qwen3-inference transformer::tests::test_model_loading

# Debug build for development
cargo build -p qwen3-cli

# Check formatting
cargo fmt --check

# Run clippy
cargo clippy --all-targets --all-features
```

## Coding Conventions

## Environment Setup Details

## Solutions to common repo bugs or problems

## Planning Phase Guidelines

### PocketFlow Integration
- Every time you enter the planning phase for a feature that includes "PocketFlow", you MUST check the most up to date documentation using the "context7" MCP server.
