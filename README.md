# pico-qwen

## Project Overview

**pico-qwen** is a production-ready, low-resource LLM inference system built as an experimental, feature-enriched version of [qwen3-rs by Ilya Builuk](https://github.com/reinterpretcat/qwen3-rs). Designed specifically for very low-resource systems like MiniPCs, SBCs, and low-power Home Automation servers, while maintaining educational clarity and minimal dependencies.

## ✅ Phase 1: Core Infrastructure Extension - COMPLETED

Phase 1 has been successfully completed, providing the foundation for all subsequent features:

### ✅ Completed Features
- **Extended Model Configuration** - Advanced configuration with quantization levels and CPU-specific optimizations
- **Advanced Quantization System** - Support for INT4, INT8, FP16, FP32 with configurable group sizes
- **CPU Feature Detection** - Runtime CPU optimization for Intel N100, Intel i9-14900HX, Raspberry Pi 4/5, and generic processors
- **Extended Transformer Support** - Configuration-based transformer loading with optimization
- **Comprehensive Testing** - Well-tested infrastructure components
- **High-end CPU Support** - Added Intel i9-14900HX (24 cores, 32 threads, 36MB cache) with FP16 optimization

### 🔧 New Configuration System
```rust
use qwen3_inference::{ExtendedModelConfig, CpuTarget, QuantizationLevel};

// Auto-detect optimal configuration
let config = ExtendedModelConfig::new(model_config);

// Target specific hardware
let config = ExtendedModelConfig::for_cpu_target(model_config, CpuTarget::RaspberryPi4);
let config = ExtendedModelConfig::for_cpu_target(model_config, CpuTarget::IntelI9_14900HX);

// Save/load configurations
config.save_to_file("model_config.toml")?;
let loaded = ExtendedModelConfig::from_file("model_config.toml")?;
```

## 🎯 Project Goals - Updated Roadmap

### ✅ Phase 1: Core Infrastructure Extension - **COMPLETED**
- ✅ Extended model configuration system
- ✅ Advanced quantization (INT4/INT8/FP16/FP32)
- ✅ CPU-specific optimization strategies
- ✅ Memory usage estimation and validation

### 🚧 Phase 2: Low-Requirements API Server - **IN PROGRESS**
- [ ] REST API with streaming support
- [ ] Model pooling and LRU eviction
- [ ] Memory pressure monitoring
- [ ] HTTP endpoints for chat and generation

### 🚧 Phase 3: Minimalist Chat-WebUI
- [ ] Progressive enhancement web interface
- [ ] Mobile-first responsive design
- [ ] Offline capability with service worker
- [ ] Real-time streaming responses

### 🚧 Phase 4: MCP Multi-Agent System
- [ ] PocketFlow-based agent orchestration
- [ ] Web search and research agents
- [ ] Sequential thinking capabilities
- [ ] Tool cost estimation and management

### 🚧 Phase 5: Hybrid Cloud/Edge Inference
- [ ] Cloud provider abstraction (OpenAI, Anthropic)
- [ ] Fallback to local inference
- [ ] Health monitoring and failover
- [ ] Cost estimation and routing

### 🚧 Phase 6: CPU-Specific Optimization
- [ ] Runtime CPU feature detection
- [ ] SIMD optimization (AVX2, AVX-512, NEON)
- [ ] Cache-aware data blocking
- [ ] Target-specific builds

### 🚧 Phase 7: Deployment & Service Management
- [ ] systemd service integration
- [ ] Configuration management
- [ ] Logging and monitoring
- [ ] Cross-platform packaging

## 📊 Resource Requirements

| Component | Memory | CPU | Storage | Target Systems |
|-----------|--------|-----|---------|----------------|
| API Server | 50MB | Low | 10MB | MiniPCs, SBCs |
| 7B Model (INT4) | 4GB | Moderate | 4GB | Raspberry Pi 4 |
| 7B Model (INT8) | 8GB | Moderate | 8GB | Intel N100 |
| WebUI | 5MB | Very Low | 2MB | All targets |
| MCP Agents | 100MB | Low | 50MB | Network-connected |

## 🏗️ Architecture - Updated for Pico-Qwen

```
pico-qwen/
├── Cargo.toml                    # Workspace configuration
├── qwen3-cli/                   # Command-line interface
├── qwen3-export/                # Model export utilities
├── qwen3-inference/             # Core inference library (ENHANCED)
│   ├── src/
│   │   ├── configuration.rs     # Original config
│   │   ├── extended_config.rs   # NEW: Extended configuration
│   │   ├── cpu_optimizations.rs # NEW: CPU feature detection
│   │   ├── extended_transformer.rs # NEW: Enhanced transformer
│   │   ├── quantization.rs      # ENHANCED: Advanced quantization
│   │   └── ...
├── qwen3-api/                   # NEW: REST API server (Phase 2)
├── qwen3-web/                   # NEW: Web interface (Phase 3)
├── qwen3-mcp/                   # NEW: MCP agents (Phase 4)
└── docs/                        # Technical documentation
    ├── phase1_summary.md        # Phase 1 completion
    ├── plan_1.md               # Implementation plan
    └── ...
```

## 🚀 Quick Start

### Prerequisites
- Rust 1.70+
- 4GB+ RAM (8GB recommended for larger models)
- HuggingFace Qwen3 model

### Installation
```bash
# Clone the repository
git clone https://github.com/PieBru/pico-qwen.git
cd pico-qwen

# Build the workspace
cargo build --release

# Export a model (example with Qwen3-0.6B)
cargo run --release -p qwen3-cli -- export Qwen3-0.6B model.bin --group-size 64

# Run inference with new configuration system
cargo run --release -p qwen3-cli -- inference model.bin -m chat -t 0.7
```

### Configuration Example
```toml
# model_config.toml
[base]
dim = 2048
hidden_dim = 8192
n_layers = 24
vocab_size = 32000

[quantization]
level = "int8-gs64"

[cpu]
target = "intel-n100"
# Available targets: intel-n100, intel-i9-14900hx, raspberry-pi-4, raspberry-pi-5, generic-x86, generic-arm

[memory]
max_memory_mb = 8192
max_context_length = 4096
```

## 📋 Development Status

| Phase | Status | Progress |
|-------|--------|----------|
| Phase 1: Core Infrastructure | ✅ **COMPLETED** | 100% |
| Phase 2: API Server | 🚧 **IN PROGRESS** | 0% |
| Phase 3: WebUI | 📋 **PENDING** | 0% |
| Phase 4: MCP Agents | 📋 **PENDING** | 0% |
| Phase 5: Hybrid Cloud | 📋 **PENDING** | 0% |
| Phase 6: CPU Optimization | 📋 **PENDING** | 0% |
| Phase 7: Deployment | 📋 **PENDING** | 0% |

## 🔧 Development Commands

### Phase 1 Testing Commands (Arch Linux)
```bash
# Build and test Phase 1 features
cargo build --release --all
cargo test --release --all

# Test specific Phase 1 features
cargo test --test integration_tests test_quantization_levels
cargo test --test integration_tests test_cpu_target_detection
cargo test --test integration_tests test_cpu_target_parsing
cargo test --test integration_tests test_memory_limits_calculation
cargo test --test integration_tests test_cpu_info_detection

# CPU and configuration testing
cargo test --test extended_config_tests test_cpu_target_optimization
cargo test --test extended_config_tests test_configuration_serialization
cargo test --test extended_config_tests test_quantization_memory_calculation

# Real-world configuration validation
cargo run --release -p qwen3-cli -- inference /dev/null --config model_config.toml

# Check your CPU detection
cat /proc/cpuinfo | grep -E "(model name|Intel|i9-14900HX)"
lscpu | grep -E "(Model name|CPU|Thread|Core|Cache)"
```

### Quick Phase 1 Validation Script
```bash
#!/bin/bash
cd /run/media/piero/NVMe-4TB/Piero/GitForked/pico-qwen

echo "=== Phase 1 Feature Testing ==="
echo "1. Building workspace..."
cargo build --release --all

echo -e "\n2. Running all tests..."
cargo test --release --all 2>/dev/null | grep -E "(test|ok|failed)" | tail -5

echo -e "\n3. Testing your i9-14900HX configuration..."
cargo run --release -p qwen3-cli -- inference /dev/null --config model_config.toml 2>&1 | head -10

echo -e "\n4. CPU Info:"
lscpu | grep -E "(Model name|CPU|Thread|Core)" | head -5

echo -e "\n=== Phase 1 Testing Complete ==="
```

### Development workflow
cargo check --all
cargo test --all
cargo build --release -p qwen3-cli

# Run with configuration
cargo run --release -p qwen3-cli -- inference model.bin --config model_config.toml

# Cross-compilation
cargo build --release --target aarch64-unknown-linux-gnu  # ARM64
cargo build --release --target x86_64-unknown-linux-musl  # x86_64

## 📝 Project Guidelines

- **Architecture**: Modular workspace structure inherited from qwen3-rs
- **Documentation**: All docs in `./docs/` directory
- **Testing**: Comprehensive integration tests for all features
- **Dependencies**: Minimal and well-vetted crates only
- **Cross-Platform**: Works on Arch Linux, ARM64, and x86_64
- **systemd**: Ready for service management

## 🤝 Contributing

This is an experimental project. Contributions are welcome, especially for:
- Additional CPU target support
- Performance optimizations
- WebUI enhancements
- MCP agent implementations

## 📄 License

This project maintains the same license as the original qwen3-rs project.


## Project Guidelines

- **Architecture.** We keep the original "qwen3-rs" forked project posture and create a new git branch for each new feature, so it will be easy to merge our additional features back on the original project.
- **Project documentation.** Except the `README.md`, all our project documentation is contained into the `./docs/` directory. So, when we refer to the original project abstractions contained into `qwen3-rs.md`, that document file is located into `./docs/qwen3-rs.md`.
  - @docs/qwen3-rs.md is a tutorial on our project foundation, that is based on the `qwen3-rs` project, please load it into the coding agent context before planning any new feature.
- We would like to reuse as much as possible of the project code at any development stage.
- We prefer to use **Arch Linux**, and any command/feature of our project should work from the command line, as well as a **`systemd`** service managed via `systemctl`.
- We like minimalist, streamlined tools, CLI and UI. Also please use minimal dependencies.
- **Agentic features.** Some of our new features use agents and tools, they should be implemented using the [PocketFlow Framework](https://github.com/the-pocket/PocketFlow). Its documentation is contained into https://github.com/The-Pocket/PocketFlow/tree/main/docs/ , its abstractions tutorial is locally into @docs/PocketFlow.md and a cookbook documentation rich of examples is @docs/PocketFlow_Cookbook_40.md

---

# qwen3-rs Description

**qwen3-rs** is an educational Rust project for exploring and running Qwen3 language family models. It is designed to be clear, modular, and approachable for learners, with minimal dependencies and many core algorithms reimplemented from scratch for transparency.

> **Note:** Parts of this codebase, including documentation and core algorithms, were generated or assisted by large language models (LLMs) to accelerate development and improve educational clarity. As a starting reference, the project [qwen3.c](https://github.com/adriancable/qwen3.c) was used for understanding model internals and file formats.


## qwen3-rs Project Goals

- **Educational:** Learn how transformer architectures, quantization, and efficient inference work in Rust.
- **Minimal Dependencies:** Most algorithms (tokenization, quantization, sampling, etc.) are implemented from scratch—no heavy ML or Python bindings.
- **Modular:** Core library logic is separated from CLI tools for clarity and maintainability.
- **Efficiency:** Uses memory mapping and zero-copy techniques for handling large model files.

## Workspace Structure

```
qwen3-rs/
├── docs                # LLM generated docs for key components
├── Cargo.toml          # Workspace configuration
├── qwen3-cli/          # Command-line interface crate
├── qwen3-export/       # Model export crate
├── qwen3-inference/    # LLM inference crate
```

## How to Use

### 1. Get a HuggingFace Qwen3 model

```bash
git clone https://huggingface.co/Qwen/Qwen3-0.6B
# Or try larger/alternative models:
# git clone https://huggingface.co/Qwen/Qwen3-4B
# git clone https://huggingface.co/Qwen/Qwen3-8B
# git clone https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
```

### 2. Build and run the exporter

```bash
cargo build --release -p qwen3-cli

# Export a HuggingFace model to quantized checkpoint format
cargo run --release -p qwen3-cli -- export /path/to/model /path/to/output.bin --group-size 64
```

### 3. Run inference

In chat mode with default parameters:

```bash
cargo run --release -p qwen3-cli -- inference /path/to/output.bin -m chat
```

## CLI Commands and Options

### `export`
Exports a HuggingFace Qwen3 model to a custom binary format for efficient Rust inference.

**Usage:**
```bash
qwen3 export <MODEL_PATH> <OUTPUT_PATH> [--group-size <SIZE>]
```
- `MODEL_PATH`: Path to HuggingFace model directory (must contain config.json, *.safetensors, tokenizer.json)
- `OUTPUT_PATH`: Output path for the binary model file
- `--group-size`, `-g`: Quantization group size (default: 64)

### `inference`
Runs inference on a binary Qwen3 model.

**Usage:**
```bash
qwen3 inference <checkpoint> [options]
```
**Options:**
- `--temperature`, `-t <FLOAT>`: Sampling temperature (default: 1.0)
- `--topp`, `-p <FLOAT>`: Top-p nucleus sampling (default: 0.9)
- `--seed`, `-s <INT>`: Random seed
- `--context`, `-c <INT>`: Context window size (default: max_seq_len)
- `--mode`, `-m <STRING>`: Mode: `generate` or `chat` (default: chat)
- `--input`, `-i <STRING>`: Input prompt
- `--system`, `-y <STRING>`: System prompt (for chat mode)
- `--reasoning`, `-r <INT>`: Reasoning mode: 0=no thinking, 1=thinking (default: 0)

