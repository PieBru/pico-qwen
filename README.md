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

### ✅ Phase 2: Low-Requirements API Server - **COMPLETED**
- ✅ REST API with streaming support
- ✅ Model pooling and LRU eviction
- ✅ Memory pressure monitoring
- ✅ HTTP endpoints for chat and generation
- ✅ Real transformer integration
- ✅ Comprehensive test suite

### ✅ Phase 3: Minimalist Chat-WebUI - **COMPLETED**
- ✅ Progressive enhancement web interface
- ✅ Mobile-first responsive design
- ✅ Offline capability with service worker
- ✅ Real-time streaming responses

### 🚧 Phase 4: MCP Multi-Agent System
- [ ] PocketFlow-based agent orchestration
- [ ] Web search and research agents
- [ ] Sequential thinking capabilities
- [ ] Tool cost estimation and management

### ✅ Phase 5: Hybrid Cloud/Edge Inference - **DOCUMENTED**
- ✅ Cloud provider abstraction (OpenAI, Anthropic)
- ✅ Fallback to local inference
- ✅ Health monitoring and failover
- ✅ Cost estimation and routing
- 📋 **Documentation**: [PHASE5_HYBRID_CLOUD.md](docs/PHASE5_HYBRID_CLOUD.md)

### ✅ Phase 6: CPU-Specific Optimization - **DOCUMENTED**
- ✅ Runtime CPU feature detection
- ✅ SIMD optimization (AVX2, AVX-512, NEON)
- ✅ Cache-aware data blocking
- ✅ Target-specific builds
- 📋 **Documentation**: [PHASE6_CPU_OPTIMIZATION.md](docs/PHASE6_CPU_OPTIMIZATION.md)

### ✅ Phase 7: Deployment & Service Management - **DOCUMENTED**
- ✅ systemd service integration
- ✅ Configuration management
- ✅ Logging and monitoring
- ✅ Cross-platform packaging
- 📋 **Documentation**: [PHASE7_DEPLOYMENT.md](docs/PHASE7_DEPLOYMENT.md)

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
├── qwen3-web/                   # NEW: Web interface (Phase 3 - COMPLETED)
├── qwen3-mcp/                   # NEW: MCP agents (Phase 4)
└── docs/                        # Technical documentation
    ├── phase1_summary.md        # Phase 1 completion
    ├── plan_1.md               # Implementation plan
    └── ...
```

## 🚀 Quick Start - Phase 1 Testing (Arch Linux)

### Prerequisites for Arch Linux
```bash
# Install dependencies
sudo pacman -S git base-devel pkg-config openssl

# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

### Installation
```bash
# Clone the repository
git clone https://github.com/PieBru/pico-qwen.git
cd pico-qwen

# Build the workspace with Phase 1 features
cargo build --release

# Verify Phase 1 features
./scripts/test_phase1.sh
```

### Phase 1 Testing Commands (Arch Linux)
```bash
# Test CPU detection and optimization
echo "Testing CPU feature detection..."
cargo test --release --test cpu_detection_tests

# Test quantization levels
echo "Testing quantization system..."
cargo test --release --test quantization_tests

# Test configuration system
echo "Testing extended configuration..."
cargo test --release --test extended_config_tests

# Test memory optimization
echo "Testing memory optimization..."
cargo test --release --test memory_optimization_tests

# Real-world test with Intel i9-14900HX
echo "Testing i9-14900HX optimization..."
cargo test --release --test cpu_target_tests test_intel_i9_14900hx_optimization

# Quick validation script
./scripts/validate_phase1_arch.sh
```

### Download and Test Models (Arch Linux)
```bash
# Create models directory
mkdir -p ~/HuggingFace
cd ~/HuggingFace

# Download lightweight models for testing
# Note: For full 87GB download, ensure 100GB+ free space
git clone --depth 1 https://huggingface.co/Qwen/Qwen3-0.6B      # ~1.2 GB
git clone --depth 1 https://huggingface.co/Qwen/Qwen3-1.7B      # ~3.4 GB

cd ~/pico-qwen

# Export models with different configurations
# Test INT8 quantization (recommended for Intel i9-14900HX)
cargo run --release -p qwen3-cli -- export ~/HuggingFace/Qwen3-0.6B ~/HuggingFace/Qwen3-0.6B-int8.bin --group-size 64

# Test INT4 quantization (for memory-constrained systems)
cargo run --release -p qwen3-cli -- export ~/HuggingFace/Qwen3-0.6B ~/HuggingFace/Qwen3-0.6B-int4.bin --group-size 32

# Test FP16 for high-end systems
cargo run --release -p qwen3-cli -- export ~/HuggingFace/Qwen3-0.6B ~/HuggingFace/Qwen3-0.6B-fp16.bin --group-size 128
```

### Phase 1 Feature Testing
```bash
# Test CPU detection on your system
echo "=== CPU Detection ==="
lscpu | head -n 10

# Test specific CPU optimizations
echo "=== Testing Intel i9-14900HX optimization ==="
cargo run --release -p qwen3-cli -- inference ~/HuggingFace/Qwen3-0.6B-int8.bin \
  --cpu-target intel-i9-14900hx \
  --max-memory 16384 \
  --ctx-length 4096 \
  --mode chat \
  --reasoning 1

# Test memory-constrained mode
echo "=== Testing low-memory mode ==="
cargo run --release -p qwen3-cli -- inference ~/HuggingFace/Qwen3-0.6B-int4.bin \
  --max-memory 4096 \
  --ctx-length 2048 \
  --mode generate \
  --input "What is the capital of France?"

# Test cache optimization
echo "=== Testing cache-aware optimization ==="
cargo run --release -p qwen3-cli -- inference ~/HuggingFace/Qwen3-0.6B-int8.bin \
  --cache-strategy l3 \
  --parallel-strategy rayon \
  --mode chat
```

### Configuration Examples for Phase 1
```toml
# ~/.config/pico-qwen/config.toml
[server]
bind_address = "127.0.0.1"
port = 8080

[models]
directory = "~/HuggingFace"
default_quantization = "int8"
max_loaded_models = 2
context_window = 4096

[limits]
max_request_size = 10485760  # 10MB
max_tokens = 512
rate_limit = 60

# CPU-specific configuration
[cpu]
target = "intel-i9-14900hx"
quantization = "int8-gs64"
cache_strategy = "l3"
parallel_strategy = "rayon"

# Memory optimization
[memory]
max_memory_mb = 16384
max_context_length = 4096
```

### systemd Service (Arch Linux)
```bash
# Create systemd service for API server
cat > ~/.config/systemd/user/pico-qwen.service << 'EOF'
[Unit]
Description=Pico-Qwen API Server
After=network.target

[Service]
Type=simple
ExecStart=/home/%i/.cargo/bin/cargo run --release -p qwen3-api -- --config /home/%i/.config/pico-qwen/config.toml
WorkingDirectory=/home/%i/pico-qwen
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
EOF

# Enable and start service
systemctl --user daemon-reload
systemctl --user enable pico-qwen.service
systemctl --user start pico-qwen.service
```

## 📋 Development Status

| Phase | Status | Progress |
|-------|--------|----------|
| Phase 1: Core Infrastructure | ✅ **COMPLETED** | 100% |
| Phase 2: API Server | ✅ **COMPLETED** | 100% |
| Phase 3: WebUI | ✅ **COMPLETED** | 100% |
| Phase 4: MCP Agents | 📋 **PENDING** | 0% |
| Phase 5: Hybrid Cloud/Edge | ✅ **DOCUMENTED** | 100% |
| Phase 6: CPU Optimization | ✅ **DOCUMENTED** | 100% |
| Phase 7: Deployment | ✅ **DOCUMENTED** | 100% |

## 🚀 Phase 3 Testing Commands (Arch Linux)

### Prerequisites
```bash
# Ensure API server is running (Phase 2)
mkdir -p ~/HuggingFace
cd ~/pico-qwen
```

### Quick WebUI Setup
```bash
# 1. Build WebUI
cargo build --release -p qwen3-web

# 2. Run tests
cargo test --release -p qwen3-web

# 3. Start WebUI (connects to API server on port 8080)
cargo run --release -p qwen3-web -- --bind-address 127.0.0.1 --port 3000 --api-url http://localhost:8080

# 4. Open in browser
firefox http://localhost:3000
# OR
chromium http://localhost:3000
```

### WebUI Features Testing
```bash
# Test mobile responsive design
# Open Chrome DevTools (F12) > Toggle device toolbar > Test various screen sizes

# Test offline capability
# 1. Open DevTools > Network > Check "Offline"
# 2. Refresh page - should still work
# 3. Check Application > Service Workers > Should show active worker

# Test PWA installation
# 1. Open DevTools > Application > Manifest
# 2. Check if "Add to home screen" prompt appears on mobile

# Test keyboard shortcuts
# Ctrl/Cmd + / : Focus input
# Ctrl/Cmd + , : Toggle settings
# Ctrl/Cmd + ? : Show help
```

### Automated Testing
```bash
# Run all Phase 3 tests
./scripts/test_phase3_arch.sh

# Test responsive design
./scripts/test_responsive.sh

# Test offline functionality
./scripts/test_offline.sh
```

## 🚀 Phase 2 Testing Commands (Arch Linux)

### API Endpoints Testing
```bash
# Health check
curl -X GET http://localhost:8080/api/v1/health

# List models (returns empty initially)
curl -X GET http://localhost:8080/api/v1/models

# Load a model (requires exported model)
curl -X POST http://localhost:8080/api/v1/models/Qwen3-0.6B-int8/load

# Chat completion
curl -X POST http://localhost:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-0.6B-int8",
    "messages": [{"role": "user", "content": "Hello, how are you?"}],
    "max_tokens": 100,
    "temperature": 0.7
  }'

# Text generation
curl -X POST http://localhost:8080/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-0.6B-int8",
    "prompt": "The future of AI is",
    "max_tokens": 50,
    "temperature": 0.8
  }'
```

### systemd Service Setup
```bash
# Create systemd service
cat > ~/.config/systemd/user/pico-qwen-api.service << 'EOF'
[Unit]
Description=Pico-Qwen API Server
After=network.target

[Service]
Type=simple
ExecStart=/home/%i/.cargo/bin/cargo run --release -p qwen3-api -- --config /home/%i/.config/pico-qwen/api.toml
WorkingDirectory=/home/%i/pico-qwen
Restart=always
RestartSec=10
Environment=RUST_LOG=info

[Install]
WantedBy=default.target
EOF

# Enable service
systemctl --user daemon-reload
systemctl --user enable pico-qwen-api.service
systemctl --user start pico-qwen-api.service
```

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

# Real-world configuration validation (interactive chat mode)
echo "Starting Qwen3 interactive chat..."
echo "Type your message and press Enter. Use '/bye' to exit."
cargo run --release -p qwen3-cli -- inference HuggingFace/Qwen3-0.6B.bin --reasoning 1

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
echo "Starting interactive chat mode. Type 'hello' to test, then '/bye' to exit:"
timeout 10s cargo run --release -p qwen3-cli -- inference HuggingFace/Qwen3-0.6B.bin --reasoning 1 || echo "Chat mode test completed (timeout reached)"

echo -e "\n4. CPU Info:"
lscpu | grep -E "(Model name|CPU|Thread|Core)" | head -5

echo -e "\n=== Phase 1 Testing Complete ==="
```

## 🧪 Comprehensive Testing Guide - All Phases

### 📋 Testing Overview
This section provides complete testing procedures for all phases, including Phases 5-7 which have been documented for implementation.

### Phase 5-7 Testing Prerequisites
```bash
# Install system dependencies for deployment testing
sudo pacman -S git base-devel pkg-config openssl systemd docker

# Install testing tools
sudo pacman -S curl wget jq htop

# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

### 🔧 Phase 5: Hybrid Cloud/Edge Testing
```bash
echo "=== Phase 5: Hybrid Cloud/Edge Testing ==="

# 1. Test cloud provider configuration
echo "Testing cloud provider setup..."
export OPENAI_API_KEY="your-openai-key-here"
export ANTHROPIC_API_KEY="your-anthropic-key-here"

# 2. Test cloud health checks
cargo run --release -p qwen3-cli -- cloud-health --provider openai

# 3. Test cost estimation
cargo run --release -p qwen3-cli -- estimate-cost --prompt "Hello world" --tokens 100

# 4. Test hybrid inference
cargo run --release -p qwen3-cli -- inference model.bin --use-cloud openai --fallback-local

# 5. Test failover mechanism
echo "Testing cloud failover..."
cargo test -p qwen3-inference cloud::tests::test_failover_behavior
```

### ⚡ Phase 6: CPU Optimization Testing
```bash
echo "=== Phase 6: CPU-Specific Optimization Testing ==="

# 1. CPU feature detection
cargo run --release -p qwen3-cli -- cpu-info

# 2. Test SIMD optimizations
cargo run --release -p qwen3-cli -- benchmark --mode cpu-features --iterations 100

# 3. Test architecture-specific builds
# Intel with AVX2
RUSTFLAGS="-C target-feature=+avx2" cargo build --release -p qwen3-cli
# ARM with NEON
RUSTFLAGS="-C target-feature=+neon" cargo build --release -p qwen3-cli

# 4. Performance validation
cargo bench -p qwen3-inference cpu_benchmarks

# 5. Memory optimization tests
cargo test -p qwen3-inference cpu::tests::test_memory_optimization
```

### 🚀 Phase 7: Deployment Testing
```bash
echo "=== Phase 7: Deployment & Service Management Testing ==="

# 1. Test systemd service installation
sudo ./scripts/install-systemd-service.sh
echo "Testing systemd service..."
sudo systemctl status pico-qwen

# 2. Test Docker deployment
docker build -t pico-qwen:latest .
docker run -d --name pico-qwen-test -p 8080:8080 pico-qwen:latest
docker ps | grep pico-qwen-test

# 3. Test health monitoring
curl -f http://localhost:8080/health || echo "Health check failed"

# 4. Test service restart on failure
sudo pkill -f pico-qwen
sleep 5
sudo systemctl status pico-qwen | grep -i active

# 5. Test log rotation and monitoring
sudo journalctl -u pico-qwen --since "1 minute ago" | tail -10
```

### Development workflow
cargo check --all
cargo test --all
cargo build --release -p qwen3-cli

# Run with configuration
cargo run --release -p qwen3-cli -- inference HuggingFace/Qwen3-0.6B.bin --reasoning 1

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

