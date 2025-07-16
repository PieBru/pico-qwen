# pico-qwen 🚀

**Ultra-lightweight Qwen3 inference for low-resource systems**

> Started as an experimental fork of [qwen3-rs](https://github.com/reinterpretcat/qwen3-rs) designed for MiniPCs, SBCs, and low-power servers, [pico-qwen](https://github.com/PieBru/pico-qwen), then evolved as a independent project with more additional features.

## TL;DR

```bash
# Quick install (Arch Linux)
sudo pacman -S git base-devel pkg-config openssl
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

cd && git clone https://github.com/PieBru/pico-qwen.git && cd pico-qwen
cargo build --release

# Download & export a model
mkdir -p ~/HuggingFace
cd ~/HuggingFace
git clone --depth 1 https://huggingface.co/Qwen/Qwen3-0.6B
cd ~/pico-qwen
cargo run --release -p qwen3-cli -- export ~/HuggingFace/Qwen3-0.6B ~/HuggingFace/Qwen3-0.6B-int8.bin --group-size 64

# Start chatting
cargo run --release -p qwen3-cli -- inference ~/HuggingFace/Qwen3-0.6B-int8.bin --mode chat
```

## 🎯 What is pico-qwen?

**pico-qwen** extends [qwen3-rs](https://github.com/reinterpretcat/qwen3-rs) with production features for **very low-resource systems** while maintaining educational clarity:

- **Educational foundation**: Learn transformer internals in Rust
- **Low-resource optimized**: Runs on <8GB RAM systems  
- **Multiple interfaces**: CLI + REST API + WebUI
- **CPU-specific builds**: Intel N100, i9-14900HX, Raspberry Pi 4/5, more upon request
- **Hybrid cloud/local**: Fallback between cloud and local inference
- **MCP agents**: Multi-agent tool orchestration with [PocketFlow](https://github.com/The-Pocket/PocketFlow-Rust)

## 📊 Resource Requirements

| System | CPU | RAM | Storage | Use Case |
|--------|-----|-----|---------|----------|
| **Raspberry Pi 4** | ARM Cortex-A72 | 4GB | 4GB | Home automation |
| **Intel N100** | 4-core x86 | 8GB | 8GB | MiniPC |
| **Intel i9-14900HX** | 24-core x86 | 16GB+ | 16GB+ | Development |

## 🚀 Quick Start

### 1. Installation
```bash
# Arch Linux (recommended)
sudo pacman -S git base-devel pkg-config openssl
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Build from source
cargo build --release
```

### 2. Model Setup
```bash
# Download lightweight model
git clone --depth 1 https://huggingface.co/Qwen/Qwen3-0.6B ~/HuggingFace/Qwen3-0.6B

# Export with optimization
cargo run --release -p qwen3-cli -- export models/Qwen3-0.6B models/Qwen3-0.6B-int8.bin --group-size 64
```

### 3. Choose Interface

#### CLI Chat
```bash
cargo run --release -p qwen3-cli -- inference models/Qwen3-0.6B-int8.bin --mode chat
```

#### REST API
```bash
# Start server
cargo run --release -p qwen3-api

# Test endpoint
curl -X POST http://localhost:58080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen3-0.6B-int8", "messages": [{"role": "user", "content": "Hello"}]}'
```

#### WebUI
```bash
# Start web interface
cargo run --release -p qwen3-web
# Open http://localhost:53000
```

## 📋 Documentation

### Detailed Guides
- [**README_OVERVIEW.md**](README_OVERVIEW.md) - Complete project overview and architecture
- [**README_INSTALLATION.md**](README_INSTALLATION.md) - Step-by-step installation for all platforms
- [**README_USAGE.md**](README_USAGE.md) - Comprehensive usage examples and API reference
- [**README_DEVELOPMENT.md**](README_DEVELOPMENT.md) - Development setup and contribution guide
- [**README_CREDITS.md**](README_CREDITS.md) - Acknowledgments and project foundations

### Key Features
- **Phases 1-7**: Core infrastructure, API server, WebUI, MCP agents, hybrid cloud, CPU optimization, deployment
- **Quantization**: INT4, INT8, FP16, FP32 with configurable group sizes
- **CPU Optimization**: Runtime detection for Intel, ARM, and generic processors
- **Service Management**: systemd integration and Docker support

## 🔧 Configuration

### Basic Setup
```toml
# ~/.config/pico-qwen/config.toml
[server]
bind_address = "127.0.0.1"
port = 58080

[models]
directory = "~/models"
default_quantization = "int8"
max_loaded_models = 2

[cpu]
target = "auto"  # auto-detect CPU
```

### System Service
```bash
# Install as systemd service
./scripts/install-systemd.sh
sudo systemctl enable pico-qwen
sudo systemctl start pico-qwen
```

## 🤝 Contributing

We welcome contributions focused on:
- **Low-resource optimizations** for new CPU targets
- **Educational improvements** to codebase clarity
- **Deployment enhancements** for various platforms
- **MCP agent capabilities** using PocketFlow

See [README_DEVELOPMENT.md](README_DEVELOPMENT.md) for development setup.

## 📄 License

Maintains the same license as [qwen3-rs](https://github.com/reinterpretcat/qwen3-rs). See individual project repositories for specific license details.

---

**Ready to start?** → [Installation Guide](README_INSTALLATION.md) | [Usage Examples](README_USAGE.md) | [Development Setup](README_DEVELOPMENT.md)