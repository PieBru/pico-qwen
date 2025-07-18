# pico-qwen 🚀

**Ultra-lightweight Qwen3 inference for low-resource systems**

> Started as an experimental fork of [qwen3-rs](https://github.com/reinterpretcat/qwen3-rs) especially useful for MiniPCs, SBCs,low-power servers and all CPU-only light weight efficient LLM inferencing needs, [pico-qwen](https://github.com/PieBru/pico-qwen) then evolved as a independent project with additional features.

## TL;DR

```bash
# Quick install (Arch Linux)
sudo pacman -S git base-devel pkg-config openssl rust ruff

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

## Our vision and motivations, as of July 2025

Except for some LLM inference niches, such as for a single user running fully-local on a PC, GPU inference will be mainly centralized by cloud providers, or decentralized to the productivity endpoints and mainly served from more flexible fast-RAM systems.
We envision that LLM inference (and related services like research, specialized ranked search, coding IDE, etc.) which are architected to serve thousands to millions active users will be high-tech GPU-bound and served from cloud giga-structures.
On the other side, we feel there will be an ever evolving need of decentralized AI serving, thanks to its intrinsic benefits that cloud AI serving can't provide, such as:
  * privacy guaranteed by design;
  * resilience to internet and cloud services faults;
  * inference server offloading by the edge;
  * edge pre-processing before passing synthetic data to a server;
  * long-term reduced costs.

## Project development methodology: our vision as of July 2025

* **Coding agents are reliable enough to be the default:** We've built and tested coding agents for years and realised recently that reliability crossed an invisible threshold so we now prefer starting most tasks with coding agents.
* **Coding agents are going to get much better:** Coding agents have improved rapidly, and we expect this trend to continue. Imagine that in six months, 50% of the current failure modes of coding agents will get fixed, and six months after that another 50%. What will we (engineers) be spending our time on in that world? What tools would help us do that work most efficiently?
* **The SOTA will improve:** Foundational model labs and application layer startups will continuously leap-frog each other to build state-of-the-art coding agents over the next 12m+. You shouldn't have to change your workflow or behaviour to be able to try out different coding agents.

## 🎯 What is pico-qwen?

**pico-qwen** targets **very low-resource systems** while maintaining educational clarity:

- **Educational foundation**: Learn transformer internals in Rust
- **Low-resource and edge optimized**: Runs on <8GB RAM systems
- **Multiple interfaces**: CLI + REST API + MCP + WebUI
- **CPU-specific builds**: Intel N100, i9-14900HX, Raspberry Pi 4/5, and more
- **Performance, Hybrid cloud/local**: Optional automatic fallback between cloud and local inference
- **MCP agents**: Multi-agent tool orchestration with [PocketFlow](https://github.com/The-Pocket/PocketFlow-Rust)

## 📊 Resource Requirements

| System | CPU | RAM | Storage | Use Case |
|--------|-----|-----|---------|----------|
| **Raspberry Pi 4** | ARM Cortex-A72 | 4GB | 4GB | Home automation |
| **Intel N100** | 4-core x86 | 8GB | 8GB | MiniPC |
| **Intel i9-14900HX** | 24-core x86 | 16GB+ | 16GB+ | Development |
| _More to come_ | | | | |

## 🚀 Quick Start

### 1. Installation
```bash
# Arch Linux (recommended)
sudo pacman -S git base-devel pkg-config openssl rust ruff

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
- **Baseline**: Core infrastructure, API server, WebUI, MCP agents, hybrid cloud, CPU optimization, deployment
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

Licensed under the Apache License 2.0. See the LICENSE file for details.

---

**Ready to start?** → [Installation Guide](README_INSTALLATION.md) | [Usage Examples](README_USAGE.md) | [Development Setup](README_DEVELOPMENT.md)
