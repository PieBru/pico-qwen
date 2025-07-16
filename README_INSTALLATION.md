# Installation Guide

## Prerequisites

### Arch Linux (Primary)
```bash
# Install system dependencies
sudo pacman -S git base-devel pkg-config openssl

# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

### Other Linux Distributions
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install git build-essential pkg-config libssl-dev

# CentOS/RHEL/Fedora
sudo dnf install git gcc pkg-config openssl-devel

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

## Quick Installation

### 1. Clone Repository
```bash
git clone https://github.com/PieBru/pico-qwen.git
cd pico-qwen
```

### 2. Build Workspace
```bash
# Build all crates
cargo build --release

# Build specific components
cargo build --release -p qwen3-cli      # CLI only
cargo build --release -p qwen3-api     # API server only
cargo build --release -p qwen3-web     # WebUI only
```

### 3. Verify Installation
```bash
# Test CLI
cargo run --release -p qwen3-cli -- --help

# Run tests
cargo test --release --all
```

## Model Setup

### Download Models
```bash
# Create models directory
mkdir -p ~/HuggingFace
cd ~/HuggingFace

# Download lightweight models
git clone --depth 1 https://huggingface.co/Qwen/Qwen3-0.6B      # ~1.2 GB
git clone --depth 1 https://huggingface.co/Qwen/Qwen3-1.7B      # ~3.4 GB

# For larger models (ensure adequate disk space)
# git clone https://huggingface.co/Qwen/Qwen3-4B      # ~8 GB
# git clone https://huggingface.co/Qwen/Qwen3-8B      # ~16 GB
# git clone https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B  # ~16 GB (recommended - more performant than Qwen3-8B)
```

### Export Models
```bash
cd ~/pico-qwen

# Export with different quantization levels
# INT8 (recommended for most systems)
cargo run --release -p qwen3-cli -- export ~/HuggingFace/Qwen3-0.6B ~/HuggingFace/Qwen3-0.6B-int8.bin --group-size 64

# INT4 (for memory-constrained systems)
cargo run --release -p qwen3-cli -- export ~/HuggingFace/Qwen3-0.6B ~/HuggingFace/Qwen3-0.6B-int4.bin --group-size 32

# FP16 (for high-end systems)
cargo run --release -p qwen3-cli -- export ~/HuggingFace/Qwen3-0.6B ~/HuggingFace/Qwen3-0.6B-fp16.bin --group-size 128
```

## Configuration

### System Configuration Directory
```bash
# Create config directory
mkdir -p ~/.config/pico-qwen
```

### Create Configuration File
```bash
cat > ~/.config/pico-qwen/config.toml << 'EOF'
[server]
bind_address = "127.0.0.1"
port = 58080

[models]
directory = "~/HuggingFace"
default_quantization = "int8"
max_loaded_models = 2
context_window = 4096

[limits]
max_request_size = 10485760  # 10MB
max_tokens = 512
rate_limit = 60

[cpu]
target = "auto"  # auto-detect CPU
quantization = "int8-gs64"
cache_strategy = "auto"
parallel_strategy = "rayon"

[memory]
max_memory_mb = 8192
max_context_length = 4096
EOF
```

## Development Installation

### For Contributors
```bash
# Install additional development tools
sudo pacman -S clippy rustfmt

# Check code quality
cargo fmt --check
cargo clippy --all-targets --all-features

# Run comprehensive tests
cargo test --release --all

# Cross-compilation setup
rustup target add aarch64-unknown-linux-gnu
rustup target add x86_64-unknown-linux-musl
```

## Service Installation

### systemd Service (User)
```bash
# Create user service
cat > ~/.config/systemd/user/pico-qwen.service << 'EOF'
[Unit]
Description=Pico-Qwen API Server
After=network.target

[Service]
Type=simple
ExecStart=%h/.cargo/bin/cargo run --release -p qwen3-api -- --config %h/.config/pico-qwen/config.toml
WorkingDirectory=%h/pico-qwen
Restart=always
RestartSec=10
Environment=RUST_LOG=info

[Install]
WantedBy=default.target
EOF

# Enable and start
systemctl --user daemon-reload
systemctl --user enable pico-qwen.service
systemctl --user start pico-qwen.service
```

### systemd Service (System)
```bash
# Install as system service (requires root)
sudo ./scripts/install-systemd-service.sh
sudo systemctl enable pico-qwen
sudo systemctl start pico-qwen
```

## Docker Installation

### Build Docker Image
```bash
# Build from source
docker build -t pico-qwen:latest .

# Run with volume mounts
docker run -d \
  --name pico-qwen \
  -p 58080:58080 \
  -v ~/HuggingFace:/models \
  -v ~/.config/pico-qwen:/config \
  pico-qwen:latest
```

### Docker Compose
```bash
# Create docker-compose.yml
cat > docker-compose.yml << 'EOF'
version: '3.8'
services:
  pico-qwen:
    build: .
    ports:
      - "58080:58080"
    volumes:
      - ~/HuggingFace:/models
      - ~/.config/pico-qwen:/config
    environment:
      - RUST_LOG=info
    restart: unless-stopped
EOF

docker-compose up -d
```

## Verification

### Basic Tests
```bash
# Test CPU detection
cargo run --release -p qwen3-cli -- cpu-info

# Test model loading
cargo run --release -p qwen3-cli -- inference ~/HuggingFace/Qwen3-0.6B-int8.bin --mode chat --input "Hello"

# Test API server
curl -X GET http://localhost:58080/api/v1/health
```

### Resource Usage Check
```bash
# Monitor memory usage
htop -p $(pgrep pico-qwen)

# Check disk space
du -h ~/HuggingFace/
```

## Troubleshooting

### Common Issues

#### "No such file or directory" errors
```bash
# Ensure all dependencies are installed
sudo pacman -S base-devel pkg-config openssl
```

#### Memory issues during build
```bash
# Reduce parallel jobs
export CARGO_BUILD_JOBS=2
cargo build --release
```

#### Model download failures
```bash
# Use git-lfs for large models
sudo pacman -S git-lfs
git lfs install
git clone https://huggingface.co/Qwen/Qwen3-4B
```

#### Permission issues
```bash
# Fix ownership issues
sudo chown -R $USER:$USER ~/.config/pico-qwen
sudo chown -R $USER:$USER ~/HuggingFace
```