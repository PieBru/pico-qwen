# Usage Guide

## Command Line Interface (CLI)

### Basic Inference
```bash
# Interactive chat mode
cargo run --release -p qwen3-cli -- inference model.bin -m chat

# Generate mode with prompt
cargo run --release -p qwen3-cli -- inference model.bin -m generate -i "Your prompt here"

# With custom parameters
cargo run --release -p qwen3-cli -- inference model.bin -t 0.7 -p 0.8 -s 42
```

### CLI Options
```bash
# Export models
cargo run --release -p qwen3-cli -- export /path/to/huggingface/model /path/to/output.bin --group-size 64

# Available options:
# --group-size: Quantization group size (32, 64, 128, 256)
# --quantization: Quantization level (int4, int8, fp16, fp32)

# Inference options:
# --temperature, -t: Sampling temperature (0.1-2.0)
# --topp, -p: Top-p nucleus sampling (0.1-1.0)
# --seed, -s: Random seed for reproducibility
# --context, -c: Context window size
# --reasoning, -r: Enable reasoning mode (0/1)
```

## API Server Usage

### Starting the API Server
```bash
# Basic start
cargo run --release -p qwen3-api

# With custom config
cargo run --release -p qwen3-api -- --config ~/.config/pico-qwen/api.toml

# With specific port
cargo run --release -p qwen3-api -- --port 3000
```

### API Endpoints

#### Health Check
```bash
curl -X GET http://localhost:8080/api/v1/health
```

#### Model Management
```bash
# List available models
curl -X GET http://localhost:8080/api/v1/models

# Load a model
curl -X POST http://localhost:8080/api/v1/models/Qwen3-0.6B-int8/load

# Unload a model
curl -X POST http://localhost:8080/api/v1/models/Qwen3-0.6B-int8/unload

# Get model info
curl -X GET http://localhost:8080/api/v1/models/Qwen3-0.6B-int8
```

#### Chat Completion
```bash
curl -X POST http://localhost:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-0.6B-int8",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 100,
    "temperature": 0.7,
    "stream": false
  }'
```

#### Text Generation
```bash
curl -X POST http://localhost:8080/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-0.6B-int8",
    "prompt": "The future of AI is",
    "max_tokens": 50,
    "temperature": 0.8
  }'
```

#### Streaming Responses
```bash
curl -X POST http://localhost:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-0.6B-int8",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "max_tokens": 200,
    "temperature": 0.7,
    "stream": true
  }'
```

## WebUI Usage

### Starting the WebUI
```bash
# Basic start (connects to API on localhost:8080)
cargo run --release -p qwen3-web

# With custom API endpoint
cargo run --release -p qwen3-web -- --api-url http://localhost:3000

# Custom port
cargo run --release -p qwen3-web -- --port 3000
```

### WebUI Features

#### Access
- Open browser to: http://localhost:3000
- Mobile-responsive design works on all devices
- PWA support for mobile installation

#### Usage
1. **Model Selection**: Choose from loaded models dropdown
2. **Chat Interface**: Type messages and get responses
3. **Settings**: Adjust temperature, max tokens, system prompt
4. **Keyboard Shortcuts**:
   - `Ctrl/Cmd + /`: Focus input
   - `Ctrl/Cmd + ,`: Toggle settings
   - `Ctrl/Cmd + ?`: Show help

#### Offline Mode
- Works offline after first load
- Service worker caches assets
- Install as PWA on mobile devices

## Configuration Examples

### Basic Configuration
```toml
# ~/.config/pico-qwen/basic.toml
[server]
bind_address = "127.0.0.1"
port = 8080

[models]
directory = "~/HuggingFace"
default_quantization = "int8"
max_loaded_models = 2
```

### Low-Memory Configuration
```toml
# ~/.config/pico-qwen/low-memory.toml
[server]
bind_address = "127.0.0.1"
port = 8080

[models]
directory = "~/HuggingFace"
default_quantization = "int4"
max_loaded_models = 1
context_window = 2048

[memory]
max_memory_mb = 4096
max_context_length = 2048

[cpu]
target = "raspberry-pi-4"
quantization = "int4-gs32"
```

### High-Performance Configuration
```toml
# ~/.config/pico-qwen/high-performance.toml
[server]
bind_address = "0.0.0.0"
port = 8080

[models]
directory = "~/HuggingFace"
default_quantization = "fp16"
max_loaded_models = 4
context_window = 8192

[cpu]
target = "intel-i9-14900hx"
quantization = "fp16-gs128"
parallel_strategy = "rayon"
```

## Resource Optimization

### Memory Usage Guidelines
| Model Size | Quantization | RAM Required | Recommended System |
|------------|--------------|--------------|-------------------|
| 0.6B | INT4 | 2GB | Raspberry Pi 4 |
| 0.6B | INT8 | 4GB | Intel N100 |
| 1.7B | INT4 | 4GB | MiniPC |
| 1.7B | INT8 | 8GB | Standard PC |
| 4B | INT8 | 16GB | High-end PC |

### CPU Optimization
```bash
# Check your CPU features
cargo run --release -p qwen3-cli -- cpu-info

# Manual CPU target selection
cargo run --release -p qwen3-cli -- inference model.bin --cpu-target intel-i9-14900hx
cargo run --release -p qwen3-cli -- inference model.bin --cpu-target raspberry-pi-4
cargo run --release -p qwen3-cli -- inference model.bin --cpu-target generic
```

## Advanced Usage

### Batch Processing
```bash
# Process multiple prompts
for prompt in "Hello" "How are you" "Tell me a joke"; do
  echo "Prompt: $prompt"
  cargo run --release -p qwen3-cli -- inference model.bin -m generate -i "$prompt" --max-tokens 50
done
```

### API Rate Limiting
```bash
# Test rate limits
curl -X POST http://localhost:8080/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen3-0.6B-int8", "prompt": "test", "max_tokens": 10}' \
  -w "%{http_code}\n"
```

### Monitoring
```bash
# Check service status
systemctl --user status pico-qwen

# View logs
journalctl --user -u pico-qwen -f

# Monitor resource usage
htop -p $(pgrep pico-qwen)
```

## Troubleshooting Common Issues

### Model Loading Issues
```bash
# Check model file integrity
ls -la ~/HuggingFace/*.bin

# Verify model format
cargo run --release -p qwen3-cli -- info ~/HuggingFace/model.bin
```

### Memory Issues
```bash
# Check available memory
free -h

# Reduce context size
cargo run --release -p qwen3-cli -- inference model.bin --context 1024
```

### Port Conflicts
```bash
# Check port usage
sudo lsof -i :8080

# Use different port
cargo run --release -p qwen3-api -- --port 3000
```