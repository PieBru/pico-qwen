# Usage Guide

## Command Line Interface (CLI)

### Basic Inference
```bash
# Interactive chat mode - use full path to .bin file
./target/release/qwen3 inference /home/user/HuggingFace/Qwen3-0.6B-int8.bin -m chat

# Generate mode with prompt
cargo run --release -p qwen3-cli -- inference ~/HuggingFace/Qwen3-0.6B-int8.bin -m generate -i "Your prompt here"

# With custom parameters
cargo run --release -p qwen3-cli -- inference ./models/Qwen3-0.6B-int8.bin -t 0.7 -p 0.8 -s 42
```

### CLI Options
```bash
# Export models - specify full directory path and output filename
# Directory must contain: config.json, tokenizer.json, *.safetensors files
cargo run --release -p qwen3-cli -- export ~/Downloads/Qwen3-0.6B ~/HuggingFace/Qwen3-0.6B-int8 --group-size 64

# List available models - scans directory for .bin files
cargo run --release -p qwen3-cli -- models
# With options:
cargo run --release -p qwen3-cli -- models --directory ~/HuggingFace --format json
# Formats: table|json|list

# Available options:
# --group-size: Quantization group size (32, 64, 128, 256)
# --quantization: Quantization level (int4, int8, fp16, fp32)
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

# List available models without starting server
# This scans the models directory specified in config.toml
cargo run --release -p qwen3-api -- --list-models
```

### API Endpoints

#### Health Check
```bash
curl -X GET http://localhost:8080/api/v1/health
```

#### Server Status
```bash
# Get comprehensive server status including loaded models, memory usage, and statistics
curl -X GET http://localhost:8080/api/v1/status
```

**Example Response:**
```json
{
  "uptime": {
    "secs": 3600,
    "nanos": 0
  },
  "active_requests": 0,
  "loaded_models": [
    {
      "id": "Qwen3-0.6B-int8",
      "size_mb": 1245.7,
      "loaded_at": {
        "secs": 1800,
        "nanos": 0
      },
      "last_used": {
        "secs": 300,
        "nanos": 0
      },
      "inference_stats": {
        "total_requests": 42,
        "total_tokens_generated": 1847,
        "avg_tokens_per_sec": 0.0,
        "last_inference_at": {
          "secs": 300,
          "nanos": 0
        }
      },
      "memory_usage_mb": 1619.4
    }
  ],
  "total_memory_mb": 1619.4,
  "system_info": {
    "cpu_cores": 8,
    "total_memory_gb": 16.0,
    "available_memory_gb": 10.2,
    "used_memory_gb": 5.8,
    "memory_usage_percent": 36.25
  }
}
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
# First, load the model (required before first use)
curl -X POST http://localhost:8080/api/v1/models/Qwen3-0.6B-int8/load

# Then use it for chat
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
# Load model first (if not already loaded)
curl -X POST http://localhost:8080/api/v1/models/Qwen3-0.6B-int8/load

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
# Load model first
curl -X POST http://localhost:8080/api/v1/models/Qwen3-0.6B-int8/load

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

## Important Notes About File Paths

### Model Files (.bin files)
- **Location**: Store your exported models in `~/HuggingFace/` directory
- **Format**: Files must end with `.bin` extension (e.g., `Qwen3-0.6B-int8.bin`)
- **Naming**: Use descriptive names like `Qwen3-0.6B-int8.bin`, `DeepSeek-R1-0528-Qwen3-8B-int8.bin`

### Export Workflow
1. **Download HuggingFace model** to a directory (e.g., `~/Downloads/Qwen3-0.6B/`)
2. **Verify files**: Directory must contain `config.json`, `tokenizer.json`, and `.safetensors` files
3. **Export**: Run export command to create `.bin` file
4. **Use**: Use the generated `.bin` file path in CLI or API

### API Workflow
1. **Start API server**: `cargo run --release -p qwen3-api`
2. **List models**: `curl http://localhost:8080/api/v1/models`
3. **Load model**: `curl -X POST http://localhost:8080/api/v1/models/YOUR_MODEL_NAME/load`
4. **Use model**: Send requests to `/v1/chat` or `/v1/generate`

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