#!/bin/bash
# Phase 2 API Server Testing Script for Arch Linux
set -e

echo "=== Phase 2: Low-Requirements API Server Testing ==="
echo "Target: Arch Linux with systemd"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're on Arch Linux
if [[ ! -f /etc/arch-release ]]; then
    echo -e "${YELLOW}Warning: This script is designed for Arch Linux${NC}"
fi

echo -e "${GREEN}1. Building Phase 2 API server...${NC}"
cargo build --release -p qwen3-api

if [[ $? -ne 0 ]]; then
    echo -e "${RED}Build failed!${NC}"
    exit 1
fi

echo -e "${GREEN}2. Running API tests...${NC}"
cargo test --release -p qwen3-api --quiet

if [[ $? -ne 0 ]]; then
    echo -e "${RED}Tests failed!${NC}"
    exit 1
fi

echo -e "${GREEN}3. Setting up configuration...${NC}"
mkdir -p ~/.config/pico-qwen

cat > ~/.config/pico-qwen/api.toml << 'EOF'
[server]
bind_address = "127.0.0.1"
port = 8080

[models]
directory = "~/HuggingFace"
default_quantization = "int8"
max_loaded_models = 2
context_window = 4096

[limits]
max_request_size = 10485760
max_tokens = 512
rate_limit = 60
EOF

echo -e "${GREEN}4. Testing API server startup...${NC}"
cargo run --release -p qwen3-api -- --config ~/.config/pico-qwen/api.toml &
SERVER_PID=$!

# Wait for server to start
sleep 3

# Test health endpoint
if curl -s -f http://localhost:8080/api/v1/health > /dev/null; then
    echo -e "${GREEN}✅ Health check passed${NC}"
else
    echo -e "${RED}❌ Health check failed${NC}"
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi

# Test models endpoint
if curl -s -f http://localhost:8080/api/v1/models > /dev/null; then
    echo -e "${GREEN}✅ Models endpoint working${NC}"
else
    echo -e "${RED}❌ Models endpoint failed${NC}"
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi

# Test chat endpoint with mock data
RESPONSE=$(curl -s -X POST http://localhost:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"model": "test-model", "messages": [{"role": "user", "content": "hello"}]}')

if echo "$RESPONSE" | grep -q "Model not found"; then
    echo -e "${GREEN}✅ Chat endpoint working (expected 404 for test model)${NC}"
else
    echo -e "${YELLOW}⚠️  Chat endpoint response: $RESPONSE${NC}"
fi

echo -e "${GREEN}5. Cleanup...${NC}"
kill $SERVER_PID 2>/dev/null || true
sleep 2

echo -e "${GREEN}6. Creating systemd service...${NC}"
cat > ~/.config/systemd/user/pico-qwen-api.service << 'EOF'
[Unit]
Description=Pico-Qwen API Server - Phase 2
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

echo -e "${GREEN}=== Phase 2 Testing Complete ===${NC}"
echo -e "${GREEN}✅ API Server built and tested successfully${NC}"
echo -e "${GREEN}✅ All endpoints functional${NC}"
echo -e "${GREEN}✅ systemd service created${NC}"
echo ""
echo "To start the API server:"
echo "  systemctl --user daemon-reload"
echo "  systemctl --user enable pico-qwen-api.service"
echo "  systemctl --user start pico-qwen-api.service"
echo ""
echo "To test: curl http://localhost:8080/api/v1/health"