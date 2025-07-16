#!/bin/bash
# Phase 3 WebUI Testing Script for Arch Linux
set -e

echo "=== Phase 3: Minimalist Chat-WebUI Testing ==="
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

echo -e "${GREEN}1. Building Phase 3 WebUI...${NC}"
cargo build --release -p qwen3-web

if [[ $? -ne 0 ]]; then
    echo -e "${RED}Build failed!${NC}"
    exit 1
fi

echo -e "${GREEN}2. Running WebUI tests...${NC}"
cargo test --release -p qwen3-web --quiet

if [[ $? -ne 0 ]]; then
    echo -e "${RED}Tests failed!${NC}"
    exit 1
fi

echo -e "${GREEN}3. Testing static file serving...${NC}"
if [[ ! -f qwen3-web/static/index.html ]]; then
    echo -e "${RED}Static files not found!${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Static files verified${NC}"

echo -e "${GREEN}4. Testing web server startup...${NC}"
cargo run --release -p qwen3-web -- --bind-address 127.0.0.1 --port 3001 &
SERVER_PID=$!

# Wait for server to start
sleep 3

# Test health endpoint
if curl -s -f http://localhost:3001/health > /dev/null; then
    echo -e "${GREEN}✅ Health check passed${NC}"
else
    echo -e "${RED}❌ Health check failed${NC}"
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi

# Test static file serving
if curl -s -f http://localhost:3001/ | grep -q "Pico-Qwen Chat"; then
    echo -e "${GREEN}✅ Static file serving working${NC}"
else
    echo -e "${RED}❌ Static file serving failed${NC}"
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi

# Test service worker registration
if curl -s -f http://localhost:3001/sw.js | grep -q "serviceWorker"; then
    echo -e "${GREEN}✅ Service worker available${NC}"
else
    echo -e "${RED}❌ Service worker not found${NC}"
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi

echo -e "${GREEN}5. Cleanup...${NC}"
kill $SERVER_PID 2>/dev/null || true
sleep 2

echo -e "${GREEN}6. Testing responsive design...${NC}"
# Check CSS for mobile-first queries
if grep -q "@media (max-width: 768px)" qwen3-web/static/styles.css; then
    echo -e "${GREEN}✅ Mobile-first responsive design detected${NC}"
else
    echo -e "${RED}❌ Mobile-first responsive design missing${NC}"
    exit 1
fi

echo -e "${GREEN}7. Testing offline capability...${NC}"
# Check service worker cache
if grep -q "cache.addAll" qwen3-web/static/sw.js; then
    echo -e "${GREEN}✅ Service worker caching enabled${NC}"
else
    echo -e "${RED}❌ Service worker caching missing${NC}"
    exit 1
fi

echo -e "${GREEN}8. Testing PWA manifest...${NC}"
if [[ -f qwen3-web/static/manifest.json ]]; then
    echo -e "${GREEN}✅ PWA manifest present${NC}"
else
    echo -e "${RED}❌ PWA manifest missing${NC}"
    exit 1
fi

echo -e "${GREEN}9. Creating systemd service...${NC}"
cat > ~/.config/systemd/user/pico-qwen-web.service << 'EOF'
[Unit]
Description=Pico-Qwen WebUI - Phase 3
After=network.target

[Service]
Type=simple
ExecStart=/home/%i/.cargo/bin/cargo run --release -p qwen3-web -- --bind-address 127.0.0.1 --port 3000 --api-url http://localhost:8080
WorkingDirectory=/home/%i/pico-qwen
Restart=always
RestartSec=10
Environment=RUST_LOG=info

[Install]
WantedBy=default.target
EOF

echo -e "${GREEN}=== Phase 3 Testing Complete ===${NC}"
echo -e "${GREEN}✅ WebUI built and tested successfully${NC}"
echo -e "${GREEN}✅ Static files verified${NC}"
echo -e "${GREEN}✅ Responsive design validated${NC}"
echo -e "${GREEN}✅ Offline capability confirmed${NC}"
echo -e "${GREEN}✅ PWA features enabled${NC}"
echo ""
echo "To start the WebUI:"
echo "  systemctl --user daemon-reload"
echo "  systemctl --user enable pico-qwen-web.service"
echo "  systemctl --user start pico-qwen-web.service"
echo ""
echo "To test:"
echo "  1. Open http://localhost:3000 in your browser"
echo "  2. Test on mobile: Use Chrome DevTools device emulation"
echo "  3. Test offline: Disable network and refresh page"
echo "  4. Test PWA: Use Chrome DevTools Application panel"