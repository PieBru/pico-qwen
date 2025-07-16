#!/bin/bash

# Pico-Qwen systemd service installer
# This script installs pico-qwen as a system service on Linux systems

set -e

# Configuration
SERVICE_NAME="pico-qwen"
INSTALL_DIR="/opt/pico-qwen"
USER_NAME="pico-qwen"
GROUP_NAME="pico-qwen"
BINARY_NAME="qwen3-cli"
CONFIG_FILE="/etc/pico-qwen/config.toml"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Pico-Qwen systemd Service Installer ===${NC}"

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo -e "${RED}This script must be run as root${NC}"
   exit 1
fi

# Check if systemd is available
if ! command -v systemctl &> /dev/null; then
    echo -e "${RED}systemctl not found. This script requires systemd.${NC}"
    exit 1
fi

# Create user and group
echo -e "${YELLOW}Creating user and group...${NC}"
if ! id "$USER_NAME" &>/dev/null; then
    useradd --system --home-dir "$INSTALL_DIR" --shell /bin/false --user-group "$USER_NAME"
    echo -e "${GREEN}Created user: $USER_NAME${NC}"
else
    echo -e "${YELLOW}User $USER_NAME already exists${NC}"
fi

# Create directories
echo -e "${YELLOW}Creating directories...${NC}"
mkdir -p "$INSTALL_DIR/bin"
mkdir -p "$INSTALL_DIR/config"
mkdir -p "$INSTALL_DIR/models"
mkdir -p "$INSTALL_DIR/logs"

# Set permissions
echo -e "${YELLOW}Setting permissions...${NC}"
chown -R "$USER_NAME:$GROUP_NAME" "$INSTALL_DIR"
chmod 755 "$INSTALL_DIR"

# Copy binary
echo -e "${YELLOW}Installing binary...${NC}"
if [[ -f "target/release/$BINARY_NAME" ]]; then
    cp "target/release/$BINARY_NAME" "$INSTALL_DIR/bin/"
    chmod +x "$INSTALL_DIR/bin/$BINARY_NAME"
    echo -e "${GREEN}Binary installed to $INSTALL_DIR/bin/${NC}"
else
    echo -e "${RED}Binary not found at target/release/$BINARY_NAME. Please build first.${NC}"
    exit 1
fi

# Create configuration directory and file
echo -e "${YELLOW}Setting up configuration...${NC}"
mkdir -p "$(dirname "$CONFIG_FILE")"

# Create default configuration if it doesn't exist
if [[ ! -f "$CONFIG_FILE" ]]; then
    cat > "$CONFIG_FILE" << 'EOF'
# Pico-Qwen Configuration
[server]
bind_address = "127.0.0.1"
port = 8080

[models]
directory = "/opt/pico-qwen/models"
default_model = "Qwen3-0.6B-int8.bin"
max_loaded_models = 2
context_window = 4096

[limits]
max_memory_mb = 8192
max_tokens = 512
rate_limit = 60

[logging]
level = "info"
file = "/opt/pico-qwen/logs/pico-qwen.log"
rotate = true
EOF
    chown "$USER_NAME:$GROUP_NAME" "$CONFIG_FILE"
    chmod 644 "$CONFIG_FILE"
    echo -e "${GREEN}Default configuration created at $CONFIG_FILE${NC}"
fi

# Create systemd service file
echo -e "${YELLOW}Creating systemd service...${NC}"
cat > "$SERVICE_FILE" << EOF
[Unit]
Description=Pico-Qwen Inference Service
After=network.target

[Service]
Type=simple
User=$USER_NAME
Group=$GROUP_NAME
WorkingDirectory=$INSTALL_DIR
ExecStart=$INSTALL_DIR/bin/$BINARY_NAME serve --config $CONFIG_FILE
ExecReload=/bin/kill -HUP \$MAINPID
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# Resource limits
MemoryLimit=8G
CPUQuota=200%

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=$INSTALL_DIR/config $INSTALL_DIR/models $INSTALL_DIR/logs

[Install]
WantedBy=multi-user.target
EOF

echo -e "${GREEN}Service file created at $SERVICE_FILE${NC}"

# Reload systemd and enable service
echo -e "${YELLOW}Reloading systemd and enabling service...${NC}"
systemctl daemon-reload
systemctl enable "$SERVICE_NAME"

# Test configuration
echo -e "${YELLOW}Testing configuration...${NC}"
if systemctl is-enabled "$SERVICE_NAME" > /dev/null 2>&1; then
    echo -e "${GREEN}Service enabled successfully${NC}"
else
    echo -e "${RED}Failed to enable service${NC}"
    exit 1
fi

echo -e "${GREEN}=== Installation Complete ===${NC}"
echo ""
echo "Service commands:"
echo "  Start:   sudo systemctl start $SERVICE_NAME"
echo "  Stop:    sudo systemctl stop $SERVICE_NAME"
echo "  Status:  sudo systemctl status $SERVICE_NAME"
echo "  Logs:    sudo journalctl -u $SERVICE_NAME -f"
echo "  Config:  $CONFIG_FILE"
echo "  Binary:  $INSTALL_DIR/bin/$BINARY_NAME"
echo ""
echo "To start the service:"
echo "  sudo systemctl start $SERVICE_NAME"
echo ""
echo "To view logs:"
echo "  sudo journalctl -u $SERVICE_NAME -f"