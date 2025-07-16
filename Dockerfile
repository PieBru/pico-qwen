# Multi-stage Docker build for pico-qwen
FROM rust:1.80-slim-bookworm as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy workspace files
COPY Cargo.toml Cargo.lock ./
COPY qwen3-cli/Cargo.toml ./qwen3-cli/
COPY qwen3-export/Cargo.toml ./qwen3-export/
COPY qwen3-inference/Cargo.toml ./qwen3-inference/
COPY qwen3-api/Cargo.toml ./qwen3-api/
COPY qwen3-web/Cargo.toml ./qwen3-web/

# Create dummy source files to cache dependencies
RUN mkdir -p qwen3-cli/src qwen3-export/src qwen3-inference/src qwen3-api/src qwen3-web/src \
    && echo "fn main() {}" > qwen3-cli/src/main.rs \
    && echo "fn main() {}" > qwen3-export/src/main.rs \
    && echo "fn main() {}" > qwen3-api/src/main.rs \
    && echo "fn main() {}" > qwen3-web/src/main.rs \
    && echo "" > qwen3-inference/src/lib.rs

# Build dependencies
RUN cargo build --release --all-targets

# Copy actual source
COPY . .

# Build the actual application
RUN cargo build --release --all-targets

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -r -s /bin/false pico-qwen

# Create directories
RUN mkdir -p /opt/pico-qwen/{bin,config,models,logs} \
    && chown -R pico-qwen:pico-qwen /opt/pico-qwen

# Copy binaries
COPY --from=builder /app/target/release/qwen3-cli /opt/pico-qwen/bin/
COPY --from=builder /app/target/release/qwen3-api /opt/pico-qwen/bin/
COPY --from=builder /app/target/release/qwen3-web /opt/pico-qwen/bin/

# Copy configuration files
COPY config/docker.toml /opt/pico-qwen/config/config.toml
COPY config/web.toml /opt/pico-qwen/config/web.toml

# Set permissions
RUN chmod +x /opt/pico-qwen/bin/* \
    && chown -R pico-qwen:pico-qwen /opt/pico-qwen

# Expose ports
EXPOSE 8080 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Switch to non-root user
USER pico-qwen

# Set working directory
WORKDIR /opt/pico-qwen

# Default command
CMD ["/opt/pico-qwen/bin/qwen3-api", "--config", "/opt/pico-qwen/config/config.toml"]