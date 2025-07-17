# Development Guide

## Architecture Overview

### Project Structure
```
pico-qwen/
├── Cargo.toml                    # Workspace configuration
├── qwen3-cli/                   # Command-line interface
├── qwen3-export/                # Model export utilities
├── qwen3-inference/             # Core inference library (ENHANCED)
├── qwen3-api/                   # REST API server
├── qwen3-web/                   # Web interface
├── qwen3-mcp/                   # MCP agents
├── docs/                        # Technical documentation
└── scripts/                     # Development and deployment scripts
```

## Development Setup

### Prerequisites
```bash
# Arch Linux development tools
sudo pacman -S git base-devel pkg-config openssl clippy rustfmt

# Install additional tools
sudo pacman -S gdb valgrind strace

# VSCode extensions (optional)
# rust-analyzer, CodeLLDB
```

### Environment Setup
```bash
# Clone repository
git clone https://github.com/PieBru/pico-qwen.git
cd pico-qwen

# Install Rust nightly (for latest features)
rustup install nightly
rustup override set nightly

# Install additional targets for cross-compilation
rustup target add aarch64-unknown-linux-gnu
rustup target add x86_64-unknown-linux-musl
rustup target add armv7-unknown-linux-gnueabihf
```

## Development Workflow

### Building
```bash
# Build entire workspace
cargo build --release

# Build specific crate
cargo build --release -p qwen3-cli
cargo build --release -p qwen3-inference

# Build with optimizations (CPU optimizations are built-in)
cargo build --release

# Cross-compilation
RUSTFLAGS="-C target-feature=+avx2" cargo build --release -p qwen3-cli
cargo build --release --target aarch64-unknown-linux-gnu
```

### Testing
```bash
# Run all tests
cargo test --release --all

# Run specific package tests
cargo test --release -p qwen3-inference

# Run CPU optimization tests (available in qwen3-inference)
cargo test --release -p qwen3-inference --test cpu_optimization_tests -- --nocapture

# Run export package tests
cargo test --release -p qwen3-export

# Run API server tests
cargo test --release -p qwen3-api

# Run integration tests
cargo test --release --test integration_tests

# Run benchmarks
cargo bench
```

### Code Quality
```bash
# Format code
cargo fmt

# Check formatting
cargo fmt --check

# Run clippy
cargo clippy --all-targets --all-features

# Run security audit
cargo audit

# Check for outdated dependencies
cargo outdated
```

## Architecture Details

### Core Components

#### qwen3-inference (Enhanced)
```rust
// Key enhancements
- extended_config.rs    # Extended model configuration
- cpu_optimizations.rs  # CPU feature detection
- extended_transformer.rs # Enhanced transformer with optimizations
- quantization.rs       # Advanced quantization (INT4/INT8/FP16/FP32)
```

#### qwen3-api (New)
```rust
// REST API server
- src/main.rs          # Server entry point
- src/routes/          # API endpoints
- src/models/          # API models
- src/middleware/      # Rate limiting, auth
```

#### qwen3-web (New)
```rust
// Web interface
- src/main.rs          # Web server
- static/              # HTML, CSS, JS
- templates/           # Handlebars templates
```

#### qwen3-mcp (New)
```rust
// MCP agents
- src/agent.rs         # Agent orchestration
- src/tools/           # Available tools
- src/workflows.rs     # PocketFlow workflows
```

## Adding New Features

### 1. CPU Target Support
```rust
// Add to qwen3-inference/src/cpu_optimizations.rs
pub enum CpuTarget {
    IntelI9_14900HX,
    RaspberryPi4,
    GenericX86,
    // Add new target
    AmdRyzen9_5900X,
}

impl CpuTarget {
    pub fn detect() -> Self {
        // Detection logic
    }
}
```

### 2. New Quantization Level (FIXME: need more testing)
```rust
// Add to qwen3-inference/src/quantization.rs
pub enum QuantizationLevel {
    Int4,
    Int8,
    Fp16,
    Fp32,
    // Add new level
    Int2,
}
```

### 3. New API Endpoint
```rust
// Add to qwen3-api/src/routes/
use axum::{
    routing::post,
    Router,
    Json,
};

async fn new_endpoint(
    Json(payload): Json<RequestModel>
) -> Result<Json<ResponseModel>, StatusCode> {
    // Implementation
}

// Register in main.rs
let app = Router::new()
    .route("/api/v1/new", post(new_endpoint));
```

### 4. New MCP Tool (FIXME: need more testing)
```rust
// Add to qwen3-mcp/src/tools/
use pocketflow::prelude::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewTool {
    pub input: String,
}

#[async_trait]
impl Tool for NewTool {
    async fn run(
        &self,
        context: &Context
    ) -> Result<ToolOutput, ToolError> {
        // Implementation
    }
}
```

## Performance Optimization

### Profiling
```bash
# Install profiling tools
sudo pacman -S perf-tools valgrind

# CPU profiling
perf record -g cargo run --release -p qwen3-cli -- inference model.bin
cargo flamegraph --bin qwen3-cli -- inference model.bin

# Memory profiling
valgrind --tool=massif cargo run --release -p qwen3-cli -- inference model.bin

# Benchmark specific functions
cargo bench -p qwen3-inference
```

### Memory Optimization
```bash
# Check memory usage
/usr/bin/time -v cargo run --release -p qwen3-cli -- inference model.bin

# Memory map analysis
cat /proc/$(pgrep pico-qwen)/smaps
```

## Testing Strategy

### Unit Tests
```bash
# Run unit tests
cargo test --lib

# Run specific test module
cargo test -p qwen3-inference transformer::tests

# Run with output
cargo test -- --nocapture
```

### Integration Tests
```bash
# Run integration tests
cargo test --test integration_tests

# Run specific integration test
cargo test --test integration_tests test_quantization_levels
```

### End-to-End Tests
```bash
# Test full workflow - manual testing commands
cargo run --release -p qwen3-export -- /path/to/model /tmp/test_model.bin --group-size 64
cargo run --release -p qwen3-cli -- inference /tmp/test_model.bin -i "Hello world" --max-tokens 20

# Test API endpoints
cargo run --release -p qwen3-api &
sleep 2
curl -X POST http://localhost:58080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"test"}]}'

# Test WebUI
cargo run --release -p qwen3-web &
open http://localhost:58080
```

## Debugging

### Debug Build
```bash
# Build with debug symbols
cargo build

# Run with debugger
gdb target/debug/qwen3-cli
(gdb) run inference model.bin -m chat
```

### Logging
```bash
# Enable verbose logging
RUST_LOG=debug cargo run --release -p qwen3-cli -- inference model.bin

# Log to file
RUST_LOG=info cargo run --release -p qwen3-api 2>> api.log
```

### Common Issues

#### Build Failures
```bash
# Clean build
cargo clean
cargo build --release

# Check dependencies
cargo tree

# Update lockfile
cargo update
```

#### Memory Issues
```bash
# Check system memory
free -h

# Reduce parallel compilation
export CARGO_BUILD_JOBS=2

# Use release build for testing
cargo test --release
```

#### Cross-Compilation Issues
```bash
# Install cross-compilation tools
sudo pacman -S aarch64-linux-gnu-gcc

# Build for ARM
cargo build --release --target aarch64-unknown-linux-gnu
```

## Contributing Guidelines

### Pull Request Process
1. **Fork repository** on GitHub
2. **Create feature branch** from `master`
3. **Write tests** for new functionality
4. **Run full test suite** locally
5. **Update documentation** if needed
6. **Submit pull request** with clear description

### Code Style
- Follow Rust standard formatting (`cargo fmt --help`)
- Use meaningful variable names
- Add documentation comments
- Write unit tests for new features
- Keep functions small and focused

### Commit Messages
```
type(scope): description

Examples:
feat(cpu): add AMD Ryzen 9 5900X support
fix(quantization): resolve INT4 memory alignment issue
docs(readme): update installation instructions
test(api): add streaming response tests
```

### Performance Requirements
- All changes must maintain or improve performance
- Benchmark critical paths
- Memory usage should not regress
- CPU optimization should be validated on target hardware

## Release Process

### Version Management
```bash
# Update version
cargo set-version --bump minor

# Generate changelog
git log --oneline v1.0.0..HEAD > CHANGELOG.md

# Create release
git tag -a v1.1.0 -m "Release v1.1.0"
git push origin v1.1.0
```

### Distribution
```bash
# Build release artifacts
cargo build --release --all

# Create packages
./scripts/create-packages.sh

# Upload to GitHub releases
gh release create v1.1.0 --generate-notes
```

## Documentation

### Adding Documentation
```rust
/// Brief description
/// 
/// Detailed description with examples
/// 
/// # Examples
/// ```
/// use qwen3_inference::Model;
/// let model = Model::load("path").unwrap();
/// ```
/// 
/// # Errors
/// Returns `ModelError` if loading fails
pub fn load(path: &str) -> Result<Model, ModelError> {
    // Implementation
}
```

### Architecture Documentation
- Update `./docs/architecture.md` for major changes (FIXME: TBD)
- Add diagrams for complex components
- Include performance benchmarks
- Document breaking changes

## Performance Benchmarks

### Current Benchmark Status
The project currently doesn't have dedicated benchmark suites. Use these alternatives:

### Performance Testing
```bash
# Test inference performance with timing
/usr/bin/time -v cargo run --release -p qwen3-cli -- inference model.bin -i "Hello world"

# Test CPU optimization impact
cargo test --release -p qwen3-inference --test cpu_optimization_tests -- --nocapture

# Memory usage profiling
valgrind --tool=massif cargo run --release -p qwen3-cli -- inference model.bin -i "Test prompt"

# Performance comparison between builds
hyperfine --warmup 3 'cargo run --release -p qwen3-cli -- inference model.bin -i "Benchmark test" --max-tokens 50'
```

### Creating Benchmarks
To add benchmarks, create `benches/` directories in crates and add benchmark files:

```bash
# Example: Add benchmark to qwen3-inference
mkdir -p qwen3-inference/benches
touch qwen3-inference/benches/cpu_benchmarks.rs

# Then add to Cargo.toml:
# [[bench]]
# name = "cpu_benchmarks"
# harness = false
```

### Custom Benchmarks
```rust
// Add to benches/
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_new_feature(c: &mut Criterion) {
    c.bench_function("new_feature", |b| {
        b.iter(|| {
            // Benchmark code
        })
    });
}

criterion_group!(benches, bench_new_feature);
criterion_main!(benches);
```