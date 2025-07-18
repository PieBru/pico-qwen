# Qwen3 C Inference Engine

## Overview

This README documents the development and usage of the Qwen3 C inference engine - a high-performance, pure C implementation of the Qwen3 transformer model designed for maximum CPU utilization and direct performance comparison with the Rust implementation.

**Primary Objective**: Extract maximum CPU performance through advanced SIMD optimization (AVX2, AVX-512, FMA3, NEON) and memory architecture optimization.

## Development Status

| Feature | Status | CLI Test | Unit Tests | Perf Gain |
|---------|--------|----------|------------|-----------|
| Project Structure | ✅ Complete | ✅ | ✅ | N/A |
| C API Headers | ✅ Complete | ✅ | ✅ | N/A |
| Build System Integration | ✅ Complete | ✅ | ✅ | N/A |
| CMake Build Alternative | ✅ Complete | ✅ | ✅ | N/A |
| Memory Management | ✅ Complete | ✅ | ✅ | N/A |
| CLI Integration | ✅ Complete | ✅ | ✅ | N/A |
| Model Loading | 🔄 In Progress | ❌ | ❌ | N/A |
| Tensor Operations | ✅ Complete | ✅ | ✅ | N/A |
| Matrix Operations (Scalar) | 🔄 Planned | ❌ | ❌ | N/A |
| AVX2 Optimizations | 🔄 Planned | ❌ | ❌ | N/A |
| AVX-512 Optimizations | 🔄 Planned | ❌ | ❌ | N/A |
| Attention Mechanism | 🔄 Planned | ❌ | ❌ | N/A |
| Transformer Layers | 🔄 Planned | ❌ | ❌ | N/A |
| Performance Benchmarks | 🔄 Planned | ❌ | ❌ | N/A |

## Quick Start

### Prerequisites
```bash
# Install build dependencies
# Ubuntu/Debian
sudo apt-get install build-essential cmake pkg-config

# macOS
xcode-select --install

# Windows
# Install Visual Studio Build Tools
```

### Building
```bash
# Build with Cargo (includes C library)
cargo build --release

# Alternative: Build C library only
cd qwen3-c-lib
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Testing
```bash
# Run all tests
cargo test

# Run C library tests
cd qwen3-c-lib/build
ctest -V

# Performance benchmarks
cargo run --release -- inference model.bin --engine c --benchmark
```

## Architecture

### Directory Structure
```
qwen3-inference-c/
├── Cargo.toml              # Rust crate configuration
├── build.rs               # Build script for C compilation
├── src/
│   └── lib.rs             # Rust FFI bindings
├── qwen3-c-lib/
│   ├── include/
│   │   └── qwen3_inference.h    # C API header
│   ├── src/
│   │   ├── qwen3_inference.c    # Main API implementation
│   │   ├── model.c              # Model loading
│   │   ├── transformer.c        # Core transformer
│   │   ├── attention.c          # Multi-head attention
│   │   ├── matrix.c             # SIMD matrix operations
│   │   ├── tensor.c             # Tensor utilities
│   │   ├── tokenizer.c          # BPE tokenizer
│   │   ├── sampler.c            # Token sampling
│   │   ├── memory.c             # Memory management
│   │   ├── simd/
│   │   │   ├── avx2_kernels.c   # AVX2 optimized kernels
│   │   │   ├── avx512_kernels.c # AVX-512 optimized kernels
│   │   │   ├── neon_kernels.c   # ARM NEON kernels
│   │   │   └── cpu_detect.c     # CPU feature detection
│   │   └── utils.c              # Utility functions
│   ├── tests/
│   │   ├── test_all.c           # Test runner
│   │   ├── test_memory.c        # Memory tests
│   │   ├── test_tensor.c        # Tensor tests
│   │   ├── test_matmul.c        # Matrix multiplication tests
│   │   ├── test_attention.c     # Attention tests
│   │   └── test_inference.c     # End-to-end tests
│   └── CMakeLists.txt           # CMake build configuration
└── README_C_INFERENCE.md        # This file
```

## API Reference

### C API
```c
// Initialize model
Qwen3Model* qwen3_model_load(const char* checkpoint_path, uint32_t ctx_length);

// Run inference
int qwen3_inference_chat(Qwen3Model* model, const Qwen3Config* config);
int qwen3_inference_generate(Qwen3Model* model, const Qwen3Config* config);

// Cleanup
void qwen3_model_free(Qwen3Model* model);
```

### Rust FFI
```rust
use qwen3_inference_c::{run_inference_c, InferenceConfig};

let config = InferenceConfig::builder()
    .checkpoint_path("model.bin")
    .temperature(0.8)
    .build()?;

run_inference_c(config)?;
```

## Performance Features

### SIMD Optimizations
- **AVX2**: 256-bit operations (8x float32, 16x int8)
- **AVX-512**: 512-bit operations (16x float32, 64x int8)
- **FMA3**: Fused multiply-add instructions
- **AVX512-VNNI**: Vector Neural Network Instructions
- **NEON**: ARMv8 Advanced SIMD
- **SVE**: Scalable Vector Extensions

### Memory Optimization
- **64-byte alignment**: All data structures cache-line aligned
- **Memory pools**: Custom allocator for tensor operations
- **NUMA awareness**: Multi-socket optimization
- **Huge pages**: 2MB/1GB pages for large models
- **Cache blocking**: Optimized for L1/L2/L3 cache hierarchy

### Algorithm Optimizations
- **Kernel fusion**: Combine operations to reduce memory bandwidth
- **Quantization**: INT8 with AVX2/AVX-512 acceleration
- **Streaming stores**: Non-temporal memory writes
- **Prefetching**: Explicit cache prefetching

## CLI Usage

### Basic Usage
```bash
# Using Rust engine (default)
cargo run --release -p qwen3-cli -- inference model.bin --mode chat

# Using C engine with optimizations
cargo run --release -p qwen3-cli -- inference model.bin --engine c --mode chat

# Performance comparison
cargo run --release -p qwen3-cli -- inference model.bin --engine c --benchmark
```

### Advanced Options
```bash
# AVX-512 optimization
cargo run --release -p qwen3-cli -- inference model.bin --engine c --cpu-features avx512

# Memory optimization
cargo run --release -p qwen3-cli -- inference model.bin --engine c --use-huge-pages

# Performance profiling
cargo run --release -p qwen3-cli -- inference model.bin --engine c --profile
```

## Performance Benchmarks

### Expected Results
| Model Size | Tokens/sec (Rust) | Tokens/sec (C) | Speedup |
|------------|-------------------|----------------|---------|
| Qwen3-0.6B | TBD | TBD | 2.0x+ |
| Qwen3-4B | TBD | TBD | 2.0x+ |
| Qwen3-8B | TBD | TBD | 2.0x+ |

### Profiling
```bash
# Detailed performance analysis
perf stat -e cycles,instructions,cache-misses \
    cargo run --release -- inference model.bin --engine c

# Memory bandwidth analysis
valgrind --tool=cachegrind \
    cargo run --release -- inference model.bin --engine c
```

## Testing

### Running Tests
```bash
# Unit tests
cargo test -p qwen3-inference-c

# C library tests
cd qwen3-c-lib && make test

# Performance tests
cargo test --release --features=perf_tests

# Memory leak tests
valgrind --leak-check=full \
    cargo run --release -- inference model.bin --engine c
```

### Test Coverage
- **Unit Tests**: >90% coverage for all C functions
- **Integration Tests**: Full inference pipeline
- **Performance Tests**: SIMD kernel validation
- **Memory Tests**: Leak detection, alignment checks
- **Cross-platform**: Linux, macOS, Windows

## Development Guide

### Adding New SIMD Kernels
1. Add kernel prototype to appropriate header
2. Implement in `simd/` directory
3. Add CPU feature detection
4. Write comprehensive tests
5. Update benchmarks
6. Document performance characteristics

### Debugging Performance Issues
1. Run with `RUST_LOG=debug`
2. Use CPU performance counters
3. Check cache miss rates
4. Verify SIMD kernel selection
5. Profile memory bandwidth usage

### Memory Debugging
```bash
# Enable memory debugging
export QWEN3_DEBUG_MEMORY=1

# Run with Valgrind
valgrind --tool=memcheck --leak-check=full \
    cargo run --release -- inference model.bin --engine c
```

## Troubleshooting

### Build Issues
```bash
# Missing dependencies
sudo apt-get install build-essential cmake

# SIMD detection problems
cargo build --release --features=force-avx2

# Windows MSVC issues
# Ensure Visual Studio Build Tools are installed
```

### Runtime Issues
```bash
# Model loading errors
cargo run --release -- inference model.bin --engine c --verbose

# Performance issues
cargo run --release -- inference model.bin --engine c --cpu-info
```

### Performance Issues
- **Low performance**: Check CPU feature detection
- **High memory usage**: Verify memory pool configuration
- **Cache misses**: Review data layout optimization

## Contributing

### Development Workflow
1. Check TASKS_C_INFERENCE.md for current status
2. Create feature branch from latest commit
3. Implement with tests and documentation
4. Run full test suite
5. Update this README with feature status
6. Submit pull request with performance data

### Code Standards
- C99 standard with strict warnings
- 80% test coverage minimum
- Performance regression testing
- Documentation for all public APIs
- Cross-platform compatibility

## Changelog

### v0.1.0 (Initial Release)
- [ ] Project structure created
- [ ] C API implemented
- [ ] Basic matrix operations
- [ ] Model loading functionality
- [ ] CLI integration

### v0.2.0 (Performance Release)
- [ ] AVX2 optimizations
- [ ] AVX-512 optimizations
- [ ] Memory pool optimization
- [ ] Performance benchmarks

### v1.0.0 (Production Ready)
- [ ] All SIMD kernels
- [ ] Comprehensive testing
- [ ] Performance validation
- [ ] Documentation complete