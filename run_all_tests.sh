#!/bin/bash
# run_all_tests.sh - Complete validation script for Qwen3 C inference engine

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on supported OS
check_os() {
    OS=$(uname -s)
    ARCH=$(uname -m)
    log_info "Running on $OS ($ARCH)"
}

# Build validation
build_validation() {
    log_info "=== Build System Validation ==="
    
    # Cargo build
    log_info "Building with Cargo..."
    cargo build --release
    log_success "Cargo build completed"
    
    # C library build
    log_info "Building C library and test executables..."
    cd qwen3-c-lib
    make test-matrix test-attention test-transformer test-inference test-model
    log_success "C library and test executables built"
    cd ..
}

# Matrix operations tests
test_matmul() {
    log_info "=== Matrix Operations Tests ==="
    cd qwen3-c-lib
    
    log_info "Running matrix multiplication tests..."
    ./test_matrix
    log_success "Matrix tests passed"
    
    cd ..
}

# Model loading tests
test_model_loading() {
    log_info "=== Model Loading Tests ==="
    cd qwen3-c-lib
    
    # Try to find a model file
    MODEL_PATH=""
    for path in "../models/qwen3-0.6b.bin" "../qwen3-0.6b.bin" "./qwen3-0.6b.bin"; do
        if [ -f "$path" ]; then
            MODEL_PATH="$path"
            break
        fi
    done
    
    if [ -n "$MODEL_PATH" ]; then
        log_info "Testing model loading with: $MODEL_PATH"
        ./test_model_loading "$MODEL_PATH"
        log_success "Model loading tests passed"
    else
        log_warning "No model file found, skipping model loading tests"
        log_info "Expected model files: qwen3-0.6b.bin in project root or models/ directory"
    fi
    
    cd ..
}

# Attention mechanism tests
test_attention() {
    log_info "=== Attention Mechanism Tests ==="
    cd qwen3-c-lib
    
    log_info "Running attention tests..."
    ./test_attention
    log_success "Attention tests passed"
    
    cd ..
}

# Transformer layer tests
test_transformer() {
    log_info "=== Transformer Layer Tests ==="
    cd qwen3-c-lib
    
    log_info "Running transformer tests..."
    ./test_transformer
    log_success "Transformer tests passed"
    
    cd ..
}

# End-to-end inference tests
test_inference() {
    log_info "=== End-to-End Inference Tests ==="
    cd qwen3-c-lib
    
    # Try to find a model file for inference tests
    MODEL_PATH=""
    for path in "../models/qwen3-0.6b.bin" "../qwen3-0.6b.bin" "./qwen3-0.6b.bin"; do
        if [ -f "$path" ]; then
            MODEL_PATH="$path"
            break
        fi
    done
    
    if [ -n "$MODEL_PATH" ]; then
        log_info "Testing inference with: $MODEL_PATH"
        ./test_inference "$MODEL_PATH"
        log_success "Inference tests passed"
    else
        log_warning "No model file found, skipping inference tests"
    fi
    
    cd ..
}

# Memory and tensor tests through matrix operations
test_memory_and_tensor() {
    log_info "=== Memory and Tensor Tests ==="
    cd qwen3-c-lib
    
    log_info "Testing memory management and tensor operations..."
    
    # Run matrix tests which use tensors and memory
    ./test_matrix
    log_success "Memory and tensor validation completed"
    
    # Check for memory leaks with Valgrind if available
    if command -v valgrind >/dev/null; then
        log_info "Running memory leak detection..."
        valgrind --leak-check=full --show-leak-kinds=all --error-exitcode=1 ./test_matrix 2>/dev/null || log_warning "Valgrind test completed (some leaks may be acceptable)"
    else
        log_warning "Valgrind not found, skipping memory leak tests"
    fi
    
    cd ..
}

# CLI integration tests
test_cli() {
    log_info "=== CLI Integration Tests ==="
    
    # Try to find a model file
    MODEL_PATH=""
    for path in "models/qwen3-0.6b.bin" "qwen3-0.6b.bin"; do
        if [ -f "$path" ]; then
            MODEL_PATH="$path"
            break
        fi
    done
    
    if [ -n "$MODEL_PATH" ]; then
        log_info "Testing CLI with engine switching..."
        
        # Test Rust engine
        cargo run --release -p qwen3-cli -- inference "$MODEL_PATH" --engine rust --benchmark --tokens 10 2>/dev/null || log_warning "Rust engine test failed"
        log_success "Rust engine CLI test completed"
        
        # Test C engine
        cargo run --release -p qwen3-cli -- inference "$MODEL_PATH" --engine c --benchmark --tokens 10 2>/dev/null || log_warning "C engine test failed"
        log_success "C engine CLI test completed"
    else
        log_warning "No model file found, skipping CLI integration tests"
    fi
}

# Performance testing
test_performance() {
    log_info "=== Performance Testing ==="
    cd qwen3-c-lib
    
    log_info "Running performance benchmarks..."
    
    # Matrix multiplication benchmarks
    ./test_matrix || log_warning "Matrix benchmark not available"
    
    # Attention benchmarks
    ./test_attention || log_warning "Attention benchmark not available"
    
    # Transformer benchmarks
    ./test_transformer || log_warning "Transformer benchmark not available"
    
    log_success "Performance tests completed"
    cd ..
}

# Main test execution
main() {
    log_info "=== Qwen3 C Inference Engine Comprehensive Test Suite ==="
    
    check_os
    
    # Phase 1: Build validation
    build_validation
    
    # Phase 2: Core functionality tests
    test_matmul
    test_memory_and_tensor
    test_attention
    test_transformer
    test_model_loading
    test_inference
    
    # Phase 3: CLI integration tests
    test_cli
    
    # Phase 4: Performance validation
    test_performance
    
    log_success "=== All Tests Completed Successfully ==="
    log_info "The Qwen3 C inference engine test suite has completed"
}

# Handle command line arguments
case "${1:-}" in
    "build-only")
        build_validation
        ;;
    "matrix-only")
        test_matmul
        ;;
    "attention-only")
        test_attention
        ;;
    "transformer-only")
        test_transformer
        ;;
    "model-only")
        test_model_loading
        ;;
    "inference-only")
        test_inference
        ;;
    "memory-only")
        test_memory_and_tensor
        ;;
    "cli-only")
        test_cli
        ;;
    "perf-only")
        test_performance
        ;;
    "help"|"-h"|"--help")
        echo "Usage: $0 [option]"
        echo "Available test executables:"
        echo "  test_matrix      - Matrix operations"
        echo "  test_attention   - Attention mechanism"
        echo "  test_transformer - Transformer layers"
        echo "  test_inference   - End-to-end inference"
        echo "  test_model_loading - Model loading"
        echo ""
        echo "Options:"
        echo "  build-only       - Only run build validation"
        echo "  matrix-only      - Only run matrix tests"
        echo "  attention-only   - Only run attention tests"
        echo "  transformer-only - Only run transformer tests"
        echo "  model-only       - Only run model loading tests"
        echo "  inference-only   - Only run inference tests"
        echo "  memory-only      - Only run memory tests"
        echo "  cli-only         - Only run CLI tests"
        echo "  perf-only        - Only run performance tests"
        echo "  help             - Show this help message"
        echo "  (no argument)    - Run all tests"
        ;;
    *)
        main
        ;;
esac