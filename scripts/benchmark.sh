#!/bin/bash

# Qwen3 C Inference Engine Performance Benchmarking Script
# This script runs comprehensive performance tests across different CPU architectures

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BUILD_DIR="build"
RESULTS_DIR="benchmark_results"
ITERATIONS=10

# Create results directory
mkdir -p "$RESULTS_DIR"

echo -e "${BLUE}=== Qwen3 C Inference Engine Performance Benchmarking ===${NC}"
echo

# Check for required commands
if ! command -v cmake &> /dev/null; then
    echo -e "${RED}Error: cmake is required but not installed${NC}"
    exit 1
fi

if ! command -v gcc &> /dev/null && ! command -v clang &> /dev/null; then
    echo -e "${RED}Error: gcc or clang is required${NC}"
    exit 1
fi

# Function to detect CPU features
detect_cpu_features() {
    echo -e "${YELLOW}Detecting CPU features...${NC}"
    
    # Check for AVX-512
    if grep -q "avx512" /proc/cpuinfo 2>/dev/null; then
        echo -e "${GREEN}✓ AVX-512 detected${NC}"
        return 3
    elif grep -q "avx2" /proc/cpuinfo 2>/dev/null; then
        echo -e "${GREEN}✓ AVX2 detected${NC}"
        return 2
    elif grep -q "avx" /proc/cpuinfo 2>/dev/null; then
        echo -e "${GREEN}✓ AVX detected${NC}"
        return 1
    else
        echo -e "${YELLOW}! Using scalar operations${NC}"
        return 0
    fi
}

# Function to build the project
build_project() {
    echo -e "${YELLOW}Building project...${NC}"
    
    # Clean build
    rm -rf "$BUILD_DIR"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    
    # Configure with aggressive optimization
    cmake ../qwen3-c-lib \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_FLAGS="-O3 -march=native -mtune=native -funroll-loops -ffast-math" \
        -DCMAKE_C_COMPILER=gcc
    
    make -j$(nproc)
    cd ..
    
    echo -e "${GREEN}✓ Build completed${NC}"
}

# Function to run matrix benchmarks
run_matrix_benchmarks() {
    echo -e "${BLUE}Running Matrix Operation Benchmarks...${NC}"
    
    local matrix_sizes=(32 64 128 256 512 1024)
    local results_file="$RESULTS_DIR/matrix_benchmarks.csv"
    
    echo "Size,GFLOPS,Time_us,CPU_Feature" > "$results_file"
    
    for size in "${matrix_sizes[@]}"; do
        echo -e "${YELLOW}Testing ${size}x${size} matrix multiplication...${NC}"
        
        # Run the benchmark
        if [ -f "$BUILD_DIR/test_matrix" ]; then
            output=$(cd "$BUILD_DIR" && ./test_matrix 2>/dev/null | grep -E "${size}x${size}:" || echo "")
            if [[ $output =~ ([0-9.]+)\ us\ \(([0-9.]+)\ GFLOPS\) ]]; then
                time_us="${BASH_REMATCH[1]}"
                gflops="${BASH_REMATCH[2]}"
                cpu_feature=$(cd "$BUILD_DIR" && ./test_matrix 2>/dev/null | grep "CPU Features:" | awk '{print $3}' || echo "unknown")
                
                echo "$size,$gflops,$time_us,$cpu_feature" >> "$results_file"
                echo -e "  ${GREEN}Size: $size, GFLOPS: $gflops, Time: ${time_us}us${NC}"
            fi
        fi
    done
}

# Function to run attention benchmarks
run_attention_benchmarks() {
    echo -e "${BLUE}Running Attention Mechanism Benchmarks...${NC}"
    
    local seq_lengths=(128 256 512 1024 2048)
    local results_file="$RESULTS_DIR/attention_benchmarks.csv"
    
    echo "Seq_Length,Heads,Head_Dim,Time_us,CPU_Feature" > "$results_file"
    
    for seq_len in "${seq_lengths[@]}"; do
        echo -e "${YELLOW}Testing attention with seq_len=$seq_len...${NC}"
        
        # Run the benchmark
        if [ -f "$BUILD_DIR/test_attention" ]; then
            output=$(cd "$BUILD_DIR" && ./test_attention 2>/dev/null | grep -E "Attention benchmark" || echo "")
            if [[ $output =~ ([0-9.]+)\ us ]]; then
                time_us="${BASH_REMATCH[1]}"
                cpu_feature=$(cd "$BUILD_DIR" && ./test_attention 2>/dev/null | grep "CPU Features:" | awk '{print $3}' || echo "unknown")
                
                echo "$seq_len,8,64,$time_us,$cpu_feature" >> "$results_file"
                echo -e "  ${GREEN}Seq: $seq_len, Time: ${time_us}us${NC}"
            fi
        fi
    done
}

# Function to run memory benchmarks
run_memory_benchmarks() {
    echo -e "${BLUE}Running Memory Management Benchmarks...${NC}"
    
    local alloc_sizes=(1024 4096 16384 65536 262144 1048576)
    local results_file="$RESULTS_DIR/memory_benchmarks.csv"
    
    echo "Alloc_Size,Time_us,Throughput_MB_s" > "$results_file"
    
    for size in "${alloc_sizes[@]}"; do
        echo -e "${YELLOW}Testing memory allocation of ${size} bytes...${NC}"
        
        # Run memory benchmark (this would need a dedicated test)
        # For now, we'll use a simple timing approach
        start_time=$(date +%s%N)
        
        # Simulate allocation/deallocation
        for i in $(seq 1 1000); do
            ptr=$(malloc $size 2>/dev/null || echo "")
            if [ -n "$ptr" ]; then
                free $ptr 2>/dev/null || true
            fi
        done
        
        end_time=$(date +%s%N)
        time_ns=$((end_time - start_time))
        time_us=$((time_ns / 1000))
        throughput=$((1000 * size * 1000 / time_us / 1024 / 1024))
        
        echo "$size,$time_us,$throughput" >> "$results_file"
        echo -e "  ${GREEN}Size: $size, Time: ${time_us}us, Throughput: ${throughput} MB/s${NC}"
    done
}

# Function to generate summary report
generate_summary() {
    echo -e "${BLUE}Generating Summary Report...${NC}"
    
    local summary_file="$RESULTS_DIR/summary_report.txt"
    
    cat > "$summary_file" << EOF
Qwen3 C Inference Engine Performance Summary
===========================================

Generated: $(date)
CPU: $(grep 'model name' /proc/cpuinfo | head -1 | cut -d':' -f2 | xargs)
Architecture: $(uname -m)
Kernel: $(uname -r)

Matrix Operations Performance:
$(cat "$RESULTS_DIR/matrix_benchmarks.csv" 2>/dev/null || echo "No matrix benchmark data")

Attention Mechanism Performance:
$(cat "$RESULTS_DIR/attention_benchmarks.csv" 2>/dev/null || echo "No attention benchmark data")

Memory Management Performance:
$(cat "$RESULTS_DIR/memory_benchmarks.csv" 2>/dev/null || echo "No memory benchmark data")

Peak Performance Achieved:
$(grep -E '^[0-9]+,' "$RESULTS_DIR/matrix_benchmarks.csv" 2>/dev/null | sort -t',' -k2 -nr | head -1 || echo "No data")

EOF
    
    echo -e "${GREEN}✓ Summary report generated: $summary_file${NC}"
}

# Function to run all tests
run_all_benchmarks() {
    detect_cpu_features
    build_project
    
    # Run individual benchmarks
    run_matrix_benchmarks
    run_attention_benchmarks
    run_memory_benchmarks
    
    # Generate summary
    generate_summary
    
    echo -e "${GREEN}=== All benchmarks completed ===${NC}"
    echo -e "Results saved in: ${YELLOW}$RESULTS_DIR${NC}"
}

# Function to display help
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  --matrix      Run only matrix benchmarks"
    echo "  --attention   Run only attention benchmarks"
    echo "  --memory      Run only memory benchmarks"
    echo "  --all         Run all benchmarks (default)"
    echo "  --help        Show this help message"
    echo
    echo "Examples:"
    echo "  $0 --all      # Run all benchmarks"
    echo "  $0 --matrix   # Run only matrix benchmarks"
}

# Main execution
case "${1:-all}" in
    --matrix)
        detect_cpu_features
        build_project
        run_matrix_benchmarks
        ;;
    --attention)
        detect_cpu_features
        build_project
        run_attention_benchmarks
        ;;
    --memory)
        detect_cpu_features
        build_project
        run_memory_benchmarks
        ;;
    --all|all)
        run_all_benchmarks
        ;;
    --help|help|-h)
        show_help
        ;;
    *)
        echo -e "${RED}Error: Unknown option '$1'${NC}"
        show_help
        exit 1
        ;;
esac