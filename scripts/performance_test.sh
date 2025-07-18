#!/bin/bash

# Qwen3 C Inference Engine Performance Regression Testing
# This script runs performance regression tests and compares against baselines

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
BUILD_DIR="build"
BASELINE_DIR="benchmark_baselines"
RESULTS_DIR="performance_results"
TOLERANCE=0.85  # 15% regression tolerance

# Create directories
mkdir -p "$BASELINE_DIR" "$RESULTS_DIR"

echo -e "${BLUE}=== Qwen3 C Inference Engine Performance Regression Testing ===${NC}"
echo

# Performance thresholds (GFLOPS for different matrix sizes)
declare -A PERFORMANCE_THRESHOLDS=(
    [32]=2.0
    [64]=4.0
    [128]=8.0
    [256]=16.0
    [512]=25.0
)

# Function to build project
build_project() {
    echo -e "${YELLOW}Building project with regression testing flags...${NC}"
    
    rm -rf "$BUILD_DIR"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    
    cmake ../qwen3-c-lib \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_FLAGS="-O3 -march=native -mtune=native -funroll-loops -ffast-math -fno-omit-frame-pointer"
    
    make -j$(nproc)
    cd ..
    
    echo -e "${GREEN}✓ Build completed${NC}"
}

# Function to run single benchmark
run_single_benchmark() {
    local type=$1
    local size=$2
    local iterations=$3
    
    case $type in
        "matrix")
            if [ -f "$BUILD_DIR/test_matrix" ]; then
                cd "$BUILD_DIR"
                ./test_matrix | grep -E "${size}x${size}:" | awk '{print $3}' || echo "0"
                cd ..
            else
                echo "0"
            fi
            ;;
        "attention")
            if [ -f "$BUILD_DIR/test_attention" ]; then
                cd "$BUILD_DIR"
                ./test_attention | grep -E "Attention benchmark" | awk '{print $3}' || echo "0"
                cd ..
            else
                echo "0"
            fi
            ;;
    esac
}

# Function to establish baseline
establish_baseline() {
    echo -e "${YELLOW}Establishing performance baseline...${NC}"
    
    build_project
    
    local baseline_file="$BASELINE_DIR/baseline_$(date +%Y%m%d_%H%M%S).csv"
    echo "Type,Size,GFLOPS,Time_us,CPU_Feature" > "$baseline_file"
    
    # Matrix benchmarks
    for size in 32 64 128 256 512; do
        echo -e "${YELLOW}Measuring ${size}x${size} matrix baseline...${NC}"
        gflops=$(run_single_benchmark "matrix" $size 10)
        if [[ $gflops =~ ^[0-9]+(.[0-9]+)?$ ]]; then
            echo "matrix,$size,$gflops,0,$(grep 'model name' /proc/cpuinfo | head -1 | cut -d':' -f2 | xargs)" >> "$baseline_file"
        fi
    done
    
    # Attention benchmark
    echo -e "${YELLOW}Measuring attention mechanism baseline...${NC}"
    time_us=$(run_single_benchmark "attention" 512 1)
    echo "attention,512,0,$time_us,$(grep 'model name' /proc/cpuinfo | head -1 | cut -d':' -f2 | xargs)" >> "$baseline_file"
    
    echo -e "${GREEN}✓ Baseline established: $baseline_file${NC}"
}

# Function to run regression tests
run_regression_tests() {
    echo -e "${YELLOW}Running performance regression tests...${NC}"
    
    build_project
    
    local results_file="$RESULTS_DIR/regression_$(date +%Y%m%d_%H%M%S).csv"
    local report_file="$RESULTS_DIR/regression_report.txt"
    
    echo "Type,Size,Current_GFLOPS,Baseline_GFLOPS,Ratio,Status" > "$results_file"
    
    # Find latest baseline
    local latest_baseline=$(ls -t "$BASELINE_DIR"/baseline_*.csv 2>/dev/null | head -1)
    
    if [ -z "$latest_baseline" ]; then
        echo -e "${RED}Error: No baseline found. Run --baseline first.${NC}"
        exit 1
    fi
    
    local regression_count=0
    
    # Run matrix benchmarks
    while IFS=',' read -r type size baseline_gflops time_us cpu_feature; do
        if [[ $type == "matrix" ]]; then
            echo -e "${YELLOW}Testing ${type} ${size}x${size} regression...${NC}"
            
            current_gflops=$(run_single_benchmark "matrix" $size 10)
            if [[ $current_gflops =~ ^[0-9]+(.[0-9]+)?$ ]] && [[ $baseline_gflops =~ ^[0-9]+(.[0-9]+)?$ ]]; then
                ratio=$(echo "scale=3; $current_gflops / $baseline_gflops" | bc -l)
                
                # Check if performance regressed
                if (( $(echo "$ratio < $TOLERANCE" | bc -l) )); then
                    status="REGRESSION"
                    ((regression_count++))
                    echo -e "  ${RED}REGRESSION: ${ratio} (baseline: ${baseline_gflops}, current: ${current_gflops})${NC}"
                else
                    status="PASS"
                    echo -e "  ${GREEN}PASS: ${ratio} (baseline: ${baseline_gflops}, current: ${current_gflops})${NC}"
                fi
                
                echo "$type,$size,$current_gflops,$baseline_gflops,$ratio,$status" >> "$results_file"
            fi
        fi
    done < "$latest_baseline"
    
    # Run attention benchmark
    while IFS=',' read -r type size baseline_gflops baseline_time_us cpu_feature; do
        if [[ $type == "attention" ]]; then
            echo -e "${YELLOW}Testing ${type} regression...${NC}"
            
            current_time_us=$(run_single_benchmark "attention" 512 1)
            if [[ $current_time_us =~ ^[0-9]+(.[0-9]+)?$ ]] && [[ $baseline_time_us =~ ^[0-9]+(.[0-9]+)?$ ]]; then
                ratio=$(echo "scale=3; $baseline_time_us / $current_time_us" | bc -l)
                
                # For attention, we want ratio >= TOLERANCE (faster is better)
                if (( $(echo "$ratio < $TOLERANCE" | bc -l) )); then
                    status="REGRESSION"
                    ((regression_count++))
                    echo -e "  ${RED}REGRESSION: ${ratio} (baseline: ${baseline_time_us}us, current: ${current_time_us}us)${NC}"
                else
                    status="PASS"
                    echo -e "  ${GREEN}PASS: ${ratio} (baseline: ${baseline_time_us}us, current: ${current_time_us}us)${NC}"
                fi
                
                echo "$type,$size,0,$current_time_us,$ratio,$status" >> "$results_file"
            fi
        fi
    done < "$latest_baseline"
    
    # Generate report
    cat > "$report_file" << EOF
Qwen3 C Inference Engine Performance Regression Report
=====================================================

Generated: $(date)
CPU: $(grep 'model name' /proc/cpuinfo | head -1 | cut -d':' -f2 | xargs)
Baseline: $latest_baseline
Tolerance: $(( (1 - TOLERANCE) * 100 ))%

Summary:
- Total tests: $(wc -l < "$results_file")
- Regressions: $regression_count
- Passes: $(( $(wc -l < "$results_file") - regression_count ))

Detailed Results:
$(cat "$results_file" | column -t -s, 2>/dev/null || cat "$results_file")

Performance Analysis:
$(if [ $regression_count -gt 0 ]; then
    echo "⚠️  WARNING: $regression_count performance regressions detected!"
    echo "Investigate the following issues:"
    grep "REGRESSION" "$results_file" | while IFS=',' read -r type size current baseline ratio status; do
        echo "  - $type $size: ${ratio}x baseline performance"
    done
else
    echo "✅ All performance tests passed within tolerance"
fi)

Recommendations:
$(if [ $regression_count -gt 0 ]; then
    echo "- Review recent code changes for performance impacts"
    echo "- Run detailed profiling to identify bottlenecks"
    echo "- Consider compiler optimization flags"
    echo "- Check for memory alignment issues"
else
    echo "- Performance is stable, continue development"
    echo "- Consider establishing new baseline for improved performance"
fi)

EOF
    
    if [ $regression_count -gt 0 ]; then
        echo -e "${RED}❌ Performance regression detected: $regression_count tests failed${NC}"
        echo -e "${YELLOW}Check report: $report_file${NC}"
        exit 1
    else
        echo -e "${GREEN}✅ All performance tests passed${NC}"
        echo -e "Report: $report_file"
    fi
}

# Function to show performance trends
show_trends() {
    echo -e "${BLUE}Performance Trends Analysis${NC}"
    
    local latest_baseline=$(ls -t "$BASELINE_DIR"/baseline_*.csv 2>/dev/null | head -1)
    
    if [ -z "$latest_baseline" ]; then
        echo -e "${YELLOW}No baseline data available for trend analysis${NC}"
        return
    fi
    
    echo -e "${GREEN}Latest baseline: $(basename "$latest_baseline")${NC}"
    echo
    
    echo "Matrix Performance (GFLOPS):"
    grep "^matrix" "$latest_baseline" | while IFS=',' read -r type size gflops time_us cpu; do
        local threshold=${PERFORMANCE_THRESHOLDS[$size]:-1.0}
        if (( $(echo "$gflops < $threshold" | bc -l) )); then
            echo -e "  ${RED}  ${size}x${size}: ${gflops} GFLOPS (below threshold: ${threshold})${NC}"
        else
            echo -e "  ${GREEN}  ${size}x${size}: ${gflops} GFLOPS ✓${NC}"
        fi
    done
}

# Function to display help
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  --baseline    Establish new performance baseline"
    echo "  --test        Run regression tests against baseline"
    echo "  --trends      Show performance trends"
    echo "  --help        Show this help message"
    echo
    echo "Examples:"
    echo "  $0 --baseline    # Create new baseline"
    echo "  $0 --test        # Run regression tests"
    echo "  $0 --trends      # Show performance trends"
}

# Main execution
case "${1:-help}" in
    --baseline)
        establish_baseline
        ;;
    --test)
        run_regression_tests
        ;;
    --trends)
        show_trends
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