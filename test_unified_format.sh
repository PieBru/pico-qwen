#!/bin/bash

# Test script to verify unified binary format compatibility
set -e

echo "=== Qwen3 Unified Binary Format Test ==="
echo

# Test 1: Create a test model with Rust export
echo "📦 Creating test model with Rust export..."
cargo run --release -p qwen3-cli -- export \
    ./test-data/qwen3-0.6B \
    ./test-model-unified.bin \
    --group-size 64 \
    --rope-theta 10000.0

echo "✅ Test model created successfully"
echo

# Test 2: Verify C engine can read the format
echo "🔍 Testing C engine compatibility..."
cd qwen3-c-lib
./test_model_loading ../test-model-unified.bin 2>/dev/null || {
    echo "❌ C engine failed to read unified format"
    exit 1
}
cd ..
echo "✅ C engine successfully reads unified format"
echo

# Test 3: Verify Rust engine can read the format
echo "🔍 Testing Rust engine compatibility..."
cargo run --release -p qwen3-cli -- info ./test-model-unified.bin || {
    echo "❌ Rust engine failed to read unified format"
    exit 1
}
echo "✅ Rust engine successfully reads unified format"
echo

echo "=== All tests passed! ==="
echo "✅ Unified binary format is working correctly"
echo "✅ Both C and Rust engines use the same .bin format"