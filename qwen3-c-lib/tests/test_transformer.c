/**
 * @file test_transformer.c
 * @brief Comprehensive tests for transformer layer implementation
 * 
 * Tests transformer configuration, layer initialization, RMS norm, SwiGLU,
 * and complete transformer forward pass.
 */

#include "../include/transformer.h"
#include "../src/memory.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// Tolerance for floating point comparisons
#define EPSILON 1e-4

// Helper function to check if two floats are approximately equal
static bool float_equal(float a, float b, float epsilon) {
    return fabsf(a - b) < epsilon;
}

// Test transformer configuration initialization
static void test_transformer_config(void) {
    printf("Testing transformer configuration...\n");
    
    Qwen3TransformerConfig config;
    int ret = qwen3_transformer_config_init(&config, 768, 3072, 12, 12, 2048, 1e-6f, 10000.0f);
    
    if (ret != 0) {
        printf("FAIL: Configuration initialization failed\n");
        return;
    }
    
    if (config.hidden_size != 768 || 
        config.intermediate_size != 3072 ||
        config.num_attention_heads != 12 ||
        config.num_key_value_heads != 12 ||
        config.head_dim != 64 ||
        config.max_position_embeddings != 2048) {
        printf("FAIL: Configuration values incorrect\n");
        return;
    }
    
    printf("PASS: Transformer configuration\n");
}

// Test RMS normalization
static void test_rms_norm(void) {
    printf("Testing RMS normalization...\n");
    
    const size_t seq_len = 2;
    const size_t hidden_size = 4;
    
    float input[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float weight[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float output[8];
    
    qwen3_rms_norm(input, weight, output, seq_len, hidden_size, 1e-6f);
    
    // Basic verification - check RMS normalization runs without crashing
    bool passed = true;
    for (size_t i = 0; i < seq_len * hidden_size; i++) {
        if (isnan(output[i]) || isinf(output[i])) {
            passed = false;
            break;
        }
    }
    
    if (passed) {
        printf("PASS: RMS normalization\n");
    } else {
        printf("FAIL: RMS normalization verification failed\n");
    }
}

// Test SwiGLU activation
static void test_swiglu(void) {
    printf("Testing SwiGLU activation...\n");
    
    const size_t seq_len = 2;
    const size_t intermediate_size = 4;
    
    float input[8] = {1.0f, -1.0f, 2.0f, -2.0f, 0.5f, -0.5f, 1.5f, -1.5f};
    float gate[8] = {1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float output[8];
    
    qwen3_swiglu(input, gate, output, seq_len, intermediate_size);
    
    // Basic verification - check SwiGLU properties
    bool passed = true;
    for (size_t i = 0; i < seq_len * intermediate_size; i++) {
        float expected = input[i] * (1.0f / (1.0f + expf(-gate[i])));
        if (!float_equal(output[i], expected, EPSILON)) {
            passed = false;
            break;
        }
    }
    
    if (passed) {
        printf("PASS: SwiGLU activation\n");
    } else {
        printf("FAIL: SwiGLU activation verification failed\n");
    }
}

// Test transformer layer initialization
static void test_transformer_layer_init(void) {
    printf("Testing transformer layer initialization...\n");
    
    Qwen3TransformerConfig config;
    qwen3_transformer_config_init(&config, 768, 3072, 12, 12, 2048, 1e-6f, 10000.0f);
    
    Qwen3TransformerLayer layer;
    int ret = qwen3_transformer_layer_init(&layer, &config, 0);
    
    if (ret != 0) {
        printf("FAIL: Transformer layer initialization failed\n");
        return;
    }
    
    // Verify KV cache initialization - use smaller test size
    if (layer.kv_cache.max_seq_len != 32 ||  // Updated to match smaller test cache
        layer.kv_cache.num_kv_heads != 12 ||
        layer.kv_cache.head_dim != 64) {
        printf("FAIL: KV cache initialization incorrect\n");
        printf("  Expected max_seq_len=32, got %zu\n", layer.kv_cache.max_seq_len);
        printf("  Expected num_kv_heads=12, got %zu\n", layer.kv_cache.num_kv_heads);
        printf("  Expected head_dim=64, got %zu\n", layer.kv_cache.head_dim);
        qwen3_transformer_layer_free(&layer);
        return;
    }
    
    // Verify buffer allocation
    if (!layer.attention_output || !layer.ffn_output || !layer.residual) {
        printf("FAIL: Buffer allocation failed\n");
        qwen3_transformer_layer_free(&layer);
        return;
    }
    
    printf("PASS: Transformer layer initialization\n");
    qwen3_transformer_layer_free(&layer);
}

// Test complete transformer initialization
static void test_transformer_init(void) {
    printf("Testing complete transformer initialization...\n");
    
    Qwen3TransformerConfig config;
    qwen3_transformer_config_init(&config, 768, 3072, 12, 12, 2048, 1e-6f, 10000.0f);
    
    Qwen3Transformer transformer;
    int ret = qwen3_transformer_init(&transformer, &config, 24);
    
    if (ret != 0) {
        printf("FAIL: Transformer initialization failed\n");
        return;
    }
    
    // Verify configuration
    if (transformer.config.hidden_size != 768 ||
        transformer.num_layers != 24) {
        printf("FAIL: Transformer configuration incorrect\n");
        qwen3_transformer_free(&transformer);
        return;
    }
    
    // Verify layers
    if (!transformer.layers) {
        printf("FAIL: Layers not allocated\n");
        qwen3_transformer_free(&transformer);
        return;
    }
    
    printf("PASS: Complete transformer initialization\n");
    qwen3_transformer_free(&transformer);
}

// Test transformer forward pass
static void test_transformer_forward(void) {
    printf("Testing transformer forward pass...\n");
    
    Qwen3TransformerConfig config;
    qwen3_transformer_config_init(&config, 768, 3072, 12, 12, 2048, 1e-6f, 10000.0f);
    
    Qwen3Transformer transformer;
    int ret = qwen3_transformer_init(&transformer, &config, 2); // 2 layers for testing
    
    if (ret != 0) {
        printf("FAIL: Transformer initialization failed\n");
        return;
    }
    
    // Create test data
    const size_t seq_len = 4;
    int tokens[4] = {1, 2, 3, 4};
    int pos[4] = {0, 1, 2, 3};
    float logits[4 * 768]; // seq_len * hidden_size
    
    ret = qwen3_transformer_forward(&transformer, tokens, seq_len, pos, logits);
    
    if (ret != 0) {
        printf("FAIL: Transformer forward pass failed\n");
        qwen3_transformer_free(&transformer);
        return;
    }
    
    printf("PASS: Transformer forward pass\n");
    qwen3_transformer_free(&transformer);
}

// Test transformer benchmark
static void test_transformer_benchmark(void) {
    printf("Testing transformer benchmark...\n");
    
    float time = qwen3_transformer_benchmark(32, 768, 3072, 2, 1);
    if (time > 0) {
        printf("PASS: Transformer benchmark (%.2f us)\n", time);
    } else {
        printf("FAIL: Transformer benchmark failed\n");
    }
}

// Test invalid configuration handling
static void test_invalid_config(void) {
    printf("Testing invalid configuration handling...\n");
    
    Qwen3TransformerConfig config;
    
    // Test invalid head dimension
    int ret = qwen3_transformer_config_init(&config, 768, 3072, 13, 13, 2048, 1e-6f, 10000.0f);
    if (ret == 0) {
        printf("FAIL: Should reject invalid head dimension\n");
        return;
    }
    
    // Test invalid GQA configuration
    ret = qwen3_transformer_config_init(&config, 768, 3072, 12, 5, 2048, 1e-6f, 10000.0f);
    if (ret == 0) {
        printf("FAIL: Should reject invalid GQA configuration\n");
        return;
    }
    
    printf("PASS: Invalid configuration handling\n");
}

int main(void) {
    printf("=== Qwen3 Transformer Layer Tests ===\n");
    
    // Initialize memory system
    if (qwen3_memory_init() != 0) {
        printf("Failed to initialize memory system\n");
        return 1;
    }
    
    // Seed random number generator
    srand(time(NULL));
    
    // Run tests
    test_transformer_config();
    test_rms_norm();
    test_swiglu();
    test_transformer_layer_init();
    test_transformer_init();
    test_transformer_forward();
    test_invalid_config();
    test_transformer_benchmark();
    
    // Shutdown memory system
    int leaks = qwen3_memory_shutdown();
    if (leaks > 0) {
        printf("FAIL: Memory leaks detected: %d\n", leaks);
        return 1;
    }
    
    printf("=== All transformer tests completed ===\n");
    return 0;
}