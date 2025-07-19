/**
 * @file model.c
 * @brief Model loading and management for Qwen3 C inference engine
 * 
 * Implements loading of Qwen3 models from binary checkpoint files with validation,
 * memory mapping, and efficient weight loading for maximum CPU performance.
 */

#include "../include/qwen3_inference.h"
#include "../include/model_internal.h"
#include "memory.h"
#include "../include/tensor.h"
#include "../include/tokenizer.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>

// Model file format constants - Qwen3 format with metadata
#define QWEN3_MAGIC 0x5157454E  // "QWEN" in little-endian
#define QWEN3_VERSION 1
#define MAX_MODEL_SIZE (1024LL * 1024LL * 1024LL * 10LL)  // 10GB max

// Internal model structure  
struct Qwen3Model {
    Qwen3ModelConfig config;
    void* mapped_file;
    size_t file_size;
    
    // Model weights (quantized)
    Qwen3QuantizedTensor token_embedding;      // [vocab_size, dim]
    Qwen3Tensor* final_norm;                   // [dim] - RMS final weight
    
    // Layer weights (arrays of quantized tensors)
    Qwen3QuantizedTensor* attn_norm;           // [n_layers, dim]
    Qwen3QuantizedTensor* ffn_norm;            // [n_layers, dim]
    
    // Attention weights
    Qwen3QuantizedTensor* wq;                  // [n_layers, dim, dim]
    Qwen3QuantizedTensor* wk;                  // [n_layers, dim, n_kv_heads * head_dim]
    Qwen3QuantizedTensor* wv;                  // [n_layers, dim, n_kv_heads * head_dim]
    Qwen3QuantizedTensor* wo;                  // [n_layers, dim, dim]
    
    // Feed-forward weights
    Qwen3QuantizedTensor* w1;                  // [n_layers, hidden_dim, dim]
    Qwen3QuantizedTensor* w2;                  // [n_layers, dim, hidden_dim]
    Qwen3QuantizedTensor* w3;                  // [n_layers, hidden_dim, dim]
    
    // Classifier weights (optional)
    Qwen3QuantizedTensor* classifier;          // [vocab_size, dim]
    
    // Tokenizer data
    Qwen3Tokenizer* tokenizer;
    
    // Tokenizer arrays
    char** vocab;
    float* vocab_scores;
    uint32_t vocab_size;
    
    // Memory management
    Qwen3MemoryArena* weights_arena;
    Qwen3MemoryArena* tokenizer_arena;
};

// Thread-local error storage
static __thread char last_error[256];

static void set_error(const char* format, ...) {
    va_list args;
    va_start(args, format);
    vsnprintf(last_error, sizeof(last_error), format, args);
    va_end(args);
}

const char* qwen3_get_last_error_internal(void) {
    return last_error[0] ? last_error : "No error";
}

static bool read_uint32(FILE* file, uint32_t* value) {
    if (fread(value, sizeof(uint32_t), 1, file) != 1) {
        set_error("Failed to read uint32 from file: %s", strerror(errno));
        return false;
    }
    return true;
}


static bool read_float(FILE* file, float* value) {
    if (fread(value, sizeof(float), 1, file) != 1) {
        set_error("Failed to read float from file: %s", strerror(errno));
        return false;
    }
    return true;
}


static bool validate_model_config(const Qwen3ModelConfig* config) {
    if (config->vocab_size == 0 || config->vocab_size > 1000000) {
        set_error("Invalid vocab_size: %u", config->vocab_size);
        return false;
    }
    
    if (config->dim == 0 || config->dim > 16384) {
        set_error("Invalid dim: %u", config->dim);
        return false;
    }
    
    if (config->hidden_dim == 0 || config->hidden_dim > 65536) {
        set_error("Invalid hidden_dim: %u", config->hidden_dim);
        return false;
    }
    
    if (config->n_layers == 0 || config->n_layers > 100) {
        set_error("Invalid n_layers: %u", config->n_layers);
        return false;
    }
    
    if (config->n_heads == 0 || config->n_heads > 128) {
        set_error("Invalid n_heads: %u", config->n_heads);
        return false;
    }
    
    if (config->n_kv_heads == 0 || config->n_kv_heads > config->n_heads) {
        set_error("Invalid n_kv_heads: %u (must be <= n_heads: %u)", 
                  config->n_kv_heads, config->n_heads);
        return false;
    }
    
    if (config->max_seq_len == 0 || config->max_seq_len > 65536) {
        set_error("Invalid max_seq_len: %u", config->max_seq_len);
        return false;
    }
    
    return true;
}

static bool load_model_config(FILE* file, Qwen3ModelConfig* config) {
    uint32_t magic, version;
    
    if (!read_uint32(file, &magic)) return false;
    if (magic != QWEN3_MAGIC) {
        set_error("Invalid magic number: 0x%08X (expected 0x%08X)", magic, QWEN3_MAGIC);
        return false;
    }
    
    if (!read_uint32(file, &version)) return false;
    if (version != QWEN3_VERSION) {
        set_error("Unsupported version: %u (expected %u)", version, QWEN3_VERSION);
        return false;
    }
    
    if (!read_uint32(file, &config->vocab_size)) return false;
    if (!read_uint32(file, &config->dim)) return false;
    if (!read_uint32(file, &config->hidden_dim)) return false;
    if (!read_uint32(file, &config->n_layers)) return false;
    if (!read_uint32(file, &config->n_heads)) return false;
    if (!read_uint32(file, &config->n_kv_heads)) return false;
    if (!read_uint32(file, &config->max_seq_len)) return false;
    if (!read_float(file, &config->rope_theta)) return false;
    
    return validate_model_config(config);
}

static bool load_quantized_tensor(FILE* file, Qwen3QuantizedTensor* tensor, 
                                  Qwen3MemoryArena* arena, uint32_t num_elements, uint32_t group_size) {
    uint64_t num_groups = (num_elements + group_size - 1) / group_size;
    
    // Allocate quantized data
    size_t quantized_size = num_elements * sizeof(int8_t);
    size_t scales_size = num_groups * sizeof(float);
    
    int8_t* quantized_data = qwen3_memory_arena_alloc(arena, quantized_size, QWEN3_MEMORY_ALIGNMENT);
    float* scales = qwen3_memory_arena_alloc(arena, scales_size, QWEN3_MEMORY_ALIGNMENT);
    
    if (!quantized_data || !scales) {
        set_error("Failed to allocate memory for quantized tensor");
        return false;
    }
    
    if (fread(quantized_data, 1, quantized_size, file) != quantized_size) {
        set_error("Failed to read quantized tensor data");
        return false;
    }
    
    if (fread(scales, sizeof(float), num_groups, file) != num_groups) {
        set_error("Failed to read tensor scales");
        return false;
    }
    
    // Initialize tensor
    tensor->data = quantized_data;
    tensor->scales = scales;
    tensor->zero_points = NULL;
    tensor->group_size = group_size;
    tensor->pool = arena;
    
    return true;
}


Qwen3Model* qwen3_model_load_internal(const char* checkpoint_path, uint32_t ctx_length) {
    if (!checkpoint_path) {
        set_error("Checkpoint path is NULL");
        return NULL;
    }
    
    FILE* file = fopen(checkpoint_path, "rb");
    if (!file) {
        set_error("Failed to open checkpoint file '%s': %s", checkpoint_path, strerror(errno));
        return NULL;
    }
    
    // Get file size
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    if (file_size < 0 || file_size > MAX_MODEL_SIZE) {
        set_error("Invalid file size: %ld", file_size);
        fclose(file);
        return NULL;
    }
    
    Qwen3Model* model = calloc(1, sizeof(Qwen3Model));
    if (!model) {
        set_error("Failed to allocate model structure");
        fclose(file);
        return NULL;
    }
    
    // Initialize memory arenas
    model->weights_arena = qwen3_memory_arena_create(1024 * 1024 * 1024);  // 1GB
    model->tokenizer_arena = qwen3_memory_arena_create(64 * 1024 * 1024);   // 64MB
    
    if (!model->weights_arena || !model->tokenizer_arena) {
        set_error("Failed to create memory arenas");
        goto cleanup;
    }
    
    // Load model configuration
    if (!load_model_config(file, &model->config)) {
        goto cleanup;
    }
    
    // Override context length if specified
    if (ctx_length > 0 && ctx_length <= model->config.max_seq_len) {
        model->config.max_seq_len = ctx_length;
    }
    
    // Calculate derived values
    uint32_t head_dim = model->config.dim / model->config.n_heads;
    uint32_t group_size = 64;  // Default from Rust export
    
    // Allocate arrays for layer weights
    size_t layer_weights_size = model->config.n_layers * sizeof(Qwen3QuantizedTensor);
    model->attn_norm = qwen3_memory_arena_alloc(model->weights_arena, layer_weights_size, QWEN3_MEMORY_ALIGNMENT);
    model->ffn_norm = qwen3_memory_arena_alloc(model->weights_arena, layer_weights_size, QWEN3_MEMORY_ALIGNMENT);
    model->wq = qwen3_memory_arena_alloc(model->weights_arena, layer_weights_size, QWEN3_MEMORY_ALIGNMENT);
    model->wk = qwen3_memory_arena_alloc(model->weights_arena, layer_weights_size, QWEN3_MEMORY_ALIGNMENT);
    model->wv = qwen3_memory_arena_alloc(model->weights_arena, layer_weights_size, QWEN3_MEMORY_ALIGNMENT);
    model->wo = qwen3_memory_arena_alloc(model->weights_arena, layer_weights_size, QWEN3_MEMORY_ALIGNMENT);
    model->w1 = qwen3_memory_arena_alloc(model->weights_arena, layer_weights_size, QWEN3_MEMORY_ALIGNMENT);
    model->w2 = qwen3_memory_arena_alloc(model->weights_arena, layer_weights_size, QWEN3_MEMORY_ALIGNMENT);
    model->w3 = qwen3_memory_arena_alloc(model->weights_arena, layer_weights_size, QWEN3_MEMORY_ALIGNMENT);
    
    if (!model->attn_norm || !model->ffn_norm || !model->wq || !model->wk || 
        !model->wv || !model->wo || !model->w1 || !model->w2 || !model->w3) {
        set_error("Failed to allocate layer weight arrays");
        goto cleanup;
    }

    // Load normalization weights (fp32) - skip for now as they are part of transformer layer
    size_t norm_weights_size = model->config.n_layers * model->config.dim * sizeof(float);
    if (fseek(file, norm_weights_size * 2, SEEK_CUR) != 0) {  // attn_norm + ffn_norm
        set_error("Failed to skip norm weights");
        goto cleanup;
    }
    
    size_t qk_norm_size = model->config.n_layers * 2 * head_dim * sizeof(float);  // q_norm + k_norm
    if (fseek(file, qk_norm_size, SEEK_CUR) != 0) {
        set_error("Failed to skip qk norm weights");
        goto cleanup;
    }
    
    size_t final_norm_size = model->config.dim * sizeof(float);
    if (fseek(file, final_norm_size, SEEK_CUR) != 0) {
        set_error("Failed to skip final norm weights");
        goto cleanup;
    }
    
    // Load quantized weights
    for (uint32_t i = 0; i < model->config.n_layers; i++) {
        // WQ: [dim, dim]
        if (!load_quantized_tensor(file, &model->wq[i], model->weights_arena, 
                                 model->config.dim * model->config.dim, group_size)) {
            set_error("Failed to load WQ weights for layer %u", i);
            goto cleanup;
        }
        
        // WK: [dim, model->config.n_kv_heads * head_dim]
        if (!load_quantized_tensor(file, &model->wk[i], model->weights_arena, 
                                 model->config.dim * (model->config.n_kv_heads * head_dim), group_size)) {
            set_error("Failed to load WK weights for layer %u", i);
            goto cleanup;
        }
        
        // WV: [dim, model->config.n_kv_heads * head_dim]
        if (!load_quantized_tensor(file, &model->wv[i], model->weights_arena, 
                                 model->config.dim * (model->config.n_kv_heads * head_dim), group_size)) {
            set_error("Failed to load WV weights for layer %u", i);
            goto cleanup;
        }
        
        // WO: [dim, dim]
        if (!load_quantized_tensor(file, &model->wo[i], model->weights_arena, 
                                 model->config.dim * model->config.dim, group_size)) {
            set_error("Failed to load WO weights for layer %u", i);
            goto cleanup;
        }
        
        // W1: [hidden_dim, dim]
        if (!load_quantized_tensor(file, &model->w1[i], model->weights_arena, 
                                 model->config.hidden_dim * model->config.dim, group_size)) {
            set_error("Failed to load W1 weights for layer %u", i);
            goto cleanup;
        }
        
        // W2: [dim, hidden_dim]
        if (!load_quantized_tensor(file, &model->w2[i], model->weights_arena, 
                                 model->config.dim * model->config.hidden_dim, group_size)) {
            set_error("Failed to load W2 weights for layer %u", i);
            goto cleanup;
        }
        
        // W3: [hidden_dim, dim]
        if (!load_quantized_tensor(file, &model->w3[i], model->weights_arena, 
                                 model->config.hidden_dim * model->config.dim, group_size)) {
            set_error("Failed to load W3 weights for layer %u", i);
            goto cleanup;
        }
    }
    
    // Load token embedding
    if (!load_quantized_tensor(file, &model->token_embedding, model->weights_arena, 
                             model->config.vocab_size * model->config.dim, group_size)) {
        set_error("Failed to load token embedding table");
        goto cleanup;
    }
    
    fclose(file);
    
    printf("Loaded Qwen3 model:\n");
    printf("  Vocab size: %u\n", model->config.vocab_size);
    printf("  Model dim: %u\n", model->config.dim);
    printf("  Hidden dim: %u\n", model->config.hidden_dim);
    printf("  Layers: %u\n", model->config.n_layers);
    printf("  Heads: %u\n", model->config.n_heads);
    printf("  KV Heads: %u\n", model->config.n_kv_heads);
    printf("  Max seq len: %u\n", model->config.max_seq_len);
    printf("  File size: %ld MB\n", file_size / 1024 / 1024);
    
    return model;
    
cleanup:
    if (file) fclose(file);
    if (model) {
        if (model->weights_arena) qwen3_memory_arena_destroy(model->weights_arena);
        if (model->tokenizer_arena) qwen3_memory_arena_destroy(model->tokenizer_arena);
        free(model);
    }
    return NULL;
}

void qwen3_model_free_internal(Qwen3Model* model) {
    if (!model) return;
    
    if (model->weights_arena) {
        qwen3_memory_arena_destroy(model->weights_arena);
    }
    
    if (model->tokenizer_arena) {
        qwen3_memory_arena_destroy(model->tokenizer_arena);
    }
    
    free(model);
}

const Qwen3ModelConfig* qwen3_model_get_config_internal(const Qwen3Model* model) {
    if (!model) {
        set_error("Model is NULL");
        return NULL;
    }
    return &model->config;
}