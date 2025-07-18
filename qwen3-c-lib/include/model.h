/**
 * @file model.h
 * @brief Model loading and management for Qwen3 C inference engine
 *
 * Handles loading of quantized binary models exported from HuggingFace format,
 * including weight tensors, configuration parameters, and validation.
 */

#ifndef QWEN3_MODEL_H
#define QWEN3_MODEL_H

#include <stdint.h>
#include <stdbool.h>
#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

// Model configuration parameters
typedef struct {
    uint32_t dim;              // Model dimension
    uint32_t hidden_dim;       // Feed-forward hidden dimension
    uint32_t n_layers;         // Number of transformer layers
    uint32_t n_heads;          // Number of attention heads
    uint32_t n_kv_heads;       // Number of key/value heads (for GQA)
    uint32_t vocab_size;       // Vocabulary size
    uint32_t max_seq_len;      // Maximum sequence length
    uint32_t head_dim;         // Dimension per attention head
    bool shared_classifier;    // Whether classifier shares embedding weights
    uint32_t group_size;       // Quantization group size
} Qwen3ModelConfig;

// Model weights container
typedef struct {
    // Configuration
    Qwen3ModelConfig config;
    
    // Embedding weights
    Qwen3QuantizedTensor* token_embedding;      // [vocab_size, dim]
    Qwen3Tensor* final_norm;                    // [dim]
    
    // Layer weights (arrays of tensors)
    Qwen3Tensor** attn_norm;                    // [n_layers, dim]
    Qwen3Tensor** ffn_norm;                     // [n_layers, dim]
    
    // Attention weights
    Qwen3QuantizedTensor** wq;                  // [n_layers, dim, dim]
    Qwen3QuantizedTensor** wk;                  // [n_layers, dim, n_kv_heads * head_dim]
    Qwen3QuantizedTensor** wv;                  // [n_layers, dim, n_kv_heads * head_dim]
    Qwen3QuantizedTensor** wo;                  // [n_layers, dim, dim]
    
    // Feed-forward weights
    Qwen3QuantizedTensor** w1;                  // [n_layers, hidden_dim, dim]
    Qwen3QuantizedTensor** w2;                  // [n_layers, dim, hidden_dim]
    Qwen3QuantizedTensor** w3;                  // [n_layers, hidden_dim, dim]
    
    // Classifier weights (optional)
 Qwen3QuantizedTensor* classifier;            // [vocab_size, dim]
    
    // Memory pool for allocations
    void* memory_pool;
} Qwen3Model;

// Model loading options
typedef struct {
    const char* checkpoint_path;  // Path to .bin model file
    uint32_t context_length;      // Context length to use (0 = use model default)
    bool validate_weights;        // Validate weights after loading
    bool use_memory_pool;         // Use memory pool for allocations
} Qwen3LoadOptions;

// Load model from binary checkpoint file
Qwen3Model* qwen3_model_load(const char* checkpoint_path, uint32_t context_length);

// Load model with detailed options
Qwen3Model* qwen3_model_load_ex(const Qwen3LoadOptions* options);

// Free model and all associated resources
void qwen3_model_free(Qwen3Model* model);

// Get model configuration
const Qwen3ModelConfig* qwen3_model_get_config(const Qwen3Model* model);

// Validate model integrity
bool qwen3_model_validate(const Qwen3Model* model);

// Get model info as string
const char* qwen3_model_get_info(const Qwen3Model* model);

// Error handling
const char* qwen3_model_get_last_error(void);

#ifdef __cplusplus
}
#endif

#endif // QWEN3_MODEL_H