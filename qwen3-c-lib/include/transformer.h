/**
 * @file transformer.h
 * @brief Transformer layer implementation for Qwen3 C inference engine
 * 
 * Implements complete transformer layer with attention, feed-forward network,
 * layer normalization, and residual connections for the Qwen3 architecture.
 */

#ifndef QWEN3_TRANSFORMER_H
#define QWEN3_TRANSFORMER_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "attention.h"
#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Transformer layer configuration
 */
typedef struct {
    size_t hidden_size;       // Hidden dimension size (e.g., 768 for 0.6B)
    size_t intermediate_size; // Feed-forward intermediate size (e.g., 3072 for 0.6B)
    size_t num_attention_heads; // Number of attention heads
    size_t num_key_value_heads; // Number of key/value heads (for GQA)
    size_t head_dim;          // Dimension per attention head
    size_t max_position_embeddings; // Maximum sequence length
    float rms_norm_eps;       // RMS normalization epsilon
    float rope_theta;         // RoPE theta base value
    bool use_sliding_window;  // Whether to use sliding window attention
    size_t sliding_window;    // Sliding window size
} Qwen3TransformerConfig;

/**
 * @brief Transformer layer weights
 */
typedef struct {
    // Attention weights
    float* attention_wq;      // Query projection weight [hidden_size, hidden_size]
    float* attention_wk;      // Key projection weight [hidden_size, num_key_value_heads * head_dim]
    float* attention_wv;      // Value projection weight [hidden_size, num_key_value_heads * head_dim]
    float* attention_wo;      // Output projection weight [hidden_size, hidden_size]
    
    // Feed-forward weights
    float* feed_forward_w1;   // First feed-forward weight [hidden_size, intermediate_size]
    float* feed_forward_w2;   // Second feed-forward weight [intermediate_size, hidden_size]
    float* feed_forward_w3;   // Gate projection weight [hidden_size, intermediate_size]
    
    // Normalization parameters
    float* attention_norm;    // Attention RMS normalization weight [hidden_size]
    float* ffn_norm;          // Feed-forward RMS normalization weight [hidden_size]
} Qwen3TransformerWeights;

/**
 * @brief Transformer layer state
 */
typedef struct {
    Qwen3KVCache kv_cache;    // Key-value cache for attention
    float* attention_output;  // Attention output buffer [seq_len, hidden_size]
    float* ffn_output;        // Feed-forward output buffer [seq_len, hidden_size]
    float* residual;          // Residual connection buffer [seq_len, hidden_size]
} Qwen3TransformerLayer;

/**
 * @brief Complete transformer model
 */
typedef struct {
    Qwen3TransformerConfig config;  // Model configuration
    Qwen3TransformerWeights weights; // Model weights
    Qwen3TransformerLayer* layers;  // Transformer layers
    size_t num_layers;              // Number of transformer layers
    float* input_embeddings;        // Input embeddings buffer
    float* output_logits;           // Output logits buffer
} Qwen3Transformer;

/**
 * @brief Initialize transformer configuration from model parameters
 * @param config Output configuration structure
 * @param hidden_size Hidden dimension size
 * @param intermediate_size Feed-forward intermediate size
 * @param num_attention_heads Number of attention heads
 * @param num_key_value_heads Number of key/value heads
 * @param max_position_embeddings Maximum sequence length
 * @param rms_norm_eps RMS normalization epsilon
 * @param rope_theta RoPE theta base value
 * @return 0 on success, non-zero on error
 */
int qwen3_transformer_config_init(Qwen3TransformerConfig* config,
                                size_t hidden_size,
                                size_t intermediate_size,
                                size_t num_attention_heads,
                                size_t num_key_value_heads,
                                size_t max_position_embeddings,
                                float rms_norm_eps,
                                float rope_theta);

/**
 * @brief Initialize transformer layer
 * @param layer Layer structure to initialize
 * @param config Transformer configuration
 * @param layer_index Index of this layer (0-based)
 * @return 0 on success, non-zero on error
 */
int qwen3_transformer_layer_init(Qwen3TransformerLayer* layer,
                               const Qwen3TransformerConfig* config,
                               size_t layer_index);

/**
 * @brief Free transformer layer memory
 * @param layer Layer structure to free
 */
void qwen3_transformer_layer_free(Qwen3TransformerLayer* layer);

/**
 * @brief Initialize complete transformer model
 * @param transformer Transformer structure to initialize
 * @param config Model configuration
 * @param num_layers Number of transformer layers
 * @return 0 on success, non-zero on error
 */
int qwen3_transformer_init(Qwen3Transformer* transformer,
                         const Qwen3TransformerConfig* config,
                         size_t num_layers);

/**
 * @brief Free transformer model memory
 * @param transformer Transformer structure to free
 */
void qwen3_transformer_free(Qwen3Transformer* transformer);

/**
 * @brief RMS normalization (Root Mean Square normalization)
 * @param input Input tensor [seq_len, hidden_size]
 * @param weight RMS normalization weight [hidden_size]
 * @param output Output tensor [seq_len, hidden_size]
 * @param seq_len Sequence length
 * @param hidden_size Hidden dimension size
 * @param eps Epsilon value for numerical stability
 */
void qwen3_rms_norm(const float* input, const float* weight,
                   float* output, size_t seq_len, size_t hidden_size,
                   float eps);

/**
 * @brief SwiGLU activation function
 * @param input Input tensor [seq_len, intermediate_size]
 * @param gate Gate tensor [seq_len, intermediate_size]
 * @param output Output tensor [seq_len, intermediate_size]
 * @param seq_len Sequence length
 * @param intermediate_size Intermediate dimension size
 */
void qwen3_swiglu(const float* input, const float* gate,
                 float* output, size_t seq_len, size_t intermediate_size);

/**
 * @brief Forward pass through a single transformer layer
 * @param layer Transformer layer
 * @param weights Layer weights
 * @param input Input activations [seq_len, hidden_size]
 * @param output Output activations [seq_len, hidden_size]
 * @param seq_len Sequence length
 * @param pos Position indices [seq_len] (for RoPE)
 * @return 0 on success, non-zero on error
 */
int qwen3_transformer_layer_forward(Qwen3TransformerLayer* layer,
                                  const Qwen3TransformerWeights* weights,
                                  const float* input,
                                  float* output,
                                  size_t seq_len,
                                  const int* pos);

/**
 * @brief Forward pass through complete transformer
 * @param transformer Complete transformer model
 * @param input_tokens Input token indices [seq_len]
 * @param seq_len Sequence length
 * @param pos Position indices [seq_len]
 * @param output_logits Output logits [seq_len, vocab_size]
 * @return 0 on success, non-zero on error
 */
int qwen3_transformer_forward(Qwen3Transformer* transformer,
                            const int* input_tokens,
                            size_t seq_len,
                            const int* pos,
                            float* output_logits);

/**
 * @brief Benchmark transformer layer performance
 * @param seq_len Sequence length to benchmark
 * @param hidden_size Hidden dimension size
 * @param intermediate_size Intermediate dimension size
 * @param num_layers Number of transformer layers
 * @param iterations Number of iterations
 * @return Average time in microseconds per layer
 */
float qwen3_transformer_benchmark(size_t seq_len,
                                size_t hidden_size,
                                size_t intermediate_size,
                                size_t num_layers,
                                size_t iterations);

#ifdef __cplusplus
}
#endif

#endif // QWEN3_TRANSFORMER_H