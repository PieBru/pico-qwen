/**
 * @file attention.h
 * @brief Multi-head attention implementation for Qwen3 C inference engine
 * 
 * Implements scaled dot-product attention with causal masking, grouped query
 * attention (GQA), and memory-efficient attention computation.
 */

#ifndef QWEN3_ATTENTION_H
#define QWEN3_ATTENTION_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

// Forward declarations
struct Qwen3Model;
struct Qwen3Tensor;

/**
 * @brief Attention configuration parameters
 */
typedef struct {
    size_t seq_len;           // Sequence length
    size_t head_dim;          // Dimension per attention head
    size_t num_heads;         // Number of attention heads
    size_t num_kv_heads;      // Number of key/value heads (for GQA)
    float scale;              // Scaling factor for attention scores (1/sqrt(head_dim))
    bool causal;              // Whether to apply causal masking
    bool use_sliding_window;  // Whether to use sliding window attention
    size_t window_size;       // Sliding window size if enabled
} Qwen3AttentionConfig;

/**
 * @brief KV cache entry for efficient attention computation
 */
typedef struct {
    float* k_cache;           // Key cache [max_seq_len, num_kv_heads * head_dim]
    float* v_cache;           // Value cache [max_seq_len, num_kv_heads * head_dim]
    size_t cache_size;        // Current cache size (number of tokens)
    size_t max_seq_len;       // Maximum sequence length
    size_t head_dim;          // Head dimension
    size_t num_kv_heads;      // Number of key/value heads
} Qwen3KVCache;

/**
 * @brief Initialize KV cache
 * @param cache Cache structure to initialize
 * @param max_seq_len Maximum sequence length
 * @param num_kv_heads Number of key/value heads
 * @param head_dim Dimension per head
 * @return 0 on success, non-zero on error
 */
int qwen3_kv_cache_init(Qwen3KVCache* cache, size_t max_seq_len, 
                       size_t num_kv_heads, size_t head_dim);

/**
 * @brief Free KV cache memory
 * @param cache Cache structure to free
 */
void qwen3_kv_cache_free(Qwen3KVCache* cache);

/**
 * @brief Clear KV cache (reset to empty state)
 * @param cache Cache structure to clear
 */
void qwen3_kv_cache_clear(Qwen3KVCache* cache);

/**
 * @brief Append new key/value vectors to cache
 * @param cache KV cache structure
 * @param k New key vectors [batch_size, num_kv_heads * head_dim]
 * @param v New value vectors [batch_size, num_kv_heads * head_dim]
 * @param batch_size Number of sequences in batch
 * @return 0 on success, non-zero on error
 */
int qwen3_kv_cache_append(Qwen3KVCache* cache, const float* k, const float* v,
                         size_t batch_size);

/**
 * @brief Get key/value vectors from cache
 * @param cache KV cache structure
 * @param seq_len Sequence length to retrieve
 * @param k_out Output key vectors [seq_len, num_kv_heads * head_dim]
 * @param v_out Output value vectors [seq_len, num_kv_heads * head_dim]
 * @return 0 on success, non-zero on error
 */
int qwen3_kv_cache_get(const Qwen3KVCache* cache, size_t seq_len,
                      float* k_out, float* v_out);

/**
 * @brief Scaled dot-product attention
 * @param q Query matrix [seq_len, num_heads * head_dim]
 * @param k Key matrix [seq_len, num_kv_heads * head_dim]
 * @param v Value matrix [seq_len, num_kv_heads * head_dim]
 * @param mask Optional attention mask [seq_len, seq_len] (NULL for none)
 * @param output Output matrix [seq_len, num_heads * head_dim]
 * @param config Attention configuration
 * @return 0 on success, non-zero on error
 */
int qwen3_attention_sdpa(const float* q, const float* k, const float* v,
                        const float* mask, float* output,
                        const Qwen3AttentionConfig* config);

/**
 * @brief Multi-head attention with KV cache
 * @param q Query matrix [batch_size, seq_len, num_heads * head_dim]
 * @param k Key matrix [batch_size, seq_len, num_kv_heads * head_dim]
 * @param v Value matrix [batch_size, seq_len, num_kv_heads * head_dim]
 * @param kv_cache KV cache structure (updated in-place)
 * @param output Output matrix [batch_size, seq_len, num_heads * head_dim]
 * @param config Attention configuration
 * @return 0 on success, non-zero on error
 */
int qwen3_attention_mha(const float* q, const float* k, const float* v,
                       Qwen3KVCache* kv_cache, float* output,
                       const Qwen3AttentionConfig* config);

/**
 * @brief Grouped Query Attention (GQA) implementation
 * @param q Query matrix [batch_size, seq_len, num_heads * head_dim]
 * @param k Key matrix [batch_size, seq_len, num_kv_heads * head_dim]
 * @param v Value matrix [batch_size, seq_len, num_kv_heads * head_dim]
 * @param kv_cache KV cache structure
 * @param output Output matrix [batch_size, seq_len, num_heads * head_dim]
 * @param config Attention configuration
 * @return 0 on success, non-zero on error
 */
int qwen3_attention_gqa(const float* q, const float* k, const float* v,
                       Qwen3KVCache* kv_cache, float* output,
                       const Qwen3AttentionConfig* config);

/**
 * @brief Generate causal attention mask
 * @param mask Output mask matrix [seq_len, seq_len]
 * @param seq_len Sequence length
 * @param sliding_window Whether to use sliding window attention
 * @param window_size Window size for sliding window attention
 */
void qwen3_attention_causal_mask(float* mask, size_t seq_len, 
                                bool sliding_window, size_t window_size);

/**
 * @brief Apply rotary positional embeddings (RoPE) to queries and keys
 * @param q Query matrix [batch_size, seq_len, num_heads * head_dim]
 * @param k Key matrix [batch_size, seq_len, num_kv_heads * head_dim]
 * @param pos Position indices [batch_size, seq_len]
 * @param seq_len Sequence length
 * @param head_dim Dimension per attention head
 * @param theta_base Base theta value for RoPE
 * @return 0 on success, non-zero on error
 */
int qwen3_attention_rope(float* q, float* k, const int* pos,
                        size_t seq_len, size_t head_dim, float theta_base);

/**
 * @brief Compute attention weights from scores
 * @param scores Input attention scores [seq_len, seq_len]
 * @param weights Output attention weights [seq_len, seq_len]
 * @param seq_len Sequence length
 * @param causal Whether to apply causal masking
 * @param temperature Temperature for softmax scaling
 * @return 0 on success, non-zero on error
 */
int qwen3_attention_weights(const float* scores, float* weights,
                           size_t seq_len, bool causal, float temperature);

/**
 * @brief Optimize attention computation for memory efficiency
 * @param config Attention configuration (updated in-place)
 * @param available_memory Available memory in bytes
 * @return Optimized configuration
 */
Qwen3AttentionConfig qwen3_attention_optimize(const Qwen3AttentionConfig* config,
                                            size_t available_memory);

/**
 * @brief Benchmark attention computation performance
 * @param seq_len Sequence length to benchmark
 * @param num_heads Number of attention heads
 * @param head_dim Dimension per head
 * @param iterations Number of iterations
 * @return Average time in microseconds
 */
float qwen3_attention_benchmark(size_t seq_len, size_t num_heads,
                              size_t head_dim, size_t iterations);

#ifdef __cplusplus
}
#endif

#endif // QWEN3_ATTENTION_H