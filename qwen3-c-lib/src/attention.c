/**
 * @file attention.c
 * @brief Multi-head attention implementation for Qwen3 C inference engine
 * 
 * Implements scaled dot-product attention with causal masking, KV caching,
 * and grouped query attention (GQA) for efficient inference.
 */

#include "../include/attention.h"
#include "../include/matrix.h"
#include "../include/tensor.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>

// Constants
#define QWEN3_ATTENTION_MAX_SEQ_LEN 32768
#define QWEN3_ATTENTION_MAX_HEADS 128

/**
 * @brief Initialize KV cache
 */
int qwen3_kv_cache_init(Qwen3KVCache* cache, size_t max_seq_len, 
                       size_t num_kv_heads, size_t head_dim) {
    if (!cache || max_seq_len == 0 || num_kv_heads == 0 || head_dim == 0) {
        return -1;
    }
    
    size_t cache_size = max_seq_len * num_kv_heads * head_dim;
    
    cache->k_cache = (float*)calloc(cache_size, sizeof(float));
    cache->v_cache = (float*)calloc(cache_size, sizeof(float));
    
    if (!cache->k_cache || !cache->v_cache) {
        free(cache->k_cache);
        free(cache->v_cache);
        return -1;
    }
    
    cache->cache_size = 0;
    cache->max_seq_len = max_seq_len;
    cache->head_dim = head_dim;
    cache->num_kv_heads = num_kv_heads;
    
    return 0;
}

/**
 * @brief Free KV cache memory
 */
void qwen3_kv_cache_free(Qwen3KVCache* cache) {
    if (!cache) return;
    
    free(cache->k_cache);
    free(cache->v_cache);
    cache->k_cache = NULL;
    cache->v_cache = NULL;
    cache->cache_size = 0;
}

/**
 * @brief Clear KV cache
 */
void qwen3_kv_cache_clear(Qwen3KVCache* cache) {
    if (!cache) return;
    cache->cache_size = 0;
}

/**
 * @brief Append new key/value vectors to cache
 */
int qwen3_kv_cache_append(Qwen3KVCache* cache, const float* k, const float* v,
                         size_t batch_size) {
    if (!cache || !k || !v || batch_size == 0) {
        return -1;
    }
    
    if (cache->cache_size + batch_size > cache->max_seq_len) {
        return -1;
    }
    
    size_t block_size = cache->num_kv_heads * cache->head_dim;
    
    // Copy new keys and values to cache
    for (size_t i = 0; i < batch_size; i++) {
        size_t cache_offset = cache->cache_size * block_size;
        size_t input_offset = i * block_size;
        
        memcpy(cache->k_cache + cache_offset, 
               k + input_offset, 
               block_size * sizeof(float));
        memcpy(cache->v_cache + cache_offset, 
               v + input_offset, 
               block_size * sizeof(float));
        
        cache->cache_size++;
    }
    
    return 0;
}

/**
 * @brief Get key/value vectors from cache
 */
int qwen3_kv_cache_get(const Qwen3KVCache* cache, size_t seq_len,
                      float* k_out, float* v_out) {
    if (!cache || !k_out || !v_out || seq_len > cache->cache_size) {
        return -1;
    }
    
    size_t block_size = cache->num_kv_heads * cache->head_dim;
    
    memcpy(k_out, cache->k_cache, seq_len * block_size * sizeof(float));
    memcpy(v_out, cache->v_cache, seq_len * block_size * sizeof(float));
    
    return 0;
}

/**
 * @brief Generate causal attention mask
 */
void qwen3_attention_causal_mask(float* mask, size_t seq_len, 
                                bool sliding_window, size_t window_size) {
    if (!mask) return;
    
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j < seq_len; j++) {
            if (sliding_window) {
                mask[i * seq_len + j] = (j <= i && j + window_size > i) ? 0.0f : -INFINITY;
            } else {
                mask[i * seq_len + j] = (j <= i) ? 0.0f : -INFINITY;
            }
        }
    }
}

/**
 * @brief Apply rotary positional embeddings (RoPE)
 */
int qwen3_attention_rope(float* q, float* k, const int* pos,
                        size_t seq_len, size_t head_dim, float theta_base) {
    if (!q || !k || !pos || seq_len == 0 || head_dim == 0) {
        return -1;
    }
    
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t d = 0; d < head_dim; d += 2) {
            float freq = 1.0f / powf(theta_base, (float)d / (float)head_dim);
            float angle = pos[i] * freq;
            float cos_val = cosf(angle);
            float sin_val = sinf(angle);
            
            size_t idx0 = i * head_dim + d;
            size_t idx1 = idx0 + 1;
            
            if (idx1 < seq_len * head_dim) {
                // Apply rotation to queries
                float q0 = q[idx0];
                float q1 = q[idx1];
                q[idx0] = q0 * cos_val - q1 * sin_val;
                q[idx1] = q0 * sin_val + q1 * cos_val;
                
                // Apply rotation to keys
                float k0 = k[idx0];
                float k1 = k[idx1];
                k[idx0] = k0 * cos_val - k1 * sin_val;
                k[idx1] = k0 * sin_val + k1 * cos_val;
            }
        }
    }
    
    return 0;
}

/**
 * @brief Compute attention weights from scores
 */
int qwen3_attention_weights(const float* scores, float* weights,
                           size_t seq_len, bool causal, float temperature) {
    if (!scores || !weights || seq_len == 0) {
        return -1;
    }
    
    // Apply temperature scaling
    float scale = 1.0f / temperature;
    
    for (size_t i = 0; i < seq_len; i++) {
        // Find max for numerical stability
        float max_score = -INFINITY;
        for (size_t j = 0; j < seq_len; j++) {
            if (!causal || j <= i) {
                float score = scores[i * seq_len + j] * scale;
                max_score = fmaxf(max_score, score);
            }
        }
        
        // Compute softmax
        float sum_exp = 0.0f;
        for (size_t j = 0; j < seq_len; j++) {
            if (!causal || j <= i) {
                float score = scores[i * seq_len + j] * scale;
                float exp_val = expf(score - max_score);
                weights[i * seq_len + j] = exp_val;
                sum_exp += exp_val;
            } else {
                weights[i * seq_len + j] = 0.0f;
            }
        }
        
        // Normalize
        for (size_t j = 0; j < seq_len; j++) {
            weights[i * seq_len + j] /= sum_exp;
        }
    }
    
    return 0;
}

/**
 * @brief Scaled dot-product attention
 */
int qwen3_attention_sdpa(const float* q, const float* k, const float* v,
                        const float* mask, float* output,
                        const Qwen3AttentionConfig* config) {
    if (!q || !k || !v || !output || !config) {
        return -1;
    }
    
    size_t seq_len = config->seq_len;
    size_t head_dim = config->head_dim;
    size_t num_heads = config->num_heads;
    size_t num_kv_heads = config->num_kv_heads;
    
    // Check GQA compatibility
    if (num_heads % num_kv_heads != 0) {
        return -1;
    }
    
    size_t group_size = num_heads / num_kv_heads;
    
    // Allocate temporary memory
    float* scores = (float*)calloc(seq_len * seq_len, sizeof(float));
    float* weights = (float*)calloc(seq_len * seq_len, sizeof(float));
    
    if (!scores || !weights) {
        free(scores);
        free(weights);
        return -1;
    }
    
    // Process each head
    for (size_t h = 0; h < num_heads; h++) {
        size_t kv_head = h / group_size;
        
        // Compute attention scores: Q * K^T / sqrt(head_dim)
        for (size_t i = 0; i < seq_len; i++) {
            for (size_t j = 0; j < seq_len; j++) {
                float dot_product = 0.0f;
                
                for (size_t d = 0; d < head_dim; d++) {
                    size_t q_idx = (i * num_heads + h) * head_dim + d;
                    size_t k_idx = (j * num_kv_heads + kv_head) * head_dim + d;
                    dot_product += q[q_idx] * k[k_idx];
                }
                
                scores[i * seq_len + j] = dot_product * config->scale;
            }
        }
        
        // Apply mask if provided
        if (mask) {
            for (size_t i = 0; i < seq_len * seq_len; i++) {
                scores[i] += mask[i];
            }
        } else if (config->causal) {
            // Apply causal mask
            qwen3_attention_causal_mask(weights, seq_len, 
                                      config->use_sliding_window, 
                                      config->window_size);
            for (size_t i = 0; i < seq_len * seq_len; i++) {
                scores[i] += weights[i];
            }
        }
        
        // Compute attention weights
        qwen3_attention_weights(scores, weights, seq_len, config->causal, 1.0f);
        
        // Compute attention output: weights * V
        for (size_t i = 0; i < seq_len; i++) {
            for (size_t d = 0; d < head_dim; d++) {
                float sum = 0.0f;
                
                for (size_t j = 0; j < seq_len; j++) {
                    size_t v_idx = (j * num_kv_heads + kv_head) * head_dim + d;
                    sum += weights[i * seq_len + j] * v[v_idx];
                }
                
                size_t out_idx = (i * num_heads + h) * head_dim + d;
                output[out_idx] = sum;
            }
        }
    }
    
    free(scores);
    free(weights);
    
    return 0;
}

/**
 * @brief Multi-head attention with KV cache
 */
int qwen3_attention_mha(const float* q, const float* k, const float* v,
                       Qwen3KVCache* kv_cache, float* output,
                       const Qwen3AttentionConfig* config) {
    if (!q || !k || !v || !kv_cache || !output || !config) {
        return -1;
    }
    
    size_t head_dim = config->head_dim;
    size_t num_kv_heads = config->num_kv_heads;
    
    // Append new key/value vectors to cache
    if (qwen3_kv_cache_append(kv_cache, k, v, 1) != 0) {
        return -1;
    }
    
    // Get full key/value vectors from cache
    float* full_k = (float*)calloc(kv_cache->cache_size * num_kv_heads * head_dim, sizeof(float));
    float* full_v = (float*)calloc(kv_cache->cache_size * num_kv_heads * head_dim, sizeof(float));
    
    if (!full_k || !full_v) {
        free(full_k);
        free(full_v);
        return -1;
    }
    
    qwen3_kv_cache_get(kv_cache, kv_cache->cache_size, full_k, full_v);
    
    // Create updated config with actual cache size
    Qwen3AttentionConfig cache_config = *config;
    cache_config.seq_len = kv_cache->cache_size;
    
    // Compute attention using cached keys/values
    int result = qwen3_attention_sdpa(q, full_k, full_v, NULL, output, &cache_config);
    
    free(full_k);
    free(full_v);
    
    return result;
}

/**
 * @brief Grouped Query Attention (GQA)
 */
int qwen3_attention_gqa(const float* q, const float* k, const float* v,
                       Qwen3KVCache* kv_cache, float* output,
                       const Qwen3AttentionConfig* config) {
    // GQA is just MHA with different head counts
    return qwen3_attention_mha(q, k, v, kv_cache, output, config);
}

/**
 * @brief Optimize attention computation for memory efficiency
 */
Qwen3AttentionConfig qwen3_attention_optimize(const Qwen3AttentionConfig* config,
                                            size_t available_memory) {
    Qwen3AttentionConfig optimized = *config;
    
    // Estimate memory usage
    size_t q_size = config->seq_len * config->num_heads * config->head_dim * sizeof(float);
    size_t k_size = config->seq_len * config->num_kv_heads * config->head_dim * sizeof(float);
    size_t v_size = config->seq_len * config->num_kv_heads * config->head_dim * sizeof(float);
    size_t scores_size = config->seq_len * config->seq_len * sizeof(float);
    
    size_t total_needed = q_size + k_size + v_size + scores_size;
    
    // If memory is tight, reduce sequence length or use windowed attention
    if (total_needed > available_memory) {
        optimized.use_sliding_window = true;
        optimized.window_size = 1024; // Default window size
    }
    
    return optimized;
}

/**
 * @brief Benchmark attention computation performance
 */
float qwen3_attention_benchmark(size_t seq_len, size_t num_heads,
                              size_t head_dim, size_t iterations) {
    Qwen3AttentionConfig config = {
        .seq_len = seq_len,
        .head_dim = head_dim,
        .num_heads = num_heads,
        .num_kv_heads = num_heads,
        .scale = 1.0f / sqrtf((float)head_dim),
        .causal = true,
        .use_sliding_window = false,
        .window_size = 0
    };
    
    size_t total_elements = seq_len * num_heads * head_dim;
    float* q = (float*)calloc(total_elements, sizeof(float));
    float* k = (float*)calloc(total_elements, sizeof(float));
    float* v = (float*)calloc(total_elements, sizeof(float));
    float* output = (float*)calloc(total_elements, sizeof(float));
    
    if (!q || !k || !v || !output) {
        free(q); free(k); free(v); free(output);
        return -1.0f;
    }
    
    // Initialize with random data
    for (size_t i = 0; i < total_elements; i++) {
        q[i] = ((float)rand() / RAND_MAX) - 0.5f;
        k[i] = ((float)rand() / RAND_MAX) - 0.5f;
        v[i] = ((float)rand() / RAND_MAX) - 0.5f;
    }
    
    clock_t start = clock();
    
    for (size_t i = 0; i < iterations; i++) {
        qwen3_attention_sdpa(q, k, v, NULL, output, &config);
    }
    
    clock_t end = clock();
    float time_us = (float)(end - start) * 1000000.0f / CLOCKS_PER_SEC / iterations;
    
    free(q); free(k); free(v); free(output);
    
    return time_us;
}