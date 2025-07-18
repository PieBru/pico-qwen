/**
 * @file inference_config.h
 * @brief Inference configuration structures for Qwen3 C inference engine
 */

#ifndef QWEN3_INFERENCE_CONFIG_H
#define QWEN3_INFERENCE_CONFIG_H

#include <stdint.h>

/**
 * @brief Inference configuration structure
 */
typedef struct {
    uint32_t vocab_size;      /**< Vocabulary size */
    uint32_t hidden_size;     /**< Hidden dimension */
    uint32_t max_seq_len;     /**< Maximum sequence length */
    uint32_t max_new_tokens;  /**< Maximum new tokens to generate */
    float temperature;        /**< Sampling temperature */
    float top_p;              /**< Top-p (nucleus) sampling */
    uint32_t top_k;           /**< Top-k sampling */
    uint32_t eos_token_id;    /**< End-of-sequence token ID */
    uint32_t* seed;           /**< Random seed (NULL for random) */
} Qwen3InferenceConfigInternal;

/**
 * @brief Inference information structure
 */
typedef struct {
    uint32_t vocab_size;
    uint32_t hidden_size;
    uint32_t num_layers;
    uint32_t num_heads;
    uint32_t num_kv_heads;
    uint32_t max_seq_len;
    float temperature;
    float top_p;
    uint32_t top_k;
} Qwen3InferenceInfoInternal;

#endif // QWEN3_INFERENCE_CONFIG_H