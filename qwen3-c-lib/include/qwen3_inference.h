/**
 * @file qwen3_inference.h
 * @brief High-performance C inference engine for Qwen3 models
 * @details This API provides a pure C implementation optimized for maximum CPU performance
 *          using advanced SIMD instructions (AVX2, AVX-512, NEON, SVE).
 */

#ifndef QWEN3_INFERENCE_H
#define QWEN3_INFERENCE_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Configuration structure for inference parameters
 */
typedef struct {
    const char* checkpoint_path;  /**< Path to the model checkpoint file */
    float temperature;            /**< Sampling temperature [0.0, inf) */
    float topp;                   /**< Top-p (nucleus) sampling [0.0, 1.0] */
    uint32_t ctx_length;          /**< Context window length, 0 = use model max */
    const char* mode;             /**< Mode: "chat" or "generate" */
    const char* prompt;           /**< Input prompt (can be NULL) */
    const char* system_prompt;    /**< System prompt for chat mode (can be NULL) */
    bool enable_thinking;         /**< Enable reasoning mode */
    uint64_t seed;                /**< Random seed for reproducibility */
} Qwen3Config;

/**
 * @brief Model configuration structure
 */
typedef struct {
    uint32_t vocab_size;    /**< Vocabulary size */
    uint32_t dim;          /**< Model dimension */
    uint32_t hidden_dim;   /**< Feed-forward hidden dimension */
    uint32_t n_layers;     /**< Number of transformer layers */
    uint32_t n_heads;      /**< Number of attention heads */
    uint32_t n_kv_heads;   /**< Number of key/value heads (for GQA) */
    uint32_t max_seq_len;  /**< Maximum sequence length */
    float rope_theta;      /**< RoPE base frequency */
} Qwen3ModelConfig;

/**
 * @brief Opaque model handle
 */
typedef struct Qwen3Model Qwen3Model;

/**
 * @brief Extended model loading options
 */
typedef struct {
    const char* checkpoint_path;  /**< Path to the .bin model file */
    uint32_t context_length;      /**< Context window length (0 for model default) */
    bool validate_weights;        /**< Validate model weights on load */
    bool use_memory_pool;         /**< Use memory pool for allocations */
} Qwen3LoadOptions;

/**
 * @brief Load a model from checkpoint
 * @param checkpoint_path Path to the .bin model file
 * @param ctx_length Context window length (0 for model default)
 * @return Model handle or NULL on error
 */
Qwen3Model* qwen3_model_load(const char* checkpoint_path, uint32_t ctx_length);

/**
 * @brief Load a model with extended options
 * @param options Extended loading options
 * @return Model handle or NULL on error
 */
Qwen3Model* qwen3_model_load_ex(const Qwen3LoadOptions* options);

/**
 * @brief Free a loaded model
 * @param model Model handle to free
 */
void qwen3_model_free(Qwen3Model* model);

/**
 * @brief Run chat mode inference
 * @param model Loaded model handle
 * @param config Inference configuration
 * @return 0 on success, non-zero on error
 */
int qwen3_inference_chat(Qwen3Model* model, const Qwen3Config* config);

/**
 * @brief Run generation mode inference
 * @param model Loaded model handle
 * @param config Inference configuration
 * @return 0 on success, non-zero on error
 */
int qwen3_inference_generate(Qwen3Model* model, const Qwen3Config* config);

/**
 * @brief Get model configuration
 * @param model Loaded model handle
 * @return Pointer to model configuration (NULL if model is invalid)
 */
const Qwen3ModelConfig* qwen3_model_get_config(const Qwen3Model* model);

/**
 * @brief Get last error message
 * @return Error message string (thread-local, valid until next API call)
 */
const char* qwen3_get_last_error(void);

/**
 * @brief CPU feature detection structure
 */
typedef struct {
    bool has_avx2;          /**< AVX2 support */
    bool has_avx512f;       /**< AVX-512 foundation support */
    bool has_avx512vl;      /**< AVX-512 vector length support */
    bool has_avx512vnni;    /**< AVX-512 VNNI support */
    bool has_fma3;          /**< FMA3 support */
    bool has_neon;          /**< ARM NEON support */
    bool has_sve;           /**< ARM SVE support */
    int cache_line_size;    /**< Cache line size in bytes */
    int l1_cache_size;      /**< L1 cache size in KB */
    int l2_cache_size;      /**< L2 cache size in KB */
} Qwen3CpuFeatures;

/**
 * @brief Detect CPU features
 * @param features Pointer to store detected features
 * @return 0 on success
 */
int qwen3_detect_cpu_features(Qwen3CpuFeatures* features);

/**
 * @brief Enable/disable SIMD optimization
 * @param features CPU features to enable
 * @return 0 on success
 */
int qwen3_enable_simd(const Qwen3CpuFeatures* features);

#ifdef __cplusplus
}
#endif

#endif // QWEN3_INFERENCE_H