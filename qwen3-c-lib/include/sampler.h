/**
 * @file sampler.h
 * @brief Token sampling implementation for Qwen3 C inference engine
 * 
 * Implements temperature scaling, top-p (nucleus) sampling, top-k sampling,
 * and token generation from probability distributions.
 */

#ifndef QWEN3_SAMPLER_H
#define QWEN3_SAMPLER_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Token-probability pair for sorting
 */
typedef struct {
    int token;
    float prob;
} TokenProb;

/**
 * @brief Sampling configuration
 */
typedef struct {
    float temperature;    // Temperature for softmax scaling
    size_t top_k;         // Top-k sampling parameter
    float top_p;          // Top-p (nucleus) sampling parameter
    unsigned int seed;    // Random seed for reproducibility
} Qwen3SamplerConfig;

/**
 * @brief Apply temperature scaling to logits
 * @param logits Input logits [vocab_size]
 * @param vocab_size Vocabulary size
 * @param temperature Temperature parameter (0.1 - 2.0 typical)
 */
void qwen3_sampler_temperature(float* logits, size_t vocab_size, float temperature);

/**
 * @brief Compute softmax probabilities from logits
 * @param logits Input logits [vocab_size]
 * @param probs Output probabilities [vocab_size]
 * @param vocab_size Vocabulary size
 */
void qwen3_sampler_softmax(float* logits, float* probs, size_t vocab_size);

/**
 * @brief Apply top-k filtering to probabilities
 * @param probs Probabilities to filter [vocab_size]
 * @param vocab_size Vocabulary size
 * @param k Number of top tokens to keep
 */
void qwen3_sampler_top_k(float* probs, size_t vocab_size, size_t k);

/**
 * @brief Apply top-p (nucleus) filtering to probabilities
 * @param probs Probabilities to filter [vocab_size]
 * @param vocab_size Vocabulary size
 * @param p Cumulative probability threshold (0.0 - 1.0)
 */
void qwen3_sampler_top_p(float* probs, size_t vocab_size, float p);

/**
 * @brief Sample a token from probability distribution
 * @param probs Probabilities [vocab_size]
 * @param vocab_size Vocabulary size
 * @param seed Random seed pointer
 * @return Sampled token index, or -1 on error
 */
int qwen3_sampler_sample(const float* probs, size_t vocab_size, unsigned int* seed);

/**
 * @brief Sample with temperature, top-k, and top-p
 * @param logits Input logits [vocab_size]
 * @param vocab_size Vocabulary size
 * @param temperature Temperature parameter
 * @param top_k Top-k parameter (0 to disable)
 * @param top_p Top-p parameter (0.0 to disable)
 * @param seed Random seed pointer
 * @return Sampled token index, or -1 on error
 */
int qwen3_sampler_sample_token(const float* logits, size_t vocab_size,
                             float temperature, size_t top_k, float top_p,
                             unsigned int* seed);

/**
 * @brief Benchmark sampling performance
 * @param vocab_size Vocabulary size
 * @param iterations Number of benchmark iterations
 * @return Average time in microseconds
 */
float qwen3_sampler_benchmark(size_t vocab_size, size_t iterations);

#ifdef __cplusplus
}
#endif

#endif // QWEN3_SAMPLER_H