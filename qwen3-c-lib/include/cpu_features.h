/**
 * @file cpu_features.h
 * @brief CPU feature detection for Qwen3 C inference engine
 * 
 * This file provides CPU feature detection using CPUID instruction
 * for x86_64 processors, and ARM feature detection for ARM processors.
 */

#ifndef QWEN3_CPU_FEATURES_H
#define QWEN3_CPU_FEATURES_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief CPU feature flags
 */
typedef enum {
    QWEN3_CPU_FEATURE_SSE2     = 1 << 0,
    QWEN3_CPU_FEATURE_SSE3     = 1 << 1,
    QWEN3_CPU_FEATURE_SSSE3    = 1 << 2,
    QWEN3_CPU_FEATURE_SSE41    = 1 << 3,
    QWEN3_CPU_FEATURE_SSE42    = 1 << 4,
    QWEN3_CPU_FEATURE_AVX      = 1 << 5,
    QWEN3_CPU_FEATURE_AVX2     = 1 << 6,
    QWEN3_CPU_FEATURE_FMA      = 1 << 7,
    QWEN3_CPU_FEATURE_AVX512F  = 1 << 8,
    QWEN3_CPU_FEATURE_AVX512VL = 1 << 9,
    QWEN3_CPU_FEATURE_AVX512BW = 1 << 10,
    QWEN3_CPU_FEATURE_AVX512DQ = 1 << 11,
    QWEN3_CPU_FEATURE_NEON     = 1 << 12,
    QWEN3_CPU_FEATURE_SVE      = 1 << 13,
} Qwen3CPUFeature;

/**
 * @brief CPU information structure
 */
typedef struct {
    uint64_t features;          // Bitmask of detected CPU features
    char vendor[13];           // CPU vendor string (null-terminated)
    char brand[49];            // CPU brand string (null-terminated)
    uint32_t family;           // CPU family
    uint32_t model;            // CPU model
    uint32_t stepping;         // CPU stepping
    uint32_t cores;            // Number of physical cores
    uint32_t threads;          // Number of logical threads
    uint32_t cache_line_size;  // Cache line size in bytes
    uint32_t l1_cache_size;    // L1 cache size in KB
    uint32_t l2_cache_size;    // L2 cache size in KB
    uint32_t l3_cache_size;    // L3 cache size in KB
} Qwen3CPUInfo;

/**
 * @brief Detect CPU features and populate CPU info structure
 * @param info Pointer to CPU info structure to populate
 * @return 0 on success, -1 on error
 */
int qwen3_cpu_detect_features(Qwen3CPUInfo* info);

/**
 * @brief Check if a specific CPU feature is supported
 * @param info CPU info structure
 * @param feature Feature to check
 * @return true if feature is supported, false otherwise
 */
bool qwen3_cpu_has_feature(const Qwen3CPUInfo* info, Qwen3CPUFeature feature);

/**
 * @brief Get a human-readable string for CPU features
 * @param features Bitmask of features
 * @param buffer Buffer to write the string to
 * @param buffer_size Size of the buffer
 * @return Pointer to the buffer
 */
char* qwen3_cpu_features_string(uint64_t features, char* buffer, size_t buffer_size);

/**
 * @brief Get the optimal matrix multiplication kernel based on CPU features
 * @param info CPU info structure
 * @return String identifying the optimal kernel
 */
const char* qwen3_cpu_get_optimal_kernel(const Qwen3CPUInfo* info);

/**
 * @brief Print CPU information to stdout
 * @param info CPU info structure
 */
void qwen3_cpu_print_info(const Qwen3CPUInfo* info);

#ifdef __cplusplus
}
#endif

#endif // QWEN3_CPU_FEATURES_H