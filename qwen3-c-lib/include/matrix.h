/**
 * @file matrix.h
 * @brief High-performance matrix operations for Qwen3 C inference engine
 * 
 * Provides scalar, AVX2, and AVX-512 optimized matrix multiplication
 * with runtime kernel selection based on CPU capabilities.
 */

#ifndef QWEN3_MATRIX_H
#define QWEN3_MATRIX_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
struct Qwen3Tensor;
struct Qwen3QuantizedTensor;

/**
 * @brief CPU feature detection flags
 */
typedef enum {
    QWEN3_CPU_SCALAR = 0,
    QWEN3_CPU_AVX2 = 1,
    QWEN3_CPU_AVX512 = 2,
    QWEN3_CPU_NEON = 3,
} Qwen3CPUFeature;

/**
 * @brief Matrix multiplication configuration
 */
typedef struct {
    Qwen3CPUFeature cpu_feature;
    size_t block_size;
    bool use_threading;
    size_t num_threads;
} Qwen3MatMulConfig;

/**
 * @brief Detect available CPU features
 * @return Bitmask of detected CPU features
 */
Qwen3CPUFeature qwen3_cpu_detect_features(void);

/**
 * @brief Initialize matrix operations with optimal configuration
 * @param config Configuration structure (NULL for auto-detect)
 * @return 0 on success, non-zero on error
 */
int qwen3_matrix_init(const Qwen3MatMulConfig* config);

/**
 * @brief Get current matrix multiplication configuration
 * @return Current configuration
 */
const Qwen3MatMulConfig* qwen3_matrix_get_config(void);

/**
 * @brief Scalar matrix multiplication (fallback)
 * @param A Input matrix A [M x K]
 * @param B Input matrix B [K x N]
 * @param C Output matrix C [M x N]
 * @param M Rows in A and C
 * @param N Columns in B and C
 * @param K Columns in A, rows in B
 * @param alpha Scaling factor for A*B
 * @param beta Scaling factor for C (C = alpha*A*B + beta*C)
 */
void qwen3_matmul_scalar(const float* A, const float* B, float* C,
                        size_t M, size_t N, size_t K,
                        float alpha, float beta);

/**
 * @brief AVX2 optimized matrix multiplication
 * @param A Input matrix A [M x K]
 * @param B Input matrix B [K x N]
 * @param C Output matrix C [M x N]
 * @param M Rows in A and C
 * @param N Columns in B and C
 * @param K Columns in A, rows in B
 * @param alpha Scaling factor for A*B
 * @param beta Scaling factor for C (C = alpha*A*B + beta*C)
 */
void qwen3_matmul_avx2(const float* A, const float* B, float* C,
                      size_t M, size_t N, size_t K,
                      float alpha, float beta);

/**
 * @brief AVX-512 optimized matrix multiplication
 * @param A Input matrix A [M x K]
 * @param B Input matrix B [K x N]
 * @param C Output matrix C [M x N]
 * @param M Rows in A and C
 * @param N Columns in B and C
 * @param K Columns in A, rows in B
 * @param alpha Scaling factor for A*B
 * @param beta Scaling factor for C (C = alpha*A*B + beta*C)
 */
void qwen3_matmul_avx512(const float* A, const float* B, float* C,
                        size_t M, size_t N, size_t K,
                        float alpha, float beta);

/**
 * @brief Quantized matrix multiplication (INT8 x INT8 -> FP32)
 * @param A Quantized input matrix A [M x K]
 * @param B Quantized input matrix B [K x N]
 * @param C Output matrix C [M x N]
 * @param M Rows in A and C
 * @param N Columns in B and C
 * @param K Columns in A, rows in B
 * @param alpha Scaling factor
 * @param beta Scaling factor for C
 */
void qwen3_matmul_quantized(const struct Qwen3QuantizedTensor* A,
                           const struct Qwen3QuantizedTensor* B,
                           float* C,
                           size_t M, size_t N, size_t K,
                           float alpha, float beta);

/**
 * @brief Auto-dispatch matrix multiplication based on CPU capabilities
 * @param A Input matrix A [M x K]
 * @param B Input matrix B [K x N]
 * @param C Output matrix C [M x N]
 * @param M Rows in A and C
 * @param N Columns in B and C
 * @param K Columns in A, rows in B
 * @param alpha Scaling factor for A*B
 * @param beta Scaling factor for C (C = alpha*A*B + beta*C)
 */
void qwen3_matmul(const float* A, const float* B, float* C,
                 size_t M, size_t N, size_t K,
                 float alpha, float beta);

/**
 * @brief Matrix-vector multiplication
 * @param A Input matrix [M x N]
 * @param x Input vector [N]
 * @param y Output vector [M]
 * @param M Rows in A
 * @param N Columns in A
 * @param alpha Scaling factor
 * @param beta Scaling factor for y
 */
void qwen3_matvec(const float* A, const float* x, float* y,
                 size_t M, size_t N, float alpha, float beta);

/**
 * @brief Transpose matrix
 * @param src Input matrix [M x N]
 * @param dst Output matrix [N x M]
 * @param M Rows in src
 * @param N Columns in src
 */
void qwen3_transpose(const float* src, float* dst, size_t M, size_t N);

/**
 * @brief Add matrices element-wise
 * @param A First matrix [M x N]
 * @param B Second matrix [M x N]
 * @param C Output matrix [M x N]
 * @param M Rows
 * @param N Columns
 */
void qwen3_add(const float* A, const float* B, float* C, size_t M, size_t N);

/**
 * @brief Scale matrix by scalar
 * @param src Input matrix [M x N]
 * @param dst Output matrix [M x N]
 * @param scalar Scaling factor
 * @param M Rows
 * @param N Columns
 */
void qwen3_scale(const float* src, float* dst, float scalar, size_t M, size_t N);

/**
 * @brief Softmax along last dimension
 * @param src Input matrix [M x N]
 * @param dst Output matrix [M x N]
 * @param M Rows
 * @param N Columns (softmax along this dimension)
 */
void qwen3_softmax(float* src, float* dst, size_t M, size_t N);

/**
 * @brief Layer normalization
 * @param x Input matrix [M x N]
 * @param gamma Scale parameters [N]
 * @param beta Shift parameters [N]
 * @param output Output matrix [M x N]
 * @param M Rows
 * @param N Columns
 * @param eps Epsilon for numerical stability
 */
void qwen3_layernorm(const float* x, const float* gamma, const float* beta,
                    float* output, size_t M, size_t N, float eps);

/**
 * @brief RMS normalization
 * @param x Input matrix [M x N]
 * @param weight RMS weight parameters [N]
 * @param output Output matrix [M x N]
 * @param M Rows
 * @param N Columns
 * @param eps Epsilon for numerical stability
 */
void qwen3_rmsnorm(const float* x, const float* weight,
                  float* output, size_t M, size_t N, float eps);

/**
 * @brief RoPE (Rotary Position Embedding) application
 * @param q Query matrix [M x N]
 * @param k Key matrix [M x N]
 * @param pos Position indices [M]
 * @param head_dim Head dimension
 * @param M Rows
 * @param N Columns
 * @param theta_base Base theta value
 */
void qwen3_rope(float* q, float* k, const int* pos,
               size_t head_dim, size_t M, size_t N, float theta_base);

/**
 * @brief Get optimal block size for matrix multiplication
 * @param M Matrix dimensions M
 * @param N Matrix dimensions N
 * @param K Matrix dimensions K
 * @return Optimal block size
 */
size_t qwen3_matmul_optimal_block_size(size_t M, size_t N, size_t K);

/**
 * @brief Benchmark matrix multiplication kernels
 * @param M Matrix dimensions M
 * @param N Matrix dimensions N
 * @param K Matrix dimensions K
 * @param iterations Number of iterations
 * @return Average time in microseconds
 */
float qwen3_matmul_benchmark(size_t M, size_t N, size_t K, size_t iterations);

#ifdef __cplusplus
}
#endif

#endif // QWEN3_MATRIX_H