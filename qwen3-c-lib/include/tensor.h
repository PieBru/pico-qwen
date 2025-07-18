/**
 * @file tensor.h
 * @brief Tensor operations and quantized data structures for Qwen3 C inference engine
 * 
 * Provides efficient tensor operations with INT8 quantization support,
 * SIMD-aligned memory layout, and optimized memory access patterns.
 */

#ifndef QWEN3_TENSOR_H
#define QWEN3_TENSOR_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

// Constants
#define QWEN3_MEMORY_ALIGNMENT 64

// Memory pool forward declaration
struct Qwen3MemoryPool;

#ifdef __cplusplus
extern "C" {
#endif

// Tensor data types
typedef enum {
    QWEN3_DTYPE_FLOAT32 = 0,
    QWEN3_DTYPE_INT8 = 1,
    QWEN3_DTYPE_INT16 = 2,
    QWEN3_DTYPE_UINT8 = 3,
} Qwen3DType;

// Tensor layout
typedef enum {
    QWEN3_LAYOUT_ROW_MAJOR = 0,
    QWEN3_LAYOUT_COL_MAJOR = 1,
} Qwen3Layout;

// Quantization parameters
typedef struct {
    float scale;
    int8_t zero_point;
    size_t group_size;
} Qwen3QuantizationParams;

// Tensor shape and dimensions
typedef struct {
    size_t dims[4];
    size_t ndims;
    size_t strides[4];
} Qwen3Shape;

// Forward declaration
typedef struct Qwen3Tensor Qwen3Tensor;

// Quantized tensor with group-wise scaling
typedef struct {
    int8_t* data;                    // Quantized data
    float* scales;                   // Per-group scaling factors
    int8_t* zero_points;             // Per-group zero points (optional)
    Qwen3Shape shape;                // Tensor shape
    size_t group_size;               // Group size for quantization
    void* pool;                      // Memory pool for allocation
} Qwen3QuantizedTensor;

// Tensor structure
struct Qwen3Tensor {
    void* data;                      // Raw tensor data
    Qwen3Shape shape;                // Tensor dimensions and strides
    Qwen3DType dtype;                // Data type
    Qwen3Layout layout;              // Memory layout
    void* pool;                      // Memory pool for this tensor
    bool owns_data;                  // Whether to free data on destruction
};

// Tensor creation and destruction
Qwen3Tensor* qwen3_tensor_create(const size_t* dims, size_t ndims, Qwen3DType dtype);
Qwen3Tensor* qwen3_tensor_create_with_pool(const size_t* dims, size_t ndims, Qwen3DType dtype, struct Qwen3MemoryPool* pool);
void qwen3_tensor_destroy(Qwen3Tensor* tensor);

// Quantized tensor operations
Qwen3QuantizedTensor* qwen3_quantized_tensor_create(const size_t* dims, size_t ndims, size_t group_size);
Qwen3QuantizedTensor* qwen3_quantized_tensor_create_with_pool(const size_t* dims, size_t ndims, size_t group_size, struct Qwen3MemoryPool* pool);
void qwen3_quantized_tensor_destroy(Qwen3QuantizedTensor* tensor);

// Tensor shape operations
Qwen3Shape qwen3_shape_create(const size_t* dims, size_t ndims);
size_t qwen3_shape_num_elements(const Qwen3Shape* shape);
size_t qwen3_shape_get_stride(const Qwen3Shape* shape, size_t dim);
bool qwen3_shape_is_broadcastable(const Qwen3Shape* a, const Qwen3Shape* b);

// Tensor data access
float qwen3_tensor_get_float(const Qwen3Tensor* tensor, const size_t* indices);
int8_t qwen3_tensor_get_int8(const Qwen3Tensor* tensor, const size_t* indices);
void qwen3_tensor_set_float(Qwen3Tensor* tensor, const size_t* indices, float value);
void qwen3_tensor_set_int8(Qwen3Tensor* tensor, const size_t* indices, int8_t value);

// Quantized tensor operations
float qwen3_quantized_tensor_get_float(const Qwen3QuantizedTensor* tensor, const size_t* indices);
void qwen3_quantized_tensor_set_float(Qwen3QuantizedTensor* tensor, const size_t* indices, float value);
void qwen3_quantized_tensor_dequantize(const Qwen3QuantizedTensor* src, Qwen3Tensor* dst);
void qwen3_quantized_tensor_quantize(const Qwen3Tensor* src, Qwen3QuantizedTensor* dst, float scale, int8_t zero_point);

// Tensor reshaping and views
Qwen3Tensor* qwen3_tensor_reshape(Qwen3Tensor* tensor, const size_t* new_dims, size_t new_ndims);
Qwen3Tensor* qwen3_tensor_view(Qwen3Tensor* tensor, const size_t* starts, const size_t* ends);

// Memory operations
void qwen3_tensor_zero(Qwen3Tensor* tensor);
void qwen3_tensor_fill(Qwen3Tensor* tensor, float value);
void qwen3_tensor_copy(const Qwen3Tensor* src, Qwen3Tensor* dst);

// Utility functions
size_t qwen3_dtype_size(Qwen3DType dtype);
bool qwen3_tensor_validate(const Qwen3Tensor* tensor);
void qwen3_tensor_print_info(const Qwen3Tensor* tensor);
void qwen3_quantized_tensor_print_info(const Qwen3QuantizedTensor* tensor);

// SIMD-optimized operations
void qwen3_tensor_add(const Qwen3Tensor* a, const Qwen3Tensor* b, Qwen3Tensor* result);
void qwen3_tensor_multiply(const Qwen3Tensor* a, const Qwen3Tensor* b, Qwen3Tensor* result);
void qwen3_tensor_scale(Qwen3Tensor* tensor, float scale);

#ifdef __cplusplus
}
#endif

#endif // QWEN3_TENSOR_H