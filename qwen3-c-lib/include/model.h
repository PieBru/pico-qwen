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

// Use the Qwen3ModelConfig from qwen3_inference.h to avoid conflicts

// Forward declarations - use the opaque types from qwen3_inference.h
struct Qwen3Model;
struct Qwen3ModelConfig;

// Model loading options - use Qwen3LoadOptions from qwen3_inference.h

// Model loading utilities (implementation in qwen3_inference.h)
// These functions are implemented in the main API and use the opaque types

#ifdef __cplusplus
}
#endif

#endif // QWEN3_MODEL_H