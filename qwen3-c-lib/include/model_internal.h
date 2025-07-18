/**
 * @file model_internal.h
 * @brief Internal model loading declarations for Qwen3 C inference engine
 */

#ifndef QWEN3_MODEL_INTERNAL_H
#define QWEN3_MODEL_INTERNAL_H

#include "../include/qwen3_inference.h"

#ifdef __cplusplus
extern "C" {
#endif

// Internal model loading functions
Qwen3Model* qwen3_model_load_internal(const char* checkpoint_path, uint32_t ctx_length);
void qwen3_model_free_internal(Qwen3Model* model);
const Qwen3ModelConfig* qwen3_model_get_config_internal(const Qwen3Model* model);
const char* qwen3_get_last_error_internal(void);

#ifdef __cplusplus
}
#endif

#endif // QWEN3_MODEL_INTERNAL_H