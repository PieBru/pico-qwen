/**
 * @file qwen3_inference_internal.c
 * @brief Internal inference implementation for Qwen3 C inference engine
 * 
 * This file contains the internal implementation details for the inference engine,
 * separate from the public API to avoid type conflicts.
 */

#include "../include/qwen3_inference.h"
#include "../include/transformer.h"
#include "../include/sampler.h"
#include "../include/tokenizer.h"
#include "../include/inference_config.h"
#include "../src/memory.h"
#include "../include/model.h"
#include "../include/model_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Internal inference structure
typedef struct Qwen3InferenceInternal {
    Qwen3InferenceConfigInternal config;
    Qwen3Model* model;
    Qwen3Tokenizer* tokenizer;
    Qwen3Transformer* transformer;
    Qwen3SamplerConfig sampler_config;
    
    // Runtime state
    int* token_buffer;
    float* logits_buffer;
    char* output_buffer;
    size_t max_tokens;
    
    // Memory management
    Qwen3MemoryArena* inference_arena;
} Qwen3InferenceInternal;

// Forward declarations
void qwen3_inference_internal_free(Qwen3InferenceInternal* inference);

/**
 * @brief Initialize inference engine with configuration
 */
int qwen3_inference_internal_init(Qwen3InferenceInternal* inference, const Qwen3InferenceConfigInternal* config) {
    if (!inference || !config) {
        return -1;
    }
    
    memset(inference, 0, sizeof(Qwen3InferenceInternal));
    inference->config = *config;
    
    // Validate configuration
    if (config->vocab_size == 0 || config->hidden_size == 0 || 
        config->max_seq_len == 0) {
        return -1;
    }
    
    // Initialize memory arena
    inference->inference_arena = qwen3_memory_arena_create(256 * 1024 * 1024); // 256MB
    if (!inference->inference_arena) {
        return -1;
    }
    
    // Initialize sampler configuration
    inference->sampler_config.temperature = config->temperature;
    inference->sampler_config.top_k = config->top_k;
    inference->sampler_config.top_p = config->top_p;
    inference->sampler_config.seed = config->seed ? *config->seed : (unsigned int)time(NULL);
    
    // Allocate buffers
    inference->max_tokens = config->max_seq_len;
    inference->token_buffer = (int*)calloc(inference->max_tokens, sizeof(int));
    inference->logits_buffer = (float*)calloc(config->vocab_size, sizeof(float));
    inference->output_buffer = (char*)calloc(inference->max_tokens * 32, sizeof(char)); // 32 chars per token max
    
    if (!inference->token_buffer || !inference->logits_buffer || !inference->output_buffer) {
        qwen3_inference_internal_free(inference);
        return -1;
    }
    
    return 0;
}

/**
 * @brief Free all inference resources
 */
void qwen3_inference_internal_free(Qwen3InferenceInternal* inference) {
    if (!inference) return;
    
    if (inference->model) {
        qwen3_model_free(inference->model);
    }
    
    if (inference->tokenizer) {
        qwen3_tokenizer_free(inference->tokenizer);
    }
    
    if (inference->transformer) {
        qwen3_transformer_free(inference->transformer);
    }
    
    if (inference->inference_arena) {
        qwen3_memory_arena_destroy(inference->inference_arena);
    }
    
    free(inference->token_buffer);
    free(inference->logits_buffer);
    free(inference->output_buffer);
    
    memset(inference, 0, sizeof(Qwen3InferenceInternal));
}

/**
 * @brief Load model from checkpoint file
 */
int qwen3_inference_internal_load_model(Qwen3InferenceInternal* inference, const char* model_path) {
    if (!inference || !model_path) {
        return -1;
    }
    
    // Load model
    inference->model = qwen3_model_load(model_path, inference->config.max_seq_len);
    if (!inference->model) {
        return -1;
    }
    
    // Initialize tokenizer
    inference->tokenizer = qwen3_tokenizer_init_from_model(inference->model);
    if (!inference->tokenizer) {
        return -1;
    }
    
    // Initialize transformer from model
    const Qwen3ModelConfig* model_config = qwen3_model_get_config(inference->model);
    if (!model_config) {
        return -1;
    }
    
    Qwen3TransformerConfig transformer_config;
    int ret = qwen3_transformer_config_init(&transformer_config,
                                          model_config->dim,
                                          model_config->hidden_dim,
                                          model_config->n_heads,
                                          model_config->n_kv_heads,
                                          model_config->max_seq_len,
                                          1e-6f,
                                          model_config->rope_theta);
    if (ret != 0) {
        return -1;
    }
    
    inference->transformer = (Qwen3Transformer*)calloc(1, sizeof(Qwen3Transformer));
    if (!inference->transformer) {
        return -1;
    }
    
    ret = qwen3_transformer_init(inference->transformer, &transformer_config, model_config->n_layers);
    if (ret != 0) {
        return -1;
    }
    
    return 0;
}

/**
 * @brief Generate tokens from prompt using the loaded model
 */
static int generate_tokens_internal(Qwen3InferenceInternal* inference, const char* prompt, 
                          size_t max_new_tokens, char* output, size_t output_size) {
    if (!inference || !prompt || !output || !inference->model || !inference->transformer) {
        return -1;
    }
    
    // Tokenize prompt
    int* prompt_tokens = NULL;
    size_t prompt_len = qwen3_tokenizer_encode(inference->tokenizer, prompt, &prompt_tokens);
    if (prompt_len <= 0 || !prompt_tokens) {
        return -1;
    }
    
    // Check if prompt fits in max sequence length
    if (prompt_len >= inference->config.max_seq_len) {
        free(prompt_tokens);
        return -1;
    }
    
    size_t total_tokens = prompt_len;
    memcpy(inference->token_buffer, prompt_tokens, prompt_len * sizeof(int));
    free(prompt_tokens);
    
    // Generate tokens
    size_t output_pos = 0;
    for (size_t step = 0; step < max_new_tokens && total_tokens < inference->config.max_seq_len; step++) {
        // Create position array
        int* pos = (int*)calloc(total_tokens, sizeof(int));
        for (size_t i = 0; i < total_tokens; i++) {
            pos[i] = (int)i;
        }
        
        // Forward pass through transformer
        int ret = qwen3_transformer_forward(inference->transformer,
                                          inference->token_buffer,
                                          total_tokens,
                                          pos,
                                          inference->logits_buffer);
        free(pos);
        
        if (ret != 0) {
            return -1;
        }
        
        // Sample next token from last position logits
        const float* last_logits = inference->logits_buffer + (total_tokens - 1) * inference->config.vocab_size;
        int next_token = qwen3_sampler_sample_token(last_logits,
                                                  inference->config.vocab_size,
                                                  inference->sampler_config.temperature,
                                                  inference->sampler_config.top_k,
                                                  inference->sampler_config.top_p,
                                                  &inference->sampler_config.seed);
        
        if (next_token < 0) {
            return -1;
        }
        
        // Check for end-of-sequence token
        if (next_token == (int)inference->config.eos_token_id) {
            break;
        }
        
        // Add token to buffer
        inference->token_buffer[total_tokens++] = next_token;
        
        // Decode token to text
        char* token_text = qwen3_tokenizer_decode(inference->tokenizer, next_token);
        if (token_text) {
            size_t token_len = strlen(token_text);
            if (output_pos + token_len < output_size - 1) {
                strcat(output + output_pos, token_text);
                output_pos += token_len;
            }
        }
        
        // Print token as it's generated (streaming)
        printf("%s", token_text ? token_text : "");
        fflush(stdout);
    }
    
    return (int)output_pos;
}

/**
 * @brief Run interactive chat mode
 */
int qwen3_inference_internal_chat(Qwen3InferenceInternal* inference, const char* system_prompt) {
    if (!inference) {
        return -1;
    }
    
    printf("=== Qwen3 Chat Mode ===\n");
    printf("Type 'quit' or 'exit' to end the conversation.\n\n");
    
    if (system_prompt) {
        printf("System: %s\n\n", system_prompt);
    }
    
    char user_input[2048];
    char response[4096];
    
    while (1) {
        printf("User: ");
        fflush(stdout);
        
        if (!fgets(user_input, sizeof(user_input), stdin)) {
            break;
        }
        
        // Remove newline
        user_input[strcspn(user_input, "\n")] = 0;
        
        // Check for exit commands
        if (strcmp(user_input, "quit") == 0 || strcmp(user_input, "exit") == 0) {
            break;
        }
        
        if (strlen(user_input) == 0) {
            continue;
        }
        
        // Format prompt for chat
        char formatted_prompt[3072];
        if (system_prompt) {
            snprintf(formatted_prompt, sizeof(formatted_prompt),
                    "<|system|>\n%s<|end|>\n<|user|>\n%s<|end|>\n<|assistant|>\n",
                    system_prompt, user_input);
        } else {
            snprintf(formatted_prompt, sizeof(formatted_prompt),
                    "<|user|>\n%s<|end|>\n<|assistant|>\n", user_input);
        }
        
        printf("Assistant: ");
        fflush(stdout);
        
        // Generate response
        int response_len = generate_tokens_internal(inference, formatted_prompt, 
                                         inference->config.max_new_tokens, 
                                         response, sizeof(response));
        
        if (response_len < 0) {
            printf("\nError generating response.\n");
        } else {
            printf("\n");
        }
    }
    
    return 0;
}

/**
 * @brief Run generation mode
 */
int qwen3_inference_internal_generate(Qwen3InferenceInternal* inference, const char* prompt, 
                                   char* output, size_t output_size) {
    if (!inference || !prompt || !output || output_size == 0) {
        return -1;
    }
    
    printf("Prompt: %s\n", prompt);
    printf("Generated: ");
    fflush(stdout);
    
    int response_len = generate_tokens_internal(inference, prompt, 
                                     inference->config.max_new_tokens, 
                                     output, output_size);
    
    printf("\n");
    return response_len;
}

/**
 * @brief Set generation parameters
 */
void qwen3_inference_internal_set_parameters(Qwen3InferenceInternal* inference, float temperature, 
                                          float top_p, size_t top_k, unsigned int seed) {
    if (!inference) return;
    
    inference->sampler_config.temperature = temperature;
    inference->sampler_config.top_p = top_p;
    inference->sampler_config.top_k = top_k;
    inference->sampler_config.seed = seed;
}