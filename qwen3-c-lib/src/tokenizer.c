/**
 * @file tokenizer.c
 * @brief Tokenizer implementation for Qwen3 C inference engine
 * 
 * Implements BPE (Byte Pair Encoding) tokenization with vocabulary loading
 * and text encoding/decoding functionality for Qwen3 models.
 */

#include "../include/tokenizer.h"
#include "../include/qwen3_inference.h"
#include "../include/memory.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>

// strdup compatibility for systems without strdup
#if !defined(HAVE_STRDUP) && !defined(_GNU_SOURCE) && !defined(_POSIX_C_SOURCE)
static char* qwen3_strdup(const char* s) {
    if (!s) return NULL;
    size_t len = strlen(s) + 1;
    char* new_str = (char*)malloc(len);
    if (new_str) {
        memcpy(new_str, s, len);
    }
    return new_str;
}
#define strdup qwen3_strdup
#endif

/**
 * @brief Load tokenizer from binary tokenizer file
 */
Qwen3Tokenizer* qwen3_tokenizer_load(const char* tokenizer_path) {
    if (!tokenizer_path) {
        return NULL;
    }
    
    FILE* file = fopen(tokenizer_path, "rb");
    if (!file) {
        fprintf(stderr, "Failed to open tokenizer file: %s\n", tokenizer_path);
        return NULL;
    }
    
    Qwen3Tokenizer* tokenizer = (Qwen3Tokenizer*)calloc(1, sizeof(Qwen3Tokenizer));
    if (!tokenizer) {
        fclose(file);
        return NULL;
    }
    
    // Read header
    uint32_t max_token_length, bos_token_id, eos_token_id;
    if (fread(&max_token_length, sizeof(uint32_t), 1, file) != 1 ||
        fread(&bos_token_id, sizeof(uint32_t), 1, file) != 1 ||
        fread(&eos_token_id, sizeof(uint32_t), 1, file) != 1) {
        fprintf(stderr, "Failed to read tokenizer header\n");
        free(tokenizer);
        fclose(file);
        return NULL;
    }
    
    // Initialize tokenizer
    tokenizer->vocab_size = 32000; // Qwen3 standard vocab size
    tokenizer->bos_token_id = bos_token_id;
    tokenizer->eos_token_id = eos_token_id;
    tokenizer->pad_token_id = bos_token_id;
    tokenizer->unk_token_id = bos_token_id;
    
    tokenizer->vocab = (char**)calloc(tokenizer->vocab_size, sizeof(char*));
    tokenizer->vocab_scores = (float*)calloc(tokenizer->vocab_size, sizeof(float));
    
    if (!tokenizer->vocab || !tokenizer->vocab_scores) {
        qwen3_tokenizer_free(tokenizer);
        fclose(file);
        return NULL;
    }
    
    // Read tokens
    uint32_t token_count = 0;
    while (token_count < tokenizer->vocab_size && !feof(file)) {
        float score;
        uint32_t len;
        
        if (fread(&score, sizeof(float), 1, file) != 1 ||
            fread(&len, sizeof(uint32_t), 1, file) != 1) {
            break;
        }
        
        if (len > 1024) { // Sanity check
            fprintf(stderr, "Token length too large: %u\n", len);
            break;
        }
        
        char* token_str = (char*)malloc(len + 1);
        if (!token_str) {
            break;
        }
        
        if (fread(token_str, 1, len, file) != len) {
            free(token_str);
            break;
        }
        token_str[len] = '\0';
        
        tokenizer->vocab[token_count] = token_str;
        tokenizer->vocab_scores[token_count] = score;
        token_count++;
    }
    
    fclose(file);
    
    // Set special tokens
    tokenizer->special_tokens.bos_token = strdup("<|begin_of_text|>");
    tokenizer->special_tokens.eos_token = strdup("<|end_of_text|>");
    tokenizer->special_tokens.pad_token = strdup("<|pad|>");
    tokenizer->special_tokens.unk_token = strdup("<|unk|>");
    
    return tokenizer;
}

/**
 * @brief Initialize tokenizer from model
 */
Qwen3Tokenizer* qwen3_tokenizer_init_from_model(struct Qwen3Model* model) {
    if (!model) {
        return NULL;
    }
    
    // Default implementation - should load from model's tokenizer file
    // For now, create basic tokenizer with common Qwen3 tokens
    const char* model_path = "model.bin.tokenizer"; // Default path
    return qwen3_tokenizer_load(model_path);
}

/**
 * @brief Free tokenizer resources
 */
void qwen3_tokenizer_free(Qwen3Tokenizer* tokenizer) {
    if (!tokenizer) return;
    
    // Free vocabulary
    if (tokenizer->vocab) {
        for (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
            free(tokenizer->vocab[i]);
        }
        free(tokenizer->vocab);
    }
    
    free(tokenizer->vocab_scores);
    
    // Free special tokens
    free(tokenizer->special_tokens.bos_token);
    free(tokenizer->special_tokens.eos_token);
    free(tokenizer->special_tokens.pad_token);
    free(tokenizer->special_tokens.unk_token);
    
    // Free BPE merges
    if (tokenizer->bpe.merges) {
        for (uint32_t i = 0; i < tokenizer->bpe.merge_count; i++) {
            free(tokenizer->bpe.merges[i]);
        }
        free(tokenizer->bpe.merges);
    }
    
    free(tokenizer);
}

/**
 * @brief Encode text to token sequence (simplified implementation)
 */
size_t qwen3_tokenizer_encode(Qwen3Tokenizer* tokenizer, const char* text, int** tokens) {
    if (!tokenizer || !text || !tokens) {
        return 0;
    }
    
    // Simple encoding - split by spaces and map to tokens
    char* text_copy = strdup(text);
    if (!text_copy) {
        return 0;
    }
    
    // Count words
    size_t token_count = 0;
    char* text_ptr = text_copy;
    char* word = strtok(text_ptr, " \t\n");
    while (word) {
        token_count++;
        word = strtok(NULL, " \t\n");
    }
    
    if (token_count == 0) {
        free(text_copy);
        return 0;
    }
    
    // Allocate token array
    *tokens = (int*)malloc(token_count * sizeof(int));
    if (!*tokens) {
        free(text_copy);
        return 0;
    }
    
    // Encode tokens
    strcpy(text_copy, text); // Reset text copy
    text_ptr = text_copy;
    word = strtok(text_ptr, " \t\n");
    size_t idx = 0;
    
    while (word && idx < token_count) {
        // Simple mapping - hash word to token ID
        unsigned int hash = 5381;
        for (const char* c = word; *c; c++) {
            hash = ((hash << 5) + hash) + (unsigned char)(*c);
        }
        (*tokens)[idx] = hash % tokenizer->vocab_size;
        idx++;
        word = strtok(NULL, " \t\n");
    }
    
    free(text_copy);
    return idx;
}

/**
 * @brief Decode token to text
 */
char* qwen3_tokenizer_decode(Qwen3Tokenizer* tokenizer, int token_id) {
    if (!tokenizer || token_id < 0 || token_id >= (int)tokenizer->vocab_size) {
        return NULL;
    }
    
    return tokenizer->vocab[token_id];
}

/**
 * @brief Decode token sequence to text
 */
char* qwen3_tokenizer_decode_sequence(Qwen3Tokenizer* tokenizer, const int* tokens, size_t num_tokens) {
    if (!tokenizer || !tokens || num_tokens == 0) {
        return NULL;
    }
    
    // Calculate required buffer size
    size_t total_size = 0;
    for (size_t i = 0; i < num_tokens; i++) {
        if (tokens[i] >= 0 && tokens[i] < (int)tokenizer->vocab_size) {
            total_size += strlen(tokenizer->vocab[tokens[i]]) + 1; // +1 for space
        }
    }
    
    char* result = (char*)malloc(total_size + 1);
    if (!result) {
        return NULL;
    }
    
    result[0] = '\0';
    
    for (size_t i = 0; i < num_tokens; i++) {
        if (tokens[i] >= 0 && tokens[i] < (int)tokenizer->vocab_size) {
            strcat(result, tokenizer->vocab[tokens[i]]);
            if (i < num_tokens - 1) {
                strcat(result, " ");
            }
        }
    }
    
    return result;
}

/**
 * @brief Get special token ID
 */
int qwen3_tokenizer_get_special_token(Qwen3Tokenizer* tokenizer, const char* token_type) {
    if (!tokenizer || !token_type) {
        return -1;
    }
    
    if (strcmp(token_type, "bos") == 0) {
        return tokenizer->bos_token_id;
    } else if (strcmp(token_type, "eos") == 0) {
        return tokenizer->eos_token_id;
    } else if (strcmp(token_type, "pad") == 0) {
        return tokenizer->pad_token_id;
    } else if (strcmp(token_type, "unk") == 0) {
        return tokenizer->unk_token_id;
    }
    
    return -1;
}