/**
 * @file tokenizer.h
 * @brief Tokenizer interface for Qwen3 C inference engine
 * 
 * Provides BPE (Byte Pair Encoding) tokenization for Qwen3 models with
 * vocabulary loading and text encoding/decoding functionality.
 */

#ifndef QWEN3_TOKENIZER_H
#define QWEN3_TOKENIZER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declaration
struct Qwen3Model;

/**
 * @brief Tokenizer structure
 */
typedef struct {
    char** vocab;
    float* vocab_scores;
    uint32_t vocab_size;
    uint32_t bos_token_id;
    uint32_t eos_token_id;
    uint32_t pad_token_id;
    uint32_t unk_token_id;
    
    // BPE merge rules
    struct {
        char** merges;
        uint32_t merge_count;
    } bpe;
    
    // Special tokens
    struct {
        char* bos_token;
        char* eos_token;
        char* pad_token;
        char* unk_token;
    } special_tokens;
    
} Qwen3Tokenizer;

/**
 * @brief Initialize tokenizer from model
 * @param model Loaded model handle
 * @return Tokenizer instance or NULL on error
 */
Qwen3Tokenizer* qwen3_tokenizer_init_from_model(struct Qwen3Model* model);

/**
 * @brief Load tokenizer from binary tokenizer file
 * @param tokenizer_path Path to .tokenizer file
 * @return Tokenizer instance or NULL on error
 */
Qwen3Tokenizer* qwen3_tokenizer_load(const char* tokenizer_path);

/**
 * @brief Free tokenizer resources
 * @param tokenizer Tokenizer to free
 */
void qwen3_tokenizer_free(Qwen3Tokenizer* tokenizer);

/**
 * @brief Encode text to token sequence
 * @param tokenizer Tokenizer instance
 * @param text Input text to encode
 * @param tokens Output array of token IDs (caller must free)
 * @return Number of tokens or -1 on error
 */
size_t qwen3_tokenizer_encode(Qwen3Tokenizer* tokenizer, const char* text, int** tokens);

/**
 * @brief Decode token to text
 * @param tokenizer Tokenizer instance
 * @param token_id Token ID to decode
 * @return Decoded text (caller must not free)
 */
char* qwen3_tokenizer_decode(Qwen3Tokenizer* tokenizer, int token_id);

/**
 * @brief Decode token sequence to text
 * @param tokenizer Tokenizer instance
 * @param tokens Array of token IDs
 * @param num_tokens Number of tokens
 * @return Decoded text (caller must free)
 */
char* qwen3_tokenizer_decode_sequence(Qwen3Tokenizer* tokenizer, const int* tokens, size_t num_tokens);

/**
 * @brief Get special token ID
 * @param tokenizer Tokenizer instance
 * @param token_type Token type ("bos", "eos", "pad", "unk")
 * @return Token ID or -1 if not found
 */
int qwen3_tokenizer_get_special_token(Qwen3Tokenizer* tokenizer, const char* token_type);

#ifdef __cplusplus
}
#endif

#endif // QWEN3_TOKENIZER_H