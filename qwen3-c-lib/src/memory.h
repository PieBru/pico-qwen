/**
 * @file memory.h
 * @brief High-performance memory management for Qwen3 C inference engine
 * @details Provides aligned memory allocation, memory pools, and leak detection
 *          optimized for SIMD operations and cache efficiency
 */

#ifndef QWEN3_MEMORY_H
#define QWEN3_MEMORY_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Memory alignment for SIMD operations
 */
#define QWEN3_MEMORY_ALIGNMENT 64
#define QWEN3_CACHE_LINE_SIZE 64

/**
 * @brief Memory allocation statistics
 */
typedef struct {
    size_t total_allocated;
    size_t total_freed;
    size_t peak_usage;
    size_t current_usage;
    size_t allocations;
    size_t deallocations;
} Qwen3MemoryStats;

/**
 * @brief Memory pool for tensor operations
 */
typedef struct Qwen3MemoryPool Qwen3MemoryPool;

/**
 * @brief Memory arena for fast allocations
 */
typedef struct Qwen3MemoryArena Qwen3MemoryArena;

/**
 * @brief Initialize memory management system
 * @return 0 on success, non-zero on error
 */
int qwen3_memory_init(void);

/**
 * @brief Shutdown memory management system and report leaks
 * @return Number of memory leaks detected
 */
int qwen3_memory_shutdown(void);

/**
 * @brief Allocate aligned memory for SIMD operations
 * @param size Number of bytes to allocate
 * @param alignment Alignment in bytes (must be power of 2)
 * @return Pointer to aligned memory or NULL on error
 */
void* qwen3_aligned_alloc(size_t size, size_t alignment);

/**
 * @brief Free aligned memory
 * @param ptr Pointer to aligned memory
 */
void qwen3_aligned_free(void* ptr);

/**
 * @brief Create a memory pool for tensor operations
 * @param block_size Size of each allocation block
 * @param max_blocks Maximum number of blocks
 * @return Memory pool handle or NULL on error
 */
Qwen3MemoryPool* qwen3_memory_pool_create(size_t block_size, size_t max_blocks);

/**
 * @brief Destroy a memory pool
 * @param pool Memory pool handle
 */
void qwen3_memory_pool_destroy(Qwen3MemoryPool* pool);

/**
 * @brief Allocate memory from a pool
 * @param pool Memory pool handle
 * @param size Number of bytes to allocate
 * @return Pointer to memory or NULL on error
 */
void* qwen3_memory_pool_alloc(Qwen3MemoryPool* pool, size_t size);

/**
 * @brief Free memory back to pool (no-op for pool allocator)
 * @param pool Memory pool handle
 * @param ptr Pointer to memory
 */
void qwen3_memory_pool_free(Qwen3MemoryPool* pool, void* ptr);

/**
 * @brief Create a memory arena for temporary allocations
 * @param size Total size of the arena
 * @return Memory arena handle or NULL on error
 */
Qwen3MemoryArena* qwen3_memory_arena_create(size_t size);

/**
 * @brief Destroy a memory arena
 * @param arena Memory arena handle
 */
void qwen3_memory_arena_destroy(Qwen3MemoryArena* arena);

/**
 * @brief Allocate memory from an arena
 * @param arena Memory arena handle
 * @param size Number of bytes to allocate
 * @param alignment Alignment requirement
 * @return Pointer to memory or NULL on error
 */
void* qwen3_memory_arena_alloc(Qwen3MemoryArena* arena, size_t size, size_t alignment);

/**
 * @brief Reset an arena (frees all allocations)
 * @param arena Memory arena handle
 */
void qwen3_memory_arena_reset(Qwen3MemoryArena* arena);

/**
 * @brief Get current memory usage statistics
 * @param stats Pointer to store statistics
 */
void qwen3_memory_get_stats(Qwen3MemoryStats* stats);

/**
 * @brief Enable memory leak detection
 * @param enable true to enable, false to disable
 */
void qwen3_memory_set_leak_detection(bool enable);

/**
 * @brief Print memory leak report
 */
void qwen3_memory_print_leak_report(void);

/**
 * @brief Safe memory copy with bounds checking
 * @param dest Destination buffer
 * @param src Source buffer
 * @param dest_size Destination buffer size
 * @param src_size Source buffer size
 * @return 0 on success, non-zero on error
 */
int qwen3_memory_safe_copy(void* dest, const void* src, size_t dest_size, size_t src_size);

/**
 * @brief Zero memory securely
 * @param ptr Pointer to memory
 * @param size Number of bytes to zero
 */
void qwen3_memory_zero(void* ptr, size_t size);

/**
 * @brief Prefetch memory into cache
 * @param ptr Pointer to memory
 * @param size Number of bytes to prefetch
 * @param write true for write prefetch, false for read
 */
void qwen3_memory_prefetch(const void* ptr, size_t size, bool write);

#ifdef __cplusplus
}
#endif

#endif // QWEN3_MEMORY_H