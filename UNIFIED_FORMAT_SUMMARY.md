# Qwen3 Unified Binary Format - Implementation Summary

## ✅ Unified Format Successfully Implemented

Both the C engine and Rust engine now use the same `.bin` file format with a unified metadata header.

## 📋 Unified Format Specification

### Header Structure (256 bytes total)
```
Offset | Field        | Type   | Description
-------|--------------|--------|-------------
0      | magic        | u32    | Magic number: 0x5157454E ("QWEN" in little-endian)
4      | version      | u32    | Format version: 1
8      | vocab_size   | u32    | Vocabulary size
12     | dim          | u32    | Model dimension
16     | hidden_dim   | u32    | Feed-forward hidden dimension
20     | n_layers     | u32    | Number of transformer layers
24     | n_heads      | u32    | Number of attention heads
28     | n_kv_heads   | u32    | Number of key/value heads (for GQA)
32     | seq_len      | u32    | Maximum sequence length
36     | rope_theta   | f32    | RoPE base frequency (e.g., 10000.0)
40-255 | padding      | bytes  | Reserved for future use (zeroed)
```

### Binary Layout
```
[0-255]   Header (256 bytes)
[256+]    Normalization weights (float32)
          - Attention norms: [n_layers × dim]
          - FFN norms: [n_layers × dim]
          - Q-norm weights: [n_layers × head_dim]
          - K-norm weights: [n_layers × head_dim]
          - Final norm: [dim]
[...]     Quantized weights (int8 + float32 scales)
          - Layer weights in order: WQ, WK, WV, WO, W1, W2, W3
          - Token embeddings: [vocab_size × dim]
          - Classifier weights (if not shared)
```

## 🔧 Changes Made

### C Engine Updates
- **File**: `qwen3-c-lib/src/model.c`
- **No changes needed** - already using the correct format
- **Magic number**: 0x5157454E ("QWEN")
- **Parameter order**: vocab_size, dim, hidden_dim, n_layers, n_heads, n_kv_heads, seq_len, rope_theta

### Rust Engine Updates
1. **Configuration Reader** (`qwen3-inference/src/configuration.rs`):
   - Updated magic number from 0x616a6331 to 0x5157454E
   - Updated parameter order to match C engine
   - Added rope_theta support
   - Reduced config size from 48 to 36 bytes

2. **Model Exporter** (`qwen3-export/src/model_exporter.rs`):
   - Updated header writing to match C engine format
   - Added CONFIG_SIZE constant (36 bytes)
   - Fixed parameter order in header

## ✅ Compatibility Verification

### Tests Passed
- **C Engine**: All model loading tests pass
- **Rust Engine**: All export tests pass
- **Cross-compatibility**: Both engines can read/write the same format

### Key Features
1. **Shared magic number**: 0x5157454E ("QWEN")
2. **Consistent parameter order**: Standardized across engines
3. **Fixed header size**: 256 bytes for both
4. **RoPE theta support**: Included in both engines
5. **Group size determination**: Determined from quantized weights

## 🚀 Usage

### Creating a unified binary (Rust):
```bash
cargo run --release -p qwen3-cli -- export /path/to/hf/model /path/to/output.bin --group-size 64
```

### Loading in C:
```c
Qwen3Model* model = qwen3_model_load("/path/to/output.bin", 0);
```

### Loading in Rust:
```rust
let transformer = TransformerBuilder::new("/path/to/output.bin").build()?;
```

## 🔍 Format Verification

You can verify the unified format using:
```bash
# Test both engines can read the same file
./test_unified_format.sh
```

## 📊 Benefits

1. **Single file format**: One `.bin` file works with both engines
2. **Transparent compatibility**: No conversion needed
3. **Future-proof**: 256-byte header allows for extensions
4. **Performance optimized**: Minimal overhead, direct memory mapping
5. **Standardized**: Consistent across all tools and engines