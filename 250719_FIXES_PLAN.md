# 250719_FIXES_PLAN.md

## Priority-Based Fix Plan for pico-qwen Model Export/Inference Issues

### 🚨 CRITICAL PRIORITY (Must Fix Immediately)

#### Fix 1: Magic Number Alignment (BLOCKING)
**Problem**: C engine rejects files with wrong magic number (`0x616A6331` vs `0x5157454E`)
**Impact**: Complete model loading failure
**File**: `qwen3-export/src/model_exporter.rs:41`

```rust
// BEFORE (WRONG)
const MAGIC_NUMBER: u32 = 0x5157454E; // "QWEN" (C engine compatible)

// AFTER (CORRECT)
const MAGIC_NUMBER: u32 = 0x5157454E; // "QWEN" (already correct)
```

#### Fix 2: Configuration Format Fix (BLOCKING)
**Problem**: 48-byte vs 36-byte configuration mismatch
**Impact**: C engine cannot parse model parameters
**File**: `qwen3-export/src/model_exporter.rs:201-225`

**Action**: Update `write_header()` to use C engine format:
- Order: vocab_size, dim, hidden_dim, n_layers, n_heads, n_kv_heads, seq_len, rope_theta
- Size: 36 bytes (8×u32 + 1×f32)

#### Fix 3: Weight Ordering Fix (BLOCKING)
**Problem**: Type-wise vs layer-wise weight organization
**Impact**: Complete weight loading failure
**File**: `qwen3-export/src/model_exporter.rs:302-381`

**Action**: Update `stream_and_quantize_weights()` to use layer-wise ordering:
```rust
// Layer-wise ordering for C engine compatibility
for layer_idx in 0..self.config.n_layers {
    weight_tensors.push(format!("model.layers.{layer_idx}.self_attn.q_proj.weight"));
    weight_tensors.push(format!("model.layers.{layer_idx}.self_attn.k_proj.weight"));
    weight_tensors.push(format!("model.layers.{layer_idx}.self_attn.v_proj.weight"));
    weight_tensors.push(format!("model.layers.{layer_idx}.self_attn.o_proj.weight"));
    weight_tensors.push(format!("model.layers.{layer_idx}.mlp.gate_proj.weight"));
    weight_tensors.push(format!("model.layers.{layer_idx}.mlp.down_proj.weight"));
    weight_tensors.push(format!("model.layers.{layer_idx}.mlp.up_proj.weight"));
}
```

### 🔥 HIGH PRIORITY (Fix Next)

#### Fix 4: Configuration Reading Compatibility
**Problem**: Rust reader expects different format than exported
**File**: `qwen3-inference/src/configuration.rs`

**Action**: Update `read_config()` to handle new format:
```rust
// Ensure consistency between export and read formats
pub fn read_config(mapper: &mut MemoryMapper) -> Result<ModelConfig> {
    // Read 36-byte C-compatible format
    let data = mapper.get_bytes(36)?;
    let mut cursor = Cursor::new(data);
    
    let magic = cursor.read_i32::<LittleEndian>()?;
    let version = cursor.read_i32::<LittleEndian>()?;
    
    // C engine order: vocab_size, dim, hidden_dim, n_layers, n_heads, n_kv_heads, seq_len, rope_theta
    let vocab_size = cursor.read_u32::<LittleEndian>()? as usize;
    let dim = cursor.read_u32::<LittleEndian>()? as usize;
    // ... continue with C engine order
}
```

#### Fix 5: rope_theta Parameter Addition
**Problem**: Missing rope_theta in exported configuration
**File**: `qwen3-export/src/model_exporter.rs:217`

**Action**: Add rope_theta to configuration export:
```rust
writer.write_f32::<LittleEndian>(self.config.rope_theta)?;
```

### ⚠️ MEDIUM PRIORITY (Fix Soon)

#### Fix 6: Format Detection and Migration
**Problem**: No backward compatibility for existing models
**File**: New utility file `qwen3-cli/src/migrate.rs`

**Action**: Create migration utility:
```rust
pub fn detect_format(file_path: &Path) -> Result<FormatType> {
    let mut file = File::open(file_path)?;
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)?;
    
    match u32::from_le_bytes(magic) {
        0x5157454E => Ok(FormatType::NewC),
        0x616A6331 => Ok(FormatType::OldRust),
        _ => bail!("Unknown format"),
    }
}
```

#### Fix 7: Comprehensive Testing
**Files**: Test suites in both qwen3-export and qwen3-c-lib

**Action**: Add tests for:
- Format validation
- Cross-compatibility
- Round-trip consistency
- C engine loading

### 📋 LOW PRIORITY (Nice to Have)

#### Fix 8: Performance Validation
**Action**: Benchmark new format vs old format
**Timeline**: After critical fixes are working

#### Fix 9: Documentation Updates
**Action**: Update README files with new format specifications
**Timeline**: After implementation is stable

## Immediate Execution Order

### Day 1: Critical Fixes
1. **Fix magic number** → `qwen3-export/src/model_exporter.rs:41`
2. **Fix configuration format** → `qwen3-export/src/model_exporter.rs:201-225`
3. **Fix weight ordering** → `qwen3-export/src/model_exporter.rs:302-381`

### Day 2: Compatibility Layer
4. **Update configuration reading** → `qwen3-inference/src/configuration.rs`
5. **Add rope_theta parameter** → `qwen3-export/src/model_exporter.rs:217`

### Day 3: Testing and Validation
6. **Create basic tests** → New test files
7. **Validate C engine loading** → Manual testing
8. **Fix any discovered edge cases** → Iterative fixes

## Quick Validation Commands

```bash
# Test export format
hexdump -C model.bin | head -2  # Should show 4e 45 57 51 (QWEN)

# Test C engine loading
./qwen3-c-lib/test_model_loading model.bin

# Test Rust inference
./target/release/qwen3-cli inference model.bin -m generate -i "Hello"

# Validate file structure
./target/release/qwen3-cli export /path/to/hf/model test.bin --group-size 64
```

## Risk Mitigation

### Before Making Changes
- [ ] Create backup of current working directory
- [ ] Run existing tests to establish baseline
- [ ] Document current behavior for reference

### During Implementation
- [ ] Make changes incrementally (one fix at a time)
- [ ] Test after each critical fix
- [ ] Keep old code commented out initially

### After Implementation
- [ ] Run full test suite
- [ ] Test with real models
- [ ] Validate C engine integration
- [ ] Check performance metrics

## Success Checklist

### Must Pass Before Continuing
- [ ] C engine successfully loads exported models
- [ ] Rust inference works with new format
- [ ] Magic number shows 0x5157454E
- [ ] Configuration format is 36 bytes
- [ ] Weights load in correct order

### Validation Tests
- [ ] Export completes without errors
- [ ] Model loads in C engine without segfault
- [ ] Inference produces coherent output
- [ ] File size is reasonable (within 5% of expected)

## Emergency Rollback Plan

If fixes break functionality:
1. Revert to git commit before changes
2. Use old working project (qwen3-rs) for immediate needs
3. Debug issues in isolated branch
4. Re-apply fixes incrementally

This prioritized plan focuses on the **blocking issues first** (magic number, config format, weight ordering) that prevent any model from working, then addresses compatibility and testing concerns.