//! Rust FFI bindings for the Qwen3 C inference engine
//!
//! Provides safe Rust wrappers around the C API for loading and running
//! Qwen3 models with maximum CPU performance.

use std::ffi::{CString, CStr};
use std::os::raw::{c_char, c_uint};

// Link to the C library
#[link(name = "qwen3_inference")]
unsafe extern "C" {
    // Model loading
    fn qwen3_model_load(checkpoint_path: *const c_char, ctx_length: c_uint) -> *mut Qwen3Model;
    fn qwen3_model_load_ex(options: *const Qwen3LoadOptions) -> *mut Qwen3Model;
    fn qwen3_model_free(model: *mut Qwen3Model);
    
    // Model information
    fn qwen3_model_get_config(model: *const Qwen3Model) -> *const Qwen3ModelConfig;
    fn qwen3_model_validate(model: *const Qwen3Model) -> bool;
    fn qwen3_model_get_info(model: *const Qwen3Model) -> *const c_char;
    
    // Error handling
    fn qwen3_get_last_error() -> *const c_char;
}

// C struct definitions
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Qwen3ModelConfig {
    pub dim: u32,
    pub hidden_dim: u32,
    pub n_layers: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub vocab_size: u32,
    pub max_seq_len: u32,
    pub head_dim: u32,
    pub shared_classifier: bool,
    pub group_size: u32,
    pub rope_theta: f32,
}

#[repr(C)]
pub struct Qwen3LoadOptions {
    pub checkpoint_path: *const c_char,
    pub context_length: u32,
    pub validate_weights: bool,
    pub use_memory_pool: bool,
}

#[repr(C)]
pub struct Qwen3Model {
    _private: [u8; 0],
}

/// Safe Rust wrapper for Qwen3 model
pub struct Qwen3ModelHandle {
    model: *mut Qwen3Model,
}

impl Qwen3ModelHandle {
    /// Load a model from file
    pub fn load(checkpoint_path: &str, context_length: Option<u32>) -> anyhow::Result<Self> {
        let path_c = CString::new(checkpoint_path)?;
        let ctx_len = context_length.unwrap_or(0);
        
        let model = unsafe { qwen3_model_load(path_c.as_ptr(), ctx_len) };
        if model.is_null() {
            let error = unsafe {
                CStr::from_ptr(qwen3_get_last_error())
                    .to_string_lossy()
                    .into_owned()
            };
            anyhow::bail!("Failed to load model: {}", error);
        }
        
        Ok(Self { model })
    }
    
    /// Load a model with detailed options
    pub fn load_with_options(options: LoadOptions) -> anyhow::Result<Self> {
        let path_c = CString::new(options.checkpoint_path)?;
        
        let c_options = Qwen3LoadOptions {
            checkpoint_path: path_c.as_ptr(),
            context_length: options.context_length.unwrap_or(0),
            validate_weights: options.validate_weights,
            use_memory_pool: options.use_memory_pool,
        };
        
        let model = unsafe { qwen3_model_load_ex(&c_options) };
        if model.is_null() {
            let error = unsafe {
                CStr::from_ptr(qwen3_get_last_error())
                    .to_string_lossy()
                    .into_owned()
            };
            anyhow::bail!("Failed to load model: {}", error);
        }
        
        Ok(Self { model })
    }
    
    /// Get model configuration
    pub fn config(&self) -> anyhow::Result<Qwen3ModelConfig> {
        let config_ptr = unsafe { qwen3_model_get_config(self.model) };
        if config_ptr.is_null() {
            anyhow::bail!("Failed to get model configuration");
        }
        
        Ok(unsafe { *config_ptr })
    }
    
    /// Validate model integrity
    pub fn validate(&self) -> bool {
        unsafe { qwen3_model_validate(self.model) }
    }
    
    /// Get model information as string
    pub fn info(&self) -> anyhow::Result<String> {
        let info_ptr = unsafe { qwen3_model_get_info(self.model) };
        if info_ptr.is_null() {
            anyhow::bail!("Failed to get model info");
        }
        
        let info = unsafe {
            CStr::from_ptr(info_ptr)
                .to_string_lossy()
                .into_owned()
        };
        Ok(info)
    }
}

impl Drop for Qwen3ModelHandle {
    fn drop(&mut self) {
        unsafe {
            qwen3_model_free(self.model);
        }
    }
}

/// Options for loading a model
#[derive(Debug, Clone)]
pub struct LoadOptions {
    pub checkpoint_path: String,
    pub context_length: Option<u32>,
    pub validate_weights: bool,
    pub use_memory_pool: bool,
}

impl LoadOptions {
    pub fn new(checkpoint_path: String) -> Self {
        Self {
            checkpoint_path,
            context_length: None,
            validate_weights: true,
            use_memory_pool: true,
        }
    }
    
    pub fn with_context_length(mut self, length: u32) -> Self {
        self.context_length = Some(length);
        self
    }
    
    pub fn with_validation(mut self, validate: bool) -> Self {
        self.validate_weights = validate;
        self
    }
    
    pub fn with_memory_pool(mut self, use_pool: bool) -> Self {
        self.use_memory_pool = use_pool;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use tempfile::NamedTempFile;
    #[allow(unused_imports)]
    use std::io::Write;
    
    #[test]
    fn test_model_handle_creation() {
        // This is a basic test to ensure the struct compiles
        // Real tests would require actual model files
        assert!(true);
    }
    
    #[test]
    fn test_load_options() {
        let options = LoadOptions::new("test.bin".to_string())
            .with_context_length(1024)
            .with_validation(true)
            .with_memory_pool(false);
        
        assert_eq!(options.checkpoint_path, "test.bin");
        assert_eq!(options.context_length, Some(1024));
        assert_eq!(options.validate_weights, true);
        assert_eq!(options.use_memory_pool, false);
    }
}