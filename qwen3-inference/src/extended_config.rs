use std::path::PathBuf;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::configuration::ModelConfig;
use crate::quantization::{QuantizationLevel, CpuTarget, MemoryLimits, CloudConfig};

/// Extended model configuration with advanced features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtendedModelConfig {
    /// Base model configuration from checkpoint
    pub base: ModelConfig,
    /// Quantization level for memory optimization
    pub quantization: QuantizationLevel,
    /// CPU-specific optimizations
    pub cpu_target: CpuTarget,
    /// Cloud provider configuration for hybrid inference
    pub cloud_config: Option<CloudConfig>,
    /// Memory usage constraints
    pub memory_limits: MemoryLimits,
    /// Model file paths
    pub model_paths: ModelPaths,
    /// Inference parameters
    pub inference_params: InferenceParameters,
    /// Advanced features configuration
    pub advanced: AdvancedConfig,
}

/// File paths for model components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPaths {
    /// Path to the main model weights
    pub model_path: PathBuf,
    /// Path to the tokenizer configuration
    pub tokenizer_path: Option<PathBuf>,
    /// Path to chat templates
    pub chat_template_path: Option<PathBuf>,
    /// Cache directory for temporary files
    pub cache_dir: Option<PathBuf>,
}

/// Inference parameters with sensible defaults
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceParameters {
    /// Temperature for sampling (0.0 = greedy, higher = more random)
    pub temperature: f32,
    /// Top-p sampling threshold (0.0-1.0)
    pub top_p: f32,
    /// Top-k sampling (0 = disabled)
    pub top_k: usize,
    /// Maximum number of tokens to generate
    pub max_tokens: usize,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Whether to use streaming responses
    pub streaming: bool,
    /// Whether to cache prompts
    pub enable_prompt_cache: bool,
    /// Context window management
    pub context_management: ContextManagement,
}

/// Context window management strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContextManagement {
    /// Fixed context window with truncation
    Fixed { max_length: usize },
    /// Sliding window with attention sink
    Sliding { window_size: usize, sink_size: usize },
    /// Dynamic context based on memory usage
    Dynamic { max_memory_ratio: f32 },
}

/// Advanced configuration options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedConfig {
    /// Enable parallel processing where possible
    pub parallel_processing: bool,
    /// Enable memory-mapped file access
    pub memory_mapped: bool,
    /// Enable KV cache optimization
    pub kv_cache_optimization: bool,
    /// Enable CPU-specific optimizations
    pub cpu_optimizations: bool,
    /// Enable GPU acceleration (if available)
    pub gpu_acceleration: bool,
    /// Logging level
    pub log_level: LogLevel,
    /// Performance monitoring
    pub performance_monitoring: bool,
    /// Auto-save configuration changes
    pub auto_save_config: bool,
}

/// Logging levels for the inference engine
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum LogLevel {
    Error,
    Warning,
    Info,
    Debug,
    Trace,
}

impl ExtendedModelConfig {
    /// Creates a new extended configuration with defaults
    pub fn new(base: ModelConfig) -> Self {
        let cpu_target = CpuTarget::detect();
        let memory_limits = MemoryLimits::for_cpu_target(cpu_target);
        let quantization = cpu_target.optimal_quantization();

        Self {
            base,
            quantization,
            cpu_target,
            cloud_config: None,
            memory_limits,
            model_paths: ModelPaths::default(),
            inference_params: InferenceParameters::default(),
            advanced: AdvancedConfig::default(),
        }
    }

    /// Creates configuration optimized for a specific CPU target
    pub fn for_cpu_target(base: ModelConfig, cpu: CpuTarget) -> Self {
        let memory_limits = MemoryLimits::for_cpu_target(cpu);
        let quantization = cpu.optimal_quantization();

        Self {
            base,
            quantization,
            cpu_target: cpu,
            cloud_config: None,
            memory_limits,
            model_paths: ModelPaths::default(),
            inference_params: InferenceParameters::default(),
            advanced: AdvancedConfig::default(),
        }
    }

    /// Enables cloud provider for hybrid inference
    pub fn with_cloud_provider(mut self, config: CloudConfig) -> Self {
        self.cloud_config = Some(config);
        self
    }

    /// Sets custom memory limits
    pub fn with_memory_limits(mut self, limits: MemoryLimits) -> Self {
        self.memory_limits = limits;
        self
    }

    /// Validates the configuration for consistency
    pub fn validate(&self) -> Result<()> {
        // Validate model paths
        if !self.model_paths.model_path.exists() {
            anyhow::bail!("Model file not found: {:?}", self.model_paths.model_path);
        }

        // Validate memory limits
        let estimated_memory = self.estimate_memory_usage()?;
        if estimated_memory > self.memory_limits.max_memory_mb {
            anyhow::bail!(
                "Estimated memory usage ({}MB) exceeds limit ({}MB)",
                estimated_memory,
                self.memory_limits.max_memory_mb
            );
        }

        // Validate quantization compatibility
        if !self.validate_quantization_compatibility() {
            anyhow::bail!("Quantization level incompatible with CPU target");
        }

        Ok(())
    }

    /// Estimates memory usage based on configuration
    pub fn estimate_memory_usage(&self) -> Result<usize> {
        let model_size = self.base.dim * self.base.vocab_size * 4; // Rough estimate
        let quantized_size = self.quantization.memory_usage(model_size);
        
        // Add overhead for KV cache, activations, etc.
        let kv_cache_size = self.base.n_layers * self.base.seq_len * self.base.dim * 2;
        let activation_size = self.base.seq_len * self.base.dim * 4;
        
        let total_bytes = quantized_size + kv_cache_size + activation_size;
        Ok(total_bytes / (1024 * 1024)) // Convert to MB
    }

    /// Validates quantization compatibility with CPU target
    fn validate_quantization_compatibility(&self) -> bool {
        match (self.quantization, self.cpu_target) {
            (QuantizationLevel::Int4 { .. }, CpuTarget::RaspberryPi4) => true,
            (QuantizationLevel::Int4 { .. }, CpuTarget::GenericArm) => true,
            (QuantizationLevel::Int8 { .. }, _) => true,
            (QuantizationLevel::Fp16, CpuTarget::RaspberryPi5) => true,
            (QuantizationLevel::Fp16, CpuTarget::IntelN100) => true,
            (QuantizationLevel::Fp32, CpuTarget::IntelN100) => true,
            (QuantizationLevel::Fp32, CpuTarget::GenericX86) => true,
            _ => false,
        }
    }

    /// Loads configuration from TOML file
    pub fn from_file(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        let content = std::fs::read_to_string(&path)
            .with_context(|| format!("Failed to read config file: {:?}", path))?;
        
        let config: Self = toml::from_str(&content)
            .with_context(|| format!("Failed to parse config file: {:?}", path))?;
        
        Ok(config)
    }

    /// Saves configuration to TOML file
    pub fn save_to_file(&self, path: impl Into<PathBuf>) -> Result<()> {
        let path = path.into();
        let content = toml::to_string_pretty(self)
            .context("Failed to serialize configuration")?;
        
        std::fs::write(&path, content)
            .with_context(|| format!("Failed to write config file: {:?}", path))?;
        
        Ok(())
    }

    /// Updates the configuration with a closure
    pub fn update_config<F>(mut self, f: F) -> Self
    where
        F: FnOnce(&mut Self),
    {
        f(&mut self);
        self
    }
}

impl Default for ModelPaths {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("model.bin"),
            tokenizer_path: Some(PathBuf::from("tokenizer.json")),
            chat_template_path: Some(PathBuf::from("chat_template.json")),
            cache_dir: Some(PathBuf::from(".cache")),
        }
    }
}

impl Default for InferenceParameters {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 50,
            max_tokens: 512,
            seed: None,
            streaming: true,
            enable_prompt_cache: true,
            context_management: ContextManagement::Fixed { max_length: 4096 },
        }
    }
}

impl Default for AdvancedConfig {
    fn default() -> Self {
        Self {
            parallel_processing: true,
            memory_mapped: true,
            kv_cache_optimization: true,
            cpu_optimizations: true,
            gpu_acceleration: false,
            log_level: LogLevel::Info,
            performance_monitoring: false,
            auto_save_config: true,
        }
    }
}

impl std::fmt::Display for LogLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LogLevel::Error => write!(f, "error"),
            LogLevel::Warning => write!(f, "warning"),
            LogLevel::Info => write!(f, "info"),
            LogLevel::Debug => write!(f, "debug"),
            LogLevel::Trace => write!(f, "trace"),
        }
    }
}

impl std::str::FromStr for LogLevel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "error" => Ok(LogLevel::Error),
            "warning" => Ok(LogLevel::Warning),
            "info" => Ok(LogLevel::Info),
            "debug" => Ok(LogLevel::Debug),
            "trace" => Ok(LogLevel::Trace),
            _ => Err(format!("Invalid log level: {}", s)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::configuration::ModelConfig;

    #[test]
    fn test_extended_config_creation() {
        let base = ModelConfig {
            dim: 2048,
            hidden_dim: 8192,
            n_layers: 24,
            n_heads: 32,
            n_kv_heads: 8,
            head_dim: 64,
            seq_len: 4096,
            vocab_size: 32000,
            group_size: 64,
            shared_classifier: true,
        };

        let config = ExtendedModelConfig::new(base);
        assert!(config.estimate_memory_usage().is_ok());
        assert!(config.validate().is_err()); // Model file doesn't exist
    }

    #[test]
    fn test_config_serialization() {
        let base = ModelConfig {
            dim: 1024,
            hidden_dim: 4096,
            n_layers: 12,
            n_heads: 16,
            n_kv_heads: 4,
            head_dim: 64,
            seq_len: 2048,
            vocab_size: 32000,
            group_size: 32,
            shared_classifier: true,
        };

        let config = ExtendedModelConfig::new(base);
        let serialized = toml::to_string(&config).unwrap();
        let deserialized: ExtendedModelConfig = toml::from_str(&serialized).unwrap();
        
        assert_eq!(config.base.dim, deserialized.base.dim);
        assert_eq!(config.quantization, deserialized.quantization);
    }

    #[test]
    fn test_quantization_compatibility() {
        let base = ModelConfig {
            dim: 1024,
            hidden_dim: 4096,
            n_layers: 12,
            n_heads: 16,
            n_kv_heads: 4,
            head_dim: 64,
            seq_len: 2048,
            vocab_size: 32000,
            group_size: 32,
            shared_classifier: true,
        };

        let config = ExtendedModelConfig::for_cpu_target(base.clone(), CpuTarget::IntelN100);
        assert!(config.validate_quantization_compatibility());

        let config = ExtendedModelConfig::for_cpu_target(base.clone(), CpuTarget::RaspberryPi4);
        assert!(config.validate_quantization_compatibility());
    }
}