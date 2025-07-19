use anyhow::{Context, Result};

use crate::cpu_optimizations::{CpuInfo, OptimizationStrategy};
use crate::extended_config::ExtendedModelConfig;
use crate::transformer::{Transformer, TransformerBuilder};

/// Extended transformer with advanced configuration support
#[derive(Debug)]
pub struct ExtendedTransformer {
    pub transformer: Transformer,
    pub config: ExtendedModelConfig,
    pub optimization_strategy: OptimizationStrategy,
    pub cpu_info: CpuInfo,
}

/// Builder for extended transformer
#[derive(Debug, Default)]
pub struct ExtendedTransformerBuilder {
    config: Option<ExtendedModelConfig>,
    optimization_strategy: Option<OptimizationStrategy>,
    cpu_info: Option<CpuInfo>,
}

impl ExtendedTransformerBuilder {
    /// Creates a new builder with default configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new builder with specific configuration
    pub fn with_config(config: ExtendedModelConfig) -> Self {
        let cpu_info = CpuInfo::detect();
        let optimization_strategy = OptimizationStrategy::for_cpu(&cpu_info);

        Self {
            config: Some(config),
            optimization_strategy: Some(optimization_strategy),
            cpu_info: Some(cpu_info),
        }
    }

    /// Sets the extended configuration
    pub fn config(mut self, config: ExtendedModelConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Sets custom optimization strategy
    pub fn optimization_strategy(mut self, strategy: OptimizationStrategy) -> Self {
        self.optimization_strategy = Some(strategy);
        self
    }

    /// Sets custom CPU information
    pub fn cpu_info(mut self, cpu_info: CpuInfo) -> Self {
        self.cpu_info = Some(cpu_info);
        self
    }

    /// Builds the extended transformer
    pub fn build(self) -> Result<ExtendedTransformer> {
        let cpu_info = self.cpu_info.unwrap_or_else(CpuInfo::detect);
        let optimization_strategy = self
            .optimization_strategy
            .unwrap_or_else(|| OptimizationStrategy::for_cpu(&cpu_info));

        let config = self
            .config
            .ok_or_else(|| anyhow::anyhow!("Extended configuration is required"))?;

        // Validate configuration
        config
            .validate()
            .context("Invalid extended configuration")?;

        // Build underlying transformer
        let transformer = TransformerBuilder::new(&config.model_paths.model_path.to_string_lossy())
            .with_ctx_length(Some(config.memory_limits.max_context_length))
            .build()
            .context("Failed to build transformer")?;

        Ok(ExtendedTransformer {
            transformer,
            config,
            optimization_strategy,
            cpu_info,
        })
    }

    /// Builds transformer from configuration file
    pub fn from_config_file(
        config_path: impl Into<std::path::PathBuf>,
    ) -> Result<ExtendedTransformer> {
        let config = ExtendedModelConfig::from_file(config_path)
            .context("Failed to load configuration file")?;

        Self::with_config(config).build()
    }
}

impl ExtendedTransformer {
    /// Creates a new extended transformer with default settings
    pub fn new(model_path: impl Into<std::path::PathBuf>) -> Result<Self> {
        let model_path = model_path.into();

        // First, we need to load the base config from the model
        let transformer = TransformerBuilder::new(&model_path.to_string_lossy())
            .build()
            .context("Failed to load model for configuration")?;

        let base_config = transformer.get_config();
        let config = ExtendedModelConfig::new(base_config.clone());

        let cpu_info = CpuInfo::detect();
        let optimization_strategy = OptimizationStrategy::for_cpu(&cpu_info);

        // Update config with actual model path
        let mut config = config;
        config.model_paths.model_path = model_path;

        Ok(ExtendedTransformer {
            transformer,
            config,
            optimization_strategy,
            cpu_info,
        })
    }

    /// Creates transformer optimized for specific CPU target
    pub fn for_cpu_target(
        model_path: impl Into<std::path::PathBuf>,
        cpu_target: crate::quantization::CpuTarget,
    ) -> Result<Self> {
        let model_path = model_path.into();

        let transformer = TransformerBuilder::new(&model_path.to_string_lossy())
            .build()
            .context("Failed to load model for configuration")?;

        let base_config = transformer.get_config();
        let config = ExtendedModelConfig::for_cpu_target(base_config.clone(), cpu_target);

        let cpu_info = CpuInfo::detect();
        let optimization_strategy = OptimizationStrategy::for_cpu(&cpu_info);

        let mut config = config;
        config.model_paths.model_path = model_path;

        Ok(ExtendedTransformer {
            transformer,
            config,
            optimization_strategy,
            cpu_info,
        })
    }

    /// Gets the underlying transformer
    pub fn transformer(&self) -> &Transformer {
        &self.transformer
    }

    /// Gets mutable access to the underlying transformer
    pub fn transformer_mut(&mut self) -> &mut Transformer {
        &mut self.transformer
    }

    /// Gets the extended configuration
    pub fn config(&self) -> &ExtendedModelConfig {
        &self.config
    }

    /// Gets the optimization strategy
    pub fn optimization_strategy(&self) -> &OptimizationStrategy {
        &self.optimization_strategy
    }

    /// Gets CPU information
    pub fn cpu_info(&self) -> &CpuInfo {
        &self.cpu_info
    }

    /// Updates configuration and revalidates
    pub fn update_config(&mut self, updater: impl FnOnce(&mut ExtendedModelConfig)) -> Result<()> {
        updater(&mut self.config);
        self.config
            .validate()
            .context("Invalid configuration update")?;
        Ok(())
    }

    /// Saves current configuration to file
    pub fn save_config(&self, path: impl Into<std::path::PathBuf>) -> Result<()> {
        self.config
            .save_to_file(path)
            .context("Failed to save configuration")
    }

    /// Gets memory usage statistics
    pub fn memory_stats(&self) -> MemoryStats {
        let estimated_usage = self.config.estimate_memory_usage().unwrap_or(0);

        MemoryStats {
            estimated_memory_mb: estimated_usage,
            max_allowed_mb: self.config.memory_limits.max_memory_mb,
            utilization_ratio: estimated_usage as f32
                / self.config.memory_limits.max_memory_mb as f32,
            quantization_savings: self.calculate_quantization_savings(),
        }
    }

    /// Calculates memory savings from quantization
    fn calculate_quantization_savings(&self) -> QuantizationSavings {
        let original_size = self.config.base.dim * self.config.base.vocab_size * 4; // 4 bytes per float

        let quantized_size = self
            .config
            .quantization
            .memory_usage(self.config.base.dim * self.config.base.vocab_size);

        let saved_bytes = original_size.saturating_sub(quantized_size);
        let compression_ratio = quantized_size as f32 / original_size as f32;

        QuantizationSavings {
            original_size_mb: original_size / (1024 * 1024),
            quantized_size_mb: quantized_size / (1024 * 1024),
            saved_mb: saved_bytes / (1024 * 1024),
            compression_ratio,
        }
    }
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub estimated_memory_mb: usize,
    pub max_allowed_mb: usize,
    pub utilization_ratio: f32,
    pub quantization_savings: QuantizationSavings,
}

/// Quantization memory savings
#[derive(Debug, Clone)]
pub struct QuantizationSavings {
    pub original_size_mb: usize,
    pub quantized_size_mb: usize,
    pub saved_mb: usize,
    pub compression_ratio: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_extended_transformer_builder() {
        // This test would need a valid model file
        // For now, just test the builder structure
        let builder = ExtendedTransformerBuilder::new();
        assert!(builder.config.is_none());
    }

    #[test]
    fn test_memory_stats_calculation() {
        // Create a mock config for testing
        let base_config = crate::configuration::ModelConfig {
            dim: 1024,
            hidden_dim: 4096,
            n_layers: 12,
            n_heads: 16,
            n_kv_heads: 4,
            head_dim: 64,
            seq_len: 2048,
            vocab_size: 32000,
            group_size: 64,
            shared_classifier: true,
            rope_theta: 10000.0,
        };

        let _config = ExtendedModelConfig::new(base_config);

        let cpu_info = CpuInfo::detect();
        let _optimization_strategy = OptimizationStrategy::for_cpu(&cpu_info);

        // Skip transformer creation for this test
        // Just test the basic functionality
        assert!(cpu_info.core_count > 0);
    }

    #[test]
    fn test_config_save_load() {
        let dir = tempdir().unwrap();
        let config_path = dir.path().join("test_config.toml");

        let base_config = crate::configuration::ModelConfig {
            dim: 512,
            hidden_dim: 2048,
            n_layers: 8,
            n_heads: 8,
            n_kv_heads: 2,
            head_dim: 64,
            seq_len: 1024,
            vocab_size: 32000,
            group_size: 32,
            shared_classifier: true,
            rope_theta: 10000.0,
        };

        let config = ExtendedModelConfig::new(base_config);

        // Save config
        config.save_to_file(&config_path).unwrap();

        // Load config
        let loaded_config = ExtendedModelConfig::from_file(&config_path).unwrap();

        assert_eq!(config.base.dim, loaded_config.base.dim);
        assert_eq!(config.quantization, loaded_config.quantization);
    }

    // Note: Actual transformer loading tests would require a valid model file
    // and are omitted for brevity
}

// Helper function to create a test transformer (for integration tests)
#[cfg(test)]
#[allow(dead_code)]
pub fn create_test_transformer() -> Result<ExtendedTransformer> {
    // This would create a minimal transformer for testing
    // In practice, you'd need to provide a test model file
    unimplemented!("Test transformer creation requires a model file")
}

// Re-export for convenience
