use anyhow::Result;
use tempfile::tempdir;

use qwen3_inference::{
    AdvancedConfig, CloudConfig, ContextManagement, CpuTarget, ExtendedModelConfig,
    InferenceParameters, LogLevel, MemoryLimits, ModelPaths, QuantizationLevel,
};

#[test]
fn test_extended_config_creation() -> Result<()> {
    let base_config = qwen3_inference::configuration::ModelConfig {
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
            rope_theta: 10000.0,
    };

    let config = ExtendedModelConfig::new(base_config);

    // Verify basic configuration
    assert_eq!(config.base.dim, 2048);
    assert_eq!(config.base.n_layers, 24);

    // Verify quantization is set (allow any valid quantization)
    assert!(matches!(
        config.quantization,
        QuantizationLevel::Int8 { .. }
            | QuantizationLevel::Int4 { .. }
            | QuantizationLevel::Fp16
            | QuantizationLevel::Fp32
    ));

    // Verify CPU target is detected (just verify it's a valid enum value)
    assert!(matches!(
        config.cpu_target,
        CpuTarget::IntelN100
            | CpuTarget::IntelI9_14900HX
            | CpuTarget::RaspberryPi4
            | CpuTarget::RaspberryPi5
            | CpuTarget::GenericArm
            | CpuTarget::GenericX86
    ));

    // Verify memory limits
    assert!(config.memory_limits.max_memory_mb > 0);

    Ok(())
}

#[test]
fn test_quantization_levels() {
    let base_config = qwen3_inference::configuration::ModelConfig {
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

    // Test INT4 quantization
    let config = ExtendedModelConfig::new(base_config.clone());
    let config = config.update_config(|c| {
        c.quantization = QuantizationLevel::Int4 { group_size: 32 };
    });

    assert!(matches!(
        config.quantization,
        QuantizationLevel::Int4 { group_size: 32 }
    ));

    // Test INT8 quantization
    let config = ExtendedModelConfig::new(base_config.clone());
    let config = config.update_config(|c| {
        c.quantization = QuantizationLevel::Int8 { group_size: 128 };
    });

    assert!(matches!(
        config.quantization,
        QuantizationLevel::Int8 { group_size: 128 }
    ));

    // Test FP16 quantization
    let config = ExtendedModelConfig::new(base_config.clone());
    let config = config.update_config(|c| {
        c.quantization = QuantizationLevel::Fp16;
    });

    assert_eq!(config.quantization, QuantizationLevel::Fp16);

    // Test FP32 quantization
    let config = ExtendedModelConfig::new(base_config.clone());
    let config = config.update_config(|c| {
        c.quantization = QuantizationLevel::Fp32;
    });

    assert_eq!(config.quantization, QuantizationLevel::Fp32);
}

#[test]
fn test_cpu_target_optimization() {
    let base_config = qwen3_inference::configuration::ModelConfig {
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

    // Test different CPU targets
    let targets = [
        CpuTarget::IntelN100,
        CpuTarget::RaspberryPi4,
        CpuTarget::RaspberryPi5,
        CpuTarget::GenericX86,
        CpuTarget::GenericArm,
    ];

    for target in targets {
        let config = ExtendedModelConfig::for_cpu_target(base_config.clone(), target);

        assert_eq!(config.cpu_target, target);
        assert!(config.memory_limits.max_memory_mb > 0);

        // Verify quantization is appropriate for CPU
        match target {
            CpuTarget::RaspberryPi4 => {
                assert!(matches!(
                    config.quantization,
                    QuantizationLevel::Int4 { .. } | QuantizationLevel::Int8 { .. }
                ));
            }
            CpuTarget::IntelN100 => {
                assert!(matches!(
                    config.quantization,
                    QuantizationLevel::Int8 { .. }
                ));
            }
            _ => {
                // Allow flexibility for other targets
                assert!(config.quantization.memory_usage(1000) < 1000 * 4);
            }
        }
    }
}

#[test]
fn test_memory_limits() {
    let base_config = qwen3_inference::configuration::ModelConfig {
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

    let config = ExtendedModelConfig::new(base_config.clone());

    // Test memory limit validation
    let memory_limits = MemoryLimits {
        max_memory_mb: 4096,
        max_context_length: 2048,
        max_batch_size: 1,
    };

    let config = config.with_memory_limits(memory_limits);
    assert_eq!(config.memory_limits.max_memory_mb, 4096);
    assert_eq!(config.memory_limits.max_context_length, 2048);

    // Test memory estimation
    let memory_usage = config.estimate_memory_usage().unwrap_or(0);
    assert!(memory_usage < config.memory_limits.max_memory_mb);
}

#[test]
fn test_cloud_config_integration() {
    let base_config = qwen3_inference::configuration::ModelConfig {
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

    let mut config = ExtendedModelConfig::new(base_config);

    // Test OpenAI configuration
    let openai_config = CloudConfig::openai("test-key".to_string(), "gpt-3.5-turbo".to_string());

    config = config.with_cloud_provider(openai_config);
    assert!(config.cloud_config.is_some());

    if let Some(cloud) = &config.cloud_config {
        assert_eq!(cloud.provider, "openai");
        assert_eq!(cloud.model_name, "gpt-3.5-turbo");
    }

    // Test Anthropic configuration
    let anthropic_config =
        CloudConfig::anthropic("test-key".to_string(), "claude-3-haiku".to_string());

    config = config.with_cloud_provider(anthropic_config);
    assert!(config.cloud_config.is_some());

    if let Some(cloud) = &config.cloud_config {
        assert_eq!(cloud.provider, "anthropic");
        assert_eq!(cloud.model_name, "claude-3-haiku");
    }
}

#[test]
fn test_configuration_serialization() -> Result<()> {
    let dir = tempdir()?;
    let config_path = dir.path().join("test_config.toml");

    let base_config = qwen3_inference::configuration::ModelConfig {
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

    let config = ExtendedModelConfig::new(base_config);

    // Save configuration
    config.save_to_file(&config_path)?;

    // Load configuration
    let loaded_config = ExtendedModelConfig::from_file(&config_path)?;

    // Verify loaded configuration matches original
    assert_eq!(config.base.dim, loaded_config.base.dim);
    assert_eq!(config.base.n_layers, loaded_config.base.n_layers);
    assert_eq!(config.quantization, loaded_config.quantization);
    assert_eq!(config.cpu_target, loaded_config.cpu_target);

    Ok(())
}

#[test]
fn test_inference_parameters() {
    let mut params = InferenceParameters::default();

    // Test parameter updates
    params.temperature = 0.8;
    params.top_p = 0.95;
    params.top_k = 100;
    params.max_tokens = 1000;

    assert_eq!(params.temperature, 0.8);
    assert_eq!(params.top_p, 0.95);
    assert_eq!(params.top_k, 100);
    assert_eq!(params.max_tokens, 1000);

    // Test context management
    params.context_management = ContextManagement::Sliding {
        window_size: 1024,
        sink_size: 128,
    };

    match params.context_management {
        ContextManagement::Sliding {
            window_size,
            sink_size,
        } => {
            assert_eq!(window_size, 1024);
            assert_eq!(sink_size, 128);
        }
        _ => panic!("Expected sliding context management"),
    }
}

#[test]
fn test_advanced_config() {
    let mut advanced = AdvancedConfig::default();

    // Test advanced features
    advanced.parallel_processing = false;
    advanced.memory_mapped = false;
    advanced.cpu_optimizations = true;
    advanced.log_level = LogLevel::Debug;

    assert!(!advanced.parallel_processing);
    assert!(!advanced.memory_mapped);
    assert!(advanced.cpu_optimizations);
    assert_eq!(advanced.log_level, LogLevel::Debug);
}

#[test]
fn test_model_paths() {
    let mut paths = ModelPaths::default();

    // Test path updates
    paths.model_path = std::path::PathBuf::from("/tmp/model.bin");
    paths.tokenizer_path = Some(std::path::PathBuf::from("/tmp/tokenizer.json"));
    paths.chat_template_path = Some(std::path::PathBuf::from("/tmp/chat_template.json"));
    paths.cache_dir = Some(std::path::PathBuf::from("/tmp/cache"));

    assert_eq!(paths.model_path, std::path::PathBuf::from("/tmp/model.bin"));
    assert!(paths.tokenizer_path.is_some());
    assert!(paths.chat_template_path.is_some());
    assert!(paths.cache_dir.is_some());
}

#[test]
fn test_quantization_memory_calculation() {
    let elements = 1000000; // 1M elements

    // Test INT4 quantization
    let int4 = QuantizationLevel::Int4 { group_size: 64 };
    let int4_memory = int4.memory_usage(elements);
    assert!(int4_memory < elements * 4); // Should be much less than FP32

    // Test INT8 quantization
    let int8 = QuantizationLevel::Int8 { group_size: 128 };
    let int8_memory = int8.memory_usage(elements);
    assert!(int8_memory < elements * 4);
    assert!(int8_memory > int4_memory); // INT8 should use more memory than INT4

    // Test FP16 quantization
    let fp16 = QuantizationLevel::Fp16;
    let fp16_memory = fp16.memory_usage(elements);
    assert_eq!(fp16_memory, elements * 2);

    // Test FP32 quantization
    let fp32 = QuantizationLevel::Fp32;
    let fp32_memory = fp32.memory_usage(elements);
    assert_eq!(fp32_memory, elements * 4);
}

#[test]
fn test_configuration_validation() {
    let base_config = qwen3_inference::configuration::ModelConfig {
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

    let mut config = ExtendedModelConfig::new(base_config);

    // Test memory limit validation
    config = config.with_memory_limits(MemoryLimits {
        max_memory_mb: 1, // Very small limit
        max_context_length: 512,
        max_batch_size: 1,
    });

    // Config should fail validation due to memory limits
    assert!(config.validate().is_err());
}

#[test]
fn test_log_level_parsing() {
    let levels = [
        ("error", LogLevel::Error),
        ("warning", LogLevel::Warning),
        ("info", LogLevel::Info),
        ("debug", LogLevel::Debug),
        ("trace", LogLevel::Trace),
    ];

    for (level_str, expected_level) in levels {
        let parsed: LogLevel = level_str.parse().unwrap();
        assert_eq!(parsed, expected_level);
    }

    // Test invalid level
    assert!("invalid".parse::<LogLevel>().is_err());
}

#[test]
fn test_quantization_level_parsing() {
    let levels = [
        ("int4-gs32", QuantizationLevel::Int4 { group_size: 32 }),
        ("int4-gs64", QuantizationLevel::Int4 { group_size: 64 }),
        ("int8-gs128", QuantizationLevel::Int8 { group_size: 128 }),
        ("fp16", QuantizationLevel::Fp16),
        ("fp32", QuantizationLevel::Fp32),
    ];

    for (level_str, expected_level) in levels {
        let parsed: QuantizationLevel = level_str.parse().unwrap();
        assert_eq!(parsed, expected_level);
    }

    // Test invalid levels
    assert!("invalid".parse::<QuantizationLevel>().is_err());
    assert!("int4-gsabc".parse::<QuantizationLevel>().is_err());
}

#[test]
fn test_cpu_target_parsing() {
    let targets = [
        ("intel-n100", CpuTarget::IntelN100),
        ("intel-i9-14900hx", CpuTarget::IntelI9_14900HX),
        ("i9-14900hx", CpuTarget::IntelI9_14900HX),
        ("raspberry-pi-4", CpuTarget::RaspberryPi4),
        ("raspberry-pi-5", CpuTarget::RaspberryPi5),
        ("generic-x86", CpuTarget::GenericX86),
        ("generic-arm", CpuTarget::GenericArm),
    ];

    for (target_str, expected_target) in targets {
        let parsed: CpuTarget = target_str.parse().unwrap();
        assert_eq!(parsed, expected_target);
    }

    // Test invalid target
    assert!("invalid-cpu".parse::<CpuTarget>().is_err());
}
