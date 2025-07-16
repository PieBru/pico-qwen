#[cfg(test)]
mod tests {
    use qwen3_inference::*;

    #[test]
    fn test_quantization_levels() {
        let int4 = QuantizationLevel::Int4 { group_size: 32 };
        let int8 = QuantizationLevel::Int8 { group_size: 64 };
        let fp16 = QuantizationLevel::Fp16;
        let fp32 = QuantizationLevel::Fp32;

        // Test basic properties
        assert_eq!(int4.bits_per_element(), 4);
        assert_eq!(int8.bits_per_element(), 8);
        assert_eq!(fp16.bits_per_element(), 16);
        assert_eq!(fp32.bits_per_element(), 32);

        // Test memory usage calculation
        let elements = 1000;
        assert!(int4.memory_usage(elements) < int8.memory_usage(elements));
        assert!(int8.memory_usage(elements) < fp16.memory_usage(elements));
        assert!(fp16.memory_usage(elements) < fp32.memory_usage(elements));
    }

    #[test]
    fn test_cpu_target_detection() {
        let cpu_target = CpuTarget::detect();
        
        // Should detect some CPU target
        assert!(!matches!(cpu_target, CpuTarget::GenericX86) || 
                !matches!(cpu_target, CpuTarget::GenericArm));
        
        // Test memory limits for different targets
        let limits = MemoryLimits::for_cpu_target(cpu_target);
        assert!(limits.max_memory_mb > 0);
    }

    #[test]
    fn test_quantization_parsing() {
        let int4: QuantizationLevel = "int4-gs32".parse().unwrap();
        let int8: QuantizationLevel = "int8-gs64".parse().unwrap();
        let fp16: QuantizationLevel = "fp16".parse().unwrap();
        let fp32: QuantizationLevel = "fp32".parse().unwrap();

        assert!(matches!(int4, QuantizationLevel::Int4 { group_size: 32 }));
        assert!(matches!(int8, QuantizationLevel::Int8 { group_size: 64 }));
        assert_eq!(fp16, QuantizationLevel::Fp16);
        assert_eq!(fp32, QuantizationLevel::Fp32);
    }

    #[test]
    fn test_cpu_target_parsing() {
        let intel_n100: CpuTarget = "intel-n100".parse().unwrap();
        let intel_i9: CpuTarget = "intel-i9-14900hx".parse().unwrap();
        let i9_short: CpuTarget = "i9-14900hx".parse().unwrap();
        let pi4: CpuTarget = "raspberry-pi-4".parse().unwrap();
        let pi5: CpuTarget = "raspberry-pi-5".parse().unwrap();

        assert_eq!(intel_n100, CpuTarget::IntelN100);
        assert_eq!(intel_i9, CpuTarget::IntelI9_14900HX);
        assert_eq!(i9_short, CpuTarget::IntelI9_14900HX);
        assert_eq!(pi4, CpuTarget::RaspberryPi4);
        assert_eq!(pi5, CpuTarget::RaspberryPi5);
    }

    #[test]
    fn test_memory_limits_calculation() {
        let cpu_info = CpuInfo::detect();
        let strategy = OptimizationStrategy::for_cpu(&cpu_info);
        
        // Test strategy properties
        assert!(strategy.simd_width > 0);
        assert!(strategy.alignment > 0);
        
        // Test tile size calculation
        let (m, n, k) = strategy.gemm_tile_size();
        assert!(m > 0 && m <= 16);
        assert!(n > 0 && n <= 16);
        assert!(k > 0 && k <= 16);
    }

    #[test]
    fn test_cpu_info_detection() {
        let cpu_info = CpuInfo::detect();
        
        // Basic assertions
        assert!(cpu_info.core_count > 0);
        assert!(cpu_info.thread_count >= cpu_info.core_count);
        assert!(cpu_info.cache_size > 0);
        
        println!("Detected CPU: {} cores, {} threads", 
                 cpu_info.core_count, cpu_info.thread_count);
    }

    #[test]
    fn test_cloud_config_creation() {
        let openai_config = CloudConfig::openai(
            "test-key".to_string(),
            "gpt-3.5-turbo".to_string()
        );
        
        assert_eq!(openai_config.provider, "openai");
        assert_eq!(openai_config.model_name, "gpt-3.5-turbo");
        assert_eq!(openai_config.api_key, "test-key");
    }
}