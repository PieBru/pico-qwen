//! Advanced quantization system for Pico-Qwen
//!
//! Provides multiple quantization levels with configurable group sizes
//! and dynamic quantization capabilities.

use serde::{Deserialize, Serialize};

/// Supported quantization levels for memory optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationLevel {
    /// 4-bit integer quantization (ultra-low memory)
    Int4 { group_size: usize },
    /// 8-bit integer quantization (balanced performance/memory)
    Int8 { group_size: usize },
    /// 16-bit floating point (better quality)
    Fp16,
    /// 32-bit floating point (full precision)
    Fp32,
}

impl QuantizationLevel {
    /// Returns the number of bits used per element
    pub fn bits_per_element(&self) -> usize {
        match self {
            QuantizationLevel::Int4 { .. } => 4,
            QuantizationLevel::Int8 { .. } => 8,
            QuantizationLevel::Fp16 => 16,
            QuantizationLevel::Fp32 => 32,
        }
    }

    /// Returns the group size for grouped quantization
    pub fn group_size(&self) -> Option<usize> {
        match self {
            QuantizationLevel::Int4 { group_size } => Some(*group_size),
            QuantizationLevel::Int8 { group_size } => Some(*group_size),
            QuantizationLevel::Fp16 | QuantizationLevel::Fp32 => None,
        }
    }

    /// Estimates memory usage for given tensor size
    pub fn memory_usage(&self, elements: usize) -> usize {
        match self {
            QuantizationLevel::Int4 { group_size } => {
                // 4 bits per element + scale factors (f32 per group)
                (elements * 4) / 8 + (elements / group_size) * 4
            }
            QuantizationLevel::Int8 { group_size } => {
                // 1 byte per element + scale factors (f32 per group)
                elements + (elements / group_size) * 4
            }
            QuantizationLevel::Fp16 => elements * 2,
            QuantizationLevel::Fp32 => elements * 4,
        }
    }

    /// Checks if this quantization level supports dynamic adjustment
    pub fn supports_dynamic(&self) -> bool {
        matches!(
            self,
            QuantizationLevel::Int4 { .. } | QuantizationLevel::Int8 { .. }
        )
    }
}

impl std::fmt::Display for QuantizationLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QuantizationLevel::Int4 { group_size } => write!(f, "int4-gs{group_size}"),
            QuantizationLevel::Int8 { group_size } => write!(f, "int8-gs{group_size}"),
            QuantizationLevel::Fp16 => write!(f, "fp16"),
            QuantizationLevel::Fp32 => write!(f, "fp32"),
        }
    }
}

impl std::str::FromStr for QuantizationLevel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.to_lowercase();

        if s == "fp32" {
            return Ok(QuantizationLevel::Fp32);
        }

        if s == "fp16" {
            return Ok(QuantizationLevel::Fp16);
        }

        if let Some(group_size_str) = s.strip_prefix("int8-gs") {
            let group_size = group_size_str
                .parse()
                .map_err(|_| format!("Invalid group size: {group_size_str}"))?;
            return Ok(QuantizationLevel::Int8 { group_size });
        }

        if let Some(group_size_str) = s.strip_prefix("int4-gs") {
            let group_size = group_size_str
                .parse()
                .map_err(|_| format!("Invalid group size: {group_size_str}"))?;
            return Ok(QuantizationLevel::Int4 { group_size });
        }

        Err(format!("Invalid quantization level: {s}"))
    }
}

/// CPU-specific optimization targets for inference
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CpuTarget {
    /// Intel N100 processor (Alder Lake-N)
    IntelN100,
    /// Intel i9-14900HX (Raptor Lake-HX, 24 cores, 32 threads, 36MB cache, AVX2)
    IntelI9_14900HX,
    /// Raspberry Pi 4 (ARM Cortex-A72)
    RaspberryPi4,
    /// Raspberry Pi 5 (ARM Cortex-A76)
    RaspberryPi5,
    /// Generic x86_64 processor
    GenericX86,
    /// Generic ARM64 processor
    GenericArm,
}

impl CpuTarget {
    /// Detects the current CPU target based on runtime detection
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            // Try to detect Intel i9-14900HX via /proc/cpuinfo
            if let Ok(cpu_info) = std::fs::read_to_string("/proc/cpuinfo") {
                if cpu_info.contains("i9-14900HX") || cpu_info.contains("14900HX") {
                    return CpuTarget::IntelI9_14900HX;
                }
            }

            // Check for Intel features
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("avx512f") {
                return CpuTarget::IntelN100;
            }

            CpuTarget::GenericX86
        }

        #[cfg(target_arch = "aarch64")]
        {
            // Raspberry Pi detection based on /proc/cpuinfo
            if let Ok(cpu_info) = std::fs::read_to_string("/proc/cpuinfo") {
                if cpu_info.contains("BCM2835") && cpu_info.contains("Cortex-A72") {
                    return CpuTarget::RaspberryPi4;
                }
                if cpu_info.contains("BCM2712") && cpu_info.contains("Cortex-A76") {
                    return CpuTarget::RaspberryPi5;
                }
            }

            CpuTarget::GenericArm
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            CpuTarget::GenericX86
        }
    }

    /// Returns the optimal quantization level for this CPU
    pub fn optimal_quantization(&self) -> QuantizationLevel {
        match self {
            CpuTarget::IntelN100 => QuantizationLevel::Int8 { group_size: 64 },
            CpuTarget::IntelI9_14900HX => QuantizationLevel::Fp16, // High-end CPU can handle FP16
            CpuTarget::RaspberryPi4 => QuantizationLevel::Int4 { group_size: 32 },
            CpuTarget::RaspberryPi5 => QuantizationLevel::Int8 { group_size: 64 },
            CpuTarget::GenericX86 => QuantizationLevel::Int8 { group_size: 128 },
            CpuTarget::GenericArm => QuantizationLevel::Int4 { group_size: 64 },
        }
    }

    /// Returns the maximum recommended memory usage in MB
    pub fn max_memory_mb(&self) -> usize {
        match self {
            CpuTarget::IntelN100 => 4096,
            CpuTarget::IntelI9_14900HX => 32768, // 32GB for high-end desktop
            CpuTarget::RaspberryPi4 => 2048,
            CpuTarget::RaspberryPi5 => 4096,
            CpuTarget::GenericX86 => 8192,
            CpuTarget::GenericArm => 2048,
        }
    }
}

impl std::fmt::Display for CpuTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CpuTarget::IntelN100 => write!(f, "intel-n100"),
            CpuTarget::IntelI9_14900HX => write!(f, "intel-i9-14900hx"),
            CpuTarget::RaspberryPi4 => write!(f, "raspberry-pi-4"),
            CpuTarget::RaspberryPi5 => write!(f, "raspberry-pi-5"),
            CpuTarget::GenericX86 => write!(f, "generic-x86"),
            CpuTarget::GenericArm => write!(f, "generic-arm"),
        }
    }
}

impl std::str::FromStr for CpuTarget {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "intel-n100" => Ok(CpuTarget::IntelN100),
            "intel-i9-14900hx" | "i9-14900hx" => Ok(CpuTarget::IntelI9_14900HX),
            "raspberry-pi-4" => Ok(CpuTarget::RaspberryPi4),
            "raspberry-pi-5" => Ok(CpuTarget::RaspberryPi5),
            "generic-x86" => Ok(CpuTarget::GenericX86),
            "generic-arm" => Ok(CpuTarget::GenericArm),
            _ => Err(format!("Invalid CPU target: {s}")),
        }
    }
}

/// Memory usage constraints for the inference engine
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MemoryLimits {
    pub max_memory_mb: usize,
    pub max_context_length: usize,
    pub max_batch_size: usize,
}

impl Default for MemoryLimits {
    fn default() -> Self {
        Self {
            max_memory_mb: 2048,
            max_context_length: 4096,
            max_batch_size: 1,
        }
    }
}

impl MemoryLimits {
    /// Creates limits based on detected CPU target
    pub fn for_cpu_target(cpu: CpuTarget) -> Self {
        let max_memory_mb = cpu.max_memory_mb();
        Self {
            max_memory_mb,
            max_context_length: 4096,
            max_batch_size: 1,
        }
    }

    /// Validates that memory usage stays within limits
    pub fn validate_memory_usage(&self, usage_mb: usize) -> bool {
        usage_mb <= self.max_memory_mb
    }
}

/// Cloud provider configuration for hybrid inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudConfig {
    pub provider: String,
    pub api_key: String,
    pub model_name: String,
    pub base_url: Option<String>,
    pub timeout_seconds: u64,
    pub max_tokens: usize,
}

impl CloudConfig {
    /// Creates a new OpenAI configuration
    pub fn openai(api_key: String, model_name: String) -> Self {
        Self {
            provider: "openai".to_string(),
            api_key,
            model_name,
            base_url: None,
            timeout_seconds: 30,
            max_tokens: 2048,
        }
    }

    /// Creates a new Anthropic configuration
    pub fn anthropic(api_key: String, model_name: String) -> Self {
        Self {
            provider: "anthropic".to_string(),
            api_key,
            model_name,
            base_url: None,
            timeout_seconds: 30,
            max_tokens: 2048,
        }
    }
}
