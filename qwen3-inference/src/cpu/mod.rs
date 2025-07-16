//! CPU-specific optimizations and feature detection

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fmt;

/// CPU feature detection and optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuInfo {
    pub vendor: String,
    pub brand: String,
    pub features: Vec<String>,
    pub cores: usize,
    pub threads: usize,
    pub cache_sizes: CacheSizes,
    pub architecture: Architecture,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheSizes {
    pub l1_data: usize,    // KB
    pub l1_instruction: usize, // KB
    pub l2: usize,         // KB
    pub l3: usize,         // KB
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Architecture {
    X86_64,
    Aarch64,
    Other(String),
}

impl fmt::Display for Architecture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Architecture::X86_64 => write!(f, "x86_64"),
            Architecture::Aarch64 => write!(f, "aarch64"),
            Architecture::Other(arch) => write!(f, "{}", arch),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CpuTarget {
    Generic,
    IntelN100,
    IntelI9_14900HX,
    RaspberryPi4,
    RaspberryPi5,
    AppleM1,
    AppleM2,
    Custom(String),
}

impl CpuTarget {
    pub fn detect() -> CpuTarget {
        #[cfg(target_arch = "x86_64")]
        {
            detect_x86_64_target()
        }
        
        #[cfg(target_arch = "aarch64")]
        {
            detect_aarch64_target()
        }
        
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            CpuTarget::Generic
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn detect_x86_64_target() -> CpuTarget {
    use raw_cpuid::CpuId;
    
    let cpuid = CpuId::new();
    
    // Try to get processor brand string
    if let Some(brand) = cpuid.get_processor_brand_string() {
        let brand_str = brand.as_str().to_lowercase();
        
        if brand_str.contains("intel") {
            if brand_str.contains("i9-14900hx") {
                return CpuTarget::IntelI9_14900HX;
            } else if brand_str.contains("n100") {
                return CpuTarget::IntelN100;
            }
        }
    }
    
    CpuTarget::Generic
}

#[cfg(target_arch = "aarch64")]
fn detect_aarch64_target() -> CpuTarget {
    // For ARM systems, use system detection
    #[cfg(target_os = "linux")]
    {
        use std::fs;
        if let Ok(cpuinfo) = fs::read_to_string("/proc/cpuinfo") {
            let cpuinfo = cpuinfo.to_lowercase();
            if cpuinfo.contains("raspberry pi 5") {
                return CpuTarget::RaspberryPi5;
            } else if cpuinfo.contains("raspberry pi 4") {
                return CpuTarget::RaspberryPi4;
            } else if cpuinfo.contains("apple") {
                if cpuinfo.contains("m2") {
                    return CpuTarget::AppleM2;
                } else if cpuinfo.contains("m1") {
                    return CpuTarget::AppleM1;
                }
            }
        }
    }
    
    CpuTarget::Generic
}

impl CpuInfo {
    pub fn detect() -> Result<Self> {
        let mut features = Vec::new();
        
        #[cfg(target_arch = "x86_64")]
        {
            use raw_cpuid::CpuId;
            let cpuid = CpuId::new();
            
            if let Some(feature_info) = cpuid.get_feature_info() {
                if feature_info.has_sse42() {
                    features.push("sse4.2".to_string());
                }
                if cfg!(target_feature = "avx2") {
                    features.push("avx2".to_string());
                }
                if cfg!(target_feature = "avx") {
                    features.push("avx".to_string());
                }
            }
        }
        
        #[cfg(target_arch = "aarch64")]
        {
            if cfg!(target_feature = "neon") {
                features.push("neon".to_string());
            }
            if cfg!(target_feature = "sve") {
                features.push("sve".to_string());
            }
        }
        
        let cores = num_cpus::get();
        let threads = num_cpus::get();
        
        Ok(CpuInfo {
            vendor: get_vendor(),
            brand: get_brand(),
            features,
            cores,
            threads,
            cache_sizes: detect_cache_sizes()?, 
            architecture: detect_architecture(),
        })
    }
    
    pub fn supports_feature(&self, feature: &str) -> bool {
        self.features.iter().any(|f| f.eq_ignore_ascii_case(feature))
    }
    
    pub fn get_optimization_level(&self) -> OptimizationLevel {
        match self.features.as_slice() {
            _ if self.supports_feature("avx512f") => OptimizationLevel::Avx512,
            _ if self.supports_feature("avx2") => OptimizationLevel::Avx2,
            _ if self.supports_feature("neon") => OptimizationLevel::Neon,
            _ => OptimizationLevel::Scalar,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationLevel {
    Scalar,
    Avx2,
    Avx512,
    Neon,
}

impl fmt::Display for OptimizationLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OptimizationLevel::Scalar => write!(f, "scalar"),
            OptimizationLevel::Avx2 => write!(f, "avx2"),
            OptimizationLevel::Avx512 => write!(f, "avx512"),
            OptimizationLevel::Neon => write!(f, "neon"),
        }
    }
}

fn get_vendor() -> String {
    #[cfg(target_arch = "x86_64")]
    {
        use raw_cpuid::CpuId;
        let cpuid = CpuId::new();
        if let Some(vendor) = cpuid.get_vendor_info() {
            vendor.as_str().to_string()
        } else {
            "unknown".to_string()
        }
    }
    
    #[cfg(target_arch = "aarch64")]
    {
        "ARM".to_string()
    }
    
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        "unknown".to_string()
    }
}

fn get_brand() -> String {
    #[cfg(target_arch = "x86_64")]
    {
        use raw_cpuid::CpuId;
        let cpuid = CpuId::new();
        if let Some(brand) = cpuid.get_processor_brand_string() {
            brand.as_str().to_string()
        } else {
            "unknown".to_string()
        }
    }
    
    #[cfg(target_arch = "aarch64")]
    {
        use std::fs;
        if let Ok(cpuinfo) = fs::read_to_string("/proc/cpuinfo") {
            for line in cpuinfo.lines() {
                if line.starts_with("Model name") || line.starts_with("model name") {
                    if let Some(name) = line.split(':').nth(1) {
                        return name.trim().to_string();
                    }
                }
            }
        }
        "unknown".to_string()
    }
    
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        "unknown".to_string()
    }
}

fn detect_cache_sizes() -> Result<CacheSizes> {
    #[cfg(target_arch = "x86_64")]
    {
        use raw_cpuid::CpuId;
        let cpuid = CpuId::new();
        
        let mut l1_data = 32;
        let mut l1_instruction = 32;
        let mut l2 = 512;
        let mut l3 = 8192;
        
        if let Some(cache_info) = cpuid.get_cache_parameters() {
            for cache in cache_info {
                match cache.level() {
                    1 => {
                        if cache.cache_type() == raw_cpuid::CacheType::Data {
                            l1_data = cache.associativity() * cache.physical_line_partitions() * cache.coherency_line_size() / 1024;
                        } else if cache.cache_type() == raw_cpuid::CacheType::Instruction {
                            l1_instruction = cache.associativity() * cache.physical_line_partitions() * cache.coherency_line_size() / 1024;
                        }
                    }
                    2 => {
                        l2 = cache.sets() * cache.associativity() * cache.physical_line_partitions() * cache.coherency_line_size() / 1024;
                    }
                    3 => {
                        l3 = cache.sets() * cache.associativity() * cache.physical_line_partitions() * cache.coherency_line_size() / 1024;
                    }
                    _ => {}
                }
            }
        }
        
        Ok(CacheSizes {
            l1_data,
            l1_instruction,
            l2,
            l3,
        })
    }
    
    #[cfg(target_arch = "aarch64")]
    {
        use std::fs;
        let mut l1_data = 32;
        let mut l2 = 512;
        let mut l3 = 8192;
        
        if let Ok(cache_info) = fs::read_to_string("/sys/devices/system/cpu/cpu0/cache/index0/size") {
            if let Some(size) = cache_info.trim().strip_suffix("K") {
                if let Ok(kb) = size.parse::<usize>() {
                    l1_data = kb;
                }
            }
        }
        
        if let Ok(cache_info) = fs::read_to_string("/sys/devices/system/cpu/cpu0/cache/index2/size") {
            if let Some(size) = cache_info.trim().strip_suffix("K") {
                if let Ok(kb) = size.parse::<usize>() {
                    l2 = kb;
                }
            }
        }
        
        if let Ok(cache_info) = fs::read_to_string("/sys/devices/system/cpu/cpu0/cache/index3/size") {
            if let Some(size) = cache_info.trim().strip_suffix("K") {
                if let Ok(kb) = size.parse::<usize>() {
                    l3 = kb;
                }
            }
        }
        
        Ok(CacheSizes {
            l1_data,
            l1_instruction: 32, // Common for ARM
            l2,
            l3,
        })
    }
    
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        Ok(CacheSizes {
            l1_data: 32,
            l1_instruction: 32,
            l2: 512,
            l3: 8192,
        })
    }
}

fn detect_architecture() -> Architecture {
    #[cfg(target_arch = "x86_64")]
    {
        Architecture::X86_64
    }
    
    #[cfg(target_arch = "aarch64")]
    {
        Architecture::Aarch64
    }
    
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        Architecture::Other(std::env::consts::ARCH.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cpu_detection() {
        let cpu_info = CpuInfo::detect().unwrap();
        assert!(!cpu_info.vendor.is_empty());
        assert!(cpu_info.cores > 0);
        assert!(cpu_info.threads > 0);
        println!("CPU Info: {:?}", cpu_info);
    }
    
    #[test]
    fn test_optimization_level() {
        let cpu_info = CpuInfo::detect().unwrap();
        let level = cpu_info.get_optimization_level();
        println!("Optimization level: {}", level);
        assert!(matches!(level, OptimizationLevel::Scalar | OptimizationLevel::Avx2 | OptimizationLevel::Avx512 | OptimizationLevel::Neon));
    }
    
    #[test]
    fn test_cpu_target_detection() {
        let target = CpuTarget::detect();
        println!("Detected CPU target: {:?}", target);
        assert!(matches!(target, CpuTarget::Generic | CpuTarget::IntelN100 | CpuTarget::IntelI9_14900HX | CpuTarget::RaspberryPi4 | CpuTarget::RaspberryPi5 | CpuTarget::AppleM1 | CpuTarget::AppleM2 | CpuTarget::Custom(_)));
    }
}