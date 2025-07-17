//! CPU-specific optimizations for Pico-Qwen
//!
//! Provides runtime CPU feature detection and optimization strategies
//! for different processor architectures.

use std::arch::x86_64::*;
use std::fs;

/// Detailed CPU information for optimization decisions
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuInfo {
    pub vendor: CpuVendor,
    pub features: Vec<CpuFeature>,
    pub cache_size: usize,
    pub memory_bandwidth: usize,
    pub core_count: usize,
    pub thread_count: usize,
    pub cpu_family: u32,
    pub cpu_model: u32,
    pub cpu_stepping: u32,
}

/// CPU vendor identification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuVendor {
    Intel,
    Amd,
    Arm,
    Unknown,
}

/// CPU features for optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuFeature {
    // x86_64 features
    Sse,
    Sse2,
    Sse3,
    Sse41,
    Sse42,
    Avx,
    Avx2,
    Fma,
    Avx512F,
    Avx512VL,
    Avx512BW,
    Avx512DQ,
    Vnni,
    Bmi1,
    Bmi2,
    Popcnt,

    // ARM features
    Neon,
    Fp16,
    Sve,
    Dotprod,
    Aes,
    Sha2,
}

/// Optimization strategy based on CPU capabilities
#[derive(Debug, Clone)]
pub struct OptimizationStrategy {
    pub quantization: crate::quantization::QuantizationLevel,
    pub simd_width: usize,
    pub use_fma: bool,
    pub use_avx512: bool,
    pub alignment: usize,
    pub cache_blocking: CacheBlockingStrategy,
    pub parallel_strategy: ParallelStrategy,
}

/// Cache blocking strategy for memory performance
#[derive(Debug, Clone)]
pub struct CacheBlockingStrategy {
    pub l1_block_size: usize,
    pub l2_block_size: usize,
    pub l3_block_size: usize,
    pub vector_width: usize,
}

/// Parallel processing strategies
#[derive(Debug, Clone)]
pub enum ParallelStrategy {
    SingleThreaded,
    RayonThreads { max_threads: usize },
    RayonPool { pool_size: usize },
    CustomPool { threads: Vec<usize> },
}

impl CpuInfo {
    /// Detects CPU information at runtime
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            detect_x86_64()
        }

        #[cfg(target_arch = "aarch64")]
        {
            detect_aarch64()
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self::generic_fallback()
        }
    }

    /// Creates a generic fallback for unknown architectures
    #[allow(dead_code)]
    pub(crate) fn generic_fallback() -> Self {
        Self {
            vendor: CpuVendor::Unknown,
            features: Vec::new(),
            cache_size: 4 * 1024 * 1024, // 4MB default
            memory_bandwidth: 25_600,    // 25.6 GB/s default
            core_count: 4,
            thread_count: 4,
            cpu_family: 0,
            cpu_model: 0,
            cpu_stepping: 0,
        }
    }

    /// Checks if a specific feature is supported
    pub fn has_feature(&self, feature: CpuFeature) -> bool {
        self.features.contains(&feature)
    }

    /// Determines optimal quantization based on CPU capabilities
    pub fn optimal_quantization(&self) -> crate::quantization::QuantizationLevel {
        use crate::quantization::QuantizationLevel;

        let total_memory_mb = self.estimate_total_memory_mb();

        match (self.vendor, total_memory_mb) {
            (CpuVendor::Intel, mem) if mem >= 8192 && self.has_feature(CpuFeature::Avx2) => {
                QuantizationLevel::Int8 { group_size: 64 }
            }
            (CpuVendor::Intel, mem) if mem >= 4096 => QuantizationLevel::Int8 { group_size: 128 },
            (CpuVendor::Amd, mem) if mem >= 8192 && self.has_feature(CpuFeature::Avx2) => {
                QuantizationLevel::Int8 { group_size: 64 }
            }
            (CpuVendor::Arm, _) if self.has_feature(CpuFeature::Neon) => {
                QuantizationLevel::Int8 { group_size: 64 }
            }
            _ => QuantizationLevel::Int4 { group_size: 64 },
        }
    }

    /// Estimates total system memory in MB
    pub fn estimate_total_memory_mb(&self) -> usize {
        // Try to read from /proc/meminfo on Linux
        #[cfg(target_os = "linux")]
        {
            if let Ok(meminfo) = fs::read_to_string("/proc/meminfo") {
                for line in meminfo.lines() {
                    if line.starts_with("MemTotal:") {
                        if let Some(kb_str) = line.split_whitespace().nth(1) {
                            if let Ok(kb) = kb_str.parse::<usize>() {
                                return kb / 1024; // Convert KB to MB
                            }
                        }
                    }
                }
            }
        }

        // Fallback based on CPU type
        match self.vendor {
            CpuVendor::Intel => 8192, // Assume 8GB for Intel
            CpuVendor::Amd => 8192,   // Assume 8GB for AMD
            CpuVendor::Arm => 4096,   // Assume 4GB for ARM (Raspberry Pi)
            CpuVendor::Unknown => 4096,
        }
    }

    /// Gets cache sizes from CPUID
    pub fn get_cache_info(&self) -> CacheInfo {
        #[cfg(target_arch = "x86_64")]
        {
            unsafe {
                let mut cache_info = CacheInfo::default();

                // Get cache topology using CPUID
                if self.has_feature(CpuFeature::Avx) {
                    // Try to get cache info from CPUID leaf 0x80000006
                    let leaf = 0x80000006;
                    let ecx = __cpuid(leaf).ecx;

                    // L2 cache size in KB (bits 16-31)
                    let l2_cache_kb = ((ecx >> 16) & 0xFFFF) as usize;
                    if l2_cache_kb > 0 {
                        cache_info.l2_cache_kb = l2_cache_kb;
                    }

                    // L3 cache size in KB (bits 0-15)
                    let l3_cache_kb = (ecx & 0xFFFF) as usize;
                    if l3_cache_kb > 0 {
                        cache_info.l3_cache_kb = l3_cache_kb;
                    }
                }

                cache_info
            }
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            CacheInfo::default()
        }
    }
}

impl OptimizationStrategy {
    /// Creates optimization strategy based on CPU info
    pub fn for_cpu(cpu_info: &CpuInfo) -> Self {
        let simd_width = determine_simd_width(cpu_info);
        let use_fma = cpu_info.has_feature(CpuFeature::Fma);
        let use_avx512 = cpu_info.has_feature(CpuFeature::Avx512F);

        let cache_info = cpu_info.get_cache_info();
        let cache_blocking = CacheBlockingStrategy::from_cache_info(&cache_info, simd_width);

        let parallel_strategy = if cpu_info.core_count >= 4 {
            ParallelStrategy::RayonThreads {
                max_threads: (cpu_info.core_count / 2).max(1),
            }
        } else {
            ParallelStrategy::SingleThreaded
        };

        Self {
            quantization: cpu_info.optimal_quantization(),
            simd_width,
            use_fma,
            use_avx512,
            alignment: simd_width * 4, // 4 bytes per float
            cache_blocking,
            parallel_strategy,
        }
    }

    /// Returns the optimal vector width for matrix operations
    pub fn vector_width(&self) -> usize {
        self.simd_width
    }

    /// Returns the optimal tile size for GEMM operations
    pub fn gemm_tile_size(&self) -> (usize, usize, usize) {
        match self.simd_width {
            16 => (8, 8, 4), // AVX-512
            8 => (4, 4, 4),  // AVX2
            4 => (4, 4, 2),  // SSE
            _ => (2, 2, 2),  // Generic
        }
    }
}

impl CacheBlockingStrategy {
    /// Creates cache blocking strategy from cache info
    pub fn from_cache_info(cache_info: &CacheInfo, vector_width: usize) -> Self {
        let l1_block_size = (cache_info.l1_cache_kb * 1024 / 4 / 3).min(64 * 1024);
        let l2_block_size = (cache_info.l2_cache_kb * 1024 / 4 / 3).min(256 * 1024);
        let l3_block_size = (cache_info.l3_cache_kb * 1024 / 4 / 3).min(1024 * 1024);

        Self {
            l1_block_size,
            l2_block_size,
            l3_block_size,
            vector_width,
        }
    }
}

/// Cache information
#[derive(Debug, Clone, Default)]
pub struct CacheInfo {
    pub l1_cache_kb: usize,
    pub l2_cache_kb: usize,
    pub l3_cache_kb: usize,
    pub cache_line_size: usize,
}

// x86_64 CPU detection
#[cfg(target_arch = "x86_64")]
fn detect_x86_64() -> CpuInfo {
    let mut features = Vec::new();
    let vendor;

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        unsafe {
            // Get feature flags
            let feature_leaf = __cpuid(1);
            let ecx = feature_leaf.ecx;
            let edx = feature_leaf.edx;

            if edx & (1 << 25) != 0 {
                features.push(CpuFeature::Sse);
            }
            if edx & (1 << 26) != 0 {
                features.push(CpuFeature::Sse2);
            }
            if ecx & (1 << 0) != 0 {
                features.push(CpuFeature::Sse3);
            }
            if ecx & (1 << 19) != 0 {
                features.push(CpuFeature::Sse41);
            }
            if ecx & (1 << 20) != 0 {
                features.push(CpuFeature::Sse42);
            }

            let extended_features = __cpuid(7);
            let ebx = extended_features.ebx;

            if ebx & (1 << 3) != 0 {
                features.push(CpuFeature::Bmi1);
            }
            if ebx & (1 << 8) != 0 {
                features.push(CpuFeature::Bmi2);
            }
            if ebx & (1 << 5) != 0 {
                features.push(CpuFeature::Avx2);
            }
            if ebx & (1 << 16) != 0 {
                features.push(CpuFeature::Avx512F);
            }
            if ebx & (1 << 17) != 0 {
                features.push(CpuFeature::Avx512DQ);
            }
            if ebx & (1 << 30) != 0 {
                features.push(CpuFeature::Avx512BW);
            }
            if ebx & (1 << 31) != 0 {
                features.push(CpuFeature::Avx512VL);
            }
            if ebx & (1 << 21) != 0 {
                features.push(CpuFeature::Avx);
            }
            if ebx & (1 << 12) != 0 {
                features.push(CpuFeature::Fma);
            }
            if ebx & (1 << 23) != 0 {
                features.push(CpuFeature::Popcnt);
            }
        }
    }

    // For now, detect Intel vs AMD based on feature availability
    #[cfg(target_arch = "x86_64")]
    {
        if features.contains(&CpuFeature::Avx512F) {
            vendor = CpuVendor::Intel;
        } else if features.contains(&CpuFeature::Avx2) {
            vendor = CpuVendor::Intel; // Assume Intel for AVX2
        } else {
            vendor = CpuVendor::Unknown;
        }
    }

    CpuInfo {
        vendor,
        features,
        cache_size: 8 * 1024 * 1024, // Default 8MB L3
        memory_bandwidth: 51_200,    // Default 51.2 GB/s
        core_count: num_cpus::get_physical(),
        thread_count: num_cpus::get(),
        cpu_family: 0,
        cpu_model: 0,
        cpu_stepping: 0,
    }
}

// ARM CPU detection
#[cfg(target_arch = "aarch64")]
fn detect_aarch64() -> CpuInfo {
    let mut features = Vec::new();
    let mut vendor = CpuVendor::Arm;

    // Read /proc/cpuinfo for ARM features
    if let Ok(cpuinfo) = fs::read_to_string("/proc/cpuinfo") {
        if cpuinfo.contains("neon") {
            features.push(CpuFeature::Neon);
        }
        if cpuinfo.contains("fp16") {
            features.push(CpuFeature::Fp16);
        }
        if cpuinfo.contains("sve") {
            features.push(CpuFeature::Sve);
        }
        if cpuinfo.contains("aes") {
            features.push(CpuFeature::Aes);
        }
        if cpuinfo.contains("sha2") {
            features.push(CpuFeature::Sha2);
        }
    }

    CpuInfo {
        vendor,
        features,
        cache_size: 4 * 1024 * 1024, // Default 4MB L3 for ARM
        memory_bandwidth: 12_800,    // Default 12.8 GB/s for ARM
        core_count: num_cpus::get_physical(),
        thread_count: num_cpus::get(),
        cpu_family: 0,
        cpu_model: 0,
        cpu_stepping: 0,
    }
}

/// Determines optimal SIMD width based on CPU features
fn determine_simd_width(cpu_info: &CpuInfo) -> usize {
    if cpu_info.has_feature(CpuFeature::Avx512F) {
        16 // AVX-512: 16 floats per vector
    } else if cpu_info.has_feature(CpuFeature::Avx2) {
        8 // AVX2: 8 floats per vector
    } else if cpu_info.has_feature(CpuFeature::Avx) {
        8 // AVX: 8 floats per vector
    } else if cpu_info.has_feature(CpuFeature::Sse) {
        4 // SSE: 4 floats per vector
    } else {
        1 // Scalar fallback
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_detection() {
        let cpu_info = CpuInfo::detect();
        assert!(cpu_info.core_count > 0);
        assert!(cpu_info.thread_count > 0);
        println!("Detected CPU: {cpu_info:?}");
    }

    #[test]
    fn test_optimization_strategy() {
        let cpu_info = CpuInfo::detect();
        let strategy = OptimizationStrategy::for_cpu(&cpu_info);
        assert!(strategy.simd_width > 0);
        println!("Optimization strategy: {strategy:?}");
    }

    #[test]
    fn test_quantization_selection() {
        let cpu_info = CpuInfo::detect();
        let quantization = cpu_info.optimal_quantization();
        println!("Optimal quantization: {quantization}");
    }
}
