use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=qwen3-c-lib/include/qwen3_inference.h");
    println!("cargo:rerun-if-changed=qwen3-c-lib/src/");
    
    let mut build = cc::Build::new();
    
    // Configuration
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let qwen3_c_lib_path = format!("{}/../qwen3-c-lib", manifest_dir);
    
    build.include(format!("{}/include", qwen3_c_lib_path));
    build.include(format!("{}/src", qwen3_c_lib_path));
    
    // Source files
    let source_files = [
        format!("{}/src/qwen3_inference.c", qwen3_c_lib_path),
        format!("{}/src/matrix.c", qwen3_c_lib_path),
        format!("{}/src/attention.c", qwen3_c_lib_path),
        format!("{}/src/transformer.c", qwen3_c_lib_path),
        format!("{}/src/tensor.c", qwen3_c_lib_path),
        format!("{}/src/tokenizer.c", qwen3_c_lib_path),
        format!("{}/src/sampler.c", qwen3_c_lib_path),
        format!("{}/src/memory.c", qwen3_c_lib_path),
        format!("{}/src/model.c", qwen3_c_lib_path),
        format!("{}/src/utils.c", qwen3_c_lib_path),
        format!("{}/src/simd/cpu_detect.c", qwen3_c_lib_path),
    ];
    
    // Add source files
    for file in source_files.iter() {
        build.file(file);
    }
    
    // Compiler flags
    build.flag("-std=c99");
    build.flag("-Wall");
    build.flag("-Wextra");
    build.flag("-pedantic");
    build.flag("-O3");
    
    // Architecture-specific flags
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    
    match target_arch.as_str() {
        "x86_64" => {
            build.flag("-march=native");
            build.flag("-mtune=native");
            build.flag("-mfma");
            build.flag("-mavx2");
            
            // Check CPU features for AVX-512 support
            if std::process::Command::new("grep")
                .arg("-q")
                .arg("avx512")
                .arg("/proc/cpuinfo")
                .status()
                .map(|s| s.success())
                .unwrap_or(false) {
                build.flag("-mavx512f");
                build.flag("-mavx512vl");
            }
        },
        "aarch64" => {
            build.flag("-march=native");
            build.flag("-mtune=native");
            build.flag("-mfpu=neon");
            build.flag("-mfpu=neon-fp-armv8");
        },
        _ => {
            println!("cargo:warning=Unknown architecture, using generic flags");
        }
    }
    
    // OS-specific flags
    if target_os == "windows" {
        build.flag("-DWIN32");
        build.flag("-D_CRT_SECURE_NO_WARNINGS");
    } else {
        build.flag("-D_POSIX_C_SOURCE=200809L");
    }
    
    // Debug vs Release
    let profile = env::var("PROFILE").unwrap_or_default();
    if profile == "debug" {
        build.flag("-g");
        build.flag("-DDEBUG");
    } else {
        build.define("NDEBUG", None);
    }
    
    // SIMD detection
    if cfg!(target_feature = "avx2") {
        build.define("QWEN3_HAS_AVX2", None);
    }
    if cfg!(target_feature = "avx512f") {
        build.define("QWEN3_HAS_AVX512", None);
    }
    if cfg!(target_feature = "neon") {
        build.define("QWEN3_HAS_NEON", None);
    }
    
    build.compile("qwen3_inference");
    
    // Link directories
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=qwen3_inference");
    
    // Link system libraries
    if target_os == "linux" {
        println!("cargo:rustc-link-lib=m");
        println!("cargo:rustc-link-lib=pthread");
    } else if target_os == "macos" {
        println!("cargo:rustc-link-lib=m");
    } else if target_os == "windows" {
        println!("cargo:rustc-link-lib=msvcrt");
    }
}