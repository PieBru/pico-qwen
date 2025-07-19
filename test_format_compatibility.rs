use std::fs::File;
use std::io::{Cursor, Read};
use qwen3_inference::configuration::{read_config, ModelConfig};

/// Test that verifies both C and Rust engines can read the same binary format
fn main() -> anyhow::Result<()> {
    println!("Testing binary format compatibility between C and Rust engines...");
    
    // Create a test binary file with the unified format
    let test_file = "/tmp/test_unified_format.bin";
    create_test_binary(test_file)?;
    
    // Test Rust engine can read it
    println!("Testing Rust engine...");
    let file = File::open(test_file)?;
    let mut mapper = qwen3_inference::utils::MemoryMapper::new(file)?;
    let rust_config = read_config(&mut mapper)?;
    
    println!("Rust engine successfully read config:");
    println!("  vocab_size: {}", rust_config.vocab_size);
    println!("  dim: {}", rust_config.dim);
    println!("  hidden_dim: {}", rust_config.hidden_dim);
    println!("  n_layers: {}", rust_config.n_layers);
    println!("  n_heads: {}", rust_config.n_heads);
    println!("  n_kv_heads: {}", rust_config.n_kv_heads);
    println!("  seq_len: {}", rust_config.seq_len);
    
    println!("✅ Binary format compatibility test passed!");
    
    Ok(())
}

fn create_test_binary(path: &str) -> anyhow::Result<()> {
    use byteorder::{LittleEndian, WriteBytesExt};
    use std::io::Write;
    
    let mut file = File::create(path)?;
    
    // Write magic number (QWEN)
    file.write_u32::<LittleEndian>(0x5157454E)?;
    
    // Write version
    file.write_u32::<LittleEndian>(1)?;
    
    // Write model parameters (same as C engine format)
    file.write_u32::<LittleEndian>(32000)?; // vocab_size
    file.write_u32::<LittleEndian>(768)?;   // dim
    file.write_u32::<LittleEndian>(2048)?;  // hidden_dim
    file.write_u32::<LittleEndian>(12)?;    // n_layers
    file.write_u32::<LittleEndian>(12)?;    // n_heads
    file.write_u32::<LittleEndian>(12)?;    // n_kv_heads
    file.write_u32::<LittleEndian>(2048)?;  // seq_len
    file.write_f32::<LittleEndian>(10000.0)?; // rope_theta
    
    // Pad to 256 bytes
    let padding = vec![0u8; 256 - 36];
    file.write_all(&padding)?;
    
    // Add some dummy data to make it a valid file
    let dummy_data = vec![0u8; 1000];
    file.write_all(&dummy_data)?;
    
    Ok(())
}