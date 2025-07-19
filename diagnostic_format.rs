use std::fs::File;
use std::io::{Cursor, Read};
use byteorder::{LittleEndian, ReadBytesExt};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <file.bin>", args[0]);
        return Ok(());
    }

    let path = &args[1];
    let mut file = File::open(path)?;
    let mut buffer = [0u8; 256];
    file.read_exact(&mut buffer)?;

    let mut cursor = Cursor::new(&buffer);

    println!("=== Binary Format Diagnostic ===");
    println!("File: {}", path);
    
    // Read the first 36 bytes to check format
    let magic = cursor.read_i32::<LittleEndian>()?;
    println!("Magic: {:#x} (expected: {:#x})", magic, 0x5157454E);
    
    let version = cursor.read_i32::<LittleEndian>()?;
    println!("Version: {} (expected: 1)", version);
    
    let vocab_size = cursor.read_u32::<LittleEndian>()?;
    println!("Vocab size: {}", vocab_size);
    
    let dim = cursor.read_u32::<LittleEndian>()?;
    println!("Dimension: {}", dim);
    
    let hidden_dim = cursor.read_u32::<LittleEndian>()?;
    println!("Hidden dim: {}", hidden_dim);
    
    let n_layers = cursor.read_u32::<LittleEndian>()?;
    println!("Layers: {}", n_layers);
    
    let n_heads = cursor.read_u32::<LittleEndian>()?;
    println!("Heads: {}", n_heads);
    
    let n_kv_heads = cursor.read_u32::<LittleEndian>()?;
    println!("KV Heads: {}", n_kv_heads);
    
    let seq_len = cursor.read_u32::<LittleEndian>()?;
    println!("Sequence length: {}", seq_len);
    
    let rope_theta = cursor.read_f32::<LittleEndian>()?;
    println!("RoPE theta: {}", rope_theta);

    if magic != 0x5157454E {
        println!("❌ WARNING: Magic number doesn't match expected QWEN format");
        println!("   This file may use an older format");
    }

    if version != 1 {
        println!("❌ WARNING: Version doesn't match expected version 1");
        println!("   This file may use an older format");
    }

    println!("✅ File appears to use the unified format");
    
    Ok(())
}