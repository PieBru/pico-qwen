use std::io::Cursor;

use crate::utils::MemoryMapper;
use anyhow::{Context, Error, Result};
use byteorder::{LittleEndian, ReadBytesExt};
use serde::{Deserialize, Serialize};

/// Magic number for validating checkpoint files (matches C engine "QWEN")
const CHECKPOINT_MAGIC: i32 = 0x5157454E;
/// Expected checkpoint version
const CHECKPOINT_VERSION: i32 = 1;
/// Size of the checkpoint header in bytes
const HEADER_SIZE: usize = 256;
/// Size of config structure in bytes (old format: 8 parameters without rope_theta, new format: 9 parameters with rope_theta)
const CONFIG_SIZE: usize = 36; // New format: 8*u32 + 1*f32 = 36 bytes
const OLD_CONFIG_SIZE: usize = 32; // Old format: 8*u32 = 32 bytes

/// Configuration struct for transformer models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub dim: usize,
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub seq_len: usize,
    pub vocab_size: usize,
    pub group_size: usize,
    pub shared_classifier: bool,
    pub rope_theta: f32,
}

/// Configuration struct for reading model parameters from checkpoint files.
/// Matches C engine format: magic, version, vocab_size, dim, hidden_dim, n_layers, n_heads, n_kv_heads, seq_len, rope_theta
#[derive(Debug, Clone, Copy)]
struct Config {
    pub magic_number: i32,
    pub version: i32,
    pub vocab_size: i32,
    pub dim: i32,
    pub hidden_dim: i32,
    pub n_layers: i32,
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub seq_len: i32,
    pub rope_theta: f32,
}

impl TryInto<ModelConfig> for Config {
    type Error = Error;

    fn try_into(self) -> Result<ModelConfig> {
        validate_config(&self).with_context(|| "Invalid model configuration")?;

        // Calculate derived parameters
        let head_dim = self.dim as usize / self.n_heads as usize;
        
        Ok(ModelConfig {
            dim: self.dim as usize,
            hidden_dim: self.hidden_dim as usize,
            n_layers: self.n_layers as usize,
            n_heads: self.n_heads as usize,
            n_kv_heads: self.n_kv_heads as usize,
            head_dim,
            seq_len: self.seq_len as usize,
            vocab_size: self.vocab_size as usize,
            group_size: 64, // Default group size, will be determined from weights
            shared_classifier: true, // Default assumption, will be determined from weights
            rope_theta: self.rope_theta, // Use the rope_theta from the config
        })
    }
}

/// Reads and validates the model configuration from checkpoint data (mapper).
    ///
    /// Supports both old format (32 bytes) and new format (36 bytes with rope_theta)
    /// This function performs bounds checking and validates the magic number and version.
    pub fn read_config(mapper: &mut MemoryMapper) -> Result<ModelConfig> {
        // Read the entire configuration block
        let config_data = match mapper.get_bytes(CONFIG_SIZE) {
            Ok(data) => data,
            Err(_) => mapper.get_bytes(OLD_CONFIG_SIZE)
                .with_context(|| "Failed to read config from checkpoint")?
        };

        let mut cursor = Cursor::new(config_data);

        // Read magic and version
        let magic_number = cursor.read_i32::<LittleEndian>()
            .with_context(|| "Failed to read magic number")?;
        let version = cursor.read_i32::<LittleEndian>()
            .with_context(|| "Failed to read version")?;

        // Validate magic and version
        if magic_number != CHECKPOINT_MAGIC {
            anyhow::bail!(
                "Invalid checkpoint magic number: expected {:#x}, got {:#x}",
                CHECKPOINT_MAGIC,
                magic_number
            );
        }

        if version != CHECKPOINT_VERSION {
            anyhow::bail!(
                "Unsupported checkpoint version: expected {}, got {}",
                CHECKPOINT_VERSION,
                version
            );
        }

        // Read the configuration parameters
        let vocab_size = cursor.read_u32::<LittleEndian>()? as i32;
        let dim = cursor.read_u32::<LittleEndian>()? as i32;
        let hidden_dim = cursor.read_u32::<LittleEndian>()? as i32;
        let n_layers = cursor.read_u32::<LittleEndian>()? as i32;
        let n_heads = cursor.read_u32::<LittleEndian>()? as i32;
        let n_kv_heads = cursor.read_u32::<LittleEndian>()? as i32;
        let seq_len = cursor.read_u32::<LittleEndian>()? as i32;

        // Handle rope_theta based on available data
        let rope_theta = if cursor.position() + 4 <= config_data.len() as u64 {
            cursor.read_f32::<LittleEndian>()
                .with_context(|| "Failed to read rope theta")?
        } else {
            10000.0 // Default rope_theta for backward compatibility
        };

        let config = Config {
            magic_number,
            version,
            vocab_size,
            dim,
            hidden_dim,
            n_layers,
            n_heads,
            n_kv_heads,
            seq_len,
            rope_theta,
        };

        let config_len = config_data.len();
    // Skip to model weights (after 256-byte header)
    mapper.skip(HEADER_SIZE - config_len)?;

        config.try_into()
    }

/// Validates the model configuration to ensure it's supported.
fn validate_config(config: &Config) -> Result<()> {
    match config.magic_number {
        CHECKPOINT_MAGIC => {}
        actual => anyhow::bail!(
            "Invalid checkpoint magic number: expected {:#x}, got {:#x}",
            CHECKPOINT_MAGIC,
            actual
        ),
    }

    match config.version {
        CHECKPOINT_VERSION => {}
        actual => anyhow::bail!(
            "Unsupported checkpoint version: expected {}, got {}",
            CHECKPOINT_VERSION,
            actual
        ),
    }

    // Validate positive dimensions
    let dimensions = [
        ("dim", config.dim),
        ("n_layers", config.n_layers),
        ("n_heads", config.n_heads),
        ("n_kv_heads", config.n_kv_heads),
        ("vocab_size", config.vocab_size),
        ("seq_len", config.seq_len),
    ];

    for (name, value) in dimensions {
        if value <= 0 {
            anyhow::bail!("Invalid {}: must be positive, got {}", name, value);
        }
    }

    Ok(())
}
