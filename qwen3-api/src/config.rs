use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Config {
    pub server: ServerConfig,
    pub models: ModelsConfig,
    pub limits: LimitsConfig,
    pub logging: LoggingConfig,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ServerConfig {
    pub bind_address: String,
    pub port: u16,
    pub cors_origins: Vec<String>,
    pub request_timeout: u64,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ModelsConfig {
    pub directory: String,
    pub max_loaded_models: usize,
    pub default_quantization: String,
    pub context_window: usize,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct LimitsConfig {
    pub max_request_size: usize,
    pub max_concurrent_requests: usize,
    pub rate_limit_per_minute: u64,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct LoggingConfig {
    pub level: String,
    pub format: String,
}

impl Config {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let config_str = std::fs::read_to_string(path)?;
        let config: Config = toml::from_str(&config_str)?;
        Ok(config)
    }

    pub fn default() -> Self {
        Self {
            server: ServerConfig {
                bind_address: "0.0.0.0".to_string(),
                port: 8080,
                cors_origins: vec!["*".to_string()],
                request_timeout: 30,
            },
            models: ModelsConfig {
                directory: "./models".to_string(),
                max_loaded_models: 2,
                default_quantization: "int8".to_string(),
                context_window: 4096,
            },
            limits: LimitsConfig {
                max_request_size: 1024 * 1024, // 1MB
                max_concurrent_requests: 100,
                rate_limit_per_minute: 60,
            },
            logging: LoggingConfig {
                level: "info".to_string(),
                format: "json".to_string(),
            },
        }
    }
}