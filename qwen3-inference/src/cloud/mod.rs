use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudConfig {
    pub providers: Vec<CloudProviderConfig>,
    pub fallback_to_local: bool,
    pub health_check_interval: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudProviderConfig {
    pub name: String,
    pub api_key: Option<String>,
    pub base_url: String,
    pub model: String,
    pub max_tokens: usize,
    pub temperature: f32,
    pub timeout: Duration,
}

#[derive(Debug, Clone)]
pub struct InferenceConfig {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub healthy: bool,
    pub latency: Duration,
    pub last_check: std::time::SystemTime,
    pub error_count: usize,
}

#[async_trait]
pub trait CloudProvider: Send + Sync {
    async fn generate(&self, prompt: &str, config: &InferenceConfig) -> Result<String>;
    async fn check_health(&self) -> HealthStatus;
    fn get_cost_estimate(&self, tokens: usize) -> f64;
    fn get_latency_estimate(&self) -> Duration;
    fn get_name(&self) -> &str;
}

pub mod openai;
pub mod anthropic;
pub mod local;