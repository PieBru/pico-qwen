use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::cloud::openai::OpenAiProvider;
use crate::cloud::anthropic::AnthropicProvider;
use crate::cloud::local::LocalProvider;

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
    #[serde(with = "humantime_serde")]
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

/// Cloud provider manager for hybrid inference
pub struct CloudManager {
    providers: HashMap<String, Arc<dyn CloudProvider>>,
    health_cache: Arc<RwLock<HashMap<String, HealthStatus>>>,
    fallback_to_local: bool,
    health_check_interval: Duration,
}

impl CloudManager {
    pub fn new(config: CloudConfig) -> Result<Self> {
        let mut providers = HashMap::new();
        
        // Add providers from configuration
        for provider_config in config.providers {
            let provider: Arc<dyn CloudProvider> = match provider_config.name.as_str() {
                "openai" => Arc::new(OpenAiProvider::new(provider_config.clone())),
                "anthropic" => Arc::new(AnthropicProvider::new(provider_config.clone())),
                "local" => Arc::new(LocalProvider::new(provider_config.clone())),
                name => return Err(anyhow::anyhow!("Unknown cloud provider: {}", name)),
            };
            providers.insert(provider_config.name.clone(), provider);
        }
        
        Ok(CloudManager {
            providers,
            health_cache: Arc::new(RwLock::new(HashMap::new())),
            fallback_to_local: config.fallback_to_local,
            health_check_interval: Duration::from_secs(config.health_check_interval),
        })
    }
    
    pub async fn generate(
        &self,
        prompt: &str,
        config: &InferenceConfig,
        preferred_provider: Option<&str>,
    ) -> Result<String> {
        // Try preferred provider first
        if let Some(provider_name) = preferred_provider {
            if let Some(provider) = self.providers.get(provider_name) {
                match provider.generate(prompt, config).await {
                    Ok(response) => return Ok(response),
                    Err(e) => {
                        log::warn!("Provider {} failed: {}", provider_name, e);
                    }
                }
            }
        }
        
        // Try all healthy providers
        let healthy_providers = self.get_healthy_providers().await;
        for provider_name in healthy_providers {
            if let Some(provider) = self.providers.get(&provider_name) {
                match provider.generate(prompt, config).await {
                    Ok(response) => return Ok(response),
                    Err(e) => {
                        log::warn!("Provider {} failed: {}", provider_name, e);
                    }
                }
            }
        }
        
        // Fallback to local if enabled
        if self.fallback_to_local {
            return Err(anyhow::anyhow!("All cloud providers failed and fallback to local not implemented"));
        }
        
        Err(anyhow::anyhow!("All cloud providers failed"))
    }
    
    pub async fn get_healthy_providers(&self) -> Vec<String> {
        let mut healthy = Vec::new();
        let health_cache = self.health_cache.read().await;
        
        for (name, _provider) in &self.providers {
            let is_healthy = health_cache
                .get(name)
                .map(|status| status.healthy)
                .unwrap_or(true); // Default to healthy if not cached
            
            if is_healthy {
                healthy.push(name.clone());
            }
        }
        
        healthy
    }
    
    pub async fn check_all_health(&self) -> HashMap<String, HealthStatus> {
        let mut results = HashMap::new();
        
        for (name, provider) in &self.providers {
            let status = provider.check_health().await;
            let status_clone = status.clone();
            results.insert(name.clone(), status);
            
            // Update cache
            let mut cache = self.health_cache.write().await;
            cache.insert(name.clone(), status_clone);
        }
        
        results
    }
    
    pub async fn get_cost_estimates(&self,
        tokens: usize,
    ) -> HashMap<String, f64> {
        let mut estimates = HashMap::new();
        
        for (name, provider) in &self.providers {
            let cost = provider.get_cost_estimate(tokens);
            estimates.insert(name.clone(), cost);
        }
        
        estimates
    }
}

pub mod openai;
pub mod anthropic;
pub mod local;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_cloud_manager_creation() {
        let config = CloudConfig {
            providers: vec![CloudProviderConfig {
                name: "openai".to_string(),
                api_key: Some("test-key".to_string()),
                base_url: "https://api.openai.com/v1".to_string(),
                model: "gpt-3.5-turbo".to_string(),
                max_tokens: 512,
                temperature: 0.7,
                timeout: Duration::from_secs(30),
            }],
            fallback_to_local: true,
            health_check_interval: 60,
        };
        
        let manager = CloudManager::new(config).unwrap();
        assert_eq!(manager.providers.len(), 1);
    }
    
    #[tokio::test]
    async fn test_health_check_caching() {
        let config = CloudConfig {
            providers: vec![],
            fallback_to_local: true,
            health_check_interval: 60,
        };
        
        let manager = CloudManager::new(config).unwrap();
        let health = manager.check_all_health().await;
        assert!(health.is_empty());
    }
}