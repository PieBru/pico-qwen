use super::*;
use std::time::Duration;
use anyhow::Result;

#[derive(Debug, Clone)]
pub struct LocalProvider {
    pub config: CloudProviderConfig,
}

impl LocalProvider {
    pub fn new(config: CloudProviderConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl CloudProvider for LocalProvider {
    async fn generate(&self, _prompt: &str, _config: &InferenceConfig
    ) -> Result<String> {
        // This is a placeholder for local inference
        // In a real implementation, this would use the local Qwen3 model
        Err(anyhow::anyhow!("Local inference not implemented in cloud provider"))
    }

    async fn check_health(&self
    ) -> HealthStatus {
        HealthStatus {
            healthy: true,
            latency: Duration::from_millis(0),
            last_check: std::time::SystemTime::now(),
            error_count: 0,
        }
    }

    fn get_cost_estimate(&self, _tokens: usize) -> f64 {
        0.0 // Local inference is free
    }

    fn get_latency_estimate(&self
    ) -> Duration {
        Duration::from_millis(100) // Fast local inference
    }

    fn get_name(&self
    ) -> &str {
        &self.config.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_local_provider() {
        let config = CloudProviderConfig {
            name: "local".to_string(),
            api_key: None,
            base_url: "http://localhost".to_string(),
            model: "local".to_string(),
            max_tokens: 512,
            temperature: 0.7,
            timeout: Duration::from_secs(30),
        };
        
        let provider = LocalProvider::new(config);
        assert_eq!(provider.get_name(), "local");
        assert_eq!(provider.get_cost_estimate(1000), 0.0);
        assert!(provider.get_latency_estimate() < Duration::from_millis(1000));
    }
    
    #[tokio::test]
    async fn test_local_health() {
        let config = CloudProviderConfig {
            name: "local".to_string(),
            api_key: None,
            base_url: "http://localhost".to_string(),
            model: "local".to_string(),
            max_tokens: 512,
            temperature: 0.7,
            timeout: Duration::from_secs(30),
        };
        
        let provider = LocalProvider::new(config);
        let health = provider.check_health().await;
        assert!(health.healthy);
    }
}