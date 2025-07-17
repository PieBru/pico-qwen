use super::*;
use serde::{Deserialize, Serialize};
use std::env;

#[derive(Debug, Clone)]
pub struct AnthropicProvider {
    pub config: CloudProviderConfig,
    pub client: reqwest::Client,
}

#[derive(Debug, Serialize, Deserialize)]
struct AnthropicRequest {
    model: String,
    messages: Vec<AnthropicMessage>,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
}

#[derive(Debug, Serialize, Deserialize)]
struct AnthropicMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicContent>,
    #[allow(dead_code)]
    usage: Option<AnthropicUsage>,
}

#[derive(Debug, Deserialize)]
struct AnthropicContent {
    #[serde(rename = "type")]
    content_type: String,
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AnthropicUsage {
    #[allow(dead_code)]
    input_tokens: usize,
    #[allow(dead_code)]
    output_tokens: usize,
}

impl AnthropicProvider {
    pub fn new(config: CloudProviderConfig) -> Self {
        let client = reqwest::Client::builder()
            .timeout(config.timeout)
            .build()
            .unwrap();

        Self { config, client }
    }

    pub fn from_env() -> Option<Self> {
        let api_key = env::var("ANTHROPIC_API_KEY").ok()?;
        let base_url = env::var("ANTHROPIC_BASE_URL")
            .unwrap_or_else(|_| "https://api.anthropic.com/v1".to_string());

        let config = CloudProviderConfig {
            name: "anthropic".to_string(),
            api_key: Some(api_key),
            base_url,
            model: "claude-3-haiku-20240307".to_string(),
            max_tokens: 512,
            temperature: 0.7,
            timeout: Duration::from_secs(30),
        };

        Some(Self::new(config))
    }
}

#[async_trait]
impl CloudProvider for AnthropicProvider {
    async fn generate(&self, prompt: &str, config: &InferenceConfig) -> Result<String> {
        let api_key = self
            .config
            .api_key
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Anthropic API key not configured"))?;

        let request = AnthropicRequest {
            model: self.config.model.clone(),
            messages: vec![AnthropicMessage {
                role: "user".to_string(),
                content: prompt.to_string(),
            }],
            max_tokens: config.max_tokens,
            temperature: config.temperature,
            top_p: config.top_p,
        };

        let response = self
            .client
            .post(format!("{}/messages", self.config.base_url))
            .header("x-api-key", api_key)
            .header("Content-Type", "application/json")
            .header("anthropic-version", "2023-06-01")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(anyhow::anyhow!(
                "Anthropic API error: {}",
                response.status()
            ));
        }

        let response_data: AnthropicResponse = response.json().await?;

        let content = response_data
            .content
            .iter()
            .find(|c| c.content_type == "text")
            .and_then(|c| c.text.clone())
            .ok_or_else(|| anyhow::anyhow!("No text content in Anthropic response"))?;

        Ok(content)
    }

    async fn check_health(&self) -> HealthStatus {
        let start = std::time::Instant::now();

        match self
            .client
            .get(format!("{}/models", self.config.base_url))
            .header(
                "x-api-key",
                self.config.api_key.as_ref().unwrap_or(&"dummy".to_string()),
            )
            .send()
            .await
        {
            Ok(response) => HealthStatus {
                healthy: response.status().is_success(),
                latency: start.elapsed(),
                last_check: std::time::SystemTime::now(),
                error_count: 0,
            },
            Err(_) => HealthStatus {
                healthy: false,
                latency: start.elapsed(),
                last_check: std::time::SystemTime::now(),
                error_count: 1,
            },
        }
    }

    fn get_cost_estimate(&self, tokens: usize) -> f64 {
        // Approximate Anthropic pricing (per 1K tokens)
        let input_cost = 0.00025; // $0.00025 per 1K input tokens (Claude 3 Haiku)
        let output_cost = 0.00125; // $0.00125 per 1K output tokens (Claude 3 Haiku)

        let estimated_output_tokens = (tokens as f64 * 0.7).min(tokens as f64);

        (tokens as f64 / 1000.0 * input_cost) + (estimated_output_tokens / 1000.0 * output_cost)
    }

    fn get_latency_estimate(&self) -> Duration {
        Duration::from_millis(800) // Typical Anthropic latency
    }

    fn get_name(&self) -> &str {
        &self.config.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_anthropic_provider_creation() {
        let config = CloudProviderConfig {
            name: "anthropic".to_string(),
            api_key: Some("test-key".to_string()),
            base_url: "https://api.anthropic.com/v1".to_string(),
            model: "claude-3-haiku-20240307".to_string(),
            max_tokens: 512,
            temperature: 0.7,
            timeout: Duration::from_secs(30),
        };

        let provider = AnthropicProvider::new(config);
        assert_eq!(provider.get_name(), "anthropic");
    }

    #[tokio::test]
    async fn test_cost_estimation() {
        let config = CloudProviderConfig {
            name: "anthropic".to_string(),
            api_key: Some("test-key".to_string()),
            base_url: "https://api.anthropic.com/v1".to_string(),
            model: "claude-3-haiku-20240307".to_string(),
            max_tokens: 512,
            temperature: 0.7,
            timeout: Duration::from_secs(30),
        };

        let provider = AnthropicProvider::new(config);
        let cost = provider.get_cost_estimate(1000);
        assert!(cost > 0.0);
    }
}
