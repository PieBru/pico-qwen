use super::*;
use serde::{Deserialize, Serialize};
use std::env;

#[derive(Debug, Clone)]
pub struct OpenAiProvider {
    pub config: CloudProviderConfig,
    pub client: reqwest::Client,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenAiRequest {
    model: String,
    messages: Vec<OpenAiMessage>,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    frequency_penalty: f32,
    presence_penalty: f32,
    stream: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenAiMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct OpenAiResponse {
    choices: Vec<OpenAiChoice>,
    #[allow(dead_code)]
    usage: Option<OpenAiUsage>,
}

#[derive(Debug, Deserialize)]
struct OpenAiChoice {
    message: OpenAiMessage,
}

#[derive(Debug, Deserialize)]
struct OpenAiUsage {
    #[allow(dead_code)]
    prompt_tokens: usize,
    #[allow(dead_code)]
    completion_tokens: usize,
    #[allow(dead_code)]
    total_tokens: usize,
}

impl OpenAiProvider {
    pub fn new(config: CloudProviderConfig) -> Self {
        let client = reqwest::Client::builder()
            .timeout(config.timeout)
            .build()
            .unwrap();

        Self { config, client }
    }

    pub fn from_env() -> Option<Self> {
        let api_key = env::var("OPENAI_API_KEY").ok()?;
        let base_url = env::var("OPENAI_BASE_URL").unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
        
        let config = CloudProviderConfig {
            name: "openai".to_string(),
            api_key: Some(api_key),
            base_url,
            model: "gpt-3.5-turbo".to_string(),
            max_tokens: 512,
            temperature: 0.7,
            timeout: Duration::from_secs(30),
        };

        Some(Self::new(config))
    }
}

#[async_trait]
impl CloudProvider for OpenAiProvider {
    async fn generate(&self, prompt: &str, config: &InferenceConfig
    ) -> Result<String> {
        let api_key = self.config.api_key.as_ref()
            .ok_or_else(|| anyhow::anyhow!("OpenAI API key not configured"))?;

        let request = OpenAiRequest {
            model: self.config.model.clone(),
            messages: vec![OpenAiMessage {
                role: "user".to_string(),
                content: prompt.to_string(),
            }],
            max_tokens: config.max_tokens,
            temperature: config.temperature,
            top_p: config.top_p,
            frequency_penalty: config.frequency_penalty,
            presence_penalty: config.presence_penalty,
            stream: false,
        };

        let response = self.client
            .post(format!("{}/chat/completions", self.config.base_url))
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(anyhow::anyhow!("OpenAI API error: {}", response.status()));
        }

        let response_data: OpenAiResponse = response.json().await?;
        response_data.choices
            .first()
            .map(|choice| choice.message.content.clone())
            .ok_or_else(|| anyhow::anyhow!("No response from OpenAI"))
    }

    async fn check_health(&self
    ) -> HealthStatus {
        let start = std::time::Instant::now();
        
        let health_status = match self.client
            .get(format!("{}/models", self.config.base_url))
            .header("Authorization", format!("Bearer {}", 
                self.config.api_key.as_ref().unwrap_or(&"dummy".to_string())))
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
        };

        health_status
    }

    fn get_cost_estimate(&self, tokens: usize
    ) -> f64 {
        // Approximate OpenAI pricing (per 1K tokens)
        let input_cost = 0.0015; // $0.0015 per 1K input tokens
        let output_cost = 0.002; // $0.002 per 1K output tokens
        
        let estimated_output_tokens = (tokens as f64 * 0.7).min(tokens as f64);
        
        (tokens as f64 / 1000.0 * input_cost) + 
        (estimated_output_tokens / 1000.0 * output_cost)
    }

    fn get_latency_estimate(&self
    ) -> Duration {
        Duration::from_millis(500) // Typical OpenAI latency
    }

    fn get_name(&self
    ) -> &str {
        &self.config.name
    }
}