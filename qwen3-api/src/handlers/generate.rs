use crate::state::AppState;
use axum::{extract::State, Json};
use qwen3_inference::{sampler::Sampler, tokenizer::Tokenizer};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct GenerateRequest {
    pub model: String,
    pub prompt: String,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub stop: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
pub struct GenerateResponse {
    pub text: String,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

pub async fn generate_handler(
    State(state): State<AppState>,
    Json(request): Json<GenerateRequest>,
) -> Result<Json<GenerateResponse>, (axum::http::StatusCode, String)> {
    // Increment active requests
    state
        .active_requests
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    // Get model
    let model = state.get_model(&request.model).ok_or_else(|| {
        (
            axum::http::StatusCode::NOT_FOUND,
            "Model not found".to_string(),
        )
    })?;

    // Update last used time
    {
        let mut transformer = model.transformer.write().await;

        // Configure generation parameters
        let max_tokens = request.max_tokens.unwrap_or(100);
        let temperature = request.temperature.unwrap_or(0.7);
        let top_p = request.top_p.unwrap_or(0.9);

        // Get underlying transformer and tokenizer
        let transformer = transformer.transformer_mut();
        let tokenizer = Tokenizer::new(
            &model.info.path.to_string_lossy(),
            transformer.config.vocab_size,
            false, // enable_thinking
        )
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

        let mut sampler = Sampler::new(
            transformer.config.vocab_size,
            temperature,
            top_p,
            42, // seed
        );

        // Generate response using API-friendly generation
        let response_text = generate_api_response(
            transformer,
            &tokenizer,
            &mut sampler,
            &request.prompt,
            max_tokens,
        )
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

        // Count tokens
        let prompt_tokens = tokenizer.encode(&request.prompt).len();
        let completion_tokens = tokenizer.encode(&response_text).len();

        let response = GenerateResponse {
            text: response_text.trim().to_string(),
            usage: Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            },
        };

        // Update model statistics
        if let Some(mut model) = state.models.get_mut(&request.model) {
            model
                .request_count
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            model.total_tokens_generated.fetch_add(
                completion_tokens as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            model.last_inference_at = Some(std::time::Instant::now());
        }

        state
            .active_requests
            .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        Ok(Json(response))
    }
}

fn generate_api_response(
    transformer: &mut qwen3_inference::transformer::Transformer,
    tokenizer: &qwen3_inference::tokenizer::Tokenizer,
    sampler: &mut qwen3_inference::sampler::Sampler,
    prompt: &str,
    max_tokens: usize,
) -> anyhow::Result<String> {
    let prompt_tokens = tokenizer.encode(prompt);
    if prompt_tokens.is_empty() {
        anyhow::bail!("Empty prompt");
    }

    let seq_len = transformer.config.seq_len;
    let mut response_tokens = Vec::new();
    let mut token = prompt_tokens[0];
    let mut pos = 0;

    // Process prompt tokens first
    for &next_token in &prompt_tokens[1..] {
        if pos >= seq_len {
            break;
        }
        let _ = transformer.forward(token, pos);
        token = next_token;
        pos += 1;
    }

    // Generate new tokens
    let mut generated_count = 0;
    while generated_count < max_tokens && pos < seq_len {
        let logits = transformer.forward(token, pos);
        let mut logits_copy = logits.to_vec();
        let next_token = sampler.sample(&mut logits_copy);

        if next_token == tokenizer.eos_token_id as usize
            || next_token == tokenizer.bos_token_id as usize
        {
            break;
        }

        response_tokens.push(next_token);
        token = next_token;
        pos += 1;
        generated_count += 1;
    }

    // Decode response tokens to text
    let response_text = response_tokens
        .iter()
        .map(|&token| tokenizer.decode(token))
        .collect::<String>();

    Ok(response_text)
}
