use axum::{
    extract::{Request, State},
    http::{HeaderMap, StatusCode},
    middleware::Next,
    response::Response,
};
use std::sync::Arc;
use tokio::sync::Mutex;
use std::collections::HashMap;
use std::time::{Duration, Instant};

#[derive(Clone)]
pub struct RateLimitLayer {
    #[allow(dead_code)]
    requests_per_minute: u64,
}

impl RateLimitLayer {
    pub fn new(requests_per_minute: u64) -> Self {
        Self { requests_per_minute }
    }
}

#[derive(Clone)]
pub struct RateLimitState {
    requests: Arc<Mutex<HashMap<String, Vec<Instant>>>>,
    requests_per_minute: u64,
}

impl RateLimitState {
    #[allow(dead_code)]
    fn new(requests_per_minute: u64) -> Self {
        Self {
            requests: Arc::new(Mutex::new(HashMap::new())),
            requests_per_minute,
        }
    }

    async fn check_rate_limit(&self, client_ip: &str
    ) -> Result<(), StatusCode> {
        let mut requests = self.requests.lock().await;
        let now = Instant::now();
        let window = Duration::from_secs(60);
        
        let client_requests = requests
            .entry(client_ip.to_string())
            .or_insert_with(Vec::new);
        
        // Remove old requests
        client_requests.retain(|&time| now.duration_since(time) < window);
        
        // Check if limit exceeded
        if client_requests.len() >= self.requests_per_minute as usize {
            return Err(StatusCode::TOO_MANY_REQUESTS);
        }
        
        // Add current request
        client_requests.push(now);
        Ok(())
    }
}

pub async fn rate_limit_middleware(
    State(state): State<RateLimitState>,
    headers: HeaderMap,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    // Get client IP from headers or use a default
    let client_ip = extract_client_ip(&headers).unwrap_or("unknown".to_string());
    
    state.check_rate_limit(&client_ip).await?;
    
    Ok(next.run(request).await)
}

fn extract_client_ip(headers: &HeaderMap) -> Option<String> {
    // Try X-Forwarded-For first
    if let Some(forwarded) = headers.get("X-Forwarded-For") {
        if let Ok(ip) = forwarded.to_str() {
            return Some(ip.split(',').next()?.trim().to_string());
        }
    }
    
    // Try X-Real-IP
    if let Some(real_ip) = headers.get("X-Real-IP") {
        if let Ok(ip) = real_ip.to_str() {
            return Some(ip.to_string());
        }
    }
    
    None
}