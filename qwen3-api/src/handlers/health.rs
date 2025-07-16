use axum::{
    extract::State,
    Json,
};
use serde::{Deserialize, Serialize};
use crate::state::AppState;

#[derive(Debug, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub timestamp: String,
    pub version: String,
    pub memory_usage: MemoryUsage,
    pub models: ModelsInfo,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MemoryUsage {
    pub active_requests: usize,
    pub loaded_models: usize,
    pub total_memory_mb: Option<u64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelsInfo {
    pub count: usize,
    pub loaded: Vec<String>,
}

pub async fn health_check(
    State(state): State<AppState>,
) -> Json<HealthResponse> {
    let active_requests = state.active_requests.load(std::sync::atomic::Ordering::Relaxed);
    let loaded_models = state.list_models();
    
    let memory_usage = MemoryUsage {
        active_requests,
        loaded_models: loaded_models.len(),
        total_memory_mb: get_memory_usage(),
    };
    
    let models = ModelsInfo {
        count: loaded_models.len(),
        loaded: loaded_models.into_iter().map(|m| m.id).collect(),
    };
    
    Json(HealthResponse {
        status: "healthy".to_string(),
        timestamp: chrono::Utc::now().to_rfc3339(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        memory_usage,
        models,
    })
}

#[cfg(target_os = "linux")]
fn get_memory_usage() -> Option<u64> {
    use std::fs;
    
    if let Ok(status) = fs::read_to_string("/proc/self/status") {
        for line in status.lines() {
            if line.starts_with("VmRSS:") {
                if let Some(kb_str) = line.split_whitespace().nth(1) {
                    if let Ok(kb) = kb_str.parse::<u64>() {
                        return Some(kb / 1024); // Convert to MB
                    }
                }
            }
        }
    }
    None
}

#[cfg(not(target_os = "linux"))]
fn get_memory_usage() -> Option<u64> {
    None
}