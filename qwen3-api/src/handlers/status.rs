use axum::{extract::State, Json};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

use crate::state::AppState;

#[derive(Debug, Serialize, Deserialize)]
pub struct ServerStatus {
    pub uptime: Duration,
    pub active_requests: usize,
    pub loaded_models: Vec<LoadedModelStatus>,
    pub total_memory_mb: f64,
    pub system_info: SystemInfo,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LoadedModelStatus {
    pub id: String,
    pub size_mb: f64,
    pub loaded_at: Duration,
    pub last_used: Duration,
    pub inference_stats: InferenceStats,
    pub memory_usage_mb: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InferenceStats {
    pub total_requests: u64,
    pub total_tokens_generated: u64,
    pub avg_tokens_per_sec: f64,
    pub last_inference_at: Option<Duration>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SystemInfo {
    pub cpu_cores: usize,
    pub total_memory_gb: f64,
    pub available_memory_gb: f64,
    pub used_memory_gb: f64,
    pub memory_usage_percent: f64,
}

// Global static for server start time
static SERVER_START_TIME: once_cell::sync::Lazy<Instant> = once_cell::sync::Lazy::new(Instant::now);

pub async fn server_status(State(state): State<AppState>) -> Json<ServerStatus> {
    let uptime = SERVER_START_TIME.elapsed();
    let active_requests = state
        .active_requests
        .load(std::sync::atomic::Ordering::Relaxed);

    // Get system memory info
    let system_info = get_system_info();

    // Get loaded models with detailed stats
    let loaded_models = get_loaded_models_status(&state);

    // Calculate total memory usage
    let total_memory_mb = loaded_models.iter().map(|m| m.memory_usage_mb).sum();

    let status = ServerStatus {
        uptime,
        active_requests,
        loaded_models,
        total_memory_mb,
        system_info,
    };

    Json(status)
}

fn get_loaded_models_status(state: &AppState) -> Vec<LoadedModelStatus> {
    let mut models = Vec::new();

    for entry in state.models.iter() {
        let model = entry.value();
        let now = Instant::now();

        // Estimate memory usage based on model size and quantization
        let memory_usage_mb = estimate_memory_usage(&model.info);

        let inference_stats = InferenceStats {
            total_requests: model
                .request_count
                .load(std::sync::atomic::Ordering::Relaxed),
            total_tokens_generated: model
                .total_tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            avg_tokens_per_sec: 0.0, // Would calculate from actual usage
            last_inference_at: model.last_inference_at.map(|t| now.duration_since(t)),
        };

        let status = LoadedModelStatus {
            id: model.info.id.clone(),
            size_mb: model.info.size as f64 / (1024.0 * 1024.0),
            loaded_at: now.duration_since(model.loaded_at),
            last_used: now.duration_since(model.last_used),
            inference_stats,
            memory_usage_mb,
        };

        models.push(status);
    }

    models
}

fn estimate_memory_usage(info: &crate::state::ModelInfo) -> f64 {
    // Rough estimation based on model size and quantization
    let base_size_mb = info.size as f64 / (1024.0 * 1024.0);

    // Add overhead for runtime data structures
    let overhead_multiplier = match info.quantization.as_str() {
        "int4" => 1.2,
        "int8" => 1.3,
        "fp16" => 1.5,
        "fp32" => 2.0,
        _ => 1.3,
    };

    base_size_mb * overhead_multiplier
}

fn get_system_info() -> SystemInfo {
    use sysinfo::System;

    let mut sys = System::new_all();
    sys.refresh_memory();

    let total_memory = sys.total_memory() as f64 / (1024.0 * 1024.0 * 1024.0);
    let available_memory = sys.available_memory() as f64 / (1024.0 * 1024.0 * 1024.0);
    let used_memory = sys.used_memory() as f64 / (1024.0 * 1024.0 * 1024.0);
    let memory_usage_percent = (used_memory / total_memory) * 100.0;

    SystemInfo {
        cpu_cores: sys.cpus().len(),
        total_memory_gb: total_memory,
        available_memory_gb: available_memory,
        used_memory_gb: used_memory,
        memory_usage_percent,
    }
}
