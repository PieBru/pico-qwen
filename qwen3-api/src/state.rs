use anyhow::Result;
use dashmap::DashMap;
use qwen3_inference::ExtendedTransformer;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub path: PathBuf,
    pub size: u64,
    pub loaded: bool,
    pub quantization: String,
    pub context_window: usize,
}

#[derive(Debug)]
pub struct LoadedModel {
    pub info: ModelInfo,
    pub transformer: Arc<RwLock<qwen3_inference::ExtendedTransformer>>,
    pub last_used: std::time::Instant,
    pub loaded_at: std::time::Instant,
    pub request_count: std::sync::atomic::AtomicU64,
    pub total_tokens_generated: std::sync::atomic::AtomicU64,
    pub last_inference_at: Option<std::time::Instant>,
}

#[derive(Debug, Clone)]
pub struct AppState {
    pub config: crate::config::Config,
    pub models: Arc<DashMap<String, LoadedModel>>,
    pub active_requests: Arc<std::sync::atomic::AtomicUsize>,
}

impl AppState {
    pub fn new(config: crate::config::Config) -> Self {
        Self {
            config,
            models: Arc::new(DashMap::new()),
            active_requests: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        }
    }

    pub async fn load_model(
        &self,
        model_id: &str,
        quantization: Option<&str>,
    ) -> Result<String> {
        let model_path = PathBuf::from(&self.config.models.directory).join(format!("{}.bin", model_id));
        
        if !model_path.exists() {
            anyhow::bail!("Model file not found: {:?}", model_path);
        }

        let transformer = ExtendedTransformer::new(&model_path)?;
        
        let info = ModelInfo {
            id: model_id.to_string(),
            path: model_path.clone(),
            size: std::fs::metadata(&model_path)?.len(),
            loaded: true,
            quantization: quantization.unwrap_or("int8").to_string(),
            context_window: self.config.models.context_window,
        };

        let loaded_model = LoadedModel {
            info: info.clone(),
            transformer: Arc::new(RwLock::new(transformer)),
            last_used: std::time::Instant::now(),
            loaded_at: std::time::Instant::now(),
            request_count: std::sync::atomic::AtomicU64::new(0),
            total_tokens_generated: std::sync::atomic::AtomicU64::new(0),
            last_inference_at: None,
        };

        self.models.insert(model_id.to_string(), loaded_model);

        Ok(info.id)
    }

    pub fn unload_model(&self, model_id: &str
    ) -> Result<()> {
        if self.models.remove(model_id).is_none() {
            anyhow::bail!("Model not found: {}", model_id);
        }
        Ok(())
    }

    pub fn get_model(&self, model_id: &str
    ) -> Option<std::sync::Arc<LoadedModel>> {
        self.models.get(model_id).map(|entry| {
            let loaded_model = entry.value();
            std::sync::Arc::new(LoadedModel {
                info: loaded_model.info.clone(),
                transformer: loaded_model.transformer.clone(),
                last_used: loaded_model.last_used,
                loaded_at: loaded_model.loaded_at,
                request_count: std::sync::atomic::AtomicU64::new(
                    loaded_model.request_count.load(std::sync::atomic::Ordering::Relaxed)
                ),
                total_tokens_generated: std::sync::atomic::AtomicU64::new(
                    loaded_model.total_tokens_generated.load(std::sync::atomic::Ordering::Relaxed)
                ),
                last_inference_at: loaded_model.last_inference_at,
            })
        })
    }

    pub fn list_models(&self
    ) -> Vec<ModelInfo> {
        self.models
            .iter()
            .map(|entry| entry.value().info.clone())
            .collect()
    }

    pub fn enforce_model_limits(&self) -> Result<()> {
        if self.models.len() > self.config.models.max_loaded_models {
            // Simple LRU eviction - remove oldest model
            let mut oldest = None;
            let mut oldest_time = std::time::Instant::now();
            
            for entry in self.models.iter() {
                if entry.last_used < oldest_time {
                    oldest = Some(entry.key().clone());
                    oldest_time = entry.last_used;
                }
            }
            
            if let Some(model_id) = oldest {
                self.unload_model(&model_id)?;
                tracing::info!("Evicted model {} due to memory limits", model_id);
            }
        }
        Ok(())
    }
}