use axum::{
    extract::State,
    Json,
};
use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::state::AppState;

#[derive(Debug, Serialize, Deserialize)]
pub struct OpenAIModel {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub owned_by: String,
    pub permission: Vec<OpenAIModelPermission>,
    pub root: String,
    pub parent: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OpenAIModelPermission {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub allow_create_engine: bool,
    pub allow_sampling: bool,
    pub allow_logprobs: bool,
    pub allow_search_indices: bool,
    pub allow_view: bool,
    pub allow_fine_tuning: bool,
    pub organization: String,
    pub group: Option<String>,
    pub is_blocking: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OpenAIModelsResponse {
    pub object: String,
    pub data: Vec<OpenAIModel>,
}

pub async fn list_openai_models(
    State(state): State<AppState>,
) -> Json<OpenAIModelsResponse> {
    let models_dir = Path::new(&state.config.models.directory);
    let mut models = Vec::new();
    
    if let Ok(entries) = std::fs::read_dir(models_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("bin") {
                if let Some(stem) = path.file_stem() {
                    let model_id = stem.to_string_lossy().to_string();
                    
                    let model = OpenAIModel {
                        id: model_id.clone(),
                        object: "model".to_string(),
                        created: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_secs() as i64,
                        owned_by: "pico-qwen".to_string(),
                        permission: vec![OpenAIModelPermission {
                            id: format!("perm_{}", model_id),
                            object: "model_permission".to_string(),
                            created: std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap()
                                .as_secs() as i64,
                            allow_create_engine: false,
                            allow_sampling: true,
                            allow_logprobs: false,
                            allow_search_indices: false,
                            allow_view: true,
                            allow_fine_tuning: false,
                            organization: "*".to_string(),
                            group: None,
                            is_blocking: false,
                        }],
                        root: model_id.clone(),
                        parent: None,
                    };
                    models.push(model);
                }
            }
        }
    }
    
    // Sort by name for consistent output
    models.sort_by(|a, b| a.id.cmp(&b.id));
    
    Json(OpenAIModelsResponse {
        object: "list".to_string(),
        data: models,
    })
}