use crate::state::{AppState, ModelInfo};
use axum::{
    extract::{Path, State},
    Json,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct LoadModelRequest {
    pub quantization: Option<String>,
    pub context_size: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LoadModelResponse {
    pub success: bool,
    pub model_id: String,
    pub message: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct UnloadModelResponse {
    pub success: bool,
    pub message: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelsListResponse {
    pub models: Vec<ModelInfo>,
    pub total: usize,
}

pub async fn list_models(State(state): State<AppState>) -> Json<ModelsListResponse> {
    let models = state.list_models();
    let total = models.len();

    Json(ModelsListResponse { models, total })
}

pub async fn load_model(
    Path(model_id): Path<String>,
    State(state): State<AppState>,
    Json(request): Json<LoadModelRequest>,
) -> Json<LoadModelResponse> {
    // Enforce model limits
    if let Err(e) = state.enforce_model_limits() {
        return Json(LoadModelResponse {
            success: false,
            model_id: model_id.clone(),
            message: format!("Failed to enforce model limits: {e}"),
        });
    }

    // Load the model
    match state
        .load_model(&model_id, request.quantization.as_deref())
        .await
    {
        Ok(model_id) => Json(LoadModelResponse {
            success: true,
            model_id,
            message: "Model loaded successfully".to_string(),
        }),
        Err(e) => Json(LoadModelResponse {
            success: false,
            model_id,
            message: format!("Failed to load model: {e}"),
        }),
    }
}

pub async fn unload_model(
    Path(model_id): Path<String>,
    State(state): State<AppState>,
) -> Json<UnloadModelResponse> {
    match state.unload_model(&model_id) {
        Ok(_) => Json(UnloadModelResponse {
            success: true,
            message: format!("Model {model_id} unloaded successfully"),
        }),
        Err(e) => Json(UnloadModelResponse {
            success: false,
            message: format!("Failed to unload model: {e}"),
        }),
    }
}
