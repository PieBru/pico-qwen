use axum::{routing::get, Router};
use tower_http::{cors::CorsLayer, services::ServeDir, trace::TraceLayer};

use crate::config::Config;
use crate::websocket;

pub async fn health() -> axum::Json<serde_json::Value> {
    axum::Json(serde_json::json!({
        "status": "healthy",
        "service": "qwen3-web"
    }))
}

pub fn create_app(config: Config) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(tower_http::cors::Any)
        .allow_methods(tower_http::cors::Any)
        .allow_headers(tower_http::cors::Any);

    Router::new()
        .route("/health", get(health))
        .route("/ws", get(websocket::websocket_handler))
        .nest_service("/", ServeDir::new("qwen3-web/static"))
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(config)
}
