use anyhow::Result;
use axum::{
    routing::get,
    Router,
};
use std::net::SocketAddr;
use tower_http::{
    cors::CorsLayer,
    trace::TraceLayer,
    limit::RequestBodyLimitLayer,
};
use tracing::info;

use crate::{config::Config, state::AppState};
use crate::handlers::{chat, generate, models, health, openai, status};

pub struct Server {
    config: Config,
    state: AppState,
}

impl Server {
    pub fn new(config: Config) -> Self {
        let state = AppState::new(config.clone());
        Self { config, state }
    }

    pub async fn run(self) -> Result<()> {
        let app = self.create_router();
        
        let addr = SocketAddr::new(
            self.config.server.bind_address.parse()?, 
            self.config.server.port
        );
        
        info!("Starting server on {}", addr);
        
        let listener = tokio::net::TcpListener::bind(addr).await?;
        axum::serve(listener, app).await?;
        
        Ok(())
    }

    pub fn create_router(&self
    ) -> Router {
        let cors = CorsLayer::new()
            .allow_origin(tower_http::cors::Any)
            .allow_methods(tower_http::cors::Any)
            .allow_headers(tower_http::cors::Any);

        Router::new()
            .route("/api/v1/health", get(health::health_check))
            .route("/api/v1/status", get(status::server_status))
            .route("/api/v1/models", get(models::list_models))
            .route("/api/v1/models/:model_id/load", axum::routing::post(models::load_model))
            .route("/api/v1/models/:model_id/unload", axum::routing::post(models::unload_model))
            .route("/api/v1/chat", axum::routing::post(chat::chat_handler))
            .route("/api/v1/generate", axum::routing::post(generate::generate_handler))
            .route("/v1/models", axum::routing::get(openai::list_openai_models))
            .layer(cors)
            .layer(TraceLayer::new_for_http())
            .layer(RequestBodyLimitLayer::new(self.config.limits.max_request_size))
                        .with_state(self.state.clone())
    }
}