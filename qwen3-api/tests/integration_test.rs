use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use serde_json::json;
use tower::ServiceExt; // for `oneshot`

use qwen3_api::{config::Config, server::Server};

#[tokio::test]
async fn test_health_check() {
    let config = Config::default();
    let server = Server::new(config);
    let app = server.create_router();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/v1/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let health: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(health["status"], "healthy");
}

#[tokio::test]
async fn test_models_list() {
    let config = Config::default();
    let server = Server::new(config);
    let app = server.create_router();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let models: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert!(models["models"].is_array());
}

#[tokio::test]
async fn test_generate_endpoint() {
    let config = Config::default();
    let server = Server::new(config);
    let app = server.create_router();

    let request_body = json!({
        "model": "test-model",
        "prompt": "Hello, world!",
        "max_tokens": 10,
        "temperature": 0.7
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/v1/generate")
                .header("content-type", "application/json")
                .body(Body::from(request_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    // Should return 404 since model doesn't exist
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_chat_endpoint() {
    let config = Config::default();
    let server = Server::new(config);
    let app = server.create_router();

    let request_body = json!({
        "model": "test-model",
        "messages": [
            {
                "role": "user",
                "content": "Hello!"
            }
        ],
        "max_tokens": 10
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/v1/chat")
                .header("content-type", "application/json")
                .body(Body::from(request_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    // Should return 404 since model doesn't exist
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_cors_headers() {
    let config = Config::default();
    let server = Server::new(config);
    let app = server.create_router();

    let response = app
        .oneshot(
            Request::builder()
                .method("OPTIONS")
                .uri("/api/v1/health")
                .header("origin", "http://localhost:3000")
                .header("access-control-request-method", "GET")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert!(response
        .headers()
        .contains_key("access-control-allow-origin"));
    assert_eq!(response.headers()["access-control-allow-origin"], "*");
}

#[tokio::test]
async fn test_invalid_json_returns_422() {
    let config = Config::default();
    let server = Server::new(config);
    let app = server.create_router();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/v1/chat")
                .header("content-type", "application/json")
                .body(Body::from("{invalid json}"))
                .unwrap(),
        )
        .await
        .unwrap();

    // Axum returns 400 for invalid JSON, not 422
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}
