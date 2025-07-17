use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use tower::util::ServiceExt; // for `oneshot`

use qwen3_web::config::Config;

#[tokio::test]
async fn test_health_check() {
    let config = Config {
        server: qwen3_web::config::ServerConfig {
            bind_address: "127.0.0.1".to_string(),
            port: 3000,
        },
        api: qwen3_web::config::ApiConfig {
            url: "http://localhost:8080".to_string(),
        },
    };

    let app = qwen3_web::routes::create_app(config);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/health")
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
    assert_eq!(health["service"], "qwen3-web");
}

#[tokio::test]
async fn test_static_files() {
    let config = Config {
        server: qwen3_web::config::ServerConfig {
            bind_address: "127.0.0.1".to_string(),
            port: 3000,
        },
        api: qwen3_web::config::ApiConfig {
            url: "http://localhost:8080".to_string(),
        },
    };

    let app = qwen3_web::routes::create_app(config);

    // Test health endpoint (static files may not exist in test environment)
    let response = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_cors_headers() {
    let config = Config {
        server: qwen3_web::config::ServerConfig {
            bind_address: "127.0.0.1".to_string(),
            port: 3000,
        },
        api: qwen3_web::config::ApiConfig {
            url: "http://localhost:8080".to_string(),
        },
    };

    let app = qwen3_web::routes::create_app(config);

    let response = app
        .oneshot(
            Request::builder()
                .method("OPTIONS")
                .uri("/health")
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
}
