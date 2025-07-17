use axum::{
    extract::Request,
    http::{HeaderMap, Method, Uri},
    middleware::Next,
    response::Response,
};
use std::time::Instant;
use tracing::{info, warn};

pub async fn logging_middleware(
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    request: Request,
    next: Next,
) -> Response {
    let start = Instant::now();
    let request_id = generate_request_id();

    info!(
        request_id = %request_id,
        method = %method,
        uri = %uri,
        user_agent = ?headers.get("user-agent"),
        "Incoming request"
    );

    let response = next.run(request).await;
    let duration = start.elapsed();

    let status = response.status();

    if status.is_server_error() {
        warn!(
            request_id = %request_id,
            status = %status,
            duration_ms = duration.as_millis(),
            "Request completed with error"
        );
    } else {
        info!(
            request_id = %request_id,
            status = %status,
            duration_ms = duration.as_millis(),
            "Request completed"
        );
    }

    response
}

fn generate_request_id() -> String {
    uuid::Uuid::new_v4().to_string()
}
