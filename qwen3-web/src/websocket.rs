use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::Response,
};
use futures::{sink::SinkExt, stream::StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::config::Config;

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WebSocketRequest {
    pub action: String,
    pub message: Option<ChatMessage>,
    pub model: Option<String>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WebSocketResponse {
    pub action: String,
    pub data: serde_json::Value,
}

pub async fn websocket_handler(ws: WebSocketUpgrade, State(config): State<Config>) -> Response {
    ws.on_upgrade(move |socket| handle_websocket(socket, config))
}

async fn handle_websocket(socket: WebSocket, config: Config) {
    let (mut sender, mut receiver) = socket.split();

    while let Some(Ok(msg)) = receiver.next().await {
        match msg {
            Message::Text(text) => {
                if let Ok(request) = serde_json::from_str::<WebSocketRequest>(&text) {
                    match request.action.as_str() {
                        "ping" => {
                            let timestamp = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap()
                                .as_secs();
                            let _ = sender
                                .send(Message::Text(
                                    serde_json::to_string(&WebSocketResponse {
                                        action: "pong".to_string(),
                                        data: json!({"timestamp": timestamp}),
                                    })
                                    .unwrap(),
                                ))
                                .await;
                        }
                        "chat" => {
                            // Forward to API server
                            let response = forward_to_api(&config, &request).await;
                            let _ = sender
                                .send(Message::Text(serde_json::to_string(&response).unwrap()))
                                .await;
                        }
                        _ => {
                            let _ = sender
                                .send(Message::Text(
                                    serde_json::to_string(&WebSocketResponse {
                                        action: "error".to_string(),
                                        data: json!({"message": "Unknown action"}),
                                    })
                                    .unwrap(),
                                ))
                                .await;
                        }
                    }
                }
            }
            Message::Close(_) => break,
            _ => {}
        }
    }
}

async fn forward_to_api(_config: &Config, _request: &WebSocketRequest) -> WebSocketResponse {
    // For now, return a mock response
    // In real implementation, this would make HTTP requests to the API server
    WebSocketResponse {
        action: "chat_response".to_string(),
        data: json!({
            "message": {
                "role": "assistant",
                "content": "This is a mock response. Connect to a real API server for actual responses."
            }
        }),
    }
}
