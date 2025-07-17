use anyhow::Result;
use clap::Parser;
use std::net::SocketAddr;
use tracing::info;

mod config;
mod routes;
mod websocket;

use crate::config::Config;

#[derive(Parser)]
#[command(name = "qwen3-web")]
#[command(about = "Minimalist chat web interface for Pico-Qwen")]
struct Args {
    #[arg(long, default_value = "0.0.0.0")]
    bind_address: String,

    #[arg(short, long, default_value = "3000")]
    port: u16,

    #[arg(long, default_value = "http://localhost:8080")]
    api_url: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();
    let config = Config {
        server: config::ServerConfig {
            bind_address: args.bind_address,
            port: args.port,
        },
        api: config::ApiConfig { url: args.api_url },
    };

    let addr = SocketAddr::new(config.server.bind_address.parse()?, config.server.port);

    let app = routes::create_app(config);

    info!("Starting web server on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
