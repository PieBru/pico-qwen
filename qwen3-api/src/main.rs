use anyhow::Result;
use clap::Parser;
use tracing::{info, error};
use qwen3_api::config::Config;
use qwen3_api::server::Server;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to configuration file
    #[arg(short, long, default_value = "config.toml")]
    config: String,
    
    /// Enable debug logging
    #[arg(short, long)]
    debug: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    
    // Initialize logging
    let log_level = if args.debug { "debug" } else { "info" };
    tracing_subscriber::fmt()
        .with_env_filter(log_level)
        .init();
    
    info!("Starting Pico-Qwen API Server");
    
    // Load configuration
    let config = Config::load(&args.config)?;
    info!("Loaded configuration from {}", args.config);
    
    // Start server
    let server = Server::new(config);
    if let Err(e) = server.run().await {
        error!("Server error: {}", e);
        return Err(e);
    }
    
    Ok(())
}