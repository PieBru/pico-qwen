use anyhow::Result;
use clap::Parser;
use qwen3_api::config::Config;
use qwen3_api::server::Server;
use tracing::{error, info};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to configuration file
    #[arg(short, long, default_value = "config.toml")]
    config: String,

    /// Enable debug logging
    #[arg(short, long)]
    debug: bool,

    /// List available models and exit
    #[arg(long)]
    list_models: bool,

    /// Directory to search for models (overrides config)
    #[arg(short, long)]
    models: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Handle list models option
    if args.list_models {
        let config = Config::load(&args.config)?;

        use std::path::Path;

        let models_dir = args.models
            .as_ref()
            .map(|s| Path::new(s))
            .unwrap_or_else(|| Path::new(&config.models.directory));

        println!("Available models in {}", models_dir.display());
        println!("┌─────────────────────────────────────────┬────────────────┬────────────┐");
        println!("│ Model Name                              │ Size           │ Format     │");
        println!("├─────────────────────────────────────────┼────────────────┼────────────┤");

        let mut found_models = false;

        if let Ok(entries) = std::fs::read_dir(models_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) == Some("bin") {
                    if let Ok(metadata) = entry.metadata() {
                        let name = path
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .unwrap_or("unknown")
                            .to_string();

                        let format = if name.contains("int4") {
                            "INT4"
                        } else if name.contains("int8") {
                            "INT8"
                        } else if name.contains("fp16") {
                            "FP16"
                        } else {
                            "BINARY"
                        }
                        .to_string();

                        let size_mb = metadata.len() as f64 / (1024.0 * 1024.0);
                        println!(
                            "│ {:<39} │ {:<14} │ {:<10} │",
                            name,
                            format!("{:.1} MB", size_mb),
                            format
                        );
                        found_models = true;
                    }
                }
            }
        }

        if found_models {
            println!("└─────────────────────────────────────────┴────────────────┴────────────┘");
        } else {
            println!("No .bin models found in {}", config.models.directory);
        }

        return Ok(());
    }

    // Initialize logging
    let log_level = if args.debug { "debug" } else { "info" };
    tracing_subscriber::fmt().with_env_filter(log_level).init();

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
