# Pico-Qwen Implementation Plan

## 🎯 Project Vision

**Pico-Qwen** extends [qwen3-rs](https://github.com/reinterpretcat/qwen3-rs) to create a production-ready, low-resource LLM system for MiniPCs, SBCs, and home automation servers while maintaining educational clarity and minimal dependencies.

## 🔍 Architecture Foundation

The existing qwen3-rs architecture provides an excellent foundation with:
- **Modular workspace structure** (CLI, Export, Inference crates)
- **Clean abstractions** for transformer components
- **INT8 quantization** with group-wise scaling
- **Memory-mapped model loading** for efficiency
- **Educational transparency** in implementations

## 📋 Phase-Based Implementation

### 🏗️ Phase 1: Core Infrastructure Extension
**Branch: `feature/infrastructure`**
**Duration: 1-2 weeks**

#### 1.1 Enhanced Model Configuration
```rust
// qwen3-inference/src/configuration.rs
pub struct ExtendedModelConfig {
    pub base: ModelConfig,           // Existing config
    pub quantization: QuantizationLevel,  // INT4, INT8, FP16, FP32
    pub cpu_optimizations: CpuTarget,     // Intel-N100, RPi-4, Generic
    pub cloud_config: Option<CloudConfig>, // Hybrid inference
    pub memory_limits: MemoryLimits,      // Resource constraints
}
```

#### 1.2 Advanced Quantization System
```rust
// qwen3-inference/src/tensor.rs
pub enum QuantizationLevel {
    Int4 { group_size: usize },  // Ultra-low memory
    Int8 { group_size: usize },  // Balanced (existing)
    Fp16,                        // Better quality
    Fp32,                        // Full precision
}

impl QuantizedTensor {
    pub fn quantize_dynamic(
        data: &[f32], 
        level: QuantizationLevel,
        cpu_target: CpuTarget
    ) -> Self;
}
```

### 🌐 Phase 2: Low-Requirements API Server
**Branch: `feature/api-server`**
**Duration: 2-3 weeks**
**Dependencies: Phase 1**

#### 2.1 New Crate: `qwen3-api`
```
qwen3-api/
├── Cargo.toml
├── src/
│   ├── main.rs          # Axum server entry
│   ├── server.rs        # HTTP server setup
│   ├── handlers/
│   │   ├── chat.rs      # POST /api/v1/chat
│   │   ├── generate.rs  # POST /api/v1/generate
│   │   ├── models.rs    # GET /api/v1/models
│   │   └── health.rs    # GET /api/v1/health
│   ├── middleware/
│   │   ├── logging.rs   # Request logging
│   │   ├── cors.rs      # CORS handling
│   │   └── rate_limit.rs # Resource protection
│   ├── config.rs        # Server configuration
│   └── state.rs         # Shared application state
```

#### 2.2 API Design
```yaml
# OpenAPI specification
endpoints:
  /api/v1/chat:
    post:
      body: { model: string, messages: array, max_tokens: int }
      response: { choices: [{ message: { content: string } }] }
      streaming: true

  /api/v1/generate:
    post:
      body: { model: string, prompt: string, temperature: float }
      response: { text: string }

  /api/v1/models:
    get:
      response: { data: [{ id: string, size: int, loaded: bool }] }

  /api/v1/models/{id}/load:
    post:
      body: { quantization: string, context_size: int }
```

#### 2.3 Memory Optimization Features
- **Streaming responses** to minimize memory usage
- **Model pooling** with LRU eviction
- **Context window management** with automatic truncation
- **Memory pressure monitoring** with graceful degradation

### 💬 Phase 3: Minimalist Chat-WebUI
**Branch: `feature/webui`**
**Duration: 1-2 weeks**
**Dependencies: Phase 2**

#### 3.1 Progressive Enhancement Strategy
```
qwen3-web/
├── static/                 # Pure HTML+CSS+JS fallback
│   ├── index.html
│   ├── styles.css
│   ├── app.js
│   └── wasm/              # Optional Rust WASM
├── src/
│   ├── main.rs           # Static file server
│   ├── routes.rs         # Web routes
│   └── websocket.rs      # Real-time updates
```

#### 3.2 UI Design Principles
- **Mobile-first responsive** design
- **Minimal dependencies** (vanilla JS, no frameworks)
- **Keyboard navigation** support
- **Low-bandwidth mode** for slow connections
- **Offline capability** with service worker

### 🤖 Phase 4: MCP Multi-Agent System
**Branch: `feature/mcp-agents`**
**Duration: 3-4 weeks**
**Dependencies: Phase 2**

#### 4.1 New Crate: `qwen3-mcp`
```rust
// qwen3-mcp/src/agents/mod.rs
pub struct AgentOrchestrator {
    pub llm: Arc<dyn LlmBackend>,
    pub tools: Vec<Box<dyn Tool>>,
    pub flows: Vec<Flow>,
}

pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn execute(&self, input: &str) -> Result<String>;
    fn cost_estimate(&self, input: &str) -> f64;
}
```

#### 4.2 Agent Implementations
```rust
// qwen3-mcp/src/agents/web_search.rs
pub struct WebSearchAgent {
    pub search_provider: DuckDuckGoClient,
    pub summarizer: Arc<dyn LlmBackend>,
}

impl Tool for WebSearchAgent {
    fn execute(&self, query: &str) -> Result<String> {
        // Search, scrape, and summarize
    }
}

// qwen3-mcp/src/agents/thinking.rs
pub struct SequentialThinkingAgent {
    pub max_steps: usize,
    pub reflection_prompt: String,
}
```

#### 4.3 PocketFlow Integration
```rust
// qwen3-mcp/src/flows/research_flow.rs
pub fn create_research_flow() -> Flow {
    Flow::new("research")
        .step("search", WebSearchAgent)
        .step("analyze", AnalysisAgent)
        .step("synthesize", SynthesisAgent)
        .condition(|state| state.confidence > 0.8)
}
```

### ☁️ Phase 5: Hybrid Cloud/Edge Inference
**Branch: `feature/hybrid-inference`**
**Duration: 2-3 weeks**
**Dependencies: Phase 2**

#### 5.1 Cloud Provider Abstraction
```rust
// qwen3-inference/src/cloud/mod.rs
#[async_trait]
pub trait CloudProvider: Send + Sync {
    async fn generate(&self, prompt: &str, config: &InferenceConfig) -> Result<String>;
    fn is_available(&self) -> bool;
    fn get_cost_estimate(&self, tokens: usize) -> f64;
    fn get_latency_estimate(&self) -> Duration;
}

pub struct OpenAiProvider { /* ... */ }
pub struct AnthropicProvider { /* ... */ }
pub struct LocalProvider { /* ... */ }
```

#### 5.2 Failover System
```rust
// qwen3-inference/src/hybrid.rs
pub struct HybridInference {
    pub primary: Box<dyn CloudProvider>,
    pub fallback: Box<dyn CloudProvider>,
    pub health_monitor: HealthMonitor,
    pub routing_strategy: RoutingStrategy,
}

impl HybridInference {
    pub async fn generate(&self, prompt: &str) -> Result<String> {
        match self.health_monitor.check_primary().await {
            HealthStatus::Healthy => self.primary.generate(prompt).await,
            HealthStatus::Degraded => self.smart_routing(prompt).await,
            HealthStatus::Unavailable => self.fallback.generate(prompt).await,
        }
    }
}
```

### 🔧 Phase 6: CPU-Specific Optimization
**Branch: `feature/cpu-optimization`**
**Duration: 2-3 weeks**
**Dependencies: Phase 1**

#### 6.1 CPU Feature Detection
```rust
// qwen3-inference/src/cpu/detect.rs
pub struct CpuInfo {
    pub vendor: CpuVendor,
    pub features: Vec<CpuFeature>,
    pub cache_size: usize,
    pub memory_bandwidth: usize,
}

pub fn detect_cpu() -> CpuInfo {
    // Use CPUID instruction on x86_64
    // Use /proc/cpuinfo on ARM
}
```

#### 6.2 Target-Specific Builds
```toml
# Cargo.toml features
[features]
default = []
intel-n100 = ["avx2", "avx512", "vnni"]
raspberry-pi-4 = ["neon", "fp16"]
raspberry-pi-5 = ["neon", "fp16", "sve"]
generic = []
```

#### 6.3 Runtime Optimization Selection
```rust
// qwen3-inference/src/optimize.rs
pub fn select_optimization_strategy(cpu: &CpuInfo) -> OptimizationStrategy {
    match cpu.vendor {
        CpuVendor::Intel => intel_optimizations(cpu),
        CpuVendor::Arm => arm_optimizations(cpu),
        _ => generic_optimizations(),
    }
}
```

### 🚀 Phase 7: Deployment & Service Management
**Branch: `feature/deployment`**
**Duration: 1 week**
**Dependencies: All phases**

#### 7.1 Systemd Integration
```bash
# /etc/systemd/system/pico-qwen-api.service
[Unit]
Description=Pico Qwen API Server
Documentation=https://github.com/PieBru/pico-qwen
After=network.target

[Service]
Type=simple
User=pico-qwen
Group=pico-qwen
ExecStart=/usr/local/bin/pico-qwen-api --config /etc/pico-qwen/config.toml
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

#### 7.2 Configuration Management
```toml
# /etc/pico-qwen/config.toml
[server]
bind_address = "0.0.0.0:8080"
max_connections = 100
request_timeout = 30

[models]
directory = "/var/lib/pico-qwen/models"
default_quantization = "int8"
max_loaded_models = 2

[cloud]
providers = ["openai", "anthropic"]
fallback_to_local = true
health_check_interval = 30

[resources]
max_memory_mb = 2048
max_context_length = 4096
```

## 🔄 Integration Strategy

### Git Workflow
```bash
# Feature branch development
main
├── feature/infrastructure
├── feature/api-server
├── feature/webui
├── feature/mcp-agents
├── feature/hybrid-inference
├── feature/cpu-optimization
└── feature/deployment
```

### Testing Strategy
```bash
# Each phase includes comprehensive testing
cargo test -p qwen3-api
cargo test -p qwen3-web
cargo test -p qwen3-mcp
cargo test --features integration-tests

# Performance benchmarks
cargo bench -p qwen3-inference
```

### Backward Compatibility
- **All features are additive**
- **Existing CLI unchanged**
- **Original export format preserved**
- **New crates optional dependencies**

## 📊 Resource Requirements

| Component | Memory | CPU | Storage | Network |
|-----------|--------|-----|---------|---------|
| API Server | 50MB | Low | 10MB | 1Mbps |
| 7B Model (INT8) | 8GB | Moderate | 8GB | - |
| 7B Model (INT4) | 4GB | Moderate | 4GB | - |
| WebUI | 5MB | Very Low | 2MB | 100Kbps |
| MCP Agents | 100MB | Low | 50MB | 10Mbps |

## 🎯 Success Metrics

### Performance Targets
- **Model loading**: < 10 seconds for 7B model
- **API response**: < 500ms average
- **Cloud failover**: < 5 seconds
- **Memory usage**: < 2GB for complete system

### Compatibility Targets
- **Raspberry Pi 4**: 4GB RAM, full functionality
- **Intel N100**: 8GB RAM, optimized performance
- **Generic x86**: 4GB RAM, basic functionality
- **ARM64**: Full support with NEON optimizations

## 📅 Implementation Timeline

| Phase | Duration | Priority | Risk Level |
|-------|----------|----------|------------|
| Phase 1: Infrastructure | 1-2 weeks | High | Low |
| Phase 2: API Server | 2-3 weeks | High | Medium |
| Phase 3: WebUI | 1-2 weeks | Medium | Low |
| Phase 4: MCP Agents | 3-4 weeks | Medium | High |
| Phase 5: Hybrid Cloud | 2-3 weeks | High | Medium |
| Phase 6: CPU Optimization | 2-3 weeks | Medium | Medium |
| Phase 7: Deployment | 1 week | Low | Low |

**Total estimated duration**: 12-18 weeks

## 🔧 Development Commands

```bash
# Development workflow
cargo check --all
cargo test --all
cargo build --release -p qwen3-api
cargo run -p qwen3-cli -- --help

# Cross-compilation
cargo build --release --target aarch64-unknown-linux-gnu
cargo build --release --target x86_64-unknown-linux-musl

# Systemd testing
sudo systemctl start pico-qwen-api
sudo journalctl -u pico-qwen-api -f
```

This plan maintains the educational clarity and minimal dependencies philosophy of qwen3-rs while adding production-ready features for low-resource systems.