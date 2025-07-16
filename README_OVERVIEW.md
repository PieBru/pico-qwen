# pico-qwen Overview

## Project Description

**pico-qwen** is an experimental, feature-enriched fork of [qwen3-rs](https://github.com/reinterpretcat/qwen3-rs) specifically designed for **very low-resource systems**. While maintaining the educational clarity and minimal dependencies of the original project, pico-qwen extends the foundation with production-ready features for deployment on MiniPCs, SBCs (Single Board Computers), and low-power home automation servers.

## Merged Vision

### Educational Foundation (from qwen3-rs)
- **Educational**: Learn how transformer architectures, quantization, and efficient inference work in Rust
- **Minimal Dependencies**: Most algorithms (tokenization, quantization, sampling, etc.) are implemented from scratch—no heavy ML or Python bindings
- **Modular**: Core library logic is separated from CLI tools for clarity and maintainability
- **Efficiency**: Uses memory mapping and zero-copy techniques for handling large model files

### Production Extensions (pico-qwen enhancements)
- **Low-requirements inference API** with REST endpoints and streaming support
- **Minimalist Chat-WebUI** with progressive enhancement and mobile-first design
- **MCP (Model Context Protocol) endpoint** with multi-agentic tools support
- **Cloud/local fallback** hybrid inference system
- **Selectable quantization** (INT4, INT8, FP16, FP32) with configurable group sizes
- **CPU-specific builds** optimized for Intel N100, i9-14900HX, Raspberry Pi 4/5, and generic processors

## Key Differentiators

| Aspect | qwen3-rs | pico-qwen |
|--------|----------|-----------|
| **Primary Focus** | Educational exploration | Production deployment on low-resource systems |
| **Target Systems** | General purpose | MiniPCs, SBCs, low-power servers |
| **Resource Usage** | Standard | Optimized for <8GB RAM |
| **Interface** | CLI only | CLI + REST API + WebUI |
| **Deployment** | Manual | systemd + Docker ready |
| **Quantization** | INT8 only | INT4/INT8/FP16/FP32 |
| **CPU Optimization** | Basic | Runtime CPU feature detection |
| **Multi-Agent** | None | MCP with PocketFlow integration |
| **Cloud Integration** | None | Hybrid cloud/local fallback |

## Architecture Philosophy

pico-qwen maintains the original **forked project posture** of qwen3-rs, meaning:
- We preserve the original architecture and educational clarity
- Each new feature is developed in separate git branches for potential upstream contribution
- We reuse as much original code as possible at every development stage
- Documentation is split between project-level (README files) and detailed technical docs (./docs/ directory)

This approach ensures that pico-qwen remains a **feature-enriched extension** rather than a complete rewrite, making it easy to merge enhancements back to the original qwen3-rs project when appropriate.