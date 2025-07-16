# Credits

This project builds upon the excellent work of several open-source projects and communities. We acknowledge and thank the following foundational projects:

## Primary Foundation

### [qwen3-rs](https://github.com/reinterpretcat/qwen3-rs)
**Core foundation and educational architecture**
- Original educational Rust implementation of Qwen3 language models
- Provided the modular architecture and clear educational approach
- Maintained minimal dependencies and transparency in implementation
- Author: Ilya Builuk ([@reinterpretcat](https://github.com/reinterpretcat))

### [qwen3.c](https://github.com/adriancable/qwen3.c)
**Reference implementation**
- Original C implementation that served as architectural reference
- Provided insights into model internals and file formats
- Author: Adrian Cable

## Agent Framework

### [PocketFlow-Rust](https://github.com/The-Pocket/PocketFlow-Rust)
**Multi-agent orchestration foundation**
- Lightweight Rust implementation of the PocketFlow framework
- Enables structured agent workflows and tool orchestration
- Perfect fit for low-resource systems
- Part of [The Pocket](https://github.com/The-Pocket) ecosystem

## Model Source

### [Qwen3](https://huggingface.co/Qwen) and [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B)
**Language models**
- Alibaba's Qwen team for providing the Qwen3 model family
- DeepSeek team for providing the enhanced DeepSeek-R1-0528-Qwen3-8B variant
- Models available: Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B, Qwen3-8B, and DeepSeek-R1-0528-Qwen3-8B (more performant)
- Open-source models licensed for research and commercial use

## Ecosystem Dependencies

### Key Rust Crates
- **safetensors**: HuggingFace tensor format handling
- **rayon**: Data parallelism for performance optimization  
- **memmap2**: Memory-mapped file access for large models
- **anyhow**: Error handling with minimal overhead
- **tokio**: Async runtime for API server
- **axum**: Web framework for REST API

### Development Tools
- **Rust**: Systems programming language for performance and safety
- **Cargo**: Rust package manager and build system
- **systemd**: Linux service management for deployment
- **Docker**: Containerization for cross-platform deployment

## Community Contributions

Special thanks to:
- The Rust community for excellent tooling and libraries
- HuggingFace for model hosting and standardization
- Arch Linux community for maintaining development tools
- Contributors to all dependencies used in this project

## License Acknowledgments

This project maintains compatibility with the original qwen3-rs license terms. All dependencies are used according to their respective licenses. Please refer to individual project repositories for specific license details.

## How to Contribute

We welcome contributions that:
- Improve low-resource system performance
- Add new CPU target optimizations
- Enhance the educational aspects of the codebase
- Improve deployment and service management
- Extend MCP agent capabilities

Please see individual project repositories for contribution guidelines and issue reporting.