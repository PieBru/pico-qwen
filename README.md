# pico-qwen

## The Idea

**pico-qwen** is an experimental, born as a feature enriched version of [qwen3-rs by Ilya Builuk](https://github.com/reinterpretcat/qwen3-rs), targeting very low-resources systems. Our goals are to explore the feasibility of some additional features, developed as much as possible using Rust:
- [ ] Serve a low-requirements inference API endpoint that can be installed on systems like MiniPC, SBC, low-power Home Automation servers, etc.
- [ ] Serve a minimalist Chat-WebUI that uses the underlying inference API endpoint, initially developed in HTML+CSS+JS and then eventually in Rust to solve the X-ref browsers blocking.
- [ ] Serve a MCP endpoint with multi-agentic tools like WEB Search, WEB Research, Sequential Thinking, Rumination, etc., developed with PocketFlow to maintain a low-resuorces posture.
- [ ] Transparently allow fast LLM inference served by the cloud when online, while resiliently fall back to a local slow-but-working emergency inference when all the configured cloud servers aren't available, i.e. when the Internet is offline.
- [ ] Allow selecting the quantization level, thus being able to balance quality, performance and available computing resources.
- [ ] Allow building for a specific target CPU (i.e. Intel N100, Raspberry PI, etc.), thus exploiting the optimizations available on that system.

And more to come, if the experimental phase has success.


## Project Guidelines

- **Architecture.** We keep the original "qwen3-rs" forked project posture and create a new git branch for each new feature, so it will be easy to merge our additional features back on the original project.
- **Project documentation.** Except the `README.md`, all our project documentation is contained into the `./docs/` directory. So, when we refer to the original project abstractions contained into `qwen3-rs.md`, that document file is located into `./docs/qwen3-rs.md`.
  - @docs/qwen3-rs.md is a tutorial on our project foundation, that is based on the `qwen3-rs` project, please load it into the coding agent context before planning any new feature.
- We would like to reuse as much as possible of the project code at any development stage.
- We prefer to use **Arch Linux**, and any command/feature of our project should work from the command line, as well as a **`systemd`** service managed via `systemctl`.
- We like minimalist, streamlined tools, CLI and UI. Also please use minimal dependencies.
- **Agentic features.** Some of our new features use agents and tools, they should be implemented using the [PocketFlow Framework](https://github.com/the-pocket/PocketFlow). Its documentation is contained into https://github.com/The-Pocket/PocketFlow/tree/main/docs/ , its abstractions tutorial is locally into @docs/PocketFlow.md and a cookbook documentation rich of examples is @docs/PocketFlow_Cookbook_40.md

---

# qwen3-rs Description

**qwen3-rs** is an educational Rust project for exploring and running Qwen3 language family models. It is designed to be clear, modular, and approachable for learners, with minimal dependencies and many core algorithms reimplemented from scratch for transparency.

> **Note:** Parts of this codebase, including documentation and core algorithms, were generated or assisted by large language models (LLMs) to accelerate development and improve educational clarity. As a starting reference, the project [qwen3.c](https://github.com/adriancable/qwen3.c) was used for understanding model internals and file formats.


## qwen3-rs Project Goals

- **Educational:** Learn how transformer architectures, quantization, and efficient inference work in Rust.
- **Minimal Dependencies:** Most algorithms (tokenization, quantization, sampling, etc.) are implemented from scratch—no heavy ML or Python bindings.
- **Modular:** Core library logic is separated from CLI tools for clarity and maintainability.
- **Efficiency:** Uses memory mapping and zero-copy techniques for handling large model files.

## Workspace Structure

```
qwen3-rs/
├── docs                # LLM generated docs for key components
├── Cargo.toml          # Workspace configuration
├── qwen3-cli/          # Command-line interface crate
├── qwen3-export/       # Model export crate
├── qwen3-inference/    # LLM inference crate
```

## How to Use

### 1. Get a HuggingFace Qwen3 model

```bash
git clone https://huggingface.co/Qwen/Qwen3-0.6B
# Or try larger/alternative models:
# git clone https://huggingface.co/Qwen/Qwen3-4B
# git clone https://huggingface.co/Qwen/Qwen3-8B
# git clone https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
```

### 2. Build and run the exporter

```bash
cargo build --release -p qwen3-cli

# Export a HuggingFace model to quantized checkpoint format
cargo run --release -p qwen3-cli -- export /path/to/model /path/to/output.bin --group-size 64
```

### 3. Run inference

In chat mode with default parameters:

```bash
cargo run --release -p qwen3-cli -- inference /path/to/output.bin -m chat
```

## CLI Commands and Options

### `export`
Exports a HuggingFace Qwen3 model to a custom binary format for efficient Rust inference.

**Usage:**
```bash
qwen3 export <MODEL_PATH> <OUTPUT_PATH> [--group-size <SIZE>]
```
- `MODEL_PATH`: Path to HuggingFace model directory (must contain config.json, *.safetensors, tokenizer.json)
- `OUTPUT_PATH`: Output path for the binary model file
- `--group-size`, `-g`: Quantization group size (default: 64)

### `inference`
Runs inference on a binary Qwen3 model.

**Usage:**
```bash
qwen3 inference <checkpoint> [options]
```
**Options:**
- `--temperature`, `-t <FLOAT>`: Sampling temperature (default: 1.0)
- `--topp`, `-p <FLOAT>`: Top-p nucleus sampling (default: 0.9)
- `--seed`, `-s <INT>`: Random seed
- `--context`, `-c <INT>`: Context window size (default: max_seq_len)
- `--mode`, `-m <STRING>`: Mode: `generate` or `chat` (default: chat)
- `--input`, `-i <STRING>`: Input prompt
- `--system`, `-y <STRING>`: System prompt (for chat mode)
- `--reasoning`, `-r <INT>`: Reasoning mode: 0=no thinking, 1=thinking (default: 0)

