# Gemini Workspace

This document provides instructions for interacting with the `pico-qwen` project.

## Project Structure

The project is organized into the following directories:

*   `qwen3-cli`: A command-line interface for interacting with the model.
*   `qwen3-export`: A tool for exporting the model to various formats.
*   `qwen3-inference`: The core inference library.
*   `docs`: Project documentation.

## How to Build

To build the project, run the following command from the root directory:

```bash
cargo build --release
```

## How to Run

To run the command-line interface, use the following command:

```bash
cargo run -p qwen3-cli -- --help
```

## How to Test

To run the project's tests, use the following command:

```bash
cargo test
```
