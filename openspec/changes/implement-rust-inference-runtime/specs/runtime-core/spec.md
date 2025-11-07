# Rust Inference Runtime Core

## ADDED Requirements

### Requirement: Load Base Model with Quantization

The runtime SHALL load Qwen3-0.6B base model with INT4/INT8 quantization.

#### Scenario: Load INT4 quantized model

- **WHEN** runtime initializes with `quantization: "int4"`
- **THEN** the system loads model weights as INT4
- **AND** uses 0.3-0.4 GB VRAM
- **AND** completes loading in <5 seconds
- **AND** verifies model integrity
- **AND** runs test inference to validate

#### Scenario: Apply RoPE scaling

- **WHEN** loading model with `rope_scaling: "yarn-128k"`
- **THEN** the system applies YaRN scaling to position embeddings
- **AND** supports contexts up to 128k tokens
- **AND** maintains quality on long contexts
- **AND** falls back to NTK if YaRN unavailable

### Requirement: Hot-Swap Expert Adapters

The runtime SHALL load and unload expert adapters in <10ms.

#### Scenario: Attach LoRA adapter

- **WHEN** attaching json-parser expert
- **THEN** the runtime loads adapter weights from SafeTensors
- **AND** applies LoRA matrices to target modules
- **AND** completes in <10ms (if weights pre-mapped)
- **AND** increases VRAM by ~48MB
- **AND** adapter is ready for immediate use

#### Scenario: Compose multiple adapters

- **WHEN** attaching 6 experts (json, english, neo4j, python, rust, classifier)
- **THEN** the runtime applies all adapters to base model
- **AND** composition order follows load_order from manifests
- **AND** total loading time <60ms
- **AND** total VRAM <8GB (including base + KV cache)
- **AND** all adapters active simultaneously

#### Scenario: Detach adapters

- **WHEN** session ends
- **THEN** the runtime removes adapter matrices
- **AND** frees VRAM
- **AND** keeps base model loaded
- **AND** completes cleanup in <5ms

### Requirement: Paged KV Cache Management

The runtime SHALL manage KV cache using paged attention for efficiency.

#### Scenario: Allocate KV cache blocks

- **WHEN** starting new inference session
- **THEN** the runtime allocates paged blocks (16 tokens each)
- **AND** maps logical positions to physical blocks
- **AND** grows cache dynamically as tokens generate
- **AND** uses ~3.5GB for 128k context
- **AND** avoids memory fragmentation

#### Scenario: KV cache isolation

- **WHEN** running multiple sessions
- **THEN** each session has isolated KV cache
- **AND** changing experts invalidates cache
- **AND** cache cannot be shared between different expert sets
- **AND** clear warning if attempting to swap experts mid-generation

### Requirement: CUDA Backend Support

The runtime SHALL efficiently utilize NVIDIA GPUs via CUDA.

#### Scenario: CUDA inference

- **WHEN** running on NVIDIA GPU
- **THEN** the runtime uses CUDA kernels
- **AND** leverages tensor cores (if available)
- **AND** achieves 50-100 tokens/sec (RTX 4090)
- **AND** uses CUDA streams for parallelism
- **AND** handles CUDA OOM gracefully

