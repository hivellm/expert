# Expert SDK Specification

**Capability**: `expert-sdk`  
**Version**: 1.0.0  
**Status**: Draft

---

## ADDED Requirements

### Requirement: Package Loading

The SDK SHALL load `.expert` package files and extract their contents for inference.

#### Scenario: Load valid expert package

- **GIVEN** a valid `.expert` package file (tar.gz format)
- **WHEN** `Expert.load(path)` is called
- **THEN** the package is extracted to a temporary directory
- **AND** manifest.json is parsed and validated
- **AND** adapter weights are located
- **AND** tokenizer files are loaded
- **AND** an Expert instance is returned

#### Scenario: Load corrupted package

- **GIVEN** a corrupted `.expert` file
- **WHEN** `Expert.load(path)` is called
- **THEN** a LoadError is raised
- **AND** error message indicates corruption
- **AND** temporary files are cleaned up

#### Scenario: Load package with missing manifest

- **GIVEN** a `.expert` file without manifest.json
- **WHEN** `Expert.load(path)` is called
- **THEN** a LoadError is raised
- **AND** error message indicates missing manifest

#### Scenario: Verify package checksum

- **GIVEN** a `.expert` file with SHA256 checksum
- **WHEN** package is loaded
- **THEN** checksum is verified against package contents
- **AND** load fails if checksum mismatch

---

### Requirement: Multiple Expert Management

The SDK SHALL support loading and managing multiple experts simultaneously with efficient memory usage.

#### Scenario: Load multiple experts

- **GIVEN** an ExpertManager instance
- **WHEN** `manager.load_expert("sql", "sql.expert")` is called
- **AND** `manager.load_expert("cypher", "cypher.expert")` is called
- **THEN** both experts are loaded and accessible
- **AND** each expert can be used independently

#### Scenario: Base model sharing

- **GIVEN** two experts with same base model (e.g., Qwen3-0.6B)
- **WHEN** both experts are loaded via ExpertManager
- **THEN** base model is loaded only once
- **AND** only adapters are swapped between generations
- **AND** memory usage is < 2x first expert size

#### Scenario: Generate with different experts

- **GIVEN** ExpertManager with SQL and Cypher experts loaded
- **WHEN** `manager.generate("sql", "SELECT * FROM users")` is called
- **AND** `manager.generate("cypher", "MATCH (u:User) RETURN u")` is called
- **THEN** each query uses the correct expert
- **AND** results are generated with appropriate grammar/decoding params

#### Scenario: List loaded experts

- **GIVEN** ExpertManager with 3 loaded experts
- **WHEN** `manager.list_loaded_experts()` is called
- **THEN** returns list of expert names
- **AND** includes metadata (base model, memory usage, last used)

#### Scenario: Unload expert

- **GIVEN** ExpertManager with loaded expert
- **WHEN** `manager.unload_expert("sql")` is called
- **THEN** expert adapter is removed from memory
- **AND** base model remains if other experts use it
- **AND** subsequent generate() calls fail with clear error

#### Scenario: Adapter hot-swap

- **GIVEN** ExpertManager with base model loaded
- **WHEN** switching between experts with same base model
- **THEN** only adapter is swapped (< 100ms)
- **AND** base model stays in VRAM/memory
- **AND** no model reload occurs

#### Scenario: LRU eviction (Rust only)

- **GIVEN** ExpertManager with max_loaded_experts=3
- **WHEN** 4th expert is loaded
- **THEN** least recently used expert is unloaded
- **AND** base model is kept if other experts need it
- **AND** evicted expert can be automatically reloaded on next use

---

### Requirement: Simple Inference API

The SDK SHALL provide a simple API for generating text from prompts.

#### Scenario: Generate text from single prompt

- **GIVEN** a loaded Expert instance
- **WHEN** `expert.generate(prompt)` is called
- **THEN** ChatML template is automatically applied
- **AND** model generates response using manifest decoding params
- **AND** result is returned as InferenceResult object
- **AND** result contains generated text, token count, and metadata

#### Scenario: Generate with custom parameters

- **GIVEN** a loaded Expert instance
- **WHEN** `expert.generate(prompt, temperature=0.5, max_tokens=100)` is called
- **THEN** custom parameters override manifest defaults
- **AND** generation uses specified temperature and max_tokens

#### Scenario: Generate with schema context

- **GIVEN** an Expert instance for text2sql task
- **WHEN** `expert.generate(question, schema=db_schema)` is called
- **THEN** schema is included in system message
- **AND** ChatML template includes schema context

---

### Requirement: Streaming Inference

The SDK SHALL support streaming token-by-token generation.

#### Scenario: Stream tokens from generation

- **GIVEN** a loaded Expert instance
- **WHEN** `expert.generate_stream(prompt)` is called
- **THEN** an iterator/generator is returned
- **AND** each iteration yields a token
- **AND** tokens can be processed in real-time
- **AND** full result is available after stream completion

#### Scenario: Cancel streaming generation

- **GIVEN** an active streaming generation
- **WHEN** iterator is stopped early
- **THEN** generation is cancelled
- **AND** resources are cleaned up
- **AND** partial result is available

---

### Requirement: Batch Inference

The SDK SHALL support efficient batch inference for multiple prompts.

#### Scenario: Generate for multiple prompts

- **GIVEN** a loaded Expert instance
- **WHEN** `expert.generate_batch([prompt1, prompt2, prompt3])` is called
- **THEN** all prompts are batched together
- **AND** inference runs efficiently with padding
- **AND** results are returned in same order as inputs
- **AND** each result has corresponding metadata

#### Scenario: Batch with different lengths

- **GIVEN** prompts of varying lengths
- **WHEN** batch inference is performed
- **THEN** padding is applied automatically
- **AND** attention masks are correct
- **AND** results are not affected by padding

---

### Requirement: Device Management

The SDK SHALL automatically select appropriate compute device (CUDA, CPU, MPS).

#### Scenario: Auto-detect CUDA

- **GIVEN** a system with CUDA available
- **WHEN** Expert is loaded without device specification
- **THEN** model is loaded on CUDA device
- **AND** device info is logged

#### Scenario: Fallback to CPU

- **GIVEN** a system without CUDA
- **WHEN** Expert is loaded
- **THEN** model is loaded on CPU
- **AND** warning is logged about CPU inference

#### Scenario: Explicit device selection

- **GIVEN** a system with multiple GPUs
- **WHEN** `Expert.load(path, device="cuda:1")` is called
- **THEN** model is loaded on specified GPU
- **AND** all tensors are moved to that device

---

### Requirement: Grammar Validation

The SDK SHALL support constrained generation with grammar validation (when enabled in manifest).

#### Scenario: Load expert with grammar

- **GIVEN** an expert package with grammar.gbnf file
- **WHEN** Expert is loaded
- **THEN** grammar file is parsed
- **AND** grammar validator is initialized
- **AND** generation respects grammar constraints

#### Scenario: Generate with grammar enforcement

- **GIVEN** an expert with SQL grammar
- **WHEN** generation occurs
- **THEN** only valid SQL tokens are generated
- **AND** result is guaranteed to be valid SQL

#### Scenario: Expert without grammar

- **GIVEN** an expert without grammar.gbnf
- **WHEN** Expert is loaded
- **THEN** grammar validation is skipped
- **AND** generation proceeds without constraints

---

### Requirement: Error Handling

The SDK SHALL provide clear, actionable error messages for all failure modes.

#### Scenario: Missing dependencies

- **GIVEN** required dependency (e.g., torch) is not installed
- **WHEN** SDK is imported
- **THEN** ImportError is raised
- **AND** error message lists missing dependencies
- **AND** installation instructions are provided

#### Scenario: Invalid manifest

- **GIVEN** a package with malformed manifest.json
- **WHEN** Expert.load() is called
- **THEN** ManifestError is raised
- **AND** error message shows JSON validation errors
- **AND** line number of error is indicated

#### Scenario: Out of memory

- **GIVEN** insufficient VRAM for model
- **WHEN** generation is attempted
- **THEN** OutOfMemoryError is raised
- **AND** error message suggests reducing batch size or sequence length

---

### Requirement: Cross-Platform Support

The SDK SHALL work on Windows, Linux, and macOS.

#### Scenario: Load expert on Windows

- **GIVEN** Windows system with CUDA
- **WHEN** Expert is loaded and used
- **THEN** all features work correctly
- **AND** paths are handled with Windows separators

#### Scenario: Load expert on Linux

- **GIVEN** Linux system with CUDA
- **WHEN** Expert is loaded and used
- **THEN** all features work correctly

#### Scenario: Load expert on macOS with MPS

- **GIVEN** macOS system with Apple Silicon
- **WHEN** Expert is loaded
- **THEN** MPS backend is used automatically
- **AND** generation works correctly

---

### Requirement: Manifest-Driven Configuration

The SDK SHALL use manifest.json to configure all inference parameters.

#### Scenario: Load decoding params from manifest

- **GIVEN** manifest with decoding config (temperature, top_p, top_k)
- **WHEN** Expert is loaded
- **THEN** default generation uses manifest params
- **AND** params can be overridden per-call

#### Scenario: Load adapter config from manifest

- **GIVEN** manifest with adapter path
- **WHEN** Expert is loaded
- **THEN** correct adapter is loaded (checkpoint-specific or final)
- **AND** adapter type (LoRA/DoRA) is respected

#### Scenario: Load stop sequences from manifest

- **GIVEN** manifest with stop_sequences defined
- **WHEN** generation occurs
- **THEN** generation stops at specified sequences
- **AND** stop sequences are removed from output

---

### Requirement: Type Safety (Rust Only)

The Rust SDK SHALL provide compile-time type safety for all public APIs.

#### Scenario: Typed inference result

- **GIVEN** Rust SDK usage
- **WHEN** `expert.generate()` returns result
- **THEN** result type is `Result<InferenceResult, ExpertError>`
- **AND** compiler enforces error handling

#### Scenario: Builder pattern for configuration

- **GIVEN** Rust SDK API
- **WHEN** configuring inference
- **THEN** builder pattern is used
- **AND** invalid configurations fail at compile time

---

### Requirement: Performance Parity

The SDK SHALL perform within 5% of manual model loading approach.

#### Scenario: Single prompt latency

- **GIVEN** same model loaded via SDK vs manually
- **WHEN** single prompt inference is measured
- **THEN** SDK latency is ≤ 105% of manual approach

#### Scenario: Batch throughput

- **GIVEN** batch of 10 prompts
- **WHEN** throughput is measured
- **THEN** SDK throughput is ≥ 95% of manual approach

#### Scenario: Memory overhead

- **GIVEN** model loaded via SDK
- **WHEN** memory usage is measured
- **THEN** overhead is < 100MB additional

---

### Requirement: Documentation

The SDK SHALL include comprehensive documentation for all public APIs.

#### Scenario: API reference

- **WHEN** developer views SDK documentation
- **THEN** all public classes, methods, and functions are documented
- **AND** parameters and return types are explained
- **AND** examples are provided

#### Scenario: Migration guide

- **WHEN** developer migrates from manual approach
- **THEN** migration guide shows equivalent SDK code
- **AND** common patterns are covered
- **AND** troubleshooting tips are provided

#### Scenario: Example applications

- **WHEN** developer explores SDK
- **THEN** at least 3 example applications are provided:
  - Simple CLI tool
  - REST API server
  - Batch processing script
- **AND** examples are runnable without modification

---

## Notes

**Implementation Priority**:
1. Package Loading (critical path)
2. Simple Inference API (core value)
3. Error Handling (developer experience)
4. Streaming Inference (common use case)
5. Batch Inference (performance)
6. Grammar Validation (advanced feature)

**Testing Strategy**:
- Unit tests for each requirement
- Integration tests with real expert packages
- Performance benchmarks
- Cross-platform CI testing

**Breaking Change Policy**:
- Follow semantic versioning strictly
- Maintain backward compatibility for at least 2 minor versions
- Provide deprecation warnings 6 months before removal

