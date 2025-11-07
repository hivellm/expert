# Python Prototype CLI Specification

## ADDED Requirements

### Requirement: Load and Compose Multiple Experts

The prototype CLI SHALL load up to 10 experts and compose them for inference.

#### Scenario: Load single expert

- **WHEN** user loads json-parser expert
- **THEN** the CLI loads base model (Qwen2.5-0.5B)
- **AND** loads json-parser LoRA adapter
- **AND** applies adapter to base model
- **AND** verifies expert is ready for inference
- **AND** reports VRAM usage

#### Scenario: Load expert with dependencies

- **WHEN** user loads document-classifier
- **THEN** the CLI reads `constraints.requires` from manifest
- **AND** automatically loads json-parser, english-basic, neo4j-cypher
- **AND** loads them in correct order (sorted by load_order)
- **AND** loads document-classifier last
- **AND** verifies all 4 experts are composed
- **AND** reports total VRAM usage

### Requirement: Document Classification

The prototype SHALL classify documents using composed experts.

#### Scenario: Classify Python file

- **WHEN** user classifies a .py file
- **AND** document-classifier + dependencies are loaded
- **THEN** the CLI formats prompt with file content
- **AND** runs inference with all composed experts
- **AND** parses JSON output: {type, language, technologies}
- **AND** displays classification result
- **AND** reports inference latency

#### Scenario: Batch classification

- **WHEN** user classifies multiple files
- **THEN** the CLI processes each file
- **AND** reuses loaded experts (no reload)
- **AND** reports accuracy vs expected classifications
- **AND** measures average latency per file

### Requirement: Performance Benchmarking

The prototype SHALL measure and report performance metrics.

#### Scenario: Benchmark expert loading

- **WHEN** running benchmark
- **THEN** the CLI measures cold load time (from disk)
- **AND** measures hot load time (cached in RAM)
- **AND** measures VRAM per expert
- **AND** reports statistics in JSON format

#### Scenario: Benchmark inference

- **WHEN** running inference benchmark
- **THEN** the CLI measures prefill latency (process input)
- **AND** measures decode latency (generate tokens)
- **AND** measures tokens per second
- **AND** measures total VRAM usage
- **AND** compares against target metrics from PERFORMANCE.md

