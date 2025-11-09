# Rust Tests

This directory contains Rust unit and integration tests for the expert-cli.

## Structure

Tests are organized into grouped files by category:

- **`commands_tests.rs`** - All CLI command tests (chat, dataset, install, list, package, sign, train, update)
- **`inference_tests.rs`** - All model inference tests (qwen, generation, hot_swap, lora)
- **`core_tests.rs`** - All core functionality tests (manifest, registry, router, model detection)
- **`integration_tests.rs`** - All integration tests (end-to-end, multi-expert, dependency resolution, error messages, package integration, validation)
- **`benchmarks_tests.rs`** - All performance benchmark tests (latency, VRAM profiling)

Each file is organized into modules for better organization while keeping all tests in a single file (as required by Cargo).

## Running Tests

Run all tests:
```bash
cargo test
```

Run tests for a specific category:
```bash
cargo test --test commands_tests
cargo test --test inference_tests
cargo test --test core_tests
cargo test --test integration_tests
cargo test --test benchmarks_tests
```

Run a specific test:
```bash
cargo test --test commands_tests test_chat_oneshot_mode
```

Run tests with output:
```bash
cargo test -- --nocapture
```

## Test Organization

### Commands Tests (`commands_tests.rs`)
- `chat` - Chat command tests
- `dataset` - Dataset command tests
- `install` - Install command tests
- `list` - List command tests
- `package` - Package command tests
- `sign` - Sign command tests
- `train` - Train command tests
- `update` - Update command tests

### Inference Tests (`inference_tests.rs`)
- `qwen` - Qwen engine tests
- `generation` - Text generation and sampling tests
- `hot_swap` - Hot swap functionality tests
- `lora` - LoRA adapter tests

### Core Tests (`core_tests.rs`)
- `manifest` - Manifest parsing and validation tests
- `manifest_features` - Advanced manifest features tests
- `model_detection` - Model detection and discovery tests
- `registry` - Expert registry tests
- `router` - Router tests
- `keyword_routing` - Keyword-based routing tests
- `router_comprehensive` - Comprehensive router tests

### Integration Tests (`integration_tests.rs`)
- `end_to_end` - End-to-end integration tests
- `multi_expert` - Multi-expert scenarios
- `dependency_resolution` - Dependency resolution tests
- `error_messages` - Error message validation tests
- `data_integrity` - Data integrity tests
- `package_integration` - Package command integration tests
- `validation` - Validation integration tests

### Benchmarks Tests (`benchmarks_tests.rs`)
- `latency` - Latency benchmark tests
- `vram` - VRAM profiling tests

## Notes

- All test files must be in the `tests/` root directory (Cargo requirement)
- Tests are grouped by category in single files with module organization
- Each module contains related tests for better maintainability
- Integration tests may require actual expert manifests or fixtures
