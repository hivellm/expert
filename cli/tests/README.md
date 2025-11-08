# Rust Tests

This directory contains Rust unit and integration tests for the expert-cli.

## Structure

- `*.rs` - Rust test files
- `fixtures/` - Test fixtures (JSON manifests, etc.)
- `run_all_tests.ps1` - PowerShell script to run all tests

## Running Tests

### Run all tests
```bash
cargo test
```

### Run with output
```bash
cargo test -- --nocapture
```

### Run specific test file
```bash
cargo test --test manifest_tests
```

### Run PowerShell script (Windows)
```powershell
.\tests\run_all_tests.ps1
```

## Test Categories

- **Unit Tests**: Test individual functions and modules
- **Integration Tests**: Test command-line interface and workflows
- **Validation Tests**: Test manifest validation and error handling

## Note

Python tests are located in `tests_python/` directory.

