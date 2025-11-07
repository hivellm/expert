# Implement Expert CLI

## Why

The Expert System architecture is fully documented, but we need the core CLI tool to make it functional. Users need a standardized way to generate datasets, train experts, package them, and install from Git repositories. Without the CLI, each expert would need custom scripts, defeating the purpose of standardization.

## What Changes

- Create `expert-cli` Rust binary with core commands
- Implement dataset generation (via Cursor Agent/DeepSeek/Claude API)
- Implement LoRA training (via PyO3 Python bindings)
- Implement packaging (tar.gz with manifest.json)
- Implement Git-based installation (clone + verify)
- Implement expert validation
- Add signing with Ed25519

**BREAKING**: First implementation - no existing code to break

## Impact

- **Affected specs**: cli-dataset, cli-train, cli-package, cli-install, cli-validate, cli-sign
- **Affected code**: New project at `/expert/cli/` (Rust)
- **Users**: Enables expert creators to build and share experts
- **Timeline**: P0 milestone (4-6 weeks)
- **Dependencies**: Requires Rust nightly, PyTorch (via PyO3), Git

