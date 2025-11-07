# Build Python Prototype CLI

## Why

Before investing in full Rust runtime, validate the expert composition architecture with a simple Python CLI. This lets us test multi-expert loading, dependency resolution, and real inference to gather performance metrics and identify issues.

## What Changes

- Create simple Python CLI at `/expert/cli-prototype/`
- Load base model (Qwen2.5-0.5B as proxy)
- Load multiple experts using PEFT
- Implement dependency resolution
- Test document classification with all 6 experts
- Benchmark performance (latency, VRAM, accuracy)

**Non-breaking**: Prototype, doesn't affect production plans

## Impact

- **Affected specs**: prototype-cli
- **Affected code**: New directory `/expert/cli-prototype/` (Python)
- **Purpose**: Validate architecture before Rust investment
- **Timeline**: Week 3 of Quick Start (after experts trained)
- **Findings**: Will inform Rust runtime design (P0)

