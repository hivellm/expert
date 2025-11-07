# Train First Six Experts

## Why

Validate the Expert System architecture with real, working experts before building the full Rust runtime. These 6 experts cover the essential capabilities for document classification (the primary use case from the classify project).

## What Changes

- Create and train 6 fundamental experts:
  1. json-parser (format, load_order: 1)
  2. english-basic (language, load_order: 3)
  3. neo4j-cypher (technology, load_order: 6)
  4. python-code (technology, load_order: 6)
  5. rust-code (technology, load_order: 6)
  6. document-classifier (task, load_order: 9, requires others)

- Generate synthetic datasets for each (using Cursor Agent)
- Train LoRA adapters (r=16)
- Validate accuracy metrics
- Test multi-expert composition

**Non-breaking**: First real experts in the system

## Impact

- **Affected specs**: None (new functionality)
- **Affected code**: New experts in `/expert/experts/`
- **Dependencies**: Requires expert-cli to be functional
- **Timeline**: Week 1-3 of Quick Start path
- **Cost**: ~$15-25 for synthetic data generation

