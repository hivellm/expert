# Expert Structure Requirements

## ADDED Requirements

### Requirement: Standard Expert Directory Structure

Each expert repository SHALL follow the standardized directory structure without custom scripts.

#### Scenario: Create new expert

- **WHEN** creating a new expert (e.g., english-basic)
- **THEN** the directory SHALL contain manifest.json
- **AND** include tests/test_cases.json
- **AND** include README.md with usage examples
- **AND** include LICENSE file
- **AND** include .gitignore and .gitattributes
- **AND** NOT include scripts/ directory
- **AND** have datasets/ directory for generated data
- **AND** have weights/ directory for training output

### Requirement: Expert Dependencies

Experts SHALL declare dependencies and load order in manifest.json.

#### Scenario: Classifier depends on multiple experts

- **WHEN** creating document-classifier expert
- **THEN** manifest SHALL include `constraints.requires`:
  - `["english-basic@>=0.0.1", "json-parser@>=0.0.1", "neo4j-cypher@>=0.0.1"]`
- **AND** set `constraints.load_order: 9` (higher than dependencies)
- **AND** dependency resolution automatically loads required experts

### Requirement: Training Configuration in Manifest

All training parameters SHALL be declarative in manifest.json.

#### Scenario: Configure LoRA training

- **WHEN** expert uses LoRA adapter
- **THEN** manifest SHALL specify in `training.config`:
  - `adapter_type: "lora"`
  - `rank: 16`
  - `alpha: 16`
  - `target_modules: ["q_proj", "v_proj", "o_proj"]`
  - `epochs: 3`
  - `learning_rate: 0.0003`
  - `batch_size: 4`
- **AND** no external configuration files needed

