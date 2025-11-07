# Expert Training CLI

## ADDED Requirements

### Requirement: Train Expert from Manifest

The system SHALL train LoRA/DoRA/IA³ adapters by reading all configuration from manifest.json.

#### Scenario: Train JSON parser expert

- **WHEN** user executes `expert-cli train --manifest manifest.json --dataset datasets/json_8k.jsonl`
- **THEN** the CLI reads `training.config` from manifest.json
- **AND** loads base model specified in `base_model` section
- **AND** configures adapter (type, rank, alpha, target_modules) from manifest
- **AND** loads dataset from specified path
- **AND** trains for configured epochs with specified learning rate
- **AND** saves adapter weights to `weights/<name>.v<version>/`
- **AND** updates manifest with training metadata (date, gpu, time)

#### Scenario: Training with INT4 quantization

- **WHEN** base_model specifies `"quantization": "int4"`
- **THEN** the CLI SHALL load model with 4-bit quantization
- **AND** use bitsandbytes or equivalent
- **AND** prepare model for k-bit training
- **AND** train adapter in FP16/BF16

#### Scenario: LoRA configuration from manifest

- **WHEN** training.config specifies `"adapter_type": "lora"`
- **THEN** the CLI SHALL configure PEFT LoRA adapter
- **AND** use rank from `training.config.rank`
- **AND** use alpha from `training.config.alpha`
- **AND** target modules from `training.config.target_modules`
- **AND** apply dropout from manifest

### Requirement: Training Progress Reporting

The system SHALL provide real-time training progress feedback.

#### Scenario: Display training metrics

- **WHEN** training is in progress
- **THEN** the CLI SHALL display current epoch
- **AND** display training loss
- **AND** display evaluation loss (if eval split exists)
- **AND** estimate time remaining
- **AND** show VRAM usage

#### Scenario: Training completion

- **WHEN** training completes successfully
- **THEN** the CLI SHALL report final metrics
- **AND** display training duration
- **AND** show output directory
- **AND** suggest next command (`expert-cli validate`)

### Requirement: PyO3 Bridge to PyTorch

The system SHALL use PyO3 to bridge Rust CLI with Python ML ecosystem.

#### Scenario: Call PyTorch training from Rust

- **WHEN** CLI executes training command
- **THEN** Rust code SHALL invoke Python training function via PyO3
- **AND** pass all configuration as Python dict
- **AND** handle Python exceptions and convert to Rust Result
- **AND** stream training logs back to Rust for display

#### Scenario: Handle CUDA availability

- **WHEN** training starts
- **THEN** the CLI SHALL check if CUDA is available
- **AND** warn user if training on CPU (slow)
- **AND** allow user to confirm or cancel
- **AND** use GPU if available

