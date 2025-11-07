# Expert Packaging CLI

## ADDED Requirements

### Requirement: Package Expert as .expert File

The system SHALL bundle expert weights and metadata into distributable .expert packages.

#### Scenario: Package trained expert

- **WHEN** user executes `expert-cli package --manifest manifest.json --weights weights/json-parser.v0.0.1`
- **THEN** the CLI reads manifest.json
- **AND** collects adapter_model.safetensors from weights directory
- **AND** collects soft prompts (if any)
- **AND** includes manifest.json
- **AND** includes LICENSE file (if present)
- **AND** computes SHA-256 hash of each file
- **AND** creates tar.gz archive
- **AND** saves as `weights/<name>.v<version>.expert`
- **AND** reports package size and contents

#### Scenario: Package with soft prompts

- **WHEN** expert includes soft prompts
- **AND** soft_prompts directory exists
- **THEN** the CLI SHALL include all .pt files
- **AND** update manifest with file hashes
- **AND** verify soft prompt tensor shapes

#### Scenario: Compression options

- **WHEN** user specifies `--compression zstd`
- **THEN** the CLI SHALL use zstd compression
- **AND** create .expert file with better compression
- **AND** note compression type in package metadata

### Requirement: Package Integrity

The system SHALL ensure package integrity and completeness.

#### Scenario: Validate before packaging

- **WHEN** packaging expert
- **THEN** the CLI SHALL verify all referenced files exist
- **AND** check safetensors file is valid
- **AND** ensure manifest.json is valid JSON
- **AND** verify no required fields are missing
- **AND** fail with clear error if validation fails

#### Scenario: Compute file hashes

- **WHEN** creating package
- **THEN** the CLI SHALL compute SHA-256 of manifest.json
- **AND** compute SHA-256 of weights.safetensors
- **AND** compute SHA-256 of each soft prompt file
- **AND** store all hashes in manifest `integrity.files` section
- **AND** use hashes for signature verification

