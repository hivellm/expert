# Expert Validation CLI

## ADDED Requirements

### Requirement: Validate Trained Expert

The system SHALL validate expert quality through test case execution and metric computation.

#### Scenario: Run test cases

- **WHEN** user executes `expert-cli validate --expert weights/json-parser.v0.0.1`
- **THEN** the CLI loads expert adapter weights
- **AND** loads base model
- **AND** reads tests/test_cases.json
- **AND** runs inference on each test case
- **AND** compares output with expected result
- **AND** computes accuracy metrics
- **AND** displays pass/fail summary
- **AND** exits with code 0 if all pass, code 11 if any fail

#### Scenario: Compute metrics

- **WHEN** validation completes
- **THEN** the CLI SHALL compute task-specific metrics
- **AND** for parsing tasks: exact match accuracy
- **AND** for validation tasks: true positive/negative rates
- **AND** for extraction tasks: field match accuracy
- **AND** update manifest `evaluation.metrics` section

### Requirement: Manifest Validation

The system SHALL validate manifest.json schema and completeness.

#### Scenario: Validate manifest structure

- **WHEN** validating expert
- **THEN** the CLI SHALL check all required fields exist
- **AND** verify version is valid semver
- **AND** verify load_order is set
- **AND** verify target_modules are valid for base model
- **AND** verify rank and alpha are positive integers
- **AND** report any schema violations

#### Scenario: Compatibility check

- **WHEN** validating expert
- **THEN** the CLI SHALL verify base_model hash matches installed base
- **AND** verify rope_scaling config matches
- **AND** verify quantization is compatible
- **AND** warn if mismatches detected

