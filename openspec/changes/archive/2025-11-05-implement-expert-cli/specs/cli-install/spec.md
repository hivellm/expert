# Expert Installation CLI

## ADDED Requirements

### Requirement: Install Expert from Git Repository

The system SHALL install experts from Git repositories with automatic dependency resolution.

#### Scenario: Install from GitHub

- **WHEN** user executes `expert-cli install https://github.com/hivellm/expert-json-parser`
- **THEN** the CLI clones repository to `~/.expert/repos/expert-json-parser`
- **AND** reads manifest.json from repository
- **AND** checks for pre-built `.expert` file in `weights/`
- **AND** verifies signature if present
- **AND** checks base model compatibility
- **AND** registers expert in `~/.expert/registry.json`
- **AND** reports installation success

#### Scenario: Install specific version

- **WHEN** user executes `expert-cli install https://github.com/user/expert-name@v1.0.0`
- **THEN** the CLI clones repository
- **AND** checks out Git tag `v1.0.0`
- **AND** proceeds with installation from that version
- **AND** records installed version in registry

#### Scenario: Install with dependencies

- **WHEN** expert manifest contains `constraints.requires: ["english-basic@>=1.0.0"]`
- **THEN** the CLI SHALL check if dependencies are installed
- **AND** install missing dependencies recursively
- **AND** verify version constraints are satisfied
- **AND** respect load_order during installation
- **AND** prevent circular dependencies

### Requirement: Dependency Resolution

The system SHALL automatically resolve and install expert dependencies.

#### Scenario: Resolve dependency chain

- **WHEN** installing document-classifier
- **AND** it requires `["json-parser@>=2.0.0", "english-basic@>=1.0.0"]`
- **THEN** the CLI SHALL install json-parser first
- **AND** install english-basic second
- **AND** install document-classifier last
- **AND** verify all version constraints
- **AND** fail if any dependency cannot be satisfied

#### Scenario: Prevent circular dependencies

- **WHEN** expert A requires expert B
- **AND** expert B requires expert A
- **THEN** the CLI SHALL detect circular dependency
- **AND** display error message with dependency chain
- **AND** exit with code 4 (compatibility error)

### Requirement: Signature Verification

The system SHALL verify Ed25519 signatures on expert packages.

#### Scenario: Verify signed expert

- **WHEN** installing expert with signature in manifest
- **THEN** the CLI SHALL extract publisher's public key
- **AND** compute SHA-256 of all files
- **AND** verify Ed25519 signature
- **AND** proceed with installation if valid
- **AND** warn or fail if signature invalid (based on --verify flag)

#### Scenario: Unsigned expert warning

- **WHEN** installing expert without signature
- **THEN** the CLI SHALL warn user
- **AND** ask for confirmation (unless --skip-verify)
- **AND** proceed if user confirms

