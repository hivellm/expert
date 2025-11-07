# Expert Signing CLI

## ADDED Requirements

### Requirement: Sign Expert Package

The system SHALL cryptographically sign .expert packages using Ed25519.

#### Scenario: Sign with private key

- **WHEN** user executes `expert-cli sign --expert weights/expert.v1.0.0.expert --key ~/.expert/keys/publisher.pem`
- **THEN** the CLI loads the .expert package
- **AND** extracts all files
- **AND** computes SHA-256 hash of each file
- **AND** creates canonical message from sorted file hashes
- **AND** loads Ed25519 private key
- **AND** signs the message
- **AND** updates manifest with signature and public key
- **AND** re-packages .expert file with updated manifest
- **AND** reports signing success

#### Scenario: Generate signing keypair

- **WHEN** user executes `expert-cli keygen --output ~/.expert/keys/publisher.pem`
- **THEN** the CLI generates Ed25519 keypair
- **AND** saves private key to specified path
- **AND** saves public key to `publisher.pub` in same directory
- **AND** sets appropriate file permissions (600 for private)
- **AND** displays public key for distribution

### Requirement: Signature Verification

The system SHALL verify signatures when installing experts.

#### Scenario: Verify valid signature

- **WHEN** installing expert with signature
- **THEN** the CLI extracts publisher public key from manifest
- **AND** computes SHA-256 of all files
- **AND** verifies Ed25519 signature
- **AND** proceeds with installation if valid
- **AND** displays "✓ Signature valid" message

#### Scenario: Reject invalid signature

- **WHEN** signature verification fails
- **AND** user specified `--verify` flag
- **THEN** the CLI SHALL display error message
- **AND** show which file hash mismatched
- **AND** exit with code 5 (signature verification failed)
- **AND** NOT install the expert

