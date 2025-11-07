# Dataset Generation CLI

## ADDED Requirements

### Requirement: Generate Dataset from Manifest

The system SHALL generate synthetic training datasets by reading configuration from manifest.json and calling LLM provider APIs.

#### Scenario: Generate JSON parser dataset

- **WHEN** user executes `expert-cli dataset generate --manifest manifest.json`
- **THEN** the CLI reads `training.dataset.generation` from manifest.json
- **AND** connects to the specified provider (cursor, deepseek, claude, gpt-4o)
- **AND** generates the specified count of examples (e.g., 8000)
- **AND** applies diversity filtering with the specified threshold
- **AND** saves output to the path specified in `training.dataset.path`
- **AND** reports generation statistics (count, size, diversity score)

#### Scenario: Missing API key

- **WHEN** user executes dataset generation
- **AND** required API key environment variable is not set (e.g., DEEPSEEK_API_KEY)
- **THEN** the CLI SHALL display clear error message
- **AND** suggest setting the environment variable
- **AND** exit with code 1

#### Scenario: Batch generation with progress

- **WHEN** generating large datasets (>1000 examples)
- **THEN** the CLI SHALL generate in batches (e.g., 100 per API call)
- **AND** display progress bar showing completion percentage
- **AND** handle API rate limits gracefully (retry with backoff)
- **AND** save partial results if interrupted

### Requirement: Quality Filtering

The system SHALL apply quality filters to ensure dataset diversity and validity.

#### Scenario: Diversity enforcement

- **WHEN** generating dataset with diversity_threshold specified
- **THEN** the CLI SHALL compute embeddings for each example
- **AND** remove near-duplicates using cosine similarity
- **AND** ensure final dataset meets diversity threshold
- **AND** report how many examples were filtered out

#### Scenario: Schema validation

- **WHEN** generating instruction-following dataset
- **THEN** each example MUST have required fields (instruction, input, output)
- **AND** examples missing fields SHALL be rejected
- **AND** the CLI reports validation statistics

### Requirement: Provider Support

The system SHALL support multiple LLM providers for dataset generation.

#### Scenario: Cursor Agent provider

- **WHEN** manifest specifies `"provider": "cursor"`
- **THEN** the CLI SHALL use Cursor Agent API
- **AND** respect temperature setting from manifest
- **AND** format prompts according to Cursor's requirements

#### Scenario: DeepSeek provider

- **WHEN** manifest specifies `"provider": "deepseek"`
- **THEN** the CLI SHALL use DeepSeek Chat API
- **AND** require DEEPSEEK_API_KEY environment variable
- **AND** use cost-effective API endpoints

#### Scenario: Claude provider

- **WHEN** manifest specifies `"provider": "claude"`
- **THEN** the CLI SHALL use Anthropic Claude API
- **AND** require ANTHROPIC_API_KEY environment variable
- **AND** leverage Claude's structured output capabilities

