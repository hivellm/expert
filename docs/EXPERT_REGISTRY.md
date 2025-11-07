# Expert Registry System

The Expert Registry is a local database that tracks installed experts, base models, and their dependencies. Similar to `package.json` in npm or `Cargo.toml` in Rust.

## Registry File: `expert-registry.json`

Located at: `~/.expert/expert-registry.json` (or `%USERPROFILE%\.expert\expert-registry.json` on Windows)

### Structure

```json
{
  "version": "1.0",
  "last_updated": "2025-11-03T12:00:00Z",
  "install_dir": "~/.expert/experts",
  "models_dir": "~/.expert/models",
  
  "base_models": [
    {
      "name": "Qwen3-0.6B",
      "path": "~/.expert/models/Qwen3-0.6B",
      "sha256": "abc123...",
      "quantization": "int4",
      "size_bytes": 536870912,
      "installed_at": "2025-11-01T10:00:00Z",
      "source": "https://huggingface.co/Qwen/Qwen3-0.6B"
    }
  ],
  
  "experts": [
    {
      "name": "expert-neo4j",
      "version": "0.0.1",
      "base_model": "Qwen3-0.6B",
      "path": "~/.expert/experts/expert-neo4j",
      "source": "git+https://github.com/hivellm/expert-neo4j.git#v0.0.1",
      "installed_at": "2025-11-03T12:00:00Z",
      "adapters": [
        {
          "type": "lora",
          "path": "qwen3-06b/adapter",
          "size_bytes": 14702472,
          "sha256": "27af68..."
        }
      ],
      "capabilities": [
        "tech:neo4j",
        "query:cypher"
      ],
      "dependencies": []
    }
  ]
}
```

## Installation Commands

### Install from Git

```bash
# Install expert from Git repository
expert-cli install git+https://github.com/hivellm/expert-neo4j.git

# Install specific version/tag
expert-cli install git+https://github.com/hivellm/expert-neo4j.git#v0.0.1

# Install specific branch
expert-cli install git+https://github.com/hivellm/expert-neo4j.git#main

# Install from local path
expert-cli install file://./experts/expert-neo4j
```

### Install from .expert Package

```bash
# Install from pre-packaged .expert file
expert-cli install expert-neo4j-qwen306b.v0.0.1.expert

# Install with auto-detection of base model
expert-cli install expert-neo4j-qwen306b.v0.0.1.expert --auto-detect
```

### List Installed Experts

```bash
# List all installed experts
expert-cli list

# List with details
expert-cli list --verbose

# List only for specific base model
expert-cli list --base-model Qwen3-0.6B
```

### Uninstall Expert

```bash
# Uninstall expert
expert-cli uninstall expert-neo4j

# Uninstall and remove unused base models
expert-cli uninstall expert-neo4j --cleanup
```

## Auto-Detection of Base Models

The CLI automatically detects installed base models by scanning:

1. **Standard paths**:
   - `~/.expert/models/`
   - `./models/` (project-local)
   - Environment variable: `$EXPERT_MODELS_PATH`

2. **HuggingFace cache**:
   - `~/.cache/huggingface/hub/`

3. **Custom paths** in config:
   - `~/.expert/config.json`

### Detection Logic

```rust
fn detect_base_models() -> Vec<BaseModelInfo> {
    let mut models = Vec::new();
    
    // 1. Check registry
    if let Ok(registry) = load_registry() {
        models.extend(registry.base_models);
    }
    
    // 2. Scan standard paths
    for path in STANDARD_MODEL_PATHS {
        models.extend(scan_directory(path));
    }
    
    // 3. Check HuggingFace cache
    models.extend(scan_hf_cache());
    
    models
}
```

## Installation Workflow

### 1. Install Expert from Git

```bash
expert-cli install git+https://github.com/hivellm/expert-neo4j.git
```

**Steps:**
1. Clone Git repository to temp directory
2. Read `manifest.json` from repository
3. Check required base model
4. Auto-detect if base model is installed
5. If not found, prompt to install base model
6. Copy expert to `~/.expert/experts/expert-neo4j/`
7. Update `expert-registry.json`
8. Verify installation

### 2. Install Base Model (if needed)

```bash
expert-cli install-model Qwen3-0.6B

# Or from HuggingFace
expert-cli install-model Qwen/Qwen3-0.6B --source huggingface

# Or from local path
expert-cli install-model ./models/Qwen3-0.6B --source local
```

### 3. Verify Installation

```bash
expert-cli verify expert-neo4j
```

**Checks:**
- ✅ Expert manifest valid
- ✅ Base model present
- ✅ Adapter weights exist
- ✅ SHA256 checksums match
- ✅ Capabilities declared

## Registry Management

### Update Registry

```bash
# Rebuild registry from installed experts
expert-cli registry rebuild

# Validate registry integrity
expert-cli registry validate

# Show registry info
expert-cli registry info
```

### Export/Import

```bash
# Export registry (for backup or sharing)
expert-cli registry export > my-experts.json

# Import registry
expert-cli registry import my-experts.json
```

## Dependency Resolution

Experts can declare dependencies on other experts:

```json
{
  "name": "expert-advanced-cypher",
  "dependencies": [
    {
      "name": "expert-neo4j",
      "version": ">=0.0.1",
      "optional": false
    }
  ]
}
```

**Installation with dependencies:**
```bash
expert-cli install expert-advanced-cypher --with-dependencies
```

## Configuration File: `~/.expert/config.json`

```json
{
  "install_dir": "~/.expert/experts",
  "models_dir": "~/.expert/models",
  "cache_dir": "~/.expert/cache",
  "model_search_paths": [
    "~/.expert/models",
    "./models",
    "~/models"
  ],
  "auto_update": false,
  "verify_signatures": true,
  "allow_unsigned": false,
  "default_device": "cuda",
  "git": {
    "clone_depth": 1,
    "timeout_seconds": 300
  }
}
```

## Security

### Signature Verification

All installed experts are verified:
1. Check `signature.sig` exists
2. Verify Ed25519 signature
3. Validate SHA256 checksums
4. Check publisher trust (optional)

### Trust Management

```bash
# Add trusted publisher
expert-cli trust add --pubkey <public-key> --name "Publisher Name"

# List trusted publishers
expert-cli trust list

# Remove publisher
expert-cli trust remove <publisher-name>
```

## Example Workflows

### First-time Setup

```bash
# 1. Initialize expert system
expert-cli init

# 2. Install base model
expert-cli install-model Qwen/Qwen3-0.6B --source huggingface

# 3. Install expert
expert-cli install git+https://github.com/hivellm/expert-neo4j.git

# 4. Verify installation
expert-cli list --verbose
```

### Developer Workflow

```bash
# 1. Clone expert repository
git clone https://github.com/hivellm/expert-neo4j.git
cd expert-neo4j

# 2. Train expert
expert-cli train --manifest manifest.json --output weights --device cuda

# 3. Test locally
expert-cli install file://. --dev

# 4. Package for distribution
expert-cli package --manifest manifest.json --weights weights --model qwen3-0.6b

# 5. Sign package
expert-cli sign --expert expert-neo4j-qwen306b.v0.0.1.expert --key ~/.expert/keys/publisher.pem
```

### Team Collaboration

```bash
# Share registry with team
expert-cli registry export > team-experts.json

# Team member imports
expert-cli registry import team-experts.json --install-missing
```

## Future Enhancements

- **Registry sync**: Sync registry across machines
- **Update notifications**: Check for expert updates
- **Version constraints**: Semantic versioning support
- **Conflict resolution**: Handle version conflicts
- **Rollback**: Revert to previous versions
- **Registry mirror**: Self-hosted registry server

