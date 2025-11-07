# Git-Based Distribution

> Distribute experts via Git repositories instead of centralized registries (NPM, PyPI)

## Overview

Instead of depending on a centralized package manager, the Expert System uses **Git repositories** as the distribution mechanism. Each expert lives in its own repository containing:

- Training datasets (or generation scripts)
- Pre-trained `.expert` files (optional)
- Manifest and metadata
- Training scripts
- Documentation

**Benefits**:
- ✅ Decentralized (no single point of failure)
- ✅ Version control via Git tags
- ✅ Easy forking and contribution
- ✅ Works with GitHub, GitLab, self-hosted Git
- ✅ No registration/approval process
- ✅ Users control their own experts

---

## Repository Structure

### Standard Expert Repository

```
expert-json-parser/                    # Git repository
├── README.md                          # Expert documentation
├── manifest.json                      # Expert metadata + training config
├── LICENSE                            # License (MIT, Apache, etc.)
├── .gitignore                         # Git ignore
├── .gitattributes                    # Git LFS for large files
│
├── datasets/                          # Training data
│   └── json_8k.jsonl                  # Generated dataset (optional)
│
├── weights/
│   ├── json-parser.v2.0.0/            # Raw training output
│   └── json-parser.v2.0.0.expert      # Packaged .expert (optional)
│
└── tests/
    ├── test_json_parsing.py
    └── test_cases.json
```

**Key points:**
- **No `scripts/` directory!** All operations via `expert-cli`
- All configuration in `manifest.json`
- Datasets can be pre-generated or generated on install
- Weights optional (train locally if missing)

### Minimal Expert Repository (Recipe Only)

For contributors who only provide the training recipe:

```
expert-python-code/
├── README.md
├── manifest.json          # Contains training.dataset.generation config
├── LICENSE
└── tests/
    └── test_cases.json
```

Users clone and run `expert-cli dataset generate` + `expert-cli train` locally.

---

## Installation Flow

### CLI Usage

```bash
# Install from GitHub
expert-cli install https://github.com/hivellm/expert-json-parser

# Install from GitLab
expert-cli install https://gitlab.com/user/expert-neo4j

# Install from self-hosted Git
expert-cli install https://git.mycompany.com/experts/domain-classifier

# Install specific version (Git tag)
expert-cli install https://github.com/hivellm/expert-json-parser@v2.1.0

# Install from local path (development)
expert-cli install /path/to/expert-json-parser
```

### Installation Process

1. **Clone repository**:
   ```bash
   git clone https://github.com/hivellm/expert-json-parser ~/.expert/repos/expert-json-parser
   ```

2. **Checkout version** (if specified):
   ```bash
   cd ~/.expert/repos/expert-json-parser
   git checkout v2.1.0
   ```

3. **Check for pre-trained weights**:
   ```bash
   if [ -f weights/*.expert ]; then
     # Use pre-trained
     cp weights/*.expert ~/.expert/installed/
   else
     # Train locally
     ./scripts/train.sh
   fi
   ```

4. **Verify signature** (if present):
   ```bash
   expert-cli verify ~/.expert/installed/json-parser.v2.0.0.expert
   ```

5. **Register in local index**:
   ```bash
   # Add to ~/.expert/registry.json
   {
     "name": "json-parser",
     "version": "2.0.0",
     "source": "https://github.com/hivellm/expert-json-parser",
     "installed_at": "2025-11-02T10:30:00Z"
   }
   ```

---

## Manifest.json in Git Repo

**Location**: Root of repository

```json
{
  "name": "json-parser",
  "version": "2.0.0",
  "description": "JSON parsing and validation specialist",
  "author": "hivellm",
  "homepage": "https://github.com/hivellm/expert-json-parser",
  
  "repository": {
    "type": "git",
    "url": "https://github.com/hivellm/expert-json-parser.git"
  },
  
  "base_model": {
    "name": "Qwen3-0.6B",
    "sha256": "abc123...",
    "quantization": "int4",
    "rope_scaling": "yarn-128k"
  },
  
  "adapters": [{
    "type": "lora",
    "target_modules": ["q_proj", "v_proj", "o_proj"],
    "r": 16,
    "alpha": 16,
    "path": "weights/json-parser.v2.0.0.expert"
  }],
  
  "capabilities": ["format:json", "parsing", "validation"],
  
  "constraints": {
    "load_order": 1,
    "requires": []
  },
  
  "training": {
    "dataset": "datasets/json_8k.jsonl",
    "dataset_generation": "datasets/generate.py",
    "method": "sft",
    "epochs": 3,
    "script": "scripts/train.sh"
  },
  
  "integrity": {
    "created_at": "2025-10-30T12:00:00Z",
    "publisher": "hivellm",
    "pubkey": "ed25519:...",
    "signature": "..."
  },
  
  "license": "Apache-2.0"
}
```

---

## Git LFS for Large Files

Pre-trained `.expert` files can be large (10-80 MB). Use Git LFS:

### Setup Git LFS

```bash
cd expert-json-parser/

# Initialize Git LFS
git lfs install

# Track .expert files
git lfs track "weights/*.expert"
git lfs track "datasets/*.jsonl"

# Commit .gitattributes
git add .gitattributes
git commit -m "Add Git LFS tracking"
```

### .gitattributes

```
weights/*.expert filter=lfs diff=lfs merge=lfs -text
datasets/*.jsonl filter=lfs diff=lfs merge=lfs -text
```

---

## Versioning with Git Tags

### Semantic Versioning

```bash
# Tag a release
git tag -a v2.0.0 -m "Release 2.0.0: Improved JSON validation"
git push origin v2.0.0

# Users can install specific version
expert-cli install https://github.com/hivellm/expert-json-parser@v2.0.0
```

### Version Discovery

```bash
# List available versions
expert-cli versions https://github.com/hivellm/expert-json-parser

# Output:
# v2.1.0 (latest)
# v2.0.0
# v1.5.2
# v1.0.0
```

---

## Creating Your Own Expert Repository

### Step 1: Create Repository

```bash
# On GitHub/GitLab, create new repository: expert-myexpert

# Clone locally
git clone https://github.com/yourusername/expert-myexpert
cd expert-myexpert
```

### Step 2: Add Files

```bash
# Create structure
mkdir -p datasets scripts weights tests

# Copy template manifest
cp ~/.expert/templates/manifest.json .

# Edit manifest.json with your expert details
```

### Step 3: Generate Dataset

```bash
# Write generation script
cat > datasets/generate.py << 'EOF'
import sys
sys.path.append('../scripts')
from expert_gen import generate_dataset

generate_dataset(
    domain="my-domain",
    task="my task description",
    count=8000,
    provider="deepseek",
    output="datasets/myexpert_8k.jsonl"
)
EOF

# Run it
python datasets/generate.py
```

### Step 4: Train Expert

```bash
# Write training script
cat > scripts/train.sh << 'EOF'
#!/bin/bash
python train_expert.py \
  --name myexpert \
  --dataset datasets/myexpert_8k.jsonl \
  --method lora \
  --r 16 \
  --epochs 3 \
  --output weights/myexpert.v1.0.0.expert
EOF

chmod +x scripts/train.sh

# Train
./scripts/train.sh
```

### Step 5: Sign (Optional)

```bash
# Generate key (once)
expert-cli keygen --output ~/.expert/keys/mykey.pem

# Sign expert
expert-cli sign \
  --expert weights/myexpert.v1.0.0.expert \
  --key ~/.expert/keys/mykey.pem

# Updates manifest.json with signature
```

### Step 6: Commit & Push

```bash
# Setup Git LFS (if including pre-trained weights)
git lfs install
git lfs track "weights/*.expert"

# Commit
git add .
git commit -m "Initial release: myexpert v1.0.0"

# Tag version
git tag -a v1.0.0 -m "Release v1.0.0"

# Push
git push origin main
git push origin v1.0.0
```

### Step 7: Share

```bash
# Others can now install:
expert-cli install https://github.com/yourusername/expert-myexpert
```

---

## Marketplace as Git Index

Instead of centralized registry, marketplace is a **Git repository of links**:

### Marketplace Repository Structure

```
expert-marketplace/
├── README.md                          # Marketplace docs
├── index.json                         # All experts index
├── experts/
│   ├── json-parser.json               # Expert entry
│   ├── neo4j-cypher.json
│   ├── python-code.json
│   └── ...
└── categories/
    ├── languages.json
    ├── formats.json
    └── technologies.json
```

### Expert Entry (experts/json-parser.json)

```json
{
  "name": "json-parser",
  "repository": "https://github.com/hivellm/expert-json-parser",
  "description": "JSON parsing and validation specialist",
  "author": "hivellm",
  "latest_version": "2.1.0",
  "downloads": 1523,
  "rating": 4.8,
  "verified": true,
  "tags": ["json", "parsing", "validation", "format"],
  "added_at": "2025-09-15T00:00:00Z",
  "updated_at": "2025-11-01T00:00:00Z"
}
```

### Marketplace Index (index.json)

```json
{
  "version": "1.0",
  "updated_at": "2025-11-02T12:00:00Z",
  "experts_count": 127,
  "experts": [
    {
      "name": "json-parser",
      "repository": "https://github.com/hivellm/expert-json-parser",
      "latest": "2.1.0"
    },
    // ... more experts
  ]
}
```

### Using Marketplace

```bash
# Update marketplace index
expert-cli marketplace update

# Search
expert-cli search "json parsing"
# Output:
# json-parser (v2.1.0) - JSON parsing specialist
#   Repository: https://github.com/hivellm/expert-json-parser
#   Rating: ⭐ 4.8 | Downloads: 1.5k
#   Install: expert-cli install https://github.com/hivellm/expert-json-parser

# Browse by category
expert-cli marketplace browse --category formats

# Submit your expert to marketplace (creates PR)
expert-cli marketplace submit https://github.com/yourusername/expert-myexpert
```

---

## Benefits Over Centralized Registry

| Aspect | Git-Based | NPM/PyPI |
|--------|-----------|----------|
| **Setup** | No registration | Requires account |
| **Approval** | None (instant) | Review process |
| **Hosting** | Self-hosted or GitHub | Centralized servers |
| **Versioning** | Git tags (built-in) | Manual version bump |
| **Collaboration** | PRs, forks | Limited |
| **Offline** | Clone once, use forever | Requires registry access |
| **Cost** | Free (Git hosting) | Bandwidth costs |
| **Censorship** | Resistant (decentralized) | Can be removed |

---

## Advanced: Multi-Expert Repositories

For related experts, use a monorepo:

```
hivellm-experts/                       # One repository
├── README.md
├── experts/
│   ├── english-basic/
│   │   ├── manifest.json
│   │   ├── datasets/
│   │   ├── scripts/
│   │   └── weights/
│   ├── json-parser/
│   │   ├── manifest.json
│   │   └── ...
│   └── neo4j-cypher/
│       └── ...
└── shared/
    ├── generate_dataset.py
    └── train_expert.py
```

**Installation**:
```bash
# Install specific expert from monorepo
expert-cli install https://github.com/hivellm/hivellm-experts/experts/json-parser

# Install all experts from monorepo
expert-cli install https://github.com/hivellm/hivellm-experts --all
```

---

## Security Considerations

### 1. Verify Sources

Only install from trusted repositories:

```bash
# Whitelist trusted orgs
expert-cli config set trusted_orgs "hivellm,mycompany"

# Warn on untrusted install
expert-cli install https://github.com/random-user/expert-something
# ⚠️  Warning: Installing from untrusted source. Verify code before training.
```

### 2. Code Review Before Training

```bash
# Clone but don't train automatically
expert-cli install --no-train https://github.com/user/expert-x

# Review code
cd ~/.expert/repos/expert-x
cat scripts/train.sh  # Check for malicious code

# Train manually if safe
./scripts/train.sh
```

### 3. Signed Commits

Require GPG-signed commits for marketplace submission:

```bash
# Marketplace only accepts signed commits
git commit -S -m "Release v1.0.0"
```

---

## Migration from NPM/PyPI (Future)

If later a centralized registry is desired:

1. Git repos remain source of truth
2. Registry mirrors Git repos (like Nixpkgs)
3. `expert-cli install expert-name` fetches from Git via registry
4. Direct Git URLs still work

**Best of both worlds**: Convenience of registry + decentralization of Git

---

## Example: Full Workflow

### As Expert Creator

```bash
# 1. Create repo from template
git clone https://github.com/hivellm/expert-repository-template expert-rust-code
cd expert-rust-code

# 2. Edit manifest.json (set name, domain, task, training config)
vim manifest.json

# 3. Generate dataset (reads from manifest.json)
expert-cli dataset generate --manifest manifest.json

# 4. Train (reads from manifest.json)
expert-cli train --manifest manifest.json

# 5. Validate
expert-cli validate --expert weights/rust-code.v1.0.0

# 6. Package
expert-cli package --manifest manifest.json

# 7. Sign
expert-cli sign --expert weights/rust-code.v1.0.0.expert

# 8. Commit & tag
git add .
git commit -m "Release v1.0.0"
git tag v1.0.0
git push origin main v1.0.0

# 9. Submit to marketplace
expert-cli marketplace submit https://github.com/me/expert-rust-code
```

**All commands via `expert-cli`** - no custom scripts!

### As User

```bash
# 1. Search marketplace
expert-cli search "rust code analysis"

# 2. Install
expert-cli install https://github.com/me/expert-rust-code

# 3. Use
expert-cli infer --expert rust-code --file main.rs

# 4. Update
expert-cli update rust-code  # Pulls latest from Git
```

---

## CLI Commands

```bash
# Install from Git
expert-cli install <git-url>[@version]

# Install from marketplace (resolves to Git URL)
expert-cli install <expert-name>

# Update expert (git pull)
expert-cli update <expert-name>

# List installed
expert-cli list

# Remove expert
expert-cli remove <expert-name>

# Show expert info
expert-cli info <expert-name>

# Marketplace
expert-cli marketplace update          # Update marketplace index
expert-cli marketplace search <query>
expert-cli marketplace submit <git-url>

# Development
expert-cli create <expert-name>        # Scaffold new expert repo
expert-cli test <expert-path>
expert-cli sign <expert-path>
```

---

## See Also

- [EXPERT_FORMAT.md](EXPERT_FORMAT.md) - `.expert` package specification
- [QUICKSTART.md](../QUICKSTART.md) - Train your first experts
- [MARKETPLACE.md](MARKETPLACE.md) - Marketplace guidelines (future)

