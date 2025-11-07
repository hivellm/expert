# PowerShell Scripts

Build, setup, and training automation scripts for Windows.

---

## Build Scripts

### `rebuild-force.ps1`
**Purpose**: Force rebuild of the CLI (clean build).

**Usage**:
```powershell
.\scripts\rebuild-force.ps1
```

**What it does**:
- Cleans previous build artifacts
- Runs `cargo clean`
- Rebuilds in release mode with all optimizations

---

### `rebuild-quick.ps1`
**Purpose**: Quick incremental rebuild.

**Usage**:
```powershell
.\scripts\rebuild-quick.ps1
```

**What it does**:
- Incremental build (faster)
- Skips cleaning
- Useful for quick iterations during development

---

### `rebuild-with-dlls.ps1`
**Purpose**: Rebuild and copy required Python DLLs.

**Usage**:
```powershell
.\scripts\rebuild-with-dlls.ps1
```

**What it does**:
- Builds the CLI
- Copies Python DLLs to release directory
- Ensures all runtime dependencies are present

---

### `copy-python-dlls.ps1`
**Purpose**: Copy Python DLLs to release directory.

**Usage**:
```powershell
.\scripts\copy-python-dlls.ps1
```

**What it does**:
- Copies `python3.dll`, `python312.dll`, `vcruntime140.dll` etc.
- Required for standalone CLI execution
- Run after building if DLLs are missing

---

## Setup Scripts

### `setup_windows.ps1`
**Purpose**: Complete Windows development environment setup.

**Usage**:
```powershell
.\scripts\setup_windows.ps1
```

**What it does**:
- Installs Python dependencies
- Sets up virtual environment
- Configures CUDA paths
- Downloads base model
- Verifies installation

**Requirements**:
- Python 3.12 installed
- CUDA Toolkit 12.1+ installed
- Rust toolchain installed

---

## Training Scripts

### `train_windows.ps1`
**Purpose**: Training wrapper for Windows with proper environment.

**Usage**:
```powershell
.\scripts\train_windows.ps1 -Manifest path\to\manifest.json
```

**Options**:
- `-Manifest`: Path to expert manifest.json
- `-Output`: Output directory (default: weights)
- `-Device`: Device to use (default: auto)

**What it does**:
- Activates Python virtual environment
- Sets up CUDA environment variables
- Runs training via CLI
- Handles errors and logging

---

### `train.ps1`
**Purpose**: Simple training script.

**Usage**:
```powershell
.\scripts\train.ps1
```

**What it does**:
- Quick training launcher
- Uses default settings
- Minimal configuration

---

## Common Workflows

### 1. Initial Setup (First Time)

```powershell
# Setup environment
.\scripts\setup_windows.ps1

# Test build
.\scripts\rebuild-quick.ps1
```

### 2. Development Workflow

```powershell
# Make code changes...

# Quick rebuild
.\scripts\rebuild-quick.ps1

# If you need clean build
.\scripts\rebuild-force.ps1
```

### 3. Training Workflow

```powershell
# Train an expert
.\scripts\train_windows.ps1 -Manifest ..\experts\expert-json\manifest.json

# Or use simple version
cd ..\experts\expert-json
..\..\cli\scripts\train.ps1
```

### 4. Troubleshooting DLL Issues

```powershell
# If you get DLL errors
.\scripts\copy-python-dlls.ps1

# Or rebuild with DLLs
.\scripts\rebuild-with-dlls.ps1
```

---

## Script Locations

All PowerShell scripts are now in `cli/scripts/`:

```
cli/scripts/
├── rebuild-force.ps1          # Clean rebuild
├── rebuild-quick.ps1          # Fast incremental build
├── rebuild-with-dlls.ps1      # Build + copy DLLs
├── copy-python-dlls.ps1       # Copy DLLs only
├── setup_windows.ps1          # Complete setup
├── train_windows.ps1          # Training wrapper
└── train.ps1                  # Simple training
```

---

## Notes

### Why PowerShell Scripts?

PowerShell scripts handle Windows-specific tasks:
- DLL management
- Path configuration (Windows paths with backslashes)
- Virtual environment activation
- CUDA environment setup
- MSBuild integration

### Execution Policy

If scripts don't run, you may need to allow script execution:

```powershell
# Check current policy
Get-ExecutionPolicy

# Allow scripts (run as Administrator)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Python Scripts vs PowerShell Scripts

- **Python scripts** (`*.py`): Cross-platform utilities
- **PowerShell scripts** (`*.ps1`): Windows-specific automation

Both types are now organized in `cli/scripts/` for consistency.

---

Last Updated: November 3, 2025

