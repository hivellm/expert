# Installation Guide - Expert CLI

## Quick Installation

### Linux / macOS

```bash
# Clone repository
git clone https://github.com/hivellm/expert.git
cd expert/cli

# Run installation script
./INSTALL.sh

# Or with CUDA support
./INSTALL.sh --with-cuda
```

### Windows

```powershell
# Clone repository
git clone https://github.com/hivellm/expert.git
cd expert/cli

# Run installation script
.\install.ps1

# Or with CUDA support
.\install.ps1 -WithCUDA
```

---

## Detailed Installation

### Prerequisites

#### All Platforms
- **Rust**: Nightly toolchain (1.85+)
  - Install: https://rustup.rs
  - Or: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- **Python**: 3.11 or later
  - Install: https://www.python.org

#### Windows (for CUDA)
- **Visual Studio 2022**: Community Edition or higher
  - With C++ development tools
  - Install: https://visualstudio.microsoft.com
- **CUDA Toolkit**: 12.6 or later
  - Install: https://developer.nvidia.com/cuda-downloads
- **cuDNN**: 9.14 or later
  - Install: https://developer.nvidia.com/cudnn

#### Linux (for CUDA)
- **CUDA Toolkit**: 12.6 or later
  - Install: https://developer.nvidia.com/cuda-downloads
- **cuDNN**: 9.14 or later
  - Install: https://developer.nvidia.com/cudnn

---

## Installation Steps

### 1. Install Rust Nightly

```bash
# Install rustup (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install nightly toolchain
rustup install nightly
rustup default nightly
```

### 2. Install Python Dependencies

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# or
.\venv\Scripts\Activate.ps1  # Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# For CUDA support (optional)
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 3. Build Expert CLI

#### CPU-only Build

```bash
cargo build --release
```

Binary location: `./target/release/expert-cli` (Linux/macOS) or `.\target\release\expert-cli.exe` (Windows)

#### CUDA Build (Windows)

```powershell
# Use the provided build script
.\build-cuda.ps1
```

#### CUDA Build (Linux)

```bash
# Ensure CUDA_PATH is set
export CUDA_PATH=/usr/local/cuda-12.6
cargo build --release --features cuda
```

### 4. Setup Environment Variables

#### Linux / macOS

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
export EXPERT_HOME="/path/to/expert/cli"
export PATH="$EXPERT_HOME/target/release:$PATH"
```

Then reload:
```bash
source ~/.bashrc  # or ~/.zshrc
```

#### Windows

Add to user PATH:

```powershell
$expertPath = "C:\path\to\expert\cli\target\release"
$currentPath = [Environment]::GetEnvironmentVariable("Path", "User")
[Environment]::SetEnvironmentVariable("Path", "$currentPath;$expertPath", "User")

# Set EXPERT_HOME
[Environment]::SetEnvironmentVariable("EXPERT_HOME", "C:\path\to\expert\cli", "User")
```

Restart your terminal for changes to take effect.

---

## Verify Installation

```bash
# Check version
expert-cli --version

# Show help
expert-cli --help

# Test training command
expert-cli train --help

# Test chat command
expert-cli chat --help
```

---

## Automated Installation Scripts

### Linux / macOS: `INSTALL.sh`

```bash
# Basic installation
./INSTALL.sh

# With CUDA support
./INSTALL.sh --with-cuda

# Skip environment variable setup
./INSTALL.sh --skip-env-vars
```

Features:
- ✅ Checks Rust and Python installations
- ✅ Installs Rust nightly if needed
- ✅ Creates Python virtual environment
- ✅ Installs Python dependencies
- ✅ Builds expert-cli binary
- ✅ Adds to PATH automatically
- ✅ Sets EXPERT_HOME environment variable

### Windows: `install.ps1`

```powershell
# Basic installation
.\install.ps1

# With CUDA support
.\install.ps1 -WithCUDA

# Skip environment variable setup
.\install.ps1 -SkipEnvVars
```

Features:
- ✅ Checks Rust and Python installations
- ✅ Installs Rust nightly if needed
- ✅ Creates Python virtual environment
- ✅ Installs Python dependencies
- ✅ Builds expert-cli binary (with or without CUDA)
- ✅ Adds to user PATH automatically
- ✅ Sets EXPERT_HOME environment variable

---

## Package Managers

### Debian/Ubuntu

Download `.deb` package from releases:

```bash
# Install package
sudo dpkg -i expert-cli_0.2.3_amd64.deb

# Fix dependencies if needed
sudo apt-get install -f
```

### Cargo Install (from crates.io)

```bash
# Install from crates.io (when published)
cargo install expert-cli

# With CUDA support
cargo install expert-cli --features cuda
```

---

## Troubleshooting

### Rust Nightly Not Found

```bash
rustup install nightly
rustup default nightly
```

### Python Dependencies Fail

```bash
# Upgrade pip
pip install --upgrade pip

# Install with verbose output
pip install -r requirements.txt -v
```

### CUDA Build Fails (Windows)

1. Verify Visual Studio 2022 is installed with C++ tools
2. Check CUDA Toolkit is in `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6`
3. Check cuDNN is in `C:\Program Files\NVIDIA\CUDNN\v9.14`
4. Run `build-cuda.ps1` instead of `cargo build`

### CUDA Build Fails (Linux)

1. Set CUDA_PATH: `export CUDA_PATH=/usr/local/cuda-12.6`
2. Add CUDA to LD_LIBRARY_PATH: `export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH`
3. Verify CUDA: `nvcc --version`

### Binary Not in PATH

**Linux/macOS:**
```bash
# Use full path temporarily
./target/release/expert-cli --help

# Or add to current session
export PATH="$(pwd)/target/release:$PATH"
```

**Windows:**
```powershell
# Use full path temporarily
.\target\release\expert-cli.exe --help

# Or add to current session
$env:Path = "$(pwd)\target\release;$env:Path"
```

---

## Uninstallation

### Linux / macOS

```bash
# Remove binary
rm ./target/release/expert-cli

# Remove from PATH (edit ~/.bashrc or ~/.zshrc)
# Remove EXPERT_HOME lines

# Remove virtual environment
rm -rf venv
```

### Windows

```powershell
# Remove binary
rm .\target\release\expert-cli.exe

# Remove from PATH
$currentPath = [Environment]::GetEnvironmentVariable("Path", "User")
$newPath = $currentPath -replace ";?C:\\path\\to\\expert\\cli\\target\\release", ""
[Environment]::SetEnvironmentVariable("Path", $newPath, "User")

# Remove EXPERT_HOME
[Environment]::SetEnvironmentVariable("EXPERT_HOME", $null, "User")

# Remove virtual environment
rm -Recurse -Force venv_windows
```

### Debian/Ubuntu

```bash
sudo apt-get remove expert-cli
```

---

## Next Steps

After installation, see:
- [Quick Start Guide](../README.md#quick-start)
- [Training Documentation](./TRAINING.md)
- [CLI Reference](./CLI_REFERENCE.md)
- [Examples](../examples/)

---

## Support

- **GitHub Issues**: https://github.com/hivellm/expert/issues
- **Documentation**: https://github.com/hivellm/expert#readme
- **Email**: team@hivellm.org

