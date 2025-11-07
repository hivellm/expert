#!/bin/bash
# Expert CLI Installation Script

set -e

# Parse arguments
SKIP_ENV_VARS=false
WITH_CUDA=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --skip-env-vars) SKIP_ENV_VARS=true ;;
        --with-cuda) WITH_CUDA=true ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Expert CLI Installation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo

# Check Rust
echo "Checking Rust installation..."
if ! command -v cargo &> /dev/null; then
    echo "❌ Rust not found"
    echo "Install with: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    exit 1
fi

# Check rustup nightly
echo "Checking Rust nightly..."
if ! rustup toolchain list | grep -q nightly; then
    echo "Installing Rust nightly..."
    rustup install nightly
fi

rustup default nightly
echo "✓ Rust nightly set as default"
echo

# Check Python
echo "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found"
    echo "Install Python 3.11+ from https://www.python.org"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo "✓ Python $PYTHON_VERSION found"
echo

# Install Python dependencies
echo "Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        echo "Creating Python virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    pip3 install --upgrade pip
    pip3 install -r requirements.txt
    
    if [ $? -eq 0 ]; then
        echo "✓ Python dependencies installed"
    else
        echo "⚠  Some dependencies failed to install"
        if [ "$WITH_CUDA" = true ]; then
            echo "Installing PyTorch with CUDA support..."
            pip3 install torch --index-url https://download.pytorch.org/whl/cu121
        fi
    fi
else
    echo "❌ requirements.txt not found"
    exit 1
fi

echo

# Build CLI
if [ "$WITH_CUDA" = true ]; then
    echo "Building Expert CLI with CUDA support..."
    echo "Note: CUDA support on Linux requires CUDA Toolkit 12.6+"
    cargo build --release --features cuda
else
    echo "Building Expert CLI (CPU only)..."
    cargo build --release
fi

if [ $? -eq 0 ]; then
    echo "✓ Expert CLI built successfully"
    echo
    echo "Binary location: target/release/expert-cli"
else
    echo "❌ Build failed"
    exit 1
fi

echo

# Setup environment variables
if [ "$SKIP_ENV_VARS" = false ]; then
    echo "Setting up environment variables..."
    
    EXPERT_PATH="$(pwd)/target/release"
    EXPERT_HOME="$(pwd)"
    
    # Determine shell config file
    if [ -n "$ZSH_VERSION" ]; then
        SHELL_CONFIG="$HOME/.zshrc"
    elif [ -n "$BASH_VERSION" ]; then
        SHELL_CONFIG="$HOME/.bashrc"
    else
        SHELL_CONFIG="$HOME/.profile"
    fi
    
    # Add to PATH if not already present
    if ! grep -q "EXPERT_HOME" "$SHELL_CONFIG" 2>/dev/null; then
        echo "" >> "$SHELL_CONFIG"
        echo "# Expert CLI" >> "$SHELL_CONFIG"
        echo "export EXPERT_HOME=\"$EXPERT_HOME\"" >> "$SHELL_CONFIG"
        echo "export PATH=\"\$EXPERT_HOME/target/release:\$PATH\"" >> "$SHELL_CONFIG"
        
        echo "✓ Added to $SHELL_CONFIG"
        echo "⚠  Run 'source $SHELL_CONFIG' or restart your terminal"
    else
        echo "✓ Environment variables already configured"
    fi
    
    # Also add to current session
    export EXPERT_HOME="$EXPERT_HOME"
    export PATH="$EXPERT_PATH:$PATH"
    
    echo "✓ Set EXPERT_HOME: $EXPERT_HOME"
    echo "✓ Added to PATH: $EXPERT_PATH"
    echo
fi

# Test installation
echo "Testing Expert CLI..."
./target/release/expert-cli --version

if [ $? -eq 0 ]; then
    echo "✓ Expert CLI works correctly"
else
    echo "⚠  Could not verify installation"
fi

echo

# Deactivate virtual environment
deactivate

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Installation Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo
echo "Run the CLI:"
echo "  expert-cli --help"
echo
echo "Or using relative path:"
echo "  ./target/release/expert-cli --help"
echo
echo "Quick start:"
echo "  expert-cli train --help"
echo "  expert-cli chat --help"
echo
echo "For CUDA support, run:"
echo "  ./INSTALL.sh --with-cuda"
echo
echo "Full documentation: ./README.md"
echo

