#!/usr/bin/env pwsh
# Expert CLI Installation Script for Windows

param(
    [switch]$WithCUDA = $false,
    [switch]$SkipEnvVars = $false
)

Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "  Expert CLI Installation - Windows" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host ""

# Check Rust
Write-Host "Checking Rust installation..." -ForegroundColor Yellow
if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Rust not found" -ForegroundColor Red
    Write-Host "Install from: https://rustup.rs" -ForegroundColor Red
    Write-Host "Or run: winget install Rustlang.Rustup" -ForegroundColor Yellow
    exit 1
}

# Check rustup nightly
Write-Host "Checking Rust nightly toolchain..." -ForegroundColor Yellow
$toolchains = rustup toolchain list
if ($toolchains -notmatch "nightly") {
    Write-Host "Installing Rust nightly..." -ForegroundColor Yellow
    rustup install nightly
}

rustup default nightly
Write-Host "✓ Rust nightly set as default" -ForegroundColor Green
Write-Host ""

# Check Python
Write-Host "Checking Python installation..." -ForegroundColor Yellow
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Python 3 not found" -ForegroundColor Red
    Write-Host "Install Python 3.11+ from: https://www.python.org" -ForegroundColor Red
    Write-Host "Or run: winget install Python.Python.3.12" -ForegroundColor Yellow
    exit 1
}

$pythonVersion = python --version
Write-Host "✓ $pythonVersion found" -ForegroundColor Green
Write-Host ""

# Install Python dependencies
Write-Host "Installing Python dependencies..." -ForegroundColor Yellow

if (-not (Test-Path "requirements.txt")) {
    Write-Host "❌ requirements.txt not found" -ForegroundColor Red
    exit 1
}

# Create virtual environment if it doesn't exist
if (-not (Test-Path "venv_windows")) {
    Write-Host "Creating Python virtual environment..." -ForegroundColor Yellow
    python -m venv venv_windows
}

# Activate virtual environment
& ".\venv_windows\Scripts\Activate.ps1"

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Python dependencies installed" -ForegroundColor Green
} else {
    Write-Host "⚠  Some dependencies failed to install" -ForegroundColor Yellow
    if ($WithCUDA) {
        Write-Host "Installing PyTorch with CUDA support..." -ForegroundColor Yellow
        pip install torch --index-url https://download.pytorch.org/whl/cu121
    }
}

Write-Host ""

# Build CLI
if ($WithCUDA) {
    Write-Host "Building Expert CLI with CUDA support..." -ForegroundColor Yellow
    Write-Host "This requires Visual Studio 2022 and CUDA Toolkit 12.6+" -ForegroundColor Gray
    Write-Host ""
    
    # Check if build-cuda.ps1 exists
    if (Test-Path "build-cuda.ps1") {
        & ".\build-cuda.ps1"
    } else {
        Write-Host "⚠  build-cuda.ps1 not found, using cargo directly" -ForegroundColor Yellow
        cargo build --release --features cuda
    }
} else {
    Write-Host "Building Expert CLI (CPU only)..." -ForegroundColor Yellow
    cargo build --release
}

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Expert CLI built successfully" -ForegroundColor Green
    Write-Host ""
    Write-Host "Binary location: .\target\release\expert-cli.exe" -ForegroundColor Cyan
} else {
    Write-Host "❌ Build failed" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Add to PATH
if (-not $SkipEnvVars) {
    Write-Host "Setting up environment variables..." -ForegroundColor Yellow
    
    $currentPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $expertPath = Join-Path $PSScriptRoot "target\release"
    
    if ($currentPath -notlike "*$expertPath*") {
        Write-Host "Adding Expert CLI to user PATH..." -ForegroundColor Yellow
        
        $newPath = "$currentPath;$expertPath"
        [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
        
        Write-Host "✓ Added to PATH: $expertPath" -ForegroundColor Green
        Write-Host "⚠  Restart your terminal for PATH changes to take effect" -ForegroundColor Yellow
    } else {
        Write-Host "✓ Expert CLI already in PATH" -ForegroundColor Green
    }
    
    # Set EXPERT_HOME environment variable
    $expertHome = $PSScriptRoot
    [Environment]::SetEnvironmentVariable("EXPERT_HOME", $expertHome, "User")
    Write-Host "✓ Set EXPERT_HOME: $expertHome" -ForegroundColor Green
    
    Write-Host ""
}

# Test installation
Write-Host "Testing Expert CLI..." -ForegroundColor Yellow
$env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")

# Test using relative path
& ".\target\release\expert-cli.exe" --version
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Expert CLI works correctly" -ForegroundColor Green
} else {
    Write-Host "⚠  Could not verify installation" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "  Installation Complete!" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host ""
Write-Host "Run the CLI:" -ForegroundColor White
Write-Host "  expert-cli --help" -ForegroundColor Yellow
Write-Host ""
Write-Host "Or using relative path:" -ForegroundColor White
Write-Host "  .\target\release\expert-cli.exe --help" -ForegroundColor Yellow
Write-Host ""
Write-Host "Quick start:" -ForegroundColor White
Write-Host "  expert-cli train --help" -ForegroundColor Yellow
Write-Host "  expert-cli chat --help" -ForegroundColor Yellow
Write-Host ""
Write-Host "For CUDA support, run:" -ForegroundColor White
Write-Host "  .\install.ps1 -WithCUDA" -ForegroundColor Yellow
Write-Host ""
Write-Host "Full documentation: .\README.md" -ForegroundColor Gray
Write-Host ""

# Deactivate virtual environment
deactivate

