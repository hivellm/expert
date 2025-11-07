# Setup Python environment for Windows with CUDA support

$ErrorActionPreference = "Stop"

Write-Host "Setting up Expert CLI for Windows with CUDA..." -ForegroundColor Cyan

# Check Python
Write-Host "`nChecking Python installation..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Python not found!" -ForegroundColor Red
    Write-Host "Install Python 3.11+ from https://www.python.org/downloads/" -ForegroundColor Red
    exit 1
}
Write-Host "Found: $pythonVersion" -ForegroundColor Green

# Create virtual environment
Write-Host "`nCreating virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv_windows") {
    Write-Host "Removing existing venv_windows..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force venv_windows
}

python -m venv venv_windows
Write-Host "Virtual environment created" -ForegroundColor Green

# Activate venv
Write-Host "`nActivating virtual environment..." -ForegroundColor Yellow
& ".\venv_windows\Scripts\Activate.ps1"

# Upgrade pip
Write-Host "`nUpgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip wheel setuptools

# Install PyTorch with CUDA 12.1 (compatible with CUDA 12.6)
Write-Host "`nInstalling PyTorch with CUDA 12.1 support..." -ForegroundColor Yellow
Write-Host "(This is compatible with your CUDA 12.6 installation)" -ForegroundColor Cyan
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
Write-Host "`nInstalling other dependencies..." -ForegroundColor Yellow
pip install transformers datasets accelerate peft bitsandbytes numpy tqdm

Write-Host "`n" + ("=" * 60) -ForegroundColor Green
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host ("=" * 60) -ForegroundColor Green

# Test CUDA
Write-Host "`nTesting CUDA availability..." -ForegroundColor Cyan
python check_cuda.py

Write-Host "`nEnvironment ready!" -ForegroundColor Green
Write-Host "To activate in future sessions: .\venv_windows\Scripts\Activate.ps1" -ForegroundColor Yellow

