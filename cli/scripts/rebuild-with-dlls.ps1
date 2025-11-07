# Rebuild expert-cli with Python DLLs auto-copy

$ErrorActionPreference = "Stop"

Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "Expert CLI - Rebuild with Python DLLs" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

# Navigate to CLI directory
$cliDir = $PSScriptRoot
Set-Location $cliDir

Write-Host "[1/3] Checking Python installation..." -ForegroundColor Yellow
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue

if (-not $pythonCmd) {
    Write-Host "  ❌ ERROR: Python not found in PATH!" -ForegroundColor Red
    Write-Host "  Please install Python 3.11+ or add it to PATH" -ForegroundColor Red
    exit 1
}

$pythonVersion = & python --version 2>&1
Write-Host "  ✓ Found: $pythonVersion" -ForegroundColor Green
Write-Host ""

Write-Host "[2/3] Cleaning previous build..." -ForegroundColor Yellow
if (Test-Path "target") {
    Remove-Item -Path "target\release\*.dll" -Force -ErrorAction SilentlyContinue
    Write-Host "  ✓ Cleaned old DLLs" -ForegroundColor Green
} else {
    Write-Host "  ✓ No previous build found" -ForegroundColor Green
}
Write-Host ""

Write-Host "[3/4] Building expert-cli in release mode..." -ForegroundColor Yellow
Write-Host "  This will automatically copy Python DLLs to target/release" -ForegroundColor Gray
Write-Host ""

wsl -d Ubuntu-24.04 -- bash -l -c "cd '$($cliDir -replace '\\', '/' -replace 'F:', '/mnt/f')' && cargo build --release"

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✅ Build completed successfully!" -ForegroundColor Green
    Write-Host ""
    
    # Copy Python DLLs
    Write-Host "[4/4] Copying Python DLLs..." -ForegroundColor Yellow
    & "$PSScriptRoot\copy-python-dlls.ps1"
    
    Write-Host ""
    Write-Host "Executable location:" -ForegroundColor Yellow
    Write-Host "  F:\Node\hivellm\expert\cli\target\release\expert-cli.exe" -ForegroundColor White
    
} else {
    Write-Host ""
    Write-Host "❌ Build failed with exit code: $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

