#!/usr/bin/env pwsh
# Build expert-cli with CUDA support

# Find Visual Studio 2022 Developer environment
$vsPath = "C:\Program Files\Microsoft Visual Studio\2022\Community"
$vcvarsPath = "$vsPath\VC\Auxiliary\Build\vcvars64.bat"

if (-not (Test-Path $vcvarsPath)) {
    Write-Host "ERROR: Visual Studio vcvars64.bat not found at: $vcvarsPath" -ForegroundColor Red
    exit 1
}

Write-Host "Setting up Visual Studio environment..." -ForegroundColor Cyan

# Setup CUDA paths
$cudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
$env:CUDA_PATH = $cudaPath
$env:CUDA_ROOT = $cudaPath
$env:CUDNN_PATH = "C:\Program Files\NVIDIA\CUDNN\v9.14"

# Add CUDA lib paths for linker
$cudaLibPath = "$cudaPath\lib\x64"
if (Test-Path $cudaLibPath) {
    $env:LIB = if ($env:LIB) { "$cudaLibPath;$env:LIB" } else { $cudaLibPath }
}

# Add CUDA bin to PATH (for DLLs)
$cudaBinPath = "$cudaPath\bin"
if (Test-Path $cudaBinPath) {
    $env:PATH = "$cudaBinPath;$env:PATH"
}

# Setup Visual Studio environment directly
Write-Host "Initializing Visual Studio 2022 x64 environment..." -ForegroundColor Gray

# Add VS2022 tools to PATH
$vsBinPath = "$vsPath\VC\Tools\MSVC"
$latestMSVC = Get-ChildItem $vsBinPath | Sort-Object Name -Descending | Select-Object -First 1
$vsToolsPath = "$vsBinPath\$($latestMSVC.Name)\bin\Hostx64\x64"
$env:PATH = "$vsToolsPath;$env:PATH"

# Add Windows SDK
$sdkPath = "C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\x64"
if (Test-Path $sdkPath) {
    $env:PATH = "$sdkPath;$env:PATH"
}

Write-Host "Building with CUDA support..." -ForegroundColor Cyan
Write-Host "CUDA_PATH: $env:CUDA_PATH" -ForegroundColor Gray
Write-Host "CUDNN_PATH: $env:CUDNN_PATH" -ForegroundColor Gray
Write-Host "PATH includes cl.exe: $(if (Get-Command cl.exe -ErrorAction SilentlyContinue) { 'YES' } else { 'NO' })" -ForegroundColor Gray
Write-Host ""

Write-Host "Running: cargo build --release --features cuda" -ForegroundColor Yellow
cargo build --release --features cuda

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✅ CUDA build completed successfully!" -ForegroundColor Green
    Write-Host "Binary: .\target\release\expert-cli.exe" -ForegroundColor Cyan
} else {
    Write-Host "`n❌ CUDA build failed!" -ForegroundColor Red
    exit $LASTEXITCODE
}

