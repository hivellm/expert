# Quick rebuild without clean

$ErrorActionPreference = "Stop"

Write-Host "Quick rebuild of expert-cli..." -ForegroundColor Cyan
Write-Host ""

$cliDir = $PSScriptRoot
Set-Location $cliDir

Write-Host "Building in release mode..." -ForegroundColor Yellow
wsl -d Ubuntu-24.04 -- bash -l -c "cd '$($cliDir -replace '\\', '/' -replace 'F:', '/mnt/f')' && cargo build --release"

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Build completed successfully!" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "Build failed with exit code: $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

