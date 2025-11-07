# Force rebuild - kills processes and rebuilds

$ErrorActionPreference = "Stop"

Write-Host "Force rebuild of expert-cli..." -ForegroundColor Cyan
Write-Host ""

$cliDir = $PSScriptRoot
Set-Location $cliDir

# Kill any running expert-cli processes
Write-Host "Checking for running processes..." -ForegroundColor Yellow
$processes = Get-Process -Name "expert-cli" -ErrorAction SilentlyContinue

if ($processes) {
    Write-Host "  Stopping $($processes.Count) expert-cli process(es)..." -ForegroundColor Yellow
    $processes | Stop-Process -Force
    Start-Sleep -Seconds 1
    Write-Host "  Processes stopped" -ForegroundColor Green
} else {
    Write-Host "  No running processes" -ForegroundColor Green
}
Write-Host ""

Write-Host "Building in release mode..." -ForegroundColor Yellow
wsl -d Ubuntu-24.04 -- bash -l -c "cd '$($cliDir -replace '\\', '/' -replace 'F:', '/mnt/f')' && cargo build --release 2>&1 | grep -E '(Compiling expert-cli|Finished|error)'"

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Build completed successfully!" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "Build failed with exit code: $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

