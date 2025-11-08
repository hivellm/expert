# PowerShell script to run expert tests
# Copy this to your expert's tests/ directory and customize as needed

$ErrorActionPreference = "Stop"

Write-Host "Running expert tests..." -ForegroundColor Cyan

# Check if Python is available
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    Write-Host "Error: Python not found in PATH" -ForegroundColor Red
    exit 1
}

# Check if pytest is installed
$pytest = python -m pytest --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: pytest not installed. Install with: pip install pytest" -ForegroundColor Red
    exit 1
}

# Run tests
Write-Host "`nRunning basic tests..." -ForegroundColor Yellow
python -m pytest test_basic.py -v

if ($LASTEXITCODE -ne 0) {
    Write-Host "Basic tests failed!" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "`nRunning hard case tests..." -ForegroundColor Yellow
python -m pytest test_hard.py -v

if ($LASTEXITCODE -ne 0) {
    Write-Host "Hard case tests failed (may be expected for known limitations)" -ForegroundColor Yellow
}

Write-Host "`nRunning comparison tests..." -ForegroundColor Yellow
python -m pytest test_comparison.py -v

if ($LASTEXITCODE -ne 0) {
    Write-Host "Comparison tests failed!" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "`nAll tests completed!" -ForegroundColor Green

