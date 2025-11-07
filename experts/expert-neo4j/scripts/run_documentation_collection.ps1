# PowerShell script to collect Neo4j official documentation examples
# Uses venv_windows from CLI for Python execution

$ErrorActionPreference = "Stop"

Write-Host "=" -NoNewline
Write-Host ("=" * 69)
Write-Host "Neo4j Official Documentation Collection"
Write-Host ("=" * 70)
Write-Host ""

# Check if venv_windows exists
$venvPython = "F:/Node/hivellm/expert/cli/venv_windows/Scripts/python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "[ERROR] venv_windows not found at: $venvPython" -ForegroundColor Red
    Write-Host "Please ensure the CLI venv_windows is set up correctly." -ForegroundColor Yellow
    exit 1
}

# Change to expert-neo4j directory
$expertDir = "F:/Node/hivellm/expert/experts/expert-neo4j"
if (-not (Test-Path $expertDir)) {
    Write-Host "[ERROR] Expert directory not found: $expertDir" -ForegroundColor Red
    exit 1
}

Set-Location $expertDir
Write-Host "Working directory: $expertDir" -ForegroundColor Cyan
Write-Host ""

# Check if beautifulsoup4 is installed
Write-Host "Checking dependencies..." -ForegroundColor Cyan
& $venvPython -c "import bs4; import requests" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARNING] Missing dependencies. Installing beautifulsoup4 and requests..." -ForegroundColor Yellow
    & $venvPython -m pip install beautifulsoup4 requests --quiet
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to install dependencies" -ForegroundColor Red
        exit 1
    }
    Write-Host "[OK] Dependencies installed" -ForegroundColor Green
}

Write-Host ""
Write-Host "Starting documentation collection..." -ForegroundColor Cyan
Write-Host ""

# Run collection script
& $venvPython scripts/collect_documentation.py

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host ("=" * 70)
    Write-Host "[OK] Documentation collection completed successfully!" -ForegroundColor Green
    Write-Host ("=" * 70)
    Write-Host ""
    Write-Host "Next steps:"
    Write-Host "  1. Review collected examples in: datasets/raw/documentation/neo4j_documentation.jsonl"
    Write-Host "  2. Run preprocessing with --include-documentation flag:"
    Write-Host "     python preprocess.py --dataset neo4j/text2cypher-2025v1 --include-documentation --output datasets"
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "[ERROR] Collection failed with exit code: $LASTEXITCODE" -ForegroundColor Red
    exit 1
}

