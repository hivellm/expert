# Test Expert Loading Script
# Usage: .\test_expert.ps1 -Package <path> -BaseModel <path>

param(
    [Parameter(Mandatory=$false)]
    [string]$Package = "f:\Node\hivellm\expert\experts\expert-neo4j\expert-neo4j-qwen306b.v0.0.1.expert",
    
    [Parameter(Mandatory=$false)]
    [string]$BaseModel = "F:/Node/hivellm/expert/models/Qwen3-0.6B",
    
    [Parameter(Mandatory=$false)]
    [string]$TestCases = "f:\Node\hivellm\expert\experts\expert-neo4j\test_cases.json",
    
    [Parameter(Mandatory=$false)]
    [string]$Device = "cuda"
)

Write-Host ""
Write-Host "╔═══════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║     Expert Loading Test Suite        ║" -ForegroundColor Cyan
Write-Host "╚═══════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Check if package exists
if (-not (Test-Path $Package)) {
    Write-Host "❌ Package not found: $Package" -ForegroundColor Red
    exit 1
}

# Check if base model exists
if (-not (Test-Path $BaseModel)) {
    Write-Host "❌ Base model not found: $BaseModel" -ForegroundColor Red
    exit 1
}

Write-Host "Package: $Package" -ForegroundColor White
Write-Host "Base Model: $BaseModel" -ForegroundColor White
Write-Host "Test Cases: $TestCases" -ForegroundColor White
Write-Host "Device: $Device" -ForegroundColor White
Write-Host ""

# Activate venv
$venvPath = "F:\Node\hivellm\expert\cli\venv_windows\Scripts\Activate.ps1"
if (Test-Path $venvPath) {
    Write-Host "Activating Python virtual environment..." -ForegroundColor Yellow
    & $venvPath
}

# Run test script
$scriptPath = "F:\Node\hivellm\expert\cli\scripts\test_expert_loading.py"

Write-Host "Running test script..." -ForegroundColor Yellow
Write-Host ""

& python $scriptPath `
    --package $Package `
    --base-model $BaseModel `
    --device $Device `
    --test-cases $TestCases

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✅ Test completed successfully!" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "❌ Test failed with exit code: $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

