# Test runner for HiveLLM Expert CLI v0.2.3
# Runs all automated tests

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  HiveLLM Expert CLI - Test Suite" -ForegroundColor White
Write-Host "  Version: 0.2.3" -ForegroundColor White
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$ErrorCount = 0

# 1. Rust Tests
Write-Host "[1/4] Running Rust tests..." -ForegroundColor Yellow
cargo test --lib --quiet 2>&1 | Select-String "test result"
if ($LASTEXITCODE -ne 0) {
    Write-Host "  [FAIL] Rust tests failed" -ForegroundColor Red
    $ErrorCount++
} else {
    Write-Host "  [OK] Rust tests passed" -ForegroundColor Green
}

# 2. Python Training Optimization Tests
Write-Host ""
Write-Host "[2/4] Running training optimization tests..." -ForegroundColor Yellow
.\venv_windows\Scripts\python.exe tests\test_training_optimizations.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "  [FAIL] Training tests failed" -ForegroundColor Red
    $ErrorCount++
} else {
    Write-Host "  [OK] Training tests passed" -ForegroundColor Green
}

# 3. Manifest Validation
Write-Host ""
Write-Host "[3/4] Validating expert manifests..." -ForegroundColor Yellow

$experts = @("expert-sql", "expert-json", "expert-typescript", "expert-neo4j")
foreach ($expert in $experts) {
    $path = "..\experts\$expert"
    if (Test-Path $path) {
        .\target\release\expert-cli.exe validate --expert $path 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  [OK] $expert validated" -ForegroundColor Green
        } else {
            Write-Host "  [FAIL] $expert validation failed" -ForegroundColor Red
            $ErrorCount++
        }
    }
}

# 4. Build Test
Write-Host ""
Write-Host "[4/4] Testing build..." -ForegroundColor Yellow
cargo build --release --quiet 2>&1 | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-Host "  [OK] Build successful" -ForegroundColor Green
} else {
    Write-Host "  [FAIL] Build failed" -ForegroundColor Red
    $ErrorCount++
}

# Summary
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
if ($ErrorCount -eq 0) {
    Write-Host "  [OK] ALL TESTS PASSED" -ForegroundColor Green -BackgroundColor Black
} else {
    Write-Host "  [FAIL] $ErrorCount test(s) failed" -ForegroundColor Red
}
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

exit $ErrorCount

