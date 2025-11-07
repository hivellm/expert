#!/usr/bin/env pwsh
# Test CLI one-shot mode with different experts

$ErrorActionPreference = "Stop"

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "CLI ONE-SHOT MODE TESTS" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

$cli = "..\target\release\expert-cli.exe"
$results = @()

# Test 1: Neo4j expert
Write-Host "[1/4] Testing Neo4j Expert..." -ForegroundColor Yellow
$output = & $cli chat --experts neo4j --prompt "Find all actors" --max-tokens 50 --device cuda 2>&1
$results += [PSCustomObject]@{
    Test = "Neo4j"
    Success = $LASTEXITCODE -eq 0
    Output = $output -join "`n"
}
if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] Neo4j test passed" -ForegroundColor Green
    Write-Host "Output: $($output | Select-Object -First 1)" -ForegroundColor Gray
} else {
    Write-Host "[FAIL] Neo4j test failed" -ForegroundColor Red
}
Write-Host ""

# Test 2: SQL expert
Write-Host "[2/4] Testing SQL Expert..." -ForegroundColor Yellow
$output = & $cli chat --experts sql --prompt "Find all users" --max-tokens 50 --device cuda 2>&1
$results += [PSCustomObject]@{
    Test = "SQL"
    Success = $LASTEXITCODE -eq 0
    Output = $output -join "`n"
}
if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] SQL test passed" -ForegroundColor Green
    Write-Host "Output: $($output | Select-Object -First 1)" -ForegroundColor Gray
} else {
    Write-Host "[FAIL] SQL test failed" -ForegroundColor Red
}
Write-Host ""

# Test 3: Multiple experts (neo4j + sql)
Write-Host "[3/4] Testing Multiple Experts (neo4j,sql)..." -ForegroundColor Yellow
$output = & $cli chat --experts neo4j,sql --prompt "Find all movies" --max-tokens 50 --device cuda 2>&1
$results += [PSCustomObject]@{
    Test = "Multiple"
    Success = $LASTEXITCODE -eq 0
    Output = $output -join "`n"
}
if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] Multiple experts test passed" -ForegroundColor Green
    Write-Host "Output: $($output | Select-Object -First 1)" -ForegroundColor Gray
} else {
    Write-Host "[FAIL] Multiple experts test failed" -ForegroundColor Red
}
Write-Host ""

# Test 4: Debug mode
Write-Host "[4/4] Testing Debug Mode..." -ForegroundColor Yellow
$output = & $cli chat --experts neo4j --prompt "Test query" --max-tokens 20 --device cuda --debug 2>&1
$results += [PSCustomObject]@{
    Test = "Debug"
    Success = $LASTEXITCODE -eq 0 -and ($output -match "Loading")
    Output = $output -join "`n"
}
if ($LASTEXITCODE -eq 0 -and ($output -match "Loading")) {
    Write-Host "[OK] Debug mode shows loading info" -ForegroundColor Green
} else {
    Write-Host "[FAIL] Debug mode test failed" -ForegroundColor Red
}
Write-Host ""

# Summary
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "SUMMARY" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan

$passed = ($results | Where-Object { $_.Success }).Count
$total = $results.Count

Write-Host "Passed: $passed/$total" -ForegroundColor $(if ($passed -eq $total) { "Green" } else { "Yellow" })
Write-Host ""

foreach ($result in $results) {
    $status = if ($result.Success) { "[PASS]" } else { "[FAIL]" }
    $color = if ($result.Success) { "Green" } else { "Red" }
    Write-Host "$status $($result.Test)" -ForegroundColor $color
}

if ($passed -eq $total) {
    Write-Host "`n[OK] All tests passed!" -ForegroundColor Green
    exit 0
} else {
    Write-Host "`n[FAIL] Some tests failed" -ForegroundColor Red
    exit 1
}

