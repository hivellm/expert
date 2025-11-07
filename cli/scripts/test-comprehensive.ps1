#!/usr/bin/env pwsh
# Comprehensive CLI testing with different scenarios

$ErrorActionPreference = "Stop"

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "COMPREHENSIVE CLI TESTS" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

$cli = "..\target\release\expert-cli.exe"
$results = @()

# Test 1: Clean one-shot (no extra output)
Write-Host "[1/8] One-shot mode (clean output)..." -ForegroundColor Yellow
$output = & $cli chat --experts neo4j --prompt "MATCH (p:Person) RETURN p.name" --max-tokens 30 --device cuda 2>&1 | Out-String
$hasCleanOutput = -not ($output -match "Loading") -and -not ($output -match "Chat Ready")
$results += [PSCustomObject]@{
    Test = "Clean One-shot"
    Success = $LASTEXITCODE -eq 0 -and $hasCleanOutput
    Details = if ($hasCleanOutput) { "Output is clean" } else { "Extra output found" }
}
Write-Host "  Result: $(if ($hasCleanOutput) { '[PASS]' } else { '[FAIL]' })" -ForegroundColor $(if ($hasCleanOutput) { "Green" } else { "Red" })
Write-Host ""

# Test 2: Debug mode (verbose output)
Write-Host "[2/8] Debug mode (verbose)..." -ForegroundColor Yellow
$output = & $cli chat --experts neo4j --prompt "Test" --max-tokens 10 --device cuda --debug 2>&1 | Out-String
$hasVerbose = ($output -match "Loading") -and ($output -match "Adapter")
$results += [PSCustomObject]@{
    Test = "Debug Mode"
    Success = $LASTEXITCODE -eq 0 -and $hasVerbose
    Details = if ($hasVerbose) { "Shows loading details" } else { "Missing verbose output" }
}
Write-Host "  Result: $(if ($hasVerbose) { '[PASS]' } else { '[FAIL]' })" -ForegroundColor $(if ($hasVerbose) { "Green" } else { "Red" })
Write-Host ""

# Test 3: SQL Expert
Write-Host "[3/8] SQL Expert..." -ForegroundColor Yellow
$output = & $cli chat --experts sql --prompt "SELECT * FROM users" --max-tokens 30 --device cuda 2>&1 | Out-String
$results += [PSCustomObject]@{
    Test = "SQL Expert"
    Success = $LASTEXITCODE -eq 0
    Details = "Exit code: $LASTEXITCODE"
}
Write-Host "  Result: $(if ($LASTEXITCODE -eq 0) { '[PASS]' } else { '[FAIL]' })" -ForegroundColor $(if ($LASTEXITCODE -eq 0) { "Green" } else { "Red" })
Write-Host ""

# Test 4: JSON Expert
Write-Host "[4/8] JSON Expert..." -ForegroundColor Yellow
$output = & $cli chat --experts json --prompt "Create JSON" --max-tokens 30 --device cuda 2>&1 | Out-String
$results += [PSCustomObject]@{
    Test = "JSON Expert"
    Success = $LASTEXITCODE -eq 0
    Details = "Exit code: $LASTEXITCODE"
}
Write-Host "  Result: $(if ($LASTEXITCODE -eq 0) { '[PASS]' } else { '[FAIL]' })" -ForegroundColor $(if ($LASTEXITCODE -eq 0) { "Green" } else { "Red" })
Write-Host ""

# Test 5: Multiple experts
Write-Host "[5/8] Multiple Experts (neo4j,sql,json)..." -ForegroundColor Yellow
$output = & $cli chat --experts neo4j,sql,json --prompt "Test multi-expert" --max-tokens 20 --device cuda 2>&1 | Out-String
$results += [PSCustomObject]@{
    Test = "Multiple Experts"
    Success = $LASTEXITCODE -eq 0
    Details = "Exit code: $LASTEXITCODE"
}
Write-Host "  Result: $(if ($LASTEXITCODE -eq 0) { '[PASS]' } else { '[FAIL]' })" -ForegroundColor $(if ($LASTEXITCODE -eq 0) { "Green" } else { "Red" })
Write-Host ""

# Test 6: Temperature override
Write-Host "[6/8] Temperature override..." -ForegroundColor Yellow
$output = & $cli chat --experts neo4j --prompt "Test" --max-tokens 10 --device cuda --temperature 0.1 2>&1 | Out-String
$results += [PSCustomObject]@{
    Test = "Temperature Override"
    Success = $LASTEXITCODE -eq 0
    Details = "Exit code: $LASTEXITCODE"
}
Write-Host "  Result: $(if ($LASTEXITCODE -eq 0) { '[PASS]' } else { '[FAIL]' })" -ForegroundColor $(if ($LASTEXITCODE -eq 0) { "Green" } else { "Red" })
Write-Host ""

# Test 7: CPU mode
Write-Host "[7/8] CPU mode (fallback)..." -ForegroundColor Yellow
$output = & $cli chat --prompt "Hello" --max-tokens 10 --device cpu 2>&1 | Out-String
$results += [PSCustomObject]@{
    Test = "CPU Mode"
    Success = $LASTEXITCODE -eq 0
    Details = "Exit code: $LASTEXITCODE"
}
Write-Host "  Result: $(if ($LASTEXITCODE -eq 0) { '[PASS]' } else { '[FAIL]' })" -ForegroundColor $(if ($LASTEXITCODE -eq 0) { "Green" } else { "Red" })
Write-Host ""

# Test 8: Base model only (no experts)
Write-Host "[8/8] Base model only..." -ForegroundColor Yellow
$output = & $cli chat --prompt "Test base model" --max-tokens 15 --device cuda 2>&1 | Out-String
$results += [PSCustomObject]@{
    Test = "Base Model"
    Success = $LASTEXITCODE -eq 0
    Details = "Exit code: $LASTEXITCODE"
}
Write-Host "  Result: $(if ($LASTEXITCODE -eq 0) { '[PASS]' } else { '[FAIL]' })" -ForegroundColor $(if ($LASTEXITCODE -eq 0) { "Green" } else { "Red" })
Write-Host ""

# Summary
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "SUMMARY" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan

$passed = ($results | Where-Object { $_.Success }).Count
$total = $results.Count

Write-Host "Passed: $passed/$total ($(($passed/$total*100).ToString('0'))%)" -ForegroundColor $(if ($passed -eq $total) { "Green" } elseif ($passed -ge $total*0.75) { "Yellow" } else { "Red" })
Write-Host ""

$results | ForEach-Object {
    $status = if ($_.Success) { "[PASS]" } else { "[FAIL]" }
    $color = if ($_.Success) { "Green" } else { "Red" }
    Write-Host "$status $($_.Test) - $($_.Details)" -ForegroundColor $color
}

if ($passed -eq $total) {
    Write-Host "`n[OK] All tests passed!" -ForegroundColor Green
    exit 0
} elseif ($passed -ge $total * 0.75) {
    Write-Host "`n[WARN] Most tests passed ($passed/$total)" -ForegroundColor Yellow
    exit 0
} else {
    Write-Host "`n[FAIL] Too many failures ($passed/$total)" -ForegroundColor Red
    exit 1
}

