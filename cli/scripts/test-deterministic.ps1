#!/usr/bin/env pwsh
# Test with temperature=0 (deterministic) to see if adapter is really being applied

$ErrorActionPreference = "Stop"

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "DETERMINISTIC TEST (temp=0.0) - Base vs Expert" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

$cli = "..\target\release\expert-cli.exe"

# Run same prompt 3x with base, then 3x with expert
# If outputs are consistent within group but different between groups = adapter is applied
# If all 6 are identical = adapter NOT applied

$prompt = "The capital of France is"

Write-Host "Testing with greedy decoding (temp=0.0, deterministic)" -ForegroundColor Yellow
Write-Host "Prompt: '$prompt'" -ForegroundColor Gray
Write-Host ""

# Base model - 3 runs
Write-Host "[BASE MODEL - 3 runs]" -ForegroundColor Blue
$baseOutputs = @()
for ($i = 1; $i -le 3; $i++) {
    $output = & $cli chat --prompt $prompt --max-tokens 20 --device cuda --temperature 0.0 2>&1 | Out-String
    $cleaned = $output.Trim()
    $baseOutputs += $cleaned
    Write-Host "  Run ${i}: $cleaned" -ForegroundColor Gray
}
Write-Host ""

# Expert - 3 runs
Write-Host "[EXPERT (neo4j) - 3 runs]" -ForegroundColor Cyan
$expertOutputs = @()
for ($i = 1; $i -le 3; $i++) {
    $output = & $cli chat --experts neo4j --prompt $prompt --max-tokens 20 --device cuda --temperature 0.0 2>&1 | Out-String
    $cleaned = $output.Trim()
    $expertOutputs += $cleaned
    Write-Host "  Run ${i}: $cleaned" -ForegroundColor Gray
}
Write-Host ""

# Analysis
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "ANALYSIS" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

# Check consistency within groups
$baseConsistent = ($baseOutputs[0] -eq $baseOutputs[1]) -and ($baseOutputs[1] -eq $baseOutputs[2])
$expertConsistent = ($expertOutputs[0] -eq $expertOutputs[1]) -and ($expertOutputs[1] -eq $expertOutputs[2])

Write-Host "Base model consistency: $(if ($baseConsistent) { 'CONSISTENT' } else { 'INCONSISTENT (random!)' })" -ForegroundColor $(if ($baseConsistent) { "Green" } else { "Red" })
Write-Host "Expert consistency: $(if ($expertConsistent) { 'CONSISTENT' } else { 'INCONSISTENT (random!)' })" -ForegroundColor $(if ($expertConsistent) { "Green" } else { "Red" })
Write-Host ""

# Check if base vs expert are different
$different = $baseOutputs[0] -ne $expertOutputs[0]
Write-Host "Base vs Expert: $(if ($different) { 'DIFFERENT' } else { 'IDENTICAL' })" -ForegroundColor $(if ($different) { "Yellow" } else { "Red" })
Write-Host ""

# Conclusion
Write-Host "VERDICT:" -ForegroundColor Cyan
if (-not $baseConsistent -or -not $expertConsistent) {
    Write-Host "[ERROR] Temperature=0 should be deterministic but outputs vary!" -ForegroundColor Red
    Write-Host "This indicates a bug in generation (sampling not respecting temp=0)" -ForegroundColor Red
} elseif ($different) {
    Write-Host "[MAYBE] Outputs differ between base and expert" -ForegroundColor Yellow
    Write-Host "BUT we need to verify adapter is actually applied (not just different seed)" -ForegroundColor Yellow
} else {
    Write-Host "[CONFIRMED] Adapter NOT applied - outputs identical" -ForegroundColor Red
    Write-Host "Adapter loaded but weights not merged into model" -ForegroundColor Red
}
Write-Host ""

