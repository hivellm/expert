#!/usr/bin/env pwsh
# Test if experts maintain generalist capabilities

$ErrorActionPreference = "Stop"

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "GENERALIST CAPABILITIES TEST" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

$cli = "..\target\release\expert-cli.exe"

# Simple prompts that have nothing to do with databases
$tests = @(
    @{
        Prompt = "What is the capital of France?"
        Expert = "neo4j"
    },
    @{
        Prompt = "How many days in a week?"
        Expert = "sql"
    },
    @{
        Prompt = "Write a hello world in Python"
        Expert = "neo4j"
    },
    @{
        Prompt = "What color is the sky?"
        Expert = "json"
    },
    @{
        Prompt = "Count from 1 to 5"
        Expert = "sql"
    }
)

$idx = 1
foreach ($test in $tests) {
    Write-Host "[$idx/$($tests.Count)] Testing: $($test.Prompt)" -ForegroundColor Yellow
    Write-Host "Expert: $($test.Expert)" -ForegroundColor Gray
    Write-Host ""
    
    # Base model
    Write-Host "[BASE]" -ForegroundColor Blue
    $base = & $cli chat --prompt $test.Prompt --max-tokens 40 --device cuda 2>&1 | Out-String
    Write-Host $base.Trim()
    Write-Host ""
    
    # With expert
    Write-Host "[EXPERT]" -ForegroundColor Cyan
    $expert = & $cli chat --experts $test.Expert --prompt $test.Prompt --max-tokens 40 --device cuda 2>&1 | Out-String
    Write-Host $expert.Trim()
    Write-Host ""
    
    # Compare
    if ($base.Trim() -eq $expert.Trim()) {
        Write-Host "IDENTICAL - Adapter not applied" -ForegroundColor Red
    } else {
        Write-Host "DIFFERENT - Adapter may be affecting output" -ForegroundColor Yellow
    }
    
    Write-Host ""
    Write-Host "================================================================================" -ForegroundColor DarkGray
    Write-Host ""
    
    $idx++
}

Write-Host "CONCLUSION:" -ForegroundColor Cyan
Write-Host "- If outputs are IDENTICAL: Adapters loaded but not applied (current state)" -ForegroundColor Yellow
Write-Host "- If outputs are DIFFERENT but still generalist: Adapters working + keeping capabilities" -ForegroundColor Green
Write-Host "- If outputs are nonsense: Adapter breaking the model" -ForegroundColor Red
Write-Host ""

