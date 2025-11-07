#!/usr/bin/env pwsh
# Test if adapters are actually being applied by comparing base vs expert outputs

$ErrorActionPreference = "Stop"

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "ADAPTER IMPACT TEST - Base Model vs Expert" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

$cli = "..\target\release\expert-cli.exe"

# Test prompts designed to show expert behavior
$tests = @(
    @{
        Name = "Neo4j - Simple MATCH"
        Prompt = "Schema: Person(name, age). Query: Find all people older than 30"
        Expert = "neo4j"
        ExpectedKeywords = @("MATCH", "WHERE", "age", "30")
    },
    @{
        Name = "Neo4j - Relationship"
        Prompt = "Schema: Person, Movie, Relationship: ACTED_IN. Query: Find actors and movies"
        Expert = "neo4j"
        ExpectedKeywords = @("MATCH", "ACTED_IN", "RETURN")
    },
    @{
        Name = "SQL - Simple SELECT"
        Prompt = "Schema: users(id, name, age). Query: Find users older than 25"
        Expert = "sql"
        ExpectedKeywords = @("SELECT", "FROM", "WHERE", "age")
    },
    @{
        Name = "SQL - JOIN"
        Prompt = "Schema: orders, customers. Query: Join orders with customers"
        Expert = "sql"
        ExpectedKeywords = @("SELECT", "JOIN", "FROM")
    }
)

foreach ($test in $tests) {
    Write-Host "================================================================================" -ForegroundColor Yellow
    Write-Host "$($test.Name)" -ForegroundColor Yellow
    Write-Host "================================================================================" -ForegroundColor Yellow
    Write-Host "Prompt: $($test.Prompt)" -ForegroundColor Gray
    Write-Host ""
    
    # Test with BASE MODEL
    Write-Host "[BASE MODEL]" -ForegroundColor Blue
    $baseOutput = & $cli chat --prompt $test.Prompt --max-tokens 50 --device cuda 2>&1 | Out-String
    Write-Host $baseOutput.Trim().Substring(0, [Math]::Min(200, $baseOutput.Trim().Length))
    Write-Host ""
    
    # Test with EXPERT
    Write-Host "[EXPERT: $($test.Expert)]" -ForegroundColor Cyan
    $expertOutput = & $cli chat --experts $test.Expert --prompt $test.Prompt --max-tokens 50 --device cuda 2>&1 | Out-String
    Write-Host $expertOutput.Trim().Substring(0, [Math]::Min(200, $expertOutput.Trim().Length))
    Write-Host ""
    
    # Compare outputs
    $areDifferent = $baseOutput -ne $expertOutput
    $hasKeywords = $true
    $missingKeywords = @()
    
    foreach ($keyword in $test.ExpectedKeywords) {
        if ($expertOutput -notmatch $keyword) {
            $hasKeywords = $false
            $missingKeywords += $keyword
        }
    }
    
    # Analysis
    Write-Host "[ANALYSIS]" -ForegroundColor Magenta
    if ($areDifferent) {
        Write-Host "  Outputs are DIFFERENT" -ForegroundColor Green
    } else {
        Write-Host "  Outputs are IDENTICAL (adapter not applied!)" -ForegroundColor Red
    }
    
    if ($hasKeywords) {
        Write-Host "  All expected keywords found" -ForegroundColor Green
    } else {
        Write-Host "  Missing keywords: $($missingKeywords -join ', ')" -ForegroundColor Yellow
    }
    
    Write-Host ""
}

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "CONCLUSION" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "If all outputs are IDENTICAL between base and expert:" -ForegroundColor Yellow
Write-Host "  -> Adapters are being LOADED but NOT APPLIED to weights" -ForegroundColor Yellow
Write-Host "  -> Need to implement adapter merging in Rust" -ForegroundColor Yellow
Write-Host ""
Write-Host "If outputs are DIFFERENT:" -ForegroundColor Green
Write-Host "  -> Adapters are working correctly!" -ForegroundColor Green
Write-Host ""

