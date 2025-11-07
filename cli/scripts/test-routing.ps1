#!/usr/bin/env pwsh
# Test intelligent routing: generic queries vs specialized queries

$ErrorActionPreference = "Stop"

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "INTELLIGENT ROUTING TEST" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

$cli = "..\target\release\expert-cli.exe"

# Test cases: mix of generic and specialized queries
$tests = @(
    @{
        Name = "Generic - Geography"
        Prompt = "What is the capital of France?"
        Expert = "neo4j"
        ExpectedBehavior = "base"
        ShouldContain = "Paris"
        ShouldNotContain = "MATCH|<|end|>"
    },
    @{
        Name = "Generic - Explanation"
        Prompt = "Explain what SQL is"
        Expert = "sql"
        ExpectedBehavior = "base"
        ShouldContain = "database|language"
        ShouldNotContain = "SELECT|<|end|>"
    },
    @{
        Name = "Specialized - SQL Query"
        Prompt = "Find all users older than 30"
        Expert = "sql"
        ExpectedBehavior = "expert"
        ShouldContain = "SELECT"
        ShouldNotContain = "<|end|>|<|endoftext|>"
    },
    @{
        Name = "Specialized - Cypher Query"
        Prompt = "MATCH all people with age > 25"
        Expert = "neo4j"
        ExpectedBehavior = "expert"
        ShouldContain = "MATCH"
        ShouldNotContain = "<|end|>|<|endoftext|>"
    },
    @{
        Name = "Specialized - Database Keywords"
        Prompt = "SELECT name FROM users WHERE active = true"
        Expert = "sql"
        ExpectedBehavior = "expert"
        ShouldContain = "SELECT|FROM|WHERE"
        ShouldNotContain = "<|end|>"
    },
    @{
        Name = "Generic - How-to"
        Prompt = "How to connect to a database?"
        Expert = "sql"
        ExpectedBehavior = "base"
        ShouldContain = "connect"
        ShouldNotContain = "SELECT|<|end|>"
    }
)

$passed = 0
$failed = 0

foreach ($test in $tests) {
    Write-Host "================================================================================" -ForegroundColor Yellow
    Write-Host "$($test.Name)" -ForegroundColor Yellow
    Write-Host "================================================================================" -ForegroundColor Yellow
    Write-Host "Prompt: '$($test.Prompt)'" -ForegroundColor Gray
    Write-Host "Expert: $($test.Expert)" -ForegroundColor Gray
    Write-Host "Expected: Use $($test.ExpectedBehavior) model" -ForegroundColor Gray
    Write-Host ""
    
    # Run with debug to see which model was used
    $debugOutput = & $cli chat --experts $test.Expert --prompt $test.Prompt --max-tokens 40 --device cuda --temperature 0.1 --debug 2>&1 | Out-String
    
    # Extract just the response (without debug info)
    $output = & $cli chat --experts $test.Expert --prompt $test.Prompt --max-tokens 40 --device cuda --temperature 0.1 2>&1 | Out-String
    $output = $output.Trim()
    
    Write-Host "Output: $($output.Substring(0, [Math]::Min(150, $output.Length)))..." -ForegroundColor White
    Write-Host ""
    
    # Validate behavior
    $usedExpert = $debugOutput -match "Using expert:"
    $usedBase = $debugOutput -match "Using base model \(generic query\)"
    
    $behaviorCorrect = $false
    if ($test.ExpectedBehavior -eq "expert" -and $usedExpert) {
        $behaviorCorrect = $true
    } elseif ($test.ExpectedBehavior -eq "base" -and $usedBase) {
        $behaviorCorrect = $true
    }
    
    # Check content
    $containsExpected = $output -match $test.ShouldContain
    $lacksUnwanted = -not ($output -match $test.ShouldNotContain)
    
    # Overall pass/fail
    if ($behaviorCorrect -and $containsExpected -and $lacksUnwanted) {
        Write-Host "[PASS]" -ForegroundColor Green
        Write-Host "  Routing: Correct ($($test.ExpectedBehavior))" -ForegroundColor Green
        Write-Host "  Content: Valid" -ForegroundColor Green
        Write-Host "  Clean: No ChatML artifacts" -ForegroundColor Green
        $passed++
    } else {
        Write-Host "[FAIL]" -ForegroundColor Red
        if (-not $behaviorCorrect) {
            Write-Host "  Routing: WRONG (expected $($test.ExpectedBehavior), got $(if ($usedExpert) {'expert'} else {'base'}))" -ForegroundColor Red
        }
        if (-not $containsExpected) {
            Write-Host "  Content: Missing expected keywords" -ForegroundColor Red
        }
        if (-not $lacksUnwanted) {
            Write-Host "  Clean: Contains ChatML artifacts" -ForegroundColor Red
        }
        $failed++
    }
    
    Write-Host ""
}

# Summary
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "SUMMARY" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
$total = $tests.Count
Write-Host "Passed: $passed/$total" -ForegroundColor $(if ($passed -eq $total) { "Green" } elseif ($passed -ge $total*0.8) { "Yellow" } else { "Red" })
Write-Host "Failed: $failed/$total" -ForegroundColor $(if ($failed -eq 0) { "Green" } else { "Red" })
Write-Host ""

if ($passed -eq $total) {
    Write-Host "[OK] Routing working perfectly!" -ForegroundColor Green
    Write-Host "  - Generic queries use base model" -ForegroundColor Green
    Write-Host "  - Specialized queries use correct expert" -ForegroundColor Green
    Write-Host "  - Output is clean (no ChatML artifacts)" -ForegroundColor Green
    exit 0
} elseif ($passed -ge $total * 0.8) {
    Write-Host "[PARTIAL] Most tests passed" -ForegroundColor Yellow
    exit 0
} else {
    Write-Host "[FAIL] Routing needs improvement" -ForegroundColor Red
    exit 1
}

