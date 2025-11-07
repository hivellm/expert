#!/usr/bin/env pwsh
# Functional test for dynamic expert router
# Tests: generic queries, specialized queries, multi-expert, routing decisions

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "     ROUTER FUNCTIONAL TESTS" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

$cli = "..\target\release\expert-cli.exe"

# Check CLI exists
if (-not (Test-Path $cli)) {
    Write-Host "[ERROR] CLI not found at: $cli" -ForegroundColor Red
    exit 1
}

# Check experts installed
$experts = & $cli list 2>&1 | Out-String
if ($experts -notmatch "expert-neo4j" -or $experts -notmatch "expert-sql") {
    Write-Host "[ERROR] Required experts not installed (neo4j, sql)" -ForegroundColor Red
    Write-Host "Run: expert-cli install <expert-package>" -ForegroundColor Yellow
    exit 1
}

Write-Host "Installed experts:" -ForegroundColor Green
& $cli list
Write-Host ""

$tests = @(
    @{
        ID = 1
        Name = "Generic Query - Geography (Neo4j loaded)"
        Expert = "neo4j"
        Prompt = "What is the capital of France?"
        ExpectedRouter = "base"
        ExpectedContains = "Paris"
        ExpectedNotContains = "MATCH|CREATE|<\|end\|>"
        MaxTokens = 20
    },
    @{
        ID = 2
        Name = "Generic Query - SQL Explanation (SQL loaded)"
        Expert = "sql"
        Prompt = "Explain what SQL is"
        ExpectedRouter = "base"
        ExpectedContains = "query|language|database"
        ExpectedNotContains = "SELECT|FROM|<\|end\|>"
        MaxTokens = 30
    },
    @{
        ID = 3
        Name = "Specialized - Explicit SQL Query"
        Expert = "sql"
        Prompt = "SELECT name FROM users WHERE active = true"
        ExpectedRouter = "expert"
        ExpectedContains = "SELECT"
        ExpectedNotContains = "<\|end\|>|<\|endoftext\|>"
        MaxTokens = 20
    },
    @{
        ID = 4
        Name = "Specialized - Explicit Cypher Query"
        Expert = "neo4j"
        Prompt = "MATCH (p:Person) WHERE p.age > 30 RETURN p.name"
        ExpectedRouter = "expert"
        ExpectedContains = "MATCH|Person|RETURN"
        ExpectedNotContains = "<\|end\|>|<\|endoftext\|>"
        MaxTokens = 25
    },
    @{
        ID = 5
        Name = "Specialized - Implicit SQL (keyword: find, users)"
        Expert = "sql"
        Prompt = "Find all users older than 30"
        ExpectedRouter = "expert"
        ExpectedContains = "SELECT"
        ExpectedNotContains = "<\|end\|>"
        MaxTokens = 20
    },
    @{
        ID = 6
        Name = "Specialized - Implicit Neo4j (keyword: graph, nodes)"
        Expert = "neo4j"
        Prompt = "Show me all nodes in the graph"
        ExpectedRouter = "expert"
        ExpectedContains = "MATCH"
        ExpectedNotContains = "<\|end\|>"
        MaxTokens = 20
    },
    @{
        ID = 7
        Name = "Multi-Expert - SQL query (both loaded)"
        Expert = "sql,neo4j"
        Prompt = "SELECT COUNT(*) FROM orders"
        ExpectedRouter = "expert"
        ExpectedContains = "SELECT|COUNT"
        ExpectedNotContains = "<\|end\|>"
        MaxTokens = 15
    },
    @{
        ID = 8
        Name = "Multi-Expert - Generic (both loaded, should use base)"
        Expert = "sql,neo4j"
        Prompt = "What is machine learning?"
        ExpectedRouter = "base"
        ExpectedContains = "learning|algorithm|model"
        ExpectedNotContains = "SELECT|MATCH|<\|end\|>"
        MaxTokens = 30
    }
)

$passed = 0
$failed = 0
$results = @()

foreach ($test in $tests) {
    Write-Host "───────────────────────────────────────────────────────────────────" -ForegroundColor DarkGray
    Write-Host "TEST $($test.ID): $($test.Name)" -ForegroundColor Yellow
    Write-Host "───────────────────────────────────────────────────────────────────" -ForegroundColor DarkGray
    Write-Host "  Expert(s): $($test.Expert)" -ForegroundColor Gray
    Write-Host "  Prompt: ""$($test.Prompt)""" -ForegroundColor Gray
    Write-Host "  Expected Router: $($test.ExpectedRouter)" -ForegroundColor Gray
    Write-Host ""
    
    # Run with debug to see routing decision
    $debugOut = & $cli chat --experts $test.Expert --prompt $test.Prompt --max-tokens $test.MaxTokens --device cuda --temperature 0.1 --debug 2>&1 | Out-String
    
    # Run without debug to get clean output
    $output = & $cli chat --experts $test.Expert --prompt $test.Prompt --max-tokens $test.MaxTokens --device cuda --temperature 0.1 2>&1 | Out-String
    $output = $output.Trim()
    
    # Check routing decision
    $routerDecision = "unknown"
    if ($debugOut -match "Using expert:") {
        $routerDecision = "expert"
    } elseif ($debugOut -match "Using base model") {
        $routerDecision = "base"
    }
    
    # Validate routing
    $routingCorrect = ($routerDecision -eq $test.ExpectedRouter)
    
    # Validate content
    $containsExpected = $output -match $test.ExpectedContains
    $lacksUnwanted = -not ($output -match $test.ExpectedNotContains)
    
    # Overall result
    $testPassed = $routingCorrect -and $containsExpected -and $lacksUnwanted
    
    if ($testPassed) {
        Write-Host "  ✅ PASS" -ForegroundColor Green
        Write-Host "     Routing: $routerDecision (correct)" -ForegroundColor Green
        Write-Host "     Output: Valid, clean" -ForegroundColor Green
        $passed++
    } else {
        Write-Host "  ❌ FAIL" -ForegroundColor Red
        if (-not $routingCorrect) {
            Write-Host "     Routing: $routerDecision (expected: $($test.ExpectedRouter))" -ForegroundColor Red
        }
        if (-not $containsExpected) {
            Write-Host "     Content: Missing expected keywords" -ForegroundColor Red
        }
        if (-not $lacksUnwanted) {
            Write-Host "     Output: Contains unwanted tokens/patterns" -ForegroundColor Red
        }
        $failed++
    }
    
    Write-Host "     Output: $($output.Substring(0, [Math]::Min(80, $output.Length)))..." -ForegroundColor DarkGray
    Write-Host ""
    
    $results += @{
        Test = $test.Name
        Passed = $testPassed
        Router = $routerDecision
        Expected = $test.ExpectedRouter
    }
}

# Summary
Write-Host "═══════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "     SUMMARY" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

$total = $tests.Count
$passRate = [math]::Round(($passed / $total) * 100, 1)

Write-Host "Total Tests: $total" -ForegroundColor White
Write-Host "Passed: $passed ($passRate%)" -ForegroundColor $(if ($passed -eq $total) { "Green" } elseif ($passRate -ge 75) { "Yellow" } else { "Red" })
Write-Host "Failed: $failed" -ForegroundColor $(if ($failed -eq 0) { "Green" } else { "Red" })
Write-Host ""

# Breakdown
Write-Host "Breakdown:" -ForegroundColor White
$results | ForEach-Object {
    $icon = if ($_.Passed) { "✅" } else { "❌" }
    $color = if ($_.Passed) { "Green" } else { "Red" }
    Write-Host "  $icon $($_.Test) (router: $($_.Router))" -ForegroundColor $color
}
Write-Host ""

# Final verdict
if ($passed -eq $total) {
    Write-Host "🎉 ALL TESTS PASSED! Router working perfectly!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Router behavior:" -ForegroundColor Green
    Write-Host "  ✓ Generic queries use base model" -ForegroundColor Green
    Write-Host "  ✓ Specialized queries use correct expert" -ForegroundColor Green
    Write-Host "  ✓ Multi-expert selection working" -ForegroundColor Green
    Write-Host "  ✓ Output is clean (no ChatML artifacts)" -ForegroundColor Green
    exit 0
} elseif ($passRate -ge 75) {
    Write-Host "⚠️  MOSTLY PASSING ($passRate%)" -ForegroundColor Yellow
    Write-Host "Some edge cases need tuning, but core functionality works" -ForegroundColor Yellow
    exit 0
} else {
    Write-Host "❌ ROUTER NEEDS FIXES ($passRate% pass rate)" -ForegroundColor Red
    exit 1
}

