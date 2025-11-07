# Compare Rust vs Python inference with identical prompts
# Tests both implementations with same parameters

param(
    [string]$Prompt = "The capital of Brazil is",
    [int]$MaxTokens = 50
)

Write-Host ""
Write-Host "=" -NoNewline; Write-Host ("=" * 100)
Write-Host "RUST vs PYTHON INFERENCE COMPARISON"
Write-Host "=" -NoNewline; Write-Host ("=" * 100)
Write-Host ""
Write-Host "Prompt: '$Prompt'"
Write-Host "Max tokens: $MaxTokens"
Write-Host "Temperature: 0.7"
Write-Host "Top-p: 0.9"
Write-Host ""

# Test 1: Python/Transformers
Write-Host "=" -NoNewline; Write-Host ("=" * 100) -ForegroundColor Yellow
Write-Host "[1/2] PYTHON/TRANSFORMERS (Reference Implementation)" -ForegroundColor Yellow
Write-Host "=" -NoNewline; Write-Host ("=" * 100) -ForegroundColor Yellow
Write-Host ""

.\venv_windows\Scripts\python.exe scripts\compare_inference.py $Prompt $MaxTokens

Write-Host ""
Write-Host "=" -NoNewline; Write-Host ("=" * 100) -ForegroundColor Cyan
Write-Host "[2/2] RUST/CANDLE (Our Implementation)" -ForegroundColor Cyan
Write-Host "=" -NoNewline; Write-Host ("=" * 100) -ForegroundColor Cyan
Write-Host ""

# Run Rust test and capture output
$rustOutput = .\target\release\expert-cli.exe chat --prompt $Prompt 2>&1 | Out-String

# Extract relevant parts
$lines = $rustOutput -split "`n"
$inGeneration = $false
$generation = ""

foreach ($line in $lines) {
    if ($line -match "Prompt tokens:") {
        Write-Host $line.Trim()
    }
    if ($line -match "Starting generation") {
        Write-Host $line.Trim()
        Write-Host ""
        Write-Host "Output: " -NoNewline
        $inGeneration = $true
        continue
    }
    if ($inGeneration -and $line -match "Assistant:") {
        Write-Host $generation
        Write-Host ""
        break
    }
    if ($inGeneration -and $line.Trim() -ne "" -and $line -notmatch "base>" -and $line -notmatch "Thinking") {
        $generation += $line
    }
}

Write-Host ""
Write-Host "=" -NoNewline; Write-Host ("=" * 100) -ForegroundColor Green
Write-Host "COMPARISON RESULT" -ForegroundColor Green
Write-Host "=" -NoNewline; Write-Host ("=" * 100) -ForegroundColor Green
Write-Host ""
Write-Host "Both implementations should generate similar quality output." -ForegroundColor White
Write-Host "Differences in exact text are expected due to probabilistic sampling." -ForegroundColor Gray
Write-Host ""

