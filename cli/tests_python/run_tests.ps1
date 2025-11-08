# PowerShell script to run Python tests

param(
    [Parameter(Mandatory=$false)]
    [string]$TestFile = "all",
    [Parameter(Mandatory=$false)]
    [switch]$Coverage = $false,
    [Parameter(Mandatory=$false)]
    [switch]$Verbose = $true
)

Write-Host "Python Tests Runner" -ForegroundColor Cyan
Write-Host "===================" -ForegroundColor Cyan
Write-Host ""

# Find Python venv
$venvPython = "F:\Node\hivellm\expert\cli\venv_windows\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    $venvPython = "python"
    Write-Host "Using system Python: $venvPython" -ForegroundColor Yellow
} else {
    Write-Host "Using venv Python: $venvPython" -ForegroundColor Green
}

Write-Host ""

# Change to cli directory
$cliDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $cliDir

# Build pytest command
$pytestArgs = @()

if ($Verbose) {
    $pytestArgs += "-v"
}

if ($Coverage) {
    $pytestArgs += "--cov=train"
    $pytestArgs += "--cov-report=html"
    $pytestArgs += "--cov-report=term"
}

if ($TestFile -ne "all") {
    $testPath = "tests_python/$TestFile"
    if (-not (Test-Path $testPath)) {
        Write-Host "ERROR: Test file not found: $testPath" -ForegroundColor Red
        exit 1
    }
    $pytestArgs += $testPath
} else {
    $pytestArgs += "tests_python/"
}

# Run tests
Write-Host "Running: pytest $($pytestArgs -join ' ')" -ForegroundColor Cyan
Write-Host ""

& $venvPython -m pytest $pytestArgs

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "All tests passed!" -ForegroundColor Green
    if ($Coverage) {
        Write-Host "Coverage report: htmlcov/index.html" -ForegroundColor Cyan
    }
} else {
    Write-Host ""
    Write-Host "Some tests failed!" -ForegroundColor Red
    exit 1
}

