# Copy Python DLLs to target/release after build

$ErrorActionPreference = "Stop"

Write-Host "Copying Python DLLs to expert-cli..." -ForegroundColor Cyan
Write-Host ""

# Find Python - get actual installation directory
$pythonVersion = & python --version 2>&1
$pythonExe = & python -c "import sys; print(sys.executable)" 2>$null

if (-not $pythonExe) {
    Write-Host "ERROR: Python not found!" -ForegroundColor Red
    exit 1
}

# Get the actual Python installation directory (where DLLs are)
$pythonRealDir = & python -c "import sys, os; print(os.path.dirname(sys.executable))" 2>$null

if (-not $pythonRealDir) {
    Write-Host "ERROR: Could not determine Python directory!" -ForegroundColor Red
    exit 1
}

$pythonDir = $pythonRealDir.Trim()

Write-Host "Python: $pythonVersion" -ForegroundColor Green
Write-Host "Executable: $pythonExe" -ForegroundColor Gray
Write-Host "DLL Directory: $pythonDir" -ForegroundColor Gray
Write-Host ""

# Verify directory exists and has DLLs
if (-not (Test-Path $pythonDir)) {
    Write-Host "ERROR: Python directory not found: $pythonDir" -ForegroundColor Red
    exit 1
}

$foundDlls = Get-ChildItem "$pythonDir\python*.dll" -ErrorAction SilentlyContinue
if (-not $foundDlls) {
    Write-Host "WARNING: No python*.dll found in $pythonDir" -ForegroundColor Yellow
    Write-Host "Checking common locations..." -ForegroundColor Yellow
    
    # Try common Python installation paths
    $commonPaths = @(
        "C:\Python312",
        "C:\Python311", 
        "F:\Python312",
        "F:\Python311",
        "$env:LOCALAPPDATA\Programs\Python\Python312",
        "$env:LOCALAPPDATA\Programs\Python\Python311"
    )
    
    foreach ($path in $commonPaths) {
        if (Test-Path $path) {
            $testDlls = Get-ChildItem "$path\python*.dll" -ErrorAction SilentlyContinue
            if ($testDlls) {
                Write-Host "  Found DLLs in: $path" -ForegroundColor Green
                $pythonDir = $path
                break
            }
        }
    }
} else {
    Write-Host "Found Python DLLs:" -ForegroundColor Green
    $foundDlls | ForEach-Object {
        Write-Host "  - $($_.Name)" -ForegroundColor Gray
    }
}

Write-Host ""

# Target directory
$targetDir = Join-Path $PSScriptRoot "target\release"
if (-not (Test-Path $targetDir)) {
    Write-Host "ERROR: target/release directory not found!" -ForegroundColor Red
    Write-Host "Please run: cargo build --release" -ForegroundColor Yellow
    exit 1
}

Write-Host "Target: $targetDir" -ForegroundColor Gray
Write-Host ""

# List of DLLs to copy
$dllsToCopy = @(
    "python3.dll",
    "python312.dll",
    "python311.dll",
    "vcruntime140.dll",
    "vcruntime140_1.dll"
)

$copiedCount = 0

Write-Host "Copying DLLs..." -ForegroundColor Yellow

foreach ($dllName in $dllsToCopy) {
    $sourceDll = Join-Path $pythonDir $dllName
    
    if (Test-Path $sourceDll) {
        $destDll = Join-Path $targetDir $dllName
        Copy-Item -Path $sourceDll -Destination $destDll -Force
        Write-Host "  ✓ Copied: $dllName" -ForegroundColor Green
        $copiedCount++
    }
}

# Also try System32 for vcruntime DLLs
$system32 = "C:\Windows\System32"
foreach ($dllName in @("vcruntime140.dll", "vcruntime140_1.dll")) {
    $sourceDll = Join-Path $system32 $dllName
    $destDll = Join-Path $targetDir $dllName
    
    if ((Test-Path $sourceDll) -and -not (Test-Path $destDll)) {
        Copy-Item -Path $sourceDll -Destination $destDll -Force
        Write-Host "  ✓ Copied: $dllName (from System32)" -ForegroundColor Green
        $copiedCount++
    }
}

Write-Host ""
if ($copiedCount -gt 0) {
    Write-Host "✅ Copied $copiedCount DLL(s) successfully!" -ForegroundColor Green
} else {
    Write-Host "⚠️  No DLLs were copied!" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "DLLs in target/release:" -ForegroundColor Cyan
Get-ChildItem "$targetDir\*.dll" -ErrorAction SilentlyContinue | ForEach-Object {
    $size = "{0:N2} KB" -f ($_.Length / 1KB)
    Write-Host "  - $($_.Name) ($size)" -ForegroundColor White
}

