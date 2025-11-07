# Expert Training Script for Windows PowerShell with CUDA

$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "Expert Training - Windows + CUDA" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Cyan

# Check if venv_windows exists
if (-not (Test-Path "venv_windows\Scripts\Activate.ps1")) {
    Write-Host "`nERROR: Windows virtual environment not found!" -ForegroundColor Red
    Write-Host "Run setup first: .\setup_windows.ps1" -ForegroundColor Yellow
    exit 1
}

# Activate virtual environment
Write-Host "`nActivating virtual environment..." -ForegroundColor Yellow
& ".\venv_windows\Scripts\Activate.ps1"

# Get Python executable path (from venv)
$pythonExe = (Get-Command python).Source
$pythonDir = Split-Path $pythonExe -Parent

# Get the base Python installation (not venv)
# The venv has a pyvenv.cfg file that points to the base installation
$pyvenvCfg = Join-Path (Split-Path $pythonDir -Parent) "pyvenv.cfg"
if (Test-Path $pyvenvCfg) {
    $basePython = (Get-Content $pyvenvCfg | Where-Object { $_ -match "home = (.+)" } | ForEach-Object { $matches[1] })
    if ($basePython) {
        Write-Host "Base Python installation: $basePython" -ForegroundColor Cyan
        # $basePython is the directory (e.g., F:\Python312)
        # This IS the Python home we want
        $pythonHome = $basePython
        $pythonBaseDir = $basePython
    } else {
        # Fallback: assume standard location
        $pythonHome = Split-Path (Split-Path $pythonDir -Parent) -Parent
        $pythonBaseDir = Split-Path $pythonExe -Parent
    }
} else {
    # Not in venv, use current Python
    $pythonHome = Split-Path $pythonDir -Parent
    $pythonBaseDir = $pythonDir
}

# Check if this is Microsoft Store Python
$isMicrosoftStore = $pythonHome -like "*WindowsApps*"

if ($isMicrosoftStore) {
    Write-Host "WARNING: Detected Microsoft Store Python installation" -ForegroundColor Yellow
    Write-Host "Microsoft Store Python may not work correctly with PyO3/Rust integration." -ForegroundColor Yellow
    Write-Host "" -ForegroundColor Yellow
    
    # Try to find the actual Python installation in Program Files
    $storeAppName = Split-Path $pythonHome -Leaf
    $programFilesPath = "C:\Program Files\WindowsApps\$storeAppName"
    
    if (Test-Path $programFilesPath) {
        Write-Host "Found Windows Store package: $programFilesPath" -ForegroundColor Cyan
        $pythonHome = $programFilesPath
        $pythonBaseDir = $programFilesPath
    } else {
        Write-Host "ERROR: Microsoft Store Python is not compatible with this tool." -ForegroundColor Red
        Write-Host "" -ForegroundColor Red
        Write-Host "SOLUTION:" -ForegroundColor Yellow
        Write-Host "1. Download Python from: https://www.python.org/downloads/" -ForegroundColor Yellow
        Write-Host "2. During installation, check 'Add Python to PATH'" -ForegroundColor Yellow
        Write-Host "3. Choose 'Install for all users' if possible" -ForegroundColor Yellow
        Write-Host "4. After installation, close this terminal and run setup again:" -ForegroundColor Yellow
        Write-Host "   .\setup_windows.ps1" -ForegroundColor Cyan
        Write-Host "" -ForegroundColor Red
        exit 1
    }
}

$pythonLibDir = Join-Path $pythonHome "Lib"

# Find Python DLL in base Python directory
Write-Host "Searching for Python DLL..." -ForegroundColor Yellow
$pythonDll = Get-ChildItem -Path $pythonBaseDir -Filter "python*.dll" -ErrorAction SilentlyContinue | Select-Object -First 1

if (-not $pythonDll) {
    # Try parent directory (some installations put DLL in Python root)
    $pythonDll = Get-ChildItem -Path $pythonHome -Filter "python*.dll" -ErrorAction SilentlyContinue | Select-Object -First 1
}

if (-not $pythonDll) {
    # Try System32 (sometimes Windows puts Python DLLs there)
    Write-Host "Checking System32..." -ForegroundColor Yellow
    $pythonDll = Get-ChildItem -Path "$env:SystemRoot\System32" -Filter "python3*.dll" -ErrorAction SilentlyContinue | Select-Object -First 1
}

if (-not $pythonDll) {
    # Last resort: search in common Python installation directories
    Write-Host "Searching common Python installation directories..." -ForegroundColor Yellow
    $commonPaths = @(
        "C:\Python312",
        "C:\Python311",
        "C:\Python310",
        "$env:LOCALAPPDATA\Programs\Python\Python312",
        "$env:LOCALAPPDATA\Programs\Python\Python311",
        "$env:LOCALAPPDATA\Programs\Python\Python310"
    )
    
    foreach ($path in $commonPaths) {
        if (Test-Path $path) {
            $pythonDll = Get-ChildItem -Path $path -Filter "python*.dll" -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($pythonDll) {
                Write-Host "Found Python installation at: $path" -ForegroundColor Green
                $pythonHome = $path
                $pythonBaseDir = $path
                break
            }
        }
    }
}

if ($pythonDll) {
    Write-Host "Python DLL found: $($pythonDll.FullName)" -ForegroundColor Green
    $pythonDllDir = Split-Path $pythonDll.FullName -Parent
} else {
    Write-Host "" -ForegroundColor Red
    Write-Host "ERROR: Python DLL not found!" -ForegroundColor Red
    Write-Host "" -ForegroundColor Red
    Write-Host "This usually happens because:" -ForegroundColor Yellow
    Write-Host "1. Python was installed from Microsoft Store (not compatible)" -ForegroundColor Yellow
    Write-Host "2. Python installation is incomplete or corrupted" -ForegroundColor Yellow
    Write-Host "" -ForegroundColor Red
    Write-Host "SOLUTION:" -ForegroundColor Yellow
    Write-Host "1. Download Python 3.11 or 3.12 from: https://www.python.org/downloads/" -ForegroundColor Yellow
    Write-Host "2. During installation:" -ForegroundColor Yellow
    Write-Host "   - Check 'Add Python to PATH'" -ForegroundColor Yellow
    Write-Host "   - Choose 'Install for all users' (recommended)" -ForegroundColor Yellow
    Write-Host "   - Use 'Customize installation' and ensure 'pip' is selected" -ForegroundColor Yellow
    Write-Host "3. After installation, close this terminal completely" -ForegroundColor Yellow
    Write-Host "4. Open a new PowerShell terminal and run:" -ForegroundColor Yellow
    Write-Host "   cd F:\Node\hivellm\expert\cli" -ForegroundColor Cyan
    Write-Host "   .\setup_windows.ps1" -ForegroundColor Cyan
    Write-Host "" -ForegroundColor Red
    exit 1
}

# Set Python environment variables for PyO3
# CRITICAL: Must point to actual Python installation (e.g., F:\Python312)
$env:PYTHONHOME = $pythonHome
$env:PYTHONPATH = $PSScriptRoot

# Add Python base directory (where DLL is) to PATH - CRITICAL for PyO3
if ($env:PATH -notlike "*$pythonDllDir*") {
    $env:PATH = "$pythonDllDir;$env:PATH"
}
# Add Python Scripts directory to PATH
if ($env:PATH -notlike "*$pythonDir*") {
    $env:PATH = "$pythonDir;$env:PATH"
}
# Add venv Scripts to PATH (for python.exe from venv)
$venvScriptsDir = ".\venv_windows\Scripts"
if ($env:PATH -notlike "*$venvScriptsDir*") {
    $env:PATH = "$PSScriptRoot\venv_windows\Scripts;$env:PATH"
}

Write-Host "`nPython Configuration:" -ForegroundColor Cyan
Write-Host "  Executable: $pythonExe" -ForegroundColor Cyan
Write-Host "  PYTHONHOME: $env:PYTHONHOME" -ForegroundColor Cyan
Write-Host "  PYTHONPATH: $env:PYTHONPATH" -ForegroundColor Cyan
Write-Host "  Python DLL: $($pythonDll.FullName)" -ForegroundColor Cyan

# Check CUDA
Write-Host "`nChecking CUDA..." -ForegroundColor Yellow
python check_cuda.py

# Define paths
$manifestPath = (Resolve-Path (Join-Path $PSScriptRoot "..\experts\expert-json-parser\manifest.json")).Path
$outputPath = Join-Path $PSScriptRoot "..\experts\expert-json-parser\weights"

# Read dataset configuration from manifest
Write-Host "`nReading manifest configuration..." -ForegroundColor Yellow
$manifest = Get-Content $manifestPath -Raw | ConvertFrom-Json
$datasetConfig = $manifest.training.dataset

# Check dataset type
$datasetType = if ($datasetConfig.type) { $datasetConfig.type } else { "single" }

if ($datasetType -eq "multi_task") {
    Write-Host "Dataset type: Multi-Task" -ForegroundColor Cyan
    
    # Validate all task files exist
    $allFilesExist = $true
    $totalExamples = 0
    
    foreach ($taskName in $datasetConfig.tasks.PSObject.Properties.Name) {
        $task = $datasetConfig.tasks.$taskName
        
        Write-Host "`n  Task: $taskName (weight=$($task.weight))" -ForegroundColor Gray
        
        foreach ($split in @("train", "valid", "test")) {
            $filePath = Join-Path $PSScriptRoot "..\experts\expert-json-parser\$($task.$split)"
            
            if (-not (Test-Path $filePath)) {
                Write-Host "    ERROR: Missing $split file: $filePath" -ForegroundColor Red
                $allFilesExist = $false
            } else {
                $lineCount = (Get-Content $filePath | Measure-Object -Line).Lines
                Write-Host "    $split : $lineCount examples" -ForegroundColor Gray
                if ($split -eq "train") {
                    $totalExamples += $lineCount
                }
            }
        }
    }
    
    if (-not $allFilesExist) {
        Write-Host "`nERROR: Missing dataset files" -ForegroundColor Red
        Write-Host "Please generate missing dataset files using premium LLMs (DeepSeek/Claude/GPT-4)" -ForegroundColor Yellow
        exit 1
    }
    
    Write-Host "`n  Total train examples (before weighting): $totalExamples" -ForegroundColor Cyan
    Write-Host "  Validation will run during training" -ForegroundColor Gray
    
    $datasetPath = ""  # Not used for multi-task
    
} else {
    # Single file or HuggingFace
    $datasetPath = $datasetConfig.path
    
    # Check if dataset is HuggingFace or local file
    $isHuggingFace = $datasetPath -notmatch '\.(jsonl|json)$' -and -not (Test-Path $datasetPath)
    
    if ($isHuggingFace) {
        Write-Host "Dataset source: HuggingFace Hub" -ForegroundColor Cyan
        Write-Host "  Dataset: $datasetPath" -ForegroundColor Cyan
        
        # Skip validation for HuggingFace datasets
        Write-Host "  Skipping grammar validation (HuggingFace dataset)" -ForegroundColor Gray
    } else {
        # Local file - resolve path and validate
        Write-Host "Dataset source: Local file" -ForegroundColor Cyan
        
        # If relative path, resolve from expert directory
        if (-not [System.IO.Path]::IsPathRooted($datasetPath)) {
            $datasetPath = (Resolve-Path (Join-Path $PSScriptRoot "..\experts\expert-json-parser\$datasetPath")).Path
        }
        
        Write-Host "  Path: $datasetPath" -ForegroundColor Cyan
        
        # Validate dataset against grammar (if enabled in manifest)
        Write-Host "`nValidating dataset against grammar..." -ForegroundColor Yellow
        python validate_grammar.py $manifestPath $datasetPath
        if ($LASTEXITCODE -ne 0) {
            Write-Host "WARNING: Some examples don't match grammar" -ForegroundColor Yellow
            Write-Host "Training will continue, but may affect output quality" -ForegroundColor Yellow
            Write-Host ""
        } else {
            Write-Host "Dataset validation passed!" -ForegroundColor Green
        }
    }
}

# Run training
Write-Host "`n" + ("=" * 70) -ForegroundColor Green
Write-Host "Starting Training..." -ForegroundColor Green
Write-Host ("=" * 70) -ForegroundColor Green

# Prepare training config for Python trainer
$baseModelPath = $manifest.base_model.name
$quantization = $manifest.base_model.quantization
$trainingConfig = $manifest.training.config

# Build config dictionary for Python
$config = @{
    base_model_name = $baseModelPath
    quantization = $quantization
    dataset_path = $datasetPath
    output_dir = $outputPath
    device = "cuda"
    training = @{
        dataset = $datasetConfig | ConvertTo-Json -Depth 10 | ConvertFrom-Json  # Deep copy
        adapter_type = $trainingConfig.adapter_type
        rank = $trainingConfig.rank
        alpha = $trainingConfig.alpha
        target_modules = $trainingConfig.target_modules
        epochs = $trainingConfig.epochs
        learning_rate = $trainingConfig.learning_rate
        batch_size = $trainingConfig.batch_size
        gradient_accumulation_steps = $trainingConfig.gradient_accumulation_steps
        warmup_steps = $trainingConfig.warmup_steps
        lr_scheduler = $trainingConfig.lr_scheduler
    }
}

# Add optional fields (for backward compatibility)
if ($manifest.training.dataset.text_field) {
    $config.text_field = $manifest.training.dataset.text_field
}
if ($manifest.training.dataset.field_mapping) {
    $config.field_mapping = $manifest.training.dataset.field_mapping
}

# Save config to temporary file
$configPath = Join-Path $PSScriptRoot "temp_train_config.json"
$config | ConvertTo-Json -Depth 10 | Set-Content $configPath

Write-Host "`nExecuting Python trainer..." -ForegroundColor Cyan
Write-Host "  Base model: $baseModelPath" -ForegroundColor Gray
Write-Host "  Dataset: $datasetPath" -ForegroundColor Gray
Write-Host "  Output: $outputPath" -ForegroundColor Gray
Write-Host ""

# Run Python trainer
python expert_trainer.py $configPath

$exitCode = $LASTEXITCODE

# Cleanup temp file
Remove-Item $configPath -ErrorAction SilentlyContinue

if ($exitCode -eq 0) {
    Write-Host "`n" + ("=" * 70) -ForegroundColor Green
    Write-Host "Training Complete!" -ForegroundColor Green
    Write-Host ("=" * 70) -ForegroundColor Green
} else {
    Write-Host "`nTraining failed with exit code: $exitCode" -ForegroundColor Red
    exit $exitCode
}

