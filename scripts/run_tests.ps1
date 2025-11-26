# Test Script for Concurrent Pytest Execution
param(
    [string]$Target = "all",
    [switch]$Parallel = $true,
    [int]$Workers = 0,
    [switch]$Coverage = $false,
    [switch]$UnitOnly = $false,
    [switch]$IntegrationOnly = $false,
    [switch]$Verbose = $false,
    [switch]$FailFast = $false,
    [string]$Markers = "",
    [switch]$ShowSlowest = $false
)

Write-Host "Running Tests with Concurrent Pytest" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$pytestCmd = "uv run python -m pytest"

if ($Verbose) {
    $pytestCmd += " -vv"
} else {
    $pytestCmd += " -v"
}

if ($Parallel) {
    if ($Workers -gt 0) {
        $pytestCmd += " -n $Workers"
        Write-Host "Running with $Workers parallel workers" -ForegroundColor Green
    } else {
        $pytestCmd += " -n auto"
        Write-Host "Running with auto-detected parallel workers" -ForegroundColor Green
    }
} else {
    Write-Host "Running sequentially" -ForegroundColor Yellow
}

if ($Coverage) {
    $pytestCmd += " --cov=packages --cov=apps --cov-report=term-missing --cov-report=html --cov-report=xml"
    Write-Host "Coverage reporting enabled" -ForegroundColor Green
}

if ($UnitOnly) {
    $pytestCmd += " -m unit"
    Write-Host "Running unit tests only" -ForegroundColor Blue
} elseif ($IntegrationOnly) {
    $pytestCmd += " -m integration"
    Write-Host "Running integration tests only" -ForegroundColor Blue
} elseif ($Markers -ne "") {
    $pytestCmd += " -m '$Markers'"
    Write-Host "Running tests with markers: $Markers" -ForegroundColor Blue
}

if ($FailFast) {
    $pytestCmd += " -x"
    Write-Host "Fail-fast mode enabled" -ForegroundColor Yellow
}

if ($ShowSlowest) {
    $pytestCmd += " --durations=10"
    Write-Host "Will show 10 slowest tests" -ForegroundColor Magenta
}

switch ($Target) {
    "all" {
        Write-Host "Running all tests" -ForegroundColor Cyan
        $pytestCmd += " tests/"
    }
    "config" {
        Write-Host "Running config tests" -ForegroundColor Cyan
        $pytestCmd += " tests/test_config.py"
    }
    "data" {
        Write-Host "Running data loader tests" -ForegroundColor Cyan
        $pytestCmd += " tests/test_data_loaders.py"
    }
    "models" {
        Write-Host "Running model tests" -ForegroundColor Cyan
        $pytestCmd += " tests/test_models.py"
    }
    "training" {
        Write-Host "Running training tests" -ForegroundColor Cyan
        $pytestCmd += " tests/test_training_pytest.py"
    }
    "utils" {
        Write-Host "Running utils tests" -ForegroundColor Cyan
        $pytestCmd += " tests/test_utils.py"
    }
    "integration" {
        Write-Host "Running integration tests" -ForegroundColor Cyan
        $pytestCmd += " tests/test_integration.py"
    }
    default {
        Write-Host "Unknown target: $Target" -ForegroundColor Red
        Write-Host "Valid targets: all, config, data, models, training, utils, integration" -ForegroundColor Yellow
        exit 1
    }
}

Write-Host ""
Write-Host "Command: $pytestCmd" -ForegroundColor DarkGray
Write-Host ""

Invoke-Expression $pytestCmd
$exitCode = $LASTEXITCODE

Write-Host ""
if ($exitCode -eq 0) {
    Write-Host "All tests passed!" -ForegroundColor Green
} else {
    Write-Host "Some tests failed!" -ForegroundColor Red
}

exit $exitCode

