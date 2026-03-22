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

# Coverage will be added per-target to scope it correctly
$coverageAdded = $false

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
        if ($Coverage) {
            $pytestCmd += " --cov=packages --cov=apps --cov-report=term-missing --cov-report=html --cov-report=xml"
            Write-Host "Coverage enabled for all packages and apps" -ForegroundColor Green
            $coverageAdded = $true
        }
    }
    "config" {
        Write-Host "Running config package tests" -ForegroundColor Cyan
        $pytestCmd += " packages/config/img_classifier_config/tests/ tests/unit/ tests/integration/"
        if ($Coverage) {
            $pytestCmd += " --cov=packages/config/img_classifier_config --cov-report=term-missing --cov-report=html --cov-report=xml"
            Write-Host "Coverage enabled for config package" -ForegroundColor Green
            $coverageAdded = $true
        }
    }
    "data" {
        Write-Host "Running data package tests" -ForegroundColor Cyan
        $pytestCmd += " packages/data/img_classifier_data/tests/ tests/unit/ tests/integration/"
        if ($Coverage) {
            $pytestCmd += " --cov=packages/data/img_classifier_data --cov-report=term-missing --cov-report=html --cov-report=xml"
            Write-Host "Coverage enabled for data package" -ForegroundColor Green
            $coverageAdded = $true
        }
    }
    "models" {
        Write-Host "Running models package tests" -ForegroundColor Cyan
        $pytestCmd += " packages/models/img_classifier_models/tests/ tests/unit/ tests/integration/"
        if ($Coverage) {
            $pytestCmd += " --cov=packages/models/img_classifier_models --cov-report=term-missing --cov-report=html --cov-report=xml"
            Write-Host "Coverage enabled for models package" -ForegroundColor Green
            $coverageAdded = $true
        }
    }
    "training" {
        Write-Host "Running training package tests" -ForegroundColor Cyan
        $pytestCmd += " packages/training/img_classifier_training/tests/ tests/unit/ tests/integration/"
        if ($Coverage) {
            $pytestCmd += " --cov=packages/training/img_classifier_training --cov-report=term-missing --cov-report=html --cov-report=xml"
            Write-Host "Coverage enabled for training package" -ForegroundColor Green
            $coverageAdded = $true
        }
    }
    "utils" {
        Write-Host "Running utils package tests" -ForegroundColor Cyan
        $pytestCmd += " packages/utils/img_classifier_utils/tests/ tests/unit/ tests/integration/"
        if ($Coverage) {
            $pytestCmd += " --cov=packages/utils/img_classifier_utils --cov-report=term-missing --cov-report=html --cov-report=xml"
            Write-Host "Coverage enabled for utils package" -ForegroundColor Green
            $coverageAdded = $true
        }
    }
    "cli" {
        Write-Host "Running CLI package tests" -ForegroundColor Cyan
        $pytestCmd += " packages/cli/img_classifier_cli/tests/ tests/unit/ tests/integration/"
        if ($Coverage) {
            $pytestCmd += " --cov=packages/cli/img_classifier_cli --cov-report=term-missing --cov-report=html --cov-report=xml"
            Write-Host "Coverage enabled for CLI package" -ForegroundColor Green
            $coverageAdded = $true
        }
    }
    "api" {
        Write-Host "Running API app tests" -ForegroundColor Cyan
        $pytestCmd += " apps/api/tests/ tests/unit/ tests/integration/"
        if ($Coverage) {
            $pytestCmd += " --cov=apps/api/img_classifier_api --cov-report=term-missing --cov-report=html --cov-report=xml"
            Write-Host "Coverage enabled for API app" -ForegroundColor Green
            $coverageAdded = $true
        }
    }
    "integration" {
        Write-Host "Running integration tests only" -ForegroundColor Cyan
        $pytestCmd += " tests/integration/"
        if ($Coverage) {
            $pytestCmd += " --cov=packages --cov=apps --cov-report=term-missing --cov-report=html --cov-report=xml"
            Write-Host "Coverage enabled for integration tests" -ForegroundColor Green
            $coverageAdded = $true
        }
    }
    "unit" {
        Write-Host "Running unit tests only" -ForegroundColor Cyan
        $pytestCmd += " tests/unit/"
        if ($Coverage) {
            $pytestCmd += " --cov=packages --cov=apps --cov-report=term-missing --cov-report=html --cov-report=xml"
            Write-Host "Coverage enabled for unit tests" -ForegroundColor Green
            $coverageAdded = $true
        }
    }
    default {
        Write-Host "Unknown target: $Target" -ForegroundColor Red
        Write-Host "Valid targets:" -ForegroundColor Yellow
        Write-Host "  all          - Run all tests" -ForegroundColor Yellow
        Write-Host "  config       - Run config package tests" -ForegroundColor Yellow
        Write-Host "  data         - Run data package tests" -ForegroundColor Yellow
        Write-Host "  models       - Run models package tests" -ForegroundColor Yellow
        Write-Host "  training     - Run training package tests" -ForegroundColor Yellow
        Write-Host "  utils        - Run utils package tests" -ForegroundColor Yellow
        Write-Host "  cli          - Run CLI package tests" -ForegroundColor Yellow
        Write-Host "  api          - Run API app tests" -ForegroundColor Yellow
        Write-Host "  integration  - Run integration tests only" -ForegroundColor Yellow
        Write-Host "  unit         - Run unit tests only" -ForegroundColor Yellow
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

