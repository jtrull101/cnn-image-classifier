# Script to run tests with coverage and combine results
$ErrorActionPreference = "Stop"

Write-Host "Running tests with comprehensive coverage report..." -ForegroundColor Cyan

# Clean up any existing coverage files
Write-Host "Cleaning up old coverage files..." -ForegroundColor Yellow
Get-ChildItem -Path . -Filter '.coverage*' -File -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
if (Test-Path "htmlcov") {
    Remove-Item -Path "htmlcov" -Recurse -Force -ErrorAction SilentlyContinue
}
if (Test-Path "coverage.xml") {
    Remove-Item -Path "coverage.xml" -Force -ErrorAction SilentlyContinue
}

# Run pytest with coverage
Write-Host "Running pytest with coverage..." -ForegroundColor Cyan
$testResult = $LASTEXITCODE
try {
    uv run python -m pytest . -n auto --cov=packages --cov=apps --cov-report=term-missing:skip-covered --cov-report=html --cov-report=xml -v
    $testResult = $LASTEXITCODE
} catch {
    Write-Host "Test execution failed: $_" -ForegroundColor Red
    $testResult = 1
}

# Check for multiple coverage files (from parallel execution)
$coverageFiles = Get-ChildItem -Path . -Filter '.coverage.*' -File -ErrorAction SilentlyContinue
if ($coverageFiles) {
    Write-Host "Found $($coverageFiles.Count) coverage data files. Combining..." -ForegroundColor Yellow
    try {
        uv run python -m coverage combine
        Write-Host "Coverage data combined successfully!" -ForegroundColor Green

        # Regenerate reports after combining
        Write-Host "Regenerating coverage reports..." -ForegroundColor Yellow
        uv run python -m coverage report --skip-covered
        uv run python -m coverage html
        uv run python -m coverage xml
    } catch {
        Write-Host "Warning: Failed to combine coverage data: $_" -ForegroundColor Yellow
    }
} else {
    # Check if single .coverage file exists
    if (Test-Path ".coverage") {
        Write-Host "Single coverage file found (no combining needed)" -ForegroundColor Green
    } else {
        Write-Host "Warning: No coverage data files found" -ForegroundColor Yellow
    }
}

# Display coverage summary location
if (Test-Path "htmlcov/index.html") {
    $htmlPath = (Resolve-Path "htmlcov/index.html").Path
    Write-Host "`nCoverage HTML report: file:///$($htmlPath.Replace('\', '/'))" -ForegroundColor Cyan
}

if (Test-Path "coverage.xml") {
    Write-Host "Coverage XML report: coverage.xml" -ForegroundColor Cyan
}

exit $testResult

