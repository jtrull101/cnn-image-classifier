# Clean build and test artifacts
# Designed to handle permission errors gracefully

Write-Host "Cleaning build artifacts..." -ForegroundColor Cyan

# Define patterns to clean
$patterns = @(
    '__pycache__',
    '*.pyc',
    '.pytest_cache',
    '.ruff_cache',
    'dist',
    'build',
    '*.egg-info',
    'node_modules'
)

# Clean build artifacts
foreach ($pattern in $patterns) {
    Get-ChildItem -Path . -Include $pattern -Recurse -Force -ErrorAction SilentlyContinue | ForEach-Object {
        try {
            Remove-Item -Path $_.FullName -Force -Recurse -ErrorAction Stop
            Write-Host "Removed: $($_.FullName)" -ForegroundColor Green
        }
        catch {
            Write-Host "Skipped (access denied): $($_.FullName)" -ForegroundColor Yellow
        }
    }
}

Write-Host "Cleaning coverage files..." -ForegroundColor Cyan

# Clean coverage files
Get-ChildItem -Path . -Filter '.coverage*' -File -ErrorAction SilentlyContinue | ForEach-Object {
    try {
        Remove-Item -Path $_.FullName -Force -ErrorAction Stop
        Write-Host "Removed: $($_.FullName)" -ForegroundColor Green
    }
    catch {
        Write-Host "Skipped: $($_.FullName)" -ForegroundColor Yellow
    }
}

Get-ChildItem -Path . -Filter 'coverage.xml' -File -ErrorAction SilentlyContinue | ForEach-Object {
    try {
        Remove-Item -Path $_.FullName -Force -ErrorAction Stop
        Write-Host "Removed: $($_.FullName)" -ForegroundColor Green
    }
    catch {
        Write-Host "Skipped: $($_.FullName)" -ForegroundColor Yellow
    }
}

Get-ChildItem -Path . -Filter 'htmlcov' -Directory -ErrorAction SilentlyContinue | ForEach-Object {
    try {
        Remove-Item -Path $_.FullName -Force -Recurse -ErrorAction Stop
        Write-Host "Removed: $($_.FullName)" -ForegroundColor Green
    }
    catch {
        Write-Host "Skipped: $($_.FullName)" -ForegroundColor Yellow
    }
}

Write-Host "Clean complete!" -ForegroundColor Green

