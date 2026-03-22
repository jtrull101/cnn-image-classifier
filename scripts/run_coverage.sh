#!/usr/bin/env bash
# Script to run tests with coverage and combine results
set -e

echo "Running tests with comprehensive coverage report..."

# Clean up any existing coverage files
echo "Cleaning up old coverage files..."
find . -maxdepth 1 -type f -name '.coverage*' -delete 2>/dev/null || true
rm -rf htmlcov 2>/dev/null || true
rm -f coverage.xml 2>/dev/null || true

# Run pytest with coverage
echo "Running pytest with coverage..."
test_result=0
uv run python -m pytest . -n auto --cov=packages --cov=apps --cov-report=term-missing:skip-covered --cov-report=html --cov-report=xml -v || test_result=$?

# Check for multiple coverage files (from parallel execution)
coverage_files=$(find . -maxdepth 1 -type f -name '.coverage.*' 2>/dev/null | wc -l)
if [ "$coverage_files" -gt 0 ]; then
    echo "Found $coverage_files coverage data files. Combining..."
    uv run python -m coverage combine || echo "Warning: Failed to combine coverage data"
    echo "Coverage data combined successfully!"

    # Regenerate reports after combining
    echo "Regenerating coverage reports..."
    uv run python -m coverage report --skip-covered || true
    uv run python -m coverage html || true
    uv run python -m coverage xml || true
else
    # Check if single .coverage file exists
    if [ -f ".coverage" ]; then
        echo "Single coverage file found (no combining needed)"
    else
        echo "Warning: No coverage data files found"
    fi
fi

# Open HTML report if available
if [ -f "htmlcov/index.html" ]; then
    echo ""
    echo "Coverage HTML report generated: htmlcov/index.html"
fi

exit $test_result

