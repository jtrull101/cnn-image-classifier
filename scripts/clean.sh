#!/usr/bin/env bash
# Clean build and test artifacts

echo "Cleaning build artifacts..."

# Define patterns to clean
patterns=(
    '__pycache__'
    '*.pyc'
    '.pytest_cache'
    '.ruff_cache'
    'dist'
    'build'
    '*.egg-info'
    'node_modules'
)

# Clean build artifacts
for pattern in "${patterns[@]}"; do
    find . -type d -name "$pattern" 2>/dev/null | while read -r dir; do
        rm -rf "$dir" 2>/dev/null && echo "Removed: $dir" || echo "Skipped (access denied): $dir"
    done

    find . -type f -name "$pattern" 2>/dev/null | while read -r file; do
        rm -f "$file" 2>/dev/null && echo "Removed: $file" || echo "Skipped (access denied): $file"
    done
done

echo "Cleaning coverage files..."

# Clean coverage files
find . -maxdepth 1 -type f -name '.coverage*' 2>/dev/null | while read -r file; do
    rm -f "$file" 2>/dev/null && echo "Removed: $file" || echo "Skipped: $file"
done

find . -maxdepth 1 -type f -name 'coverage.xml' 2>/dev/null | while read -r file; do
    rm -f "$file" 2>/dev/null && echo "Removed: $file" || echo "Skipped: $file"
done

# Clean htmlcov directory
if [ -d "htmlcov" ]; then
    rm -rf htmlcov 2>/dev/null && echo "Removed: htmlcov" || echo "Skipped: htmlcov"
fi

echo "Clean complete!"

