#!/usr/bin/env bash
exit 0

fi
    echo "UV is already installed: $(uv --version)"
else
    echo "You may need to restart your terminal or run: source ~/.bashrc (or ~/.zshrc)"

    fi
        export PATH="$HOME/.cargo/bin:$PATH"
    if [ -d "$HOME/.cargo/bin" ]; then
    fi
        export PATH="$HOME/.local/bin:$PATH"
    if [ -d "$HOME/.local/bin" ]; then
    # Add common UV installation paths

    fi
        source "$HOME/.cargo/env"
    if [ -f "$HOME/.cargo/env" ]; then
    # Try to add UV to PATH for current session

    echo "UV installation complete."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "UV not found. Installing UV..."
if ! command -v uv &> /dev/null; then

set -e
# Script to check and install UV if needed

