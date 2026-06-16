#!/bin/bash

# Setup script for pre-commit hooks in libhmm.
# Run this once after cloning to activate the git hooks.
#
# Requires: Python 3.7+, pip
# Optional: clang-format (for formatting checks)
#
# Usage: bash scripts/setup-pre-commit.sh

set -e

# ─── helpers ─────────────────────────────────────────────────────────────────
info()    { echo "[INFO]  $1"; }
warn()    { echo "[WARN]  $1"; }
error()   { echo "[ERROR] $1"; }
step()    { echo ""; echo "──── $1"; }

# ─── check Python ────────────────────────────────────────────────────────────
step "Checking prerequisites"

if command -v python3 &>/dev/null; then
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON=python
else
    error "Python is not installed. Install Python 3.7+ and try again."
    exit 1
fi
info "Python: $($PYTHON --version)"

# ─── install pre-commit ───────────────────────────────────────────────────────
step "Installing pre-commit"
if $PYTHON -m pre_commit --version &>/dev/null; then
    info "pre-commit already available: $($PYTHON -m pre_commit --version)"
else
    if command -v pre-commit &>/dev/null && ! pre-commit --version &>/dev/null; then
        warn "Ignoring broken pre-commit executable on PATH; installing via $PYTHON."
    fi
    $PYTHON -m pip install --user --upgrade pre-commit
    if ! $PYTHON -m pre_commit --version &>/dev/null; then
        error "pre-commit installed but cannot run via '$PYTHON -m pre_commit'."
        exit 1
    fi
    info "Installed: $($PYTHON -m pre_commit --version)"
fi

# ─── check optional tools ─────────────────────────────────────────────────────
step "Checking optional tools"

if command -v clang-format &>/dev/null; then
    info "clang-format: $(clang-format --version | head -1)"
else
    warn "clang-format not found. Install it for formatting support:"
    warn "  macOS:  brew install clang-format"
    warn "  Ubuntu: apt-get install clang-format"
fi

# ─── install git hooks ────────────────────────────────────────────────────────
step "Installing git hooks"

"$PYTHON" -m pre_commit install
info "Git hooks installed. They will run automatically on 'git commit'."

# ─── done ─────────────────────────────────────────────────────────────────────
step "Setup complete"
echo ""
echo "Useful commands:"
echo "  $PYTHON -m pre_commit run --all-files     # Run all hooks on all files"
echo "  $PYTHON -m pre_commit run --files FILE    # Run hooks on specific files"
echo "  $PYTHON -m pre_commit autoupdate          # Update hook versions"
echo "  git commit --no-verify         # Skip hooks (use sparingly)"
echo ""
info "See .pre-commit-config.yaml for the active hook set."
