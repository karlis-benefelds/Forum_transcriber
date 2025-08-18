#!/bin/bash

# Class Transcriber Test Runner
# This script runs all tests for the Class Transcriber project

set -e

echo "🧪 Running Class Transcriber Tests..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "🔄 Activating virtual environment..."
    source venv/bin/activate
fi

# Check if pytest is installed
if ! python -c "import pytest" 2>/dev/null; then
    echo "📦 Installing pytest..."
    pip install pytest pytest-cov
fi

# Run tests with coverage
echo "🏃 Running tests with coverage..."
python -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term

echo "✅ Tests completed!"
echo "📊 Coverage report available in htmlcov/index.html"