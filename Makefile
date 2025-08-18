# Class Transcriber Makefile
# Provides convenient commands for common development tasks

.PHONY: help install test clean lint format setup dev

# Default target
help:
	@echo "Class Transcriber - Available commands:"
	@echo ""
	@echo "  setup      - Run initial project setup"
	@echo "  install    - Install dependencies and setup environment"
	@echo "  test       - Run all tests with coverage"
	@echo "  lint       - Run code linting"
	@echo "  format     - Format code with black"
	@echo "  clean      - Clean up temporary files and cache"
	@echo "  dev        - Setup development environment"
	@echo "  run        - Run the transcriber (requires additional args)"
	@echo ""
	@echo "Examples:"
	@echo "  make setup"
	@echo "  make test"
	@echo "  make run ARGS='--help'"

# Initial project setup
setup: install
	@echo "🎉 Project setup complete!"

# Install dependencies and setup environment
install:
	@echo "🚀 Installing Class Transcriber..."
	@bash scripts/install.sh

# Run tests
test:
	@echo "🧪 Running tests..."
	@bash scripts/run_tests.sh

# Code linting
lint:
	@echo "🔍 Running linting..."
	@if command -v flake8 >/dev/null 2>&1; then \
		flake8 src/ tests/; \
	else \
		echo "flake8 not found. Installing..."; \
		pip install flake8; \
		flake8 src/ tests/; \
	fi

# Format code
format:
	@echo "🎨 Formatting code..."
	@if command -v black >/dev/null 2>&1; then \
		black src/ tests/; \
	else \
		echo "black not found. Installing..."; \
		pip install black; \
		black src/ tests/; \
	fi

# Clean up temporary files
clean:
	@echo "🧹 Cleaning up..."
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name ".coverage" -delete 2>/dev/null || true
	@find . -type f -name "*.log" -delete 2>/dev/null || true
	@rm -rf output/*.pdf output/*.csv output/*.json 2>/dev/null || true
	@echo "✅ Cleanup complete!"

# Development environment setup
dev: install
	@echo "🛠️  Setting up development environment..."
	@pip install pytest pytest-cov black flake8 mypy
	@echo "✅ Development environment ready!"

# Run the transcriber
run:
	@if [ -z "$(ARGS)" ]; then \
		echo "❌ Please provide arguments: make run ARGS='--help'"; \
		exit 1; \
	fi
	@source venv/bin/activate && python src/main.py $(ARGS)