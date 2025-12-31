.PHONY: help setup test test-unit test-dispatcher test-blackwell benchmark check-arch clean lint install

# Environment variables
ENV_NAME := moiz-mxfp4
PYTHON := python3
PIP := pip

# Colors for output
GREEN := \033[0;32m
YELLOW := \033[0;33m
NC := \033[0m # No Color

help:
	@echo "$(GREEN)MoizMXFP4 - Available Commands$(NC)"
	@echo ""
	@echo "$(YELLOW)Setup:$(NC)"
	@echo "  make setup          - Create/Update conda environment and install package"
	@echo "  make install        - Install package in editable mode (faster than setup)"
	@echo ""
	@echo "$(YELLOW)Testing:$(NC)"
	@echo "  make test           - Run ALL tests (unit + integration)"
	@echo "  make test-unit      - Run unit tests only (quantizer, modules, utils)"
	@echo "  make test-dispatcher - Test architecture detection and routing"
	@echo "  make test-blackwell - Test Blackwell-specific code (safe on all GPUs)"
	@echo "  make check-arch     - Check detected GPU architecture"
	@echo ""
	@echo "$(YELLOW)Benchmarking:$(NC)"
	@echo "  make benchmark      - Run performance benchmarks"
	@echo "  make benchmark-quick - Quick benchmark (fewer iterations)"
	@echo ""
	@echo "$(YELLOW)Development:$(NC)"
	@echo "  make lint           - Format code (black + isort)"
	@echo "  make clean          - Remove build artifacts and caches"
	@echo ""
	@echo "$(YELLOW)Quick Start:$(NC)"
	@echo "  make setup && make test    # First time setup"
	@echo "  make check-arch            # See what GPU/kernel is detected"
	@echo "  make test-dispatcher       # Verify dispatcher works"
	@echo ""

setup:
	@echo "$(GREEN)Creating/Updating Conda environment: $(ENV_NAME)...$(NC)"
	conda env update --file environment.yml --prune
	@echo "$(GREEN)Installing package in editable mode...$(NC)"
	$(PIP) install -e .
	@echo "$(GREEN)✅ Setup complete!$(NC)"
	@echo ""
	@echo "Next steps:"
	@echo "  1. conda activate $(ENV_NAME)"
	@echo "  2. make check-arch    # Check GPU detection"
	@echo "  3. make test          # Run tests"

install:
	@echo "$(GREEN)Installing package in editable mode...$(NC)"
	$(PIP) install -e .
	@echo "$(GREEN)✅ Installation complete!$(NC)"

# Testing targets
test:
	@echo "$(GREEN)Running ALL tests...$(NC)"
	PYTHONNOUSERSITE=1 pytest tests/ -v --tb=short
	@echo ""
	@echo "$(GREEN)✅ All tests passed!$(NC)"

test-unit:
	@echo "$(GREEN)Running unit tests...$(NC)"
	PYTHONNOUSERSITE=1 pytest tests/test_quantizer.py tests/test_modules.py tests/test_utils.py tests/test_fused.py -v
	@echo "$(GREEN)✅ Unit tests passed!$(NC)"

test-dispatcher:
	@echo "$(GREEN)Testing architecture dispatcher...$(NC)"
	$(PYTHON) tests/test_dispatcher.py
	@echo "$(GREEN)✅ Dispatcher test passed!$(NC)"

test-blackwell:
	@echo "$(GREEN)Testing Blackwell-specific code...$(NC)"
	@echo "$(YELLOW)(This tests the code but won't use native FP4 on non-Blackwell GPUs)$(NC)"
	$(PYTHON) tests/test_blackwell.py
	@echo "$(GREEN)✅ Blackwell code test passed!$(NC)"

check-arch:
	@echo "$(GREEN)Checking GPU architecture and kernel selection...$(NC)"
	@echo ""
	@$(PYTHON) -c "from mxfp4 import print_architecture_info; print_architecture_info()"
	@echo ""

# Benchmarking targets
benchmark:
	@echo "$(GREEN)Running performance benchmarks (100 iterations)...$(NC)"
	@echo "$(YELLOW)This may take a few minutes...$(NC)"
	@$(PYTHON) benchmarks/benchmark_linear.py
	@echo "$(GREEN)✅ Benchmarks complete!$(NC)"

benchmark-quick:
	@echo "$(GREEN)Running quick benchmark (10 iterations)...$(NC)"
	@$(PYTHON) -c "import sys; sys.path.insert(0, 'src'); from benchmarks.benchmark_linear import quick_benchmark; quick_benchmark()"

# Development targets
lint:
	@echo "$(GREEN)Running code formatters...$(NC)"
	black src/ tests/ benchmarks/ dev/
	isort src/ tests/ benchmarks/ dev/
	@echo "$(GREEN)✅ Code formatted!$(NC)"

clean:
	@echo "$(GREEN)Cleaning up build artifacts...$(NC)"
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "*.pyo" -delete 2>/dev/null || true
	find . -name "*~" -delete 2>/dev/null || true
	find . -name "*.so" -delete 2>/dev/null || true
	@echo "$(GREEN)✅ Cleanup complete!$(NC)"

# Meta target
.DEFAULT_GOAL := help
