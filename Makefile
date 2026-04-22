.PHONY: fmt lint test build dev clean

# Default target
all: fmt lint build

# Format code
fmt:
	cargo fmt
	uv run ruff format python/ tests/

# Lint and fix
lint:
	cargo clippy --fix --allow-dirty --allow-staged
	uv run ruff check --fix python/ tests/

# Run tests
test:
	cargo test
	uv run pytest tests/

# Build release
build:
	uv run maturin build --release

# Build and install for development
dev:
	uv run maturin develop

# Clean artifacts
clean:
	cargo clean
	rm -rf .ruff_cache
	rm -rf target/
