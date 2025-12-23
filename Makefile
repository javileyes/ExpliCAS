.PHONY: ci ci-release ci-msrv ci-quick lint test fmt clippy build-release help

help:
	@echo "Targets:"
	@echo "  make ci          -> fmt + lints + clippy + tests + build --release"
	@echo "  make ci-release  -> ci + test --release"
	@echo "  make ci-msrv     -> ci + MSRV (if rust-version set)"
	@echo "  make ci-quick    -> fmt + lints + tests + build --release (no clippy)"
	@echo "  make lint        -> fmt + lints + clippy"
	@echo "  make test        -> cargo test (debug) only"
	@echo "  make build-release -> cargo build --release only"

ci:
	./scripts/ci.sh --release-build

ci-release:
	./scripts/ci.sh --release-build --release-test

ci-msrv:
	./scripts/ci.sh --msrv --release-build

ci-quick:
	./scripts/ci.sh --quick --release-build

lint:
	./scripts/ci.sh --lint

test:
	./scripts/ci.sh --test

fmt:
	cargo fmt --all -- --check

clippy:
	cargo clippy --workspace --all-targets -- -D warnings

build-release:
	./scripts/ci.sh --no-fmt --no-clippy --no-tests --no-lints --release-build
