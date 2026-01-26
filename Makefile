.PHONY: ci ci-release ci-msrv ci-quick lint test fmt clippy build-release lint-allowlist lint-budget lint-limits audit-utils lint-string-compares lint-no-panic-prod help

help:
	@echo "Targets:"
	@echo "  make ci            -> fmt + lints + clippy + tests + build --release"
	@echo "  make ci-release    -> ci + test --release"
	@echo "  make ci-msrv       -> ci + MSRV (if rust-version set)"
	@echo "  make ci-quick      -> fmt + lints + tests + build --release (no clippy)"
	@echo "  make lint          -> fmt + lints + clippy"
	@echo "  make lint-budget   -> check budget instrumentation in hotspots"
	@echo "  make lint-limits   -> check presimplify_safe isolation"
	@echo "  make lint-no-panic-prod -> forbid panic! in production code"
	@echo "  make audit-utils   -> show canonical utilities registry + lint check"
	@echo "  make test          -> cargo test (debug) only"
	@echo "  make build-release -> cargo build --release only"
	@echo "  make lint-allowlist-> list remaining #[allow] attributes"

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

# List remaining local #[allow] attributes (technical debt tracking)
lint-allowlist:
	@echo "==> Local #[allow(clippy::...)] in crates:"
	@grep -rn "#\[allow(clippy::" crates/cas_engine/src crates/cas_ast/src 2>/dev/null || echo "  (none found)"
	@echo ""
	@echo "==> Crate-level #![allow] (should be 0):"
	@grep -rn "#!\[allow" crates/*/src/lib.rs crates/*/src/main.rs 2>/dev/null || echo "  ✓ None (clean)"

# Ensure hotspot modules have budget instrumentation (Phase 6)
lint-budget:
	./scripts/lint_budget_enforcement.sh

# Ensure presimplify_safe remains isolated (V1.3)
lint-limits:
	./scripts/lint_limit_presimplify.sh

# Audit canonical utilities (hold, flatten, predicates, builders, traversal)
audit-utils:
	@echo "==> Canonical Utilities Registry"
	@echo ""
	@echo "  hold      : cas_ast::hold::{strip_hold, unwrap_hold, wrap_hold}"
	@echo "  flatten   : cas_ast::views::{AddView, MulView}"
	@echo "  predicates: cas_engine::helpers::{is_zero, is_one, is_negative, get_integer*}"
	@echo "  builders  : cas_ast::views::MulBuilder, Context::build_balanced_mul"
	@echo "  traversal : cas_ast::traversal::{count_all_nodes, count_nodes_matching, count_nodes_and_max_depth}"
	@echo ""
	@echo "==> Running lint check..."
	@./scripts/lint_no_duplicate_utils.sh

# Track FunctionKind migration progress (Phase 2 of interning)
lint-string-compares:
	./scripts/lint_string_compares_progress.sh

# Forbid panic! in production code (lib/bin only, not tests)
# Exceptions must use #[allow(clippy::panic)] with justification comment
# Note: unwrap_used enforcement is Phase 2 (PR 2+ per anti_panic_inventory.md)
lint-no-panic-prod:
	@echo "==> Checking for panic! in production code..."
	@cargo clippy --workspace --lib --bins -- -D clippy::panic 2>&1 || { echo ""; echo "FAIL: Found panic! in production code. Fix or add #[allow(clippy::panic)] with justification."; exit 1; }
	@echo "✓ No panic! in production code"

# Enforce stringly-typed IR helpers (poly_result, __hold, __eq__)
# - poly_result: ENFORCE (0 violations required)
# - __hold: ENFORCE (0 violations required, as of 2026-01)
# - __eq__: ENFORCE (0 violations required, as of 2026-01)
# Canonical modules: cas_ast/src/hold.rs, cas_ast/src/eq.rs, cas_engine/src/poly_result.rs
lint-no-stringly-ir:
	@echo "==> Stringly-typed IR lint"
	@echo ""
	@HOLD_VIOLATIONS=$$(grep -rn '"__hold"' crates/cas_engine/src crates/cas_ast/src 2>/dev/null | grep -v 'hold.rs' | grep -v 'builtin.rs' | grep -v 'tests::' | wc -l | tr -d ' '); \
	POLY_VIOLATIONS=$$(grep -rn '"poly_result"' crates/cas_engine/src 2>/dev/null | grep -v 'poly_result.rs' | grep -v 'poly_store.rs' | grep -v 'tests::' | wc -l | tr -d ' '); \
	EQ_VIOLATIONS=$$(grep -rn '"__eq__"' crates/cas_engine/src crates/cas_ast/src 2>/dev/null | grep -v 'eq.rs' | grep -v 'builtin.rs' | grep -v 'tests::' | grep -v '// ' | wc -l | tr -d ' '); \
	echo "  __hold violations: $$HOLD_VIOLATIONS (enforced: 0)"; \
	echo "  poly_result violations: $$POLY_VIOLATIONS (enforced: 0)"; \
	echo "  __eq__ violations: $$EQ_VIOLATIONS (enforced: 0)"; \
	echo ""; \
	FAILED=0; \
	if [ "$$POLY_VIOLATIONS" != "0" ]; then \
		echo "❌ FAIL: poly_result violations > 0"; \
		echo "   Use helpers: is_poly_result, parse_poly_result_id, wrap_poly_result"; \
		FAILED=1; \
	else \
		echo "✓ poly_result: CLEAN (enforced)"; \
	fi; \
	if [ "$$HOLD_VIOLATIONS" != "0" ]; then \
		echo "❌ FAIL: __hold violations > 0"; \
		echo "   Use helpers: is_hold, unwrap_hold, wrap_hold, is_hold_name"; \
		FAILED=1; \
	else \
		echo "✓ __hold: CLEAN (enforced)"; \
	fi; \
	if [ "$$EQ_VIOLATIONS" != "0" ]; then \
		echo "❌ FAIL: __eq__ violations > 0"; \
		echo "   Use helpers: is_eq_name, wrap_eq, unwrap_eq from cas_ast::eq"; \
		FAILED=1; \
	else \
		echo "✓ __eq__: CLEAN (enforced)"; \
	fi; \
	if [ "$$FAILED" = "1" ]; then exit 1; fi

