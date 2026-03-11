.PHONY: ci ci-release ci-msrv ci-quick lint test fmt clippy build-release lint-allowlist lint-budget lint-limits audit-utils lint-string-compares lint-no-panic-prod bench-clean bench-engine-fast bench-engine-fast-save bench-engine-fast-compare bench-engine-fast-save-seq bench-engine-fast-compare-seq bench-engine-solve-batches bench-engine-solve-batches-save bench-engine-solve-batches-compare bench-engine-solve-hotspots-save bench-engine-solve-hotspots-compare bench-engine-solve-profile bench-engine-repl-breakdown bench-engine-repl-individual bench-engine-repl-individual-save bench-engine-repl-individual-compare bench-engine-repl-hotspots bench-engine-repl-hotspots-save bench-engine-repl-hotspots-compare bench-engine-standard-phase-subset bench-engine-root-direct bench-parser-frontend bench-parser-frontend-save bench-parser-frontend-compare bench-formatter-frontend bench-formatter-frontend-save bench-formatter-frontend-compare help

SOLVE_BATCH_FILTER = solve_modes_cached/(solve_tactic_generic_batch|solve_tactic_assume_batch)

SOLVE_HOTSPOT_FILTERS = \
	solve_hotspots_cached/generic/difference_of_squares_fraction \
	solve_hotspots_cached/generic/power_quotient_fraction \
	solve_hotspots_cached/generic/binomial_square_fraction \
	solve_eval_hotspots_cached/generic/difference_of_squares_fraction \
	solve_eval_hotspots_cached/generic/power_quotient_fraction \
	solve_eval_hotspots_cached/generic/binomial_square_fraction \
	solve_hotspots_cached/generic/x_pow_0 \
	solve_hotspots_cached/assume/x_pow_0

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
	@echo "  make bench-clean   -> remove Criterion baselines/results from target/criterion"
	@echo "  make bench-engine-fast BENCH=profile_cache [FILTER=...]"
	@echo "                     -> run a fast cas_engine Criterion bench"
	@echo "  make bench-engine-fast-save BENCH=profile_cache BASELINE=good [FILTER=...]"
	@echo "                     -> save a named Criterion baseline"
	@echo "  make bench-engine-fast-compare BENCH=profile_cache BASELINE=good [FILTER=...]"
	@echo "                     -> compare against a named baseline without overwriting it"
	@echo "  make bench-engine-fast-save-seq BENCH=profile_cache BASELINE=good FILTERS='a b c'"
	@echo "                     -> save several named baselines sequentially"
	@echo "  make bench-engine-fast-compare-seq BENCH=profile_cache BASELINE=good FILTERS='a b c'"
	@echo "                     -> compare several filters sequentially"
	@echo "  make bench-engine-solve-hotspots-save BASELINE=good"
	@echo "                     -> save the curated solve hotspot suite sequentially"
	@echo "  make bench-engine-solve-hotspots-compare BASELINE=good"
	@echo "                     -> compare the curated solve hotspot suite sequentially"
	@echo "  make bench-engine-solve-batches"
	@echo "                     -> run the main solve_tactic generic/assume guardrail pair"
	@echo "  make bench-engine-solve-batches-save BASELINE=good"
	@echo "                     -> save a named baseline for the solve_tactic guardrail pair"
	@echo "  make bench-engine-solve-batches-compare BASELINE=good"
	@echo "                     -> compare the solve_tactic guardrail pair against a named baseline"
	@echo "  make bench-engine-solve-profile MODE=hotspots-generic FILTER=solve_hotspots_cached/generic/a_pow_x_over_a [DETAIL=1] [PROBE=1] [PROBE_ITERS=2000]"
	@echo "                     -> run profile_cache with solve diagnostic env flags"
	@echo "  make bench-engine-repl-breakdown"
	@echo "                     -> run repl_end_to_end stage breakdown (parse/simplify/format)"
	@echo "  make bench-engine-repl-individual"
	@echo "                     -> rerank the 11 standard REPL batch inputs individually"
	@echo "  make bench-engine-repl-individual-save BASELINE=good"
	@echo "                     -> save a named baseline for the full 11-input REPL rerank"
	@echo "  make bench-engine-repl-individual-compare BASELINE=good"
	@echo "                     -> compare the full 11-input REPL rerank against a named baseline"
	@echo "  make bench-engine-repl-hotspots"
	@echo "                     -> run the current top-5 cached standard REPL hotspots"
	@echo "  make bench-engine-repl-hotspots-save BASELINE=good"
	@echo "                     -> save a named baseline for the current REPL hotspot suite"
	@echo "  make bench-engine-repl-hotspots-compare BASELINE=good"
	@echo "                     -> compare the current REPL hotspot suite against a named baseline"
	@echo "  make bench-engine-standard-phase-subset"
	@echo "                     -> run profile_cache standard phase subset breakdown"
	@echo "  make bench-engine-root-direct"
	@echo "                     -> run profile_cache direct root-rule benchmarks"
	@echo "  make bench-parser-frontend"
	@echo "                     -> run parser/frontend setup+parse benchmarks"
	@echo "  make bench-parser-frontend-save BASELINE=good"
	@echo "                     -> save a named baseline for parser/frontend benchmarks"
	@echo "  make bench-parser-frontend-compare BASELINE=good"
	@echo "                     -> compare parser/frontend benchmarks against a named baseline"
	@echo "  make bench-formatter-frontend"
	@echo "                     -> run formatter/frontend render+clean benchmarks"
	@echo "  make bench-formatter-frontend-save BASELINE=good"
	@echo "                     -> save a named baseline for formatter/frontend benchmarks"
	@echo "  make bench-formatter-frontend-compare BASELINE=good"
	@echo "                     -> compare formatter/frontend benchmarks against a named baseline"
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

bench-clean:
	rm -rf target/criterion

bench-engine-fast:
	@test -n "$(BENCH)" || { echo "Usage: make bench-engine-fast BENCH=profile_cache [FILTER=...]"; exit 1; }
	CAS_BENCH_FAST=1 cargo bench -p cas_engine --bench $(BENCH) $(FILTER) -- --noplot

bench-engine-fast-save:
	@test -n "$(BENCH)" || { echo "Usage: make bench-engine-fast-save BENCH=profile_cache BASELINE=good [FILTER=...]"; exit 1; }
	@test -n "$(BASELINE)" || { echo "Missing BASELINE=..."; exit 1; }
	CAS_BENCH_FAST=1 cargo bench -p cas_engine --bench $(BENCH) $(FILTER) -- --noplot --save-baseline $(BASELINE)

bench-engine-fast-compare:
	@test -n "$(BENCH)" || { echo "Usage: make bench-engine-fast-compare BENCH=profile_cache BASELINE=good [FILTER=...]"; exit 1; }
	@test -n "$(BASELINE)" || { echo "Missing BASELINE=..."; exit 1; }
	CAS_BENCH_FAST=1 cargo bench -p cas_engine --bench $(BENCH) $(FILTER) -- --noplot --baseline $(BASELINE)

bench-engine-fast-save-seq:
	@test -n "$(BENCH)" || { echo "Usage: make bench-engine-fast-save-seq BENCH=profile_cache BASELINE=good FILTERS='a b c'"; exit 1; }
	@test -n "$(BASELINE)" || { echo "Missing BASELINE=..."; exit 1; }
	@test -n "$(FILTERS)" || { echo "Missing FILTERS='a b c'"; exit 1; }
	@for filter in $(FILTERS); do \
		echo "==> $$filter"; \
		CAS_BENCH_FAST=1 cargo bench -p cas_engine --bench $(BENCH) $$filter -- --noplot --save-baseline $(BASELINE) || exit $$?; \
	done

bench-engine-fast-compare-seq:
	@test -n "$(BENCH)" || { echo "Usage: make bench-engine-fast-compare-seq BENCH=profile_cache BASELINE=good FILTERS='a b c'"; exit 1; }
	@test -n "$(BASELINE)" || { echo "Missing BASELINE=..."; exit 1; }
	@test -n "$(FILTERS)" || { echo "Missing FILTERS='a b c'"; exit 1; }
	@for filter in $(FILTERS); do \
		echo "==> $$filter"; \
		CAS_BENCH_FAST=1 cargo bench -p cas_engine --bench $(BENCH) $$filter -- --noplot --baseline $(BASELINE) || exit $$?; \
	done

bench-engine-solve-hotspots-save:
	@$(MAKE) bench-engine-fast-save-seq BENCH=profile_cache BASELINE="$(BASELINE)" FILTERS="$(SOLVE_HOTSPOT_FILTERS)"

bench-engine-solve-hotspots-compare:
	@$(MAKE) bench-engine-fast-compare-seq BENCH=profile_cache BASELINE="$(BASELINE)" FILTERS="$(SOLVE_HOTSPOT_FILTERS)"

bench-engine-solve-batches:
	CAS_BENCH_FAST=1 cargo bench -p cas_engine --bench profile_cache '$(SOLVE_BATCH_FILTER)' -- --noplot

bench-engine-solve-batches-save:
	@test -n "$(BASELINE)" || { echo "Missing BASELINE=..."; exit 1; }
	CAS_BENCH_FAST=1 cargo bench -p cas_engine --bench profile_cache '$(SOLVE_BATCH_FILTER)' -- --noplot --save-baseline $(BASELINE)

bench-engine-solve-batches-compare:
	@test -n "$(BASELINE)" || { echo "Missing BASELINE=..."; exit 1; }
	CAS_BENCH_FAST=1 cargo bench -p cas_engine --bench profile_cache '$(SOLVE_BATCH_FILTER)' -- --noplot --baseline $(BASELINE)

bench-engine-solve-profile:
	@test -n "$(MODE)" || { echo "Usage: make bench-engine-solve-profile MODE=hotspots-generic FILTER=solve_hotspots_cached/generic/a_pow_x_over_a [DETAIL=1] [PROBE=1] [PROBE_ITERS=2000]"; exit 1; }
	@test -n "$(FILTER)" || { echo "Missing FILTER=..."; exit 1; }
	CAS_BENCH_FAST=1 \
	CAS_SOLVE_BENCH_PROFILE=1 \
	CAS_SOLVE_BENCH_PROFILE_MODE=$(MODE) \
	$(if $(DETAIL),CAS_SOLVE_BENCH_PROFILE_DETAIL=1,) \
	$(if $(PROBE),CAS_SOLVE_BENCH_PROFILE_PROBE=1,) \
	$(if $(PROBE_ITERS),CAS_SOLVE_BENCH_PROFILE_PROBE_ITERS=$(PROBE_ITERS),) \
	cargo bench -p cas_engine --bench profile_cache $(FILTER) -- --noplot

bench-engine-repl-breakdown:
	CAS_BENCH_FAST=1 cargo bench -p cas_engine --bench repl_end_to_end 'repl_stage_breakdown/(parse|simplify|format)/(light/symbol_plus_literal|light/numeric_add_chain|heavy/nested_root|heavy/abs_square|gcd/scalar_multiple_fraction|gcd/common_factor_fraction|complex/gaussian_div|trig/pythagorean_chain)' -- --noplot

bench-engine-repl-individual:
	CAS_BENCH_FAST=1 cargo bench -p cas_engine --bench repl_end_to_end 'repl_individual/cached' -- --noplot

bench-engine-repl-individual-save:
	@test -n "$(BASELINE)" || { echo "Missing BASELINE=..."; exit 1; }
	CAS_BENCH_FAST=1 cargo bench -p cas_engine --bench repl_end_to_end 'repl_individual/cached' -- --noplot --save-baseline $(BASELINE)

bench-engine-repl-individual-compare:
	@test -n "$(BASELINE)" || { echo "Missing BASELINE=..."; exit 1; }
	CAS_BENCH_FAST=1 cargo bench -p cas_engine --bench repl_end_to_end 'repl_individual/cached' -- --noplot --baseline $(BASELINE)

bench-engine-repl-hotspots:
	CAS_BENCH_FAST=1 cargo bench -p cas_engine --bench repl_end_to_end 'repl_individual/cached/(03_|04_|06_|08_|10_)' -- --noplot

bench-engine-repl-hotspots-save:
	@test -n "$(BASELINE)" || { echo "Missing BASELINE=..."; exit 1; }
	CAS_BENCH_FAST=1 cargo bench -p cas_engine --bench repl_end_to_end 'repl_individual/cached/(03_|04_|06_|08_|10_)' -- --noplot --save-baseline $(BASELINE)

bench-engine-repl-hotspots-compare:
	@test -n "$(BASELINE)" || { echo "Missing BASELINE=..."; exit 1; }
	CAS_BENCH_FAST=1 cargo bench -p cas_engine --bench repl_end_to_end 'repl_individual/cached/(03_|04_|06_|08_|10_)' -- --noplot --baseline $(BASELINE)

bench-engine-standard-phase-subset:
	CAS_BENCH_FAST=1 cargo bench -p cas_engine --bench profile_cache 'standard_phase_subset_cached/(heavy/nested_root|heavy/abs_square|complex/gaussian_div)/(standard/full|standard/no_transform|standard/no_transform_no_rationalize)' -- --noplot

bench-engine-root-direct:
	CAS_BENCH_FAST=1 cargo bench -p cas_engine --bench profile_cache 'root_rule_direct' -- --noplot

bench-parser-frontend:
	CAS_BENCH_FAST=1 cargo bench -p cas_parser --bench frontend_parse -- --noplot

bench-parser-frontend-save:
	@test -n "$(BASELINE)" || { echo "Missing BASELINE=..."; exit 1; }
	CAS_BENCH_FAST=1 cargo bench -p cas_parser --bench frontend_parse -- --noplot --save-baseline $(BASELINE)

bench-parser-frontend-compare:
	@test -n "$(BASELINE)" || { echo "Missing BASELINE=..."; exit 1; }
	CAS_BENCH_FAST=1 cargo bench -p cas_parser --bench frontend_parse -- --noplot --baseline $(BASELINE)

bench-formatter-frontend:
	CAS_BENCH_FAST=1 cargo bench -p cas_formatter --bench frontend_render -- --noplot

bench-formatter-frontend-save:
	@test -n "$(BASELINE)" || { echo "Missing BASELINE=..."; exit 1; }
	CAS_BENCH_FAST=1 cargo bench -p cas_formatter --bench frontend_render -- --noplot --save-baseline $(BASELINE)

bench-formatter-frontend-compare:
	@test -n "$(BASELINE)" || { echo "Missing BASELINE=..."; exit 1; }
	CAS_BENCH_FAST=1 cargo bench -p cas_formatter --bench frontend_render -- --noplot --baseline $(BASELINE)

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
