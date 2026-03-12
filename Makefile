.PHONY: ci ci-release ci-msrv ci-quick lint test fmt clippy build-release lint-allowlist lint-budget lint-limits audit-utils lint-string-compares lint-no-panic-prod bench-clean bench-engine-fast bench-engine-fast-save bench-engine-fast-compare bench-engine-fast-save-seq bench-engine-fast-compare-seq bench-engine-solve-batches bench-engine-solve-batches-save bench-engine-solve-batches-compare bench-engine-solve-hotspots-save bench-engine-solve-hotspots-compare bench-engine-solve-profile bench-engine-repl-breakdown bench-engine-repl-individual bench-engine-repl-individual-save bench-engine-repl-individual-compare bench-engine-repl-hotspots bench-engine-repl-hotspots-save bench-engine-repl-hotspots-compare bench-engine-standard-phase-subset bench-engine-root-direct bench-parser-frontend bench-parser-frontend-save bench-parser-frontend-compare bench-formatter-frontend bench-formatter-frontend-save bench-formatter-frontend-compare bench-session-frontend bench-session-frontend-save bench-session-frontend-compare bench-session-phase-breakdown bench-session-snapshot-io bench-session-snapshot-io-save bench-session-snapshot-io-compare bench-session-snapshot-restore bench-session-snapshot-restore-save bench-session-snapshot-restore-compare bench-session-snapshot-build bench-session-snapshot-build-save bench-session-snapshot-build-compare bench-session-snapshot-store-build bench-session-snapshot-store-build-save bench-session-snapshot-store-build-compare bench-session-snapshot-load bench-session-snapshot-load-save bench-session-snapshot-load-compare bench-session-store-lookup bench-session-store-lookup-save bench-session-store-lookup-compare bench-session-resolve-frontend bench-session-resolve-frontend-save bench-session-resolve-frontend-compare bench-wire-frontend bench-wire-frontend-save bench-wire-frontend-compare bench-solver-wire-eval bench-solver-wire-eval-save bench-solver-wire-eval-compare bench-solver-wire-substitute bench-solver-wire-substitute-save bench-solver-wire-substitute-compare bench-solver-limit bench-solver-limit-save bench-solver-limit-compare bench-cli-frontend bench-cli-frontend-save bench-cli-frontend-compare bench-didactic-frontend bench-didactic-frontend-save bench-didactic-frontend-compare help

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
	@echo "  make bench-session-frontend"
	@echo "                     -> run stateful cas_session frontend benchmarks"
	@echo "  make bench-session-frontend-save BASELINE=good"
	@echo "                     -> save a named baseline for session/frontend benchmarks"
	@echo "  make bench-session-frontend-compare BASELINE=good"
	@echo "                     -> compare session/frontend benchmarks against a named baseline"
	@echo "  make bench-session-phase-breakdown"
	@echo "                     -> run persisted session phase breakdown (load/build/run)"
	@echo "  make bench-session-snapshot-io"
	@echo "                     -> run direct session snapshot load/save benchmarks"
	@echo "  make bench-session-snapshot-io-save BASELINE=good"
	@echo "                     -> save a named baseline for snapshot I/O benchmarks"
	@echo "  make bench-session-snapshot-io-compare BASELINE=good"
	@echo "                     -> compare snapshot I/O benchmarks against a named baseline"
	@echo "  make bench-session-snapshot-restore"
	@echo "                     -> run direct snapshot restore (context/store/bundle) benchmarks"
	@echo "  make bench-session-snapshot-restore-save BASELINE=good"
	@echo "                     -> save a named baseline for snapshot restore benchmarks"
	@echo "  make bench-session-snapshot-restore-compare BASELINE=good"
	@echo "                     -> compare snapshot restore benchmarks against a named baseline"
	@echo "  make bench-session-snapshot-build"
	@echo "                     -> run direct snapshot build (ContextSnapshot::from_context) benchmarks"
	@echo "  make bench-session-snapshot-build-save BASELINE=good"
	@echo "                     -> save a named baseline for snapshot build benchmarks"
	@echo "  make bench-session-snapshot-build-compare BASELINE=good"
	@echo "                     -> compare snapshot build benchmarks against a named baseline"
	@echo "  make bench-session-snapshot-store-build"
	@echo "                     -> run direct SessionStoreSnapshot build benchmarks"
	@echo "  make bench-session-snapshot-store-build-save BASELINE=good"
	@echo "                     -> save a named baseline for store snapshot build benchmarks"
	@echo "  make bench-session-snapshot-store-build-compare BASELINE=good"
	@echo "                     -> compare store snapshot build benchmarks against a named baseline"
	@echo "  make bench-session-snapshot-load"
	@echo "                     -> run compatible/incompatible session snapshot load benchmarks"
	@echo "  make bench-session-snapshot-load-save BASELINE=good"
	@echo "                     -> save a named baseline for snapshot load benchmarks"
	@echo "  make bench-session-snapshot-load-compare BASELINE=good"
	@echo "                     -> compare snapshot load benchmarks against a named baseline"
	@echo "  make bench-session-store-lookup"
	@echo "                     -> run direct session store lookup benchmarks"
	@echo "  make bench-session-store-lookup-save BASELINE=good"
	@echo "                     -> save a named baseline for session store lookup benches"
	@echo "  make bench-session-store-lookup-compare BASELINE=good"
	@echo "                     -> compare session store lookup benches against a named baseline"
	@echo "  make bench-session-resolve-frontend"
	@echo "                     -> run direct session reference-resolution benchmarks"
	@echo "  make bench-session-resolve-frontend-save BASELINE=good"
	@echo "                     -> save a named baseline for session resolution benches"
	@echo "  make bench-session-resolve-frontend-compare BASELINE=good"
	@echo "                     -> compare session resolution benches against a named baseline"
	@echo "  make bench-wire-frontend"
	@echo "                     -> run direct wire DTO build+serialize benchmarks"
	@echo "  make bench-wire-frontend-save BASELINE=good"
	@echo "                     -> save a named baseline for wire frontend benchmarks"
	@echo "  make bench-wire-frontend-compare BASELINE=good"
	@echo "                     -> compare wire frontend benchmarks against a named baseline"
	@echo "  make bench-solver-wire-eval"
	@echo "                     -> run direct cas_solver wire eval entrypoint benchmarks"
	@echo "  make bench-solver-wire-eval-save BASELINE=good"
	@echo "                     -> save a named baseline for solver wire eval benchmarks"
	@echo "  make bench-solver-wire-eval-compare BASELINE=good"
	@echo "                     -> compare solver wire eval benchmarks against a named baseline"
	@echo "  make bench-solver-wire-substitute"
	@echo "                     -> run direct cas_solver wire substitute entrypoint benchmarks"
	@echo "  make bench-solver-wire-substitute-save BASELINE=good"
	@echo "                     -> save a named baseline for solver wire substitute benchmarks"
	@echo "  make bench-solver-wire-substitute-compare BASELINE=good"
	@echo "                     -> compare solver wire substitute benchmarks against a named baseline"
	@echo "  make bench-solver-limit"
	@echo "                     -> run direct cas_solver stateless limit entrypoint benchmarks"
	@echo "  make bench-solver-limit-save BASELINE=good"
	@echo "                     -> save a named baseline for solver limit benchmarks"
	@echo "  make bench-solver-limit-compare BASELINE=good"
	@echo "                     -> compare solver limit benchmarks against a named baseline"
	@echo "  make bench-cli-frontend"
	@echo "                     -> run direct CLI clap/frontend parse benchmarks"
	@echo "  make bench-cli-frontend-save BASELINE=good"
	@echo "                     -> save a named baseline for CLI frontend benchmarks"
	@echo "  make bench-cli-frontend-compare BASELINE=good"
	@echo "                     -> compare CLI frontend benchmarks against a named baseline"
	@echo "  make bench-didactic-frontend"
	@echo "                     -> run direct cas_didactic payload/timeline render benchmarks"
	@echo "  make bench-didactic-frontend-save BASELINE=good"
	@echo "                     -> save a named baseline for didactic/frontend benchmarks"
	@echo "  make bench-didactic-frontend-compare BASELINE=good"
	@echo "                     -> compare didactic/frontend benchmarks against a named baseline"
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

bench-session-frontend:
	CAS_BENCH_FAST=1 cargo bench -p cas_session --bench frontend_session -- --noplot

bench-session-frontend-save:
	@test -n "$(BASELINE)" || { echo "Missing BASELINE=..."; exit 1; }
	CAS_BENCH_FAST=1 cargo bench -p cas_session --bench frontend_session -- --noplot --save-baseline $(BASELINE)

bench-session-frontend-compare:
	@test -n "$(BASELINE)" || { echo "Missing BASELINE=..."; exit 1; }
	CAS_BENCH_FAST=1 cargo bench -p cas_session --bench frontend_session -- --noplot --baseline $(BASELINE)

bench-session-phase-breakdown:
	CAS_BENCH_FAST=1 cargo bench -p cas_session --bench frontend_session 'frontend_session/session_phase/(load_or_new/persisted/cache_hit_seed|engine_with_context/cache_hit_seed|run_loaded/cache_hit/ref_1)' -- --noplot

bench-session-snapshot-io:
	CAS_BENCH_FAST=1 cargo bench -p cas_session_core --bench snapshot_io -- --noplot

bench-session-snapshot-io-save:
	@test -n "$(BASELINE)" || { echo "Missing BASELINE=..."; exit 1; }
	CAS_BENCH_FAST=1 cargo bench -p cas_session_core --bench snapshot_io -- --noplot --save-baseline $(BASELINE)

bench-session-snapshot-io-compare:
	@test -n "$(BASELINE)" || { echo "Missing BASELINE=..."; exit 1; }
	CAS_BENCH_FAST=1 cargo bench -p cas_session_core --bench snapshot_io -- --noplot --baseline $(BASELINE)

bench-session-snapshot-restore:
	CAS_BENCH_FAST=1 cargo bench -p cas_session_core --bench snapshot_restore -- --noplot

bench-session-snapshot-restore-save:
	@test -n "$(BASELINE)" || { echo "Missing BASELINE=..."; exit 1; }
	CAS_BENCH_FAST=1 cargo bench -p cas_session_core --bench snapshot_restore -- --noplot --save-baseline $(BASELINE)

bench-session-snapshot-restore-compare:
	@test -n "$(BASELINE)" || { echo "Missing BASELINE=..."; exit 1; }
	CAS_BENCH_FAST=1 cargo bench -p cas_session_core --bench snapshot_restore -- --noplot --baseline $(BASELINE)

bench-session-snapshot-build:
	CAS_BENCH_FAST=1 cargo bench -p cas_session_core --bench snapshot_build -- --noplot

bench-session-snapshot-build-save:
	@test -n "$(BASELINE)" || { echo "Missing BASELINE=..."; exit 1; }
	CAS_BENCH_FAST=1 cargo bench -p cas_session_core --bench snapshot_build -- --noplot --save-baseline $(BASELINE)

bench-session-snapshot-build-compare:
	@test -n "$(BASELINE)" || { echo "Missing BASELINE=..."; exit 1; }
	CAS_BENCH_FAST=1 cargo bench -p cas_session_core --bench snapshot_build -- --noplot --baseline $(BASELINE)

bench-session-snapshot-store-build:
	CAS_BENCH_FAST=1 cargo bench -p cas_session_core --bench snapshot_store_build -- --noplot

bench-session-snapshot-store-build-save:
	@test -n "$(BASELINE)" || { echo "Missing BASELINE=..."; exit 1; }
	CAS_BENCH_FAST=1 cargo bench -p cas_session_core --bench snapshot_store_build -- --noplot --save-baseline $(BASELINE)

bench-session-snapshot-store-build-compare:
	@test -n "$(BASELINE)" || { echo "Missing BASELINE=..."; exit 1; }
	CAS_BENCH_FAST=1 cargo bench -p cas_session_core --bench snapshot_store_build -- --noplot --baseline $(BASELINE)

bench-session-snapshot-load:
	CAS_BENCH_FAST=1 cargo bench -p cas_session --bench snapshot_load -- --noplot

bench-session-snapshot-load-save:
	@test -n "$(BASELINE)" || { echo "Missing BASELINE=..."; exit 1; }
	CAS_BENCH_FAST=1 cargo bench -p cas_session --bench snapshot_load -- --noplot --save-baseline $(BASELINE)

bench-session-snapshot-load-compare:
	@test -n "$(BASELINE)" || { echo "Missing BASELINE=..."; exit 1; }
	CAS_BENCH_FAST=1 cargo bench -p cas_session --bench snapshot_load -- --noplot --baseline $(BASELINE)

bench-session-store-lookup:
	CAS_BENCH_FAST=1 cargo bench -p cas_session_core --bench store_lookup -- --noplot

bench-session-store-lookup-save:
	@test -n "$(BASELINE)" || { echo "Missing BASELINE=..."; exit 1; }
	CAS_BENCH_FAST=1 cargo bench -p cas_session_core --bench store_lookup -- --noplot --save-baseline $(BASELINE)

bench-session-store-lookup-compare:
	@test -n "$(BASELINE)" || { echo "Missing BASELINE=..."; exit 1; }
	CAS_BENCH_FAST=1 cargo bench -p cas_session_core --bench store_lookup -- --noplot --baseline $(BASELINE)

bench-session-resolve-frontend:
	CAS_BENCH_FAST=1 cargo bench -p cas_session_core --bench resolve_frontend -- --noplot

bench-session-resolve-frontend-save:
	@test -n "$(BASELINE)" || { echo "Missing BASELINE=..."; exit 1; }
	CAS_BENCH_FAST=1 cargo bench -p cas_session_core --bench resolve_frontend -- --noplot --save-baseline $(BASELINE)

bench-session-resolve-frontend-compare:
	@test -n "$(BASELINE)" || { echo "Missing BASELINE=..."; exit 1; }
	CAS_BENCH_FAST=1 cargo bench -p cas_session_core --bench resolve_frontend -- --noplot --baseline $(BASELINE)

bench-wire-frontend:
	CAS_BENCH_FAST=1 cargo bench -p cas_api_models --bench frontend_wire -- --noplot

bench-wire-frontend-save:
	@test -n "$(BASELINE)" || { echo "Missing BASELINE=..."; exit 1; }
	CAS_BENCH_FAST=1 cargo bench -p cas_api_models --bench frontend_wire -- --noplot --save-baseline $(BASELINE)

bench-wire-frontend-compare:
	@test -n "$(BASELINE)" || { echo "Missing BASELINE=..."; exit 1; }
	CAS_BENCH_FAST=1 cargo bench -p cas_api_models --bench frontend_wire -- --noplot --baseline $(BASELINE)

bench-solver-wire-eval:
	CAS_BENCH_FAST=1 cargo bench -p cas_solver --bench frontend_wire_eval -- --noplot

bench-solver-wire-eval-save:
	@test -n "$(BASELINE)" || { echo "Missing BASELINE=..."; exit 1; }
	CAS_BENCH_FAST=1 cargo bench -p cas_solver --bench frontend_wire_eval -- --noplot --save-baseline $(BASELINE)

bench-solver-wire-eval-compare:
	@test -n "$(BASELINE)" || { echo "Missing BASELINE=..."; exit 1; }
	CAS_BENCH_FAST=1 cargo bench -p cas_solver --bench frontend_wire_eval -- --noplot --baseline $(BASELINE)

bench-solver-wire-substitute:
	CAS_BENCH_FAST=1 cargo bench -p cas_solver --bench frontend_wire_substitute -- --noplot

bench-solver-wire-substitute-save:
	@test -n "$(BASELINE)" || { echo "Missing BASELINE=..."; exit 1; }
	CAS_BENCH_FAST=1 cargo bench -p cas_solver --bench frontend_wire_substitute -- --noplot --save-baseline $(BASELINE)

bench-solver-wire-substitute-compare:
	@test -n "$(BASELINE)" || { echo "Missing BASELINE=..."; exit 1; }
	CAS_BENCH_FAST=1 cargo bench -p cas_solver --bench frontend_wire_substitute -- --noplot --baseline $(BASELINE)

bench-solver-limit:
	CAS_BENCH_FAST=1 cargo bench -p cas_solver --bench frontend_limit -- --noplot

bench-solver-limit-save:
	@test -n "$(BASELINE)" || { echo "Missing BASELINE=..."; exit 1; }
	CAS_BENCH_FAST=1 cargo bench -p cas_solver --bench frontend_limit -- --noplot --save-baseline $(BASELINE)

bench-solver-limit-compare:
	@test -n "$(BASELINE)" || { echo "Missing BASELINE=..."; exit 1; }
	CAS_BENCH_FAST=1 cargo bench -p cas_solver --bench frontend_limit -- --noplot --baseline $(BASELINE)

bench-cli-frontend:
	CAS_BENCH_FAST=1 cargo bench -p cas_cli --bench frontend_cli -- --noplot

bench-cli-frontend-save:
	@test -n "$(BASELINE)" || { echo "Missing BASELINE=..."; exit 1; }
	CAS_BENCH_FAST=1 cargo bench -p cas_cli --bench frontend_cli -- --noplot --save-baseline $(BASELINE)

bench-cli-frontend-compare:
	@test -n "$(BASELINE)" || { echo "Missing BASELINE=..."; exit 1; }
	CAS_BENCH_FAST=1 cargo bench -p cas_cli --bench frontend_cli -- --noplot --baseline $(BASELINE)

bench-didactic-frontend:
	CAS_BENCH_FAST=1 cargo bench -p cas_didactic --bench frontend_didactic -- --noplot

bench-didactic-frontend-save:
	@test -n "$(BASELINE)" || { echo "Missing BASELINE=..."; exit 1; }
	CAS_BENCH_FAST=1 cargo bench -p cas_didactic --bench frontend_didactic -- --noplot --save-baseline $(BASELINE)

bench-didactic-frontend-compare:
	@test -n "$(BASELINE)" || { echo "Missing BASELINE=..."; exit 1; }
	CAS_BENCH_FAST=1 cargo bench -p cas_didactic --bench frontend_didactic -- --noplot --baseline $(BASELINE)

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

bench-solver-repl-parse:
	CAS_BENCH_FAST=1 cargo bench -p cas_solver --bench frontend_repl_parse -- --noplot

bench-solver-repl-parse-save:
ifndef BASELINE
	$(error BASELINE is required, e.g. make $@ BASELINE=my_baseline)
endif
	CAS_BENCH_FAST=1 cargo bench -p cas_solver --bench frontend_repl_parse -- --noplot --save-baseline $(BASELINE)

bench-solver-repl-parse-compare:
ifndef BASELINE
	$(error BASELINE is required, e.g. make $@ BASELINE=my_baseline)
endif
	CAS_BENCH_FAST=1 cargo bench -p cas_solver --bench frontend_repl_parse -- --noplot --baseline $(BASELINE) --discard-baseline
