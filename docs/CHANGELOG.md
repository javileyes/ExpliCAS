# Changelog

All notable changes to ExpliCAS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.15.56] - 2026-02-16 - Solver TLS Elimination & Reentrancy Hardening

### Added

- **`SolveCtx` Explicit Context Threading**:
  Domain conditions are now accumulated via `Rc<RefCell<HashSet<ImplicitCondition>>>` in
  `SolveCtx`, passed explicitly through the entire solve pipeline. Eliminates dependency on
  thread-local storage for solver-semantic state.

- **`SolveDiagnostics.required`**: Domain conditions returned in-band with solve results,
  making each solve call self-contained and composable.

- **CI Lint `lint_no_solver_tls.sh`**: Prevents new `thread_local!` declarations in
  `solver/`. Allowlists 4 legacy diagnostic cells (`SOLVE_DEPTH`, `SOLVE_SEEN`,
  `SOLVE_ASSUMPTIONS`, `OUTPUT_SCOPES`). Auto-discovered by `scripts/ci.sh`.

- **Sequential Reentrancy Test** (`solver_assumptions_contract_tests.rs`):
  `sequential_solves_no_condition_leakage` — verifies `ln(x)=0` followed by `x=1`
  produces no condition leakage between solves.

- **Parallel Reentrancy Tests** (`solver_parallel_reentrancy_tests.rs`):
  - `parallel_solves_have_independent_diagnostics` — two threads solve different equations
    concurrently, asserts independent diagnostics.
  - `parallel_identical_solves_produce_consistent_results` — four threads solve `2^x=y`
    concurrently, asserts deterministic `positive(y)` conditions.

### Removed

- **`DOMAIN_ENV_SINK` TLS**: Thread-local sink for domain conditions fully removed.
  All references deleted — conditions now flow through `SolveCtx`.

### Changed

- **Legacy TLS Documentation**: Added block comment above remaining TLS declarations in
  `solver/mod.rs` and `solve_core.rs` documenting they are diagnostic/UI only and must
  not carry solver-semantic information. References TLS policy for future contributors.

### Documentation

- `CHANGELOG.md`: This entry
- `MAINTENANCE.md`: Added `lint_no_solver_tls.sh` to custom lint scripts list;
  new section 11 documenting Solver TLS Policy
- `ARCHITECTURE.md`: New section on Solver Context Architecture (`SolveCtx` design,
  TLS policy, reentrancy guarantees)

---

## [2.15.54] - 2026-02-15 - Canonicalization Consolidation & Pipeline Hardening

### Added

- **`CanonicalizeRationalDivRule`** (`rules/rational_canonicalization.rs`):
  Folds `Div(Number(p), Number(q))` → `Number(p/q)` for exact rational arithmetic.

- **`CanonicalizeNestedPowRule`** (`rules/rational_canonicalization.rs`):
  Folds `Pow(Pow(base, k), r)` → `Pow(base, k*r)` with **domain safety**:
  blocks rewrite when inner exponent is even and outer denominator is even
  (e.g., `(x²)^(1/2) ≠ x` to preserve absolute value semantics).

- **Equation-Level Cancel Primitive** (`rules/cancel_common_terms.rs`):
  Exported `cancel_common_additive_terms(ctx, lhs, rhs) → Option<CancelResult>`
  for solver pre-solve pipeline. Returns `CancelResult { new_lhs, new_rhs, cancelled_count }`
  for diagnostics.

- **Canonicalization Hardening Tests** (`tests/canonicalization_hardening_tests.rs`):
  9 tests covering rational canonicality (1A), domain-safe nested powers (1B),
  and additive term cancellation (2).

### Removed

- **`CancelCommonAdditiveTermsRule`** ("ghost rule"): Removed from simplifier
  registration — `CanonicalizeNegationRule` converts `Sub→Add(Neg)` before this
  rule could fire on `Sub` nodes. Cancellation is now exclusively equation-level.

### Changed

- **`simplifier.rs`**: Updated canonicalization tier order contract. Added
  "Sub is NOT stable" warning documenting that `CanonicalizeNegationRule`
  converts `Sub(a,b)→Add(a,Neg(b))` early and future rules must account for this.

- **`solve_core.rs`**: Solver cancel call updated to use `CancelResult` struct.

### Results

- S2: **392 symbolic, 0 incomplete, 0 mismatch** (unchanged)
- All hardening tests pass, `make ci` green

### Documentation

- `RULES.md`: Added Canonicalization & Rational Arithmetic family
- `SOLVER_SIMPLIFY_POLICY.md`: Added Equation-Level Additive Cancel section
- `MAINTENANCE.md`: Added "Sub is NOT stable" hazard to rule authoring guide
- `ARCHITECTURE.md`: Updated rule category table with new modules

---

## [2.15.52] - 2026-02-15 - S2-Driven Solver Hardening

### Added

- **`RationalRootsStrategy`** (`strategies/rational_roots.rs`):
  Solves polynomial equations of degree ≥ 3 with rational coefficients via
  Rational Root Theorem + synthetic division. Guardrails: `MAX_CANDIDATES = 200`,
  `MAX_DEGREE = 10`. Placed after `QuadraticStrategy` in pipeline.

- **Pre-Solve Identity Cancellation** (`solve_core.rs`):
  Expand+simplify pass to cancel identity noise across `LHS - RHS` boundary.
  Guard: only applies if expansion achieves >25% node reduction.

- **S2 Equation Identity Transforms Benchmark** (`metamorphic_equation_tests.rs`):
  New metamorphic test strategy verifying solver transparency to algebraic identity
  transforms. Structured `IncompleteReason` classification with verbose cross-tab
  reporting.

### Fixed

- **Factor-Split Hardening** (`quadratic.rs`):
  Non-`Discrete` sub-solve results (AllReals, Empty, Residual) now handled
  gracefully instead of crashing with `SolverError`.

### Results

- S2 Incomplete: 23 → **12** (−48%)
- New symbolic solutions: **+19**, numeric: **+6**
- Zero mismatches, zero regressions

### Documentation

- `METAMORPHIC_TESTING.md`: Added S2 section (property, execution, outcomes, baseline)
- `SOLVER_SIMPLIFY_POLICY.md`: Added Solver Strategies section

---

## [2.15.50] - 2026-02-12 - Cycle Detection Hardened + Event Registry

### Added

- **Cycle Event Registry (`cycle_events.rs`)**:
  - Thread-local `CycleEvent` registry (same pattern as `BlockedHint` in `domain.rs`)
  - Captures: phase, period, detection level, rule name, fingerprint, and truncated expression display
  - API: `register_cycle_event()`, `take_cycle_events()`, `clear_cycle_events()`
  - Deduplication by `(fingerprint, rule_name, level)`
  - 5 unit tests (register/take, dedup, dedup-different-level, clear, truncate)

- **`PipelineStats.cycle_events`**: Cycle events are automatically collected during simplification
  and exposed in pipeline statistics for analysis

- **Public re-exports**: `CycleEvent`, `CycleLevel` available from `cas_engine::{CycleEvent, CycleLevel}`

### Changed

- **CycleDetector `max_period`: 8 → 16**: Doubles the intra-node ping-pong detection window
  (ring buffer was already 64 entries, so this is safe)

- **Inter-iteration cycle detection: `VecDeque` → `HashSet`** in `Orchestrator::run_phase()`:
  Previously used a capped VecDeque of 10 recent hashes (could miss cycles of period >10).
  Now uses a `HashSet<u64>` that remembers *all* hashes seen in the phase, detecting cycles
  of any period

### Technical

- All 563 engine tests pass
- `make ci` passes (fmt, clippy, test, build --release)
- Detection events are wired at two levels:
  - **Intra-node** (`rule_application.rs`): emits with specific rule name and period
  - **Inter-iteration** (`orchestrator.rs`): emits with `"(inter-iteration)"` label

---

## [2.15.48] - 2026-01-24 - Linear Systems Solver

### Added

- **`solve_system` Command**: General solver for n×n linear equation systems
  - Syntax: `solve_system(eq1; ...; eqn; x1; ...; xn)`
  - **2×2, 3×3**: Cramer's rule
  - **n≥4**: Gaussian elimination with partial pivoting
  - Uses exact `BigRational` arithmetic (no floating-point)
  - Validates linearity via `MultiPoly` coefficient extraction
  - **Degenerate Detection**: Distinguishes "infinitely many solutions" vs "no solution"
  - See [SOLVE_SYSTEM.md](SOLVE_SYSTEM.md) for full documentation

- **Dependencies**: Added `num-rational` and `num-bigint` to `cas_cli`

### Technical

- New module: `crates/cas_cli/src/repl/commands_system.rs`
- Semicolon bypass in REPL for `solve_system` (uses `;` as internal separator)
- 15 regression tests (2×2, 3×3, 4×4)

---

## [2.15.36] - 2026-01-21 - Session Caching Hardened

### Added

- **Phase 4: Synthetic Timeline Step**
  - `CacheHitTrace` struct for tracking cache hit metadata
  - `ResolvedExpr.cache_hits` with dedup via `HashSet<EntryId>`
  - `build_cache_hit_step()` generates aggregated step: `"Used cached simplified result from #1, #3"`
  - Step prepended to timeline for traceability without repeating derivation

- **Phase 5: Memory Control with LRU Eviction**
  - `CacheConfig` struct: `max_cached_entries=100`, `max_cached_steps=5000`, `light_cache_threshold=200`
  - `VecDeque<EntryId>` for LRU ordering in `SessionStore`
  - `touch_cached(id)` method for true LRU on cache hits
  - `apply_light_cache()` drops steps when > threshold (`steps: Option<Arc<...>>`)
  - `evict_if_needed()` with proper `0 = unlimited` handling

### Changed

- `SimplifiedCache.steps` changed from `Arc<Vec<Step>>` to `Option<Arc<Vec<Step>>>` for light-cache mode
- `update_simplified()` now evicts AFTER insert (budget-correct)
- `resolve_all_with_diagnostics()` returns `cache_hits` for synthetic step generation

### Technical

- All 512 unit tests pass
- Documentation updated in `session_evaluation_caching.md`

---

## [2.14.36] - 2026-01-14 - GCD Router Unification

### Added

- **GcdGoal Enum**: Differentiates between interactive REPL usage and internal simplifier usage
  - `UserPolyGcd`: Allows full pipeline (Structural → Exact → Modp)
  - `CancelFraction`: Blocks modp methods, returns gcd=1 if budget exceeded (soundness guarantee)

- **CancelPowersDivisionRule**: Ultra-light pre-order rule for `P^m / P^n` cancellation
  - Handles all exponent cases: `m = n → 1`, `m > n → P^(m-n)`, `m < n → 1/P^(n-m)`
  - Supports negative exponents: `P^(-2)/P^(-5) → P^3`
  - Uses structural `compare_expr` (not expensive `poly_relation`)
  - O(1) stack depth - prevents overflow on complex fractions

- **Unified `gcd()` Command**: Single entry point for integer and polynomial GCD
  - Auto-dispatches to Euclidean algorithm for integers
  - Auto-dispatches to `poly_gcd` router for polynomials
  - `help gcd` shows unified documentation

- **Regression Tests** (`anti_catastrophe_tests.rs`):
  - `test_power_cancel_equal_exponents`
  - `test_power_cancel_smaller_numerator`
  - `test_power_cancel_zero_numerator_exp`
  - `test_power_cancel_negative_exponents`
  - `test_identity_neutral_add_zero`

### Fixed

- **Stack Overflow in `((x+y)^10)/((x+y)^9)`**: New pre-order rule short-circuits before heavy fraction rules
- **Identity Neutral Bug**: `(a+pi)^2 + 0` now produces same result as `(a+pi)^2`
  - Root cause: `auto_expand_scan.rs` was counting literal `0` as "other terms"
  - Fix: Filter `Number(0)` from `other_terms` before triggering auto-expand

### Documentation

- **POLY_GCD.md**: Extended with GCD Router Unification section (200+ lines)
  - Pipeline diagrams by GcdGoal
  - Usage examples for REPL and automatic fraction simplification
  - API reference for `compute_poly_gcd_unified`

---

## [2.14.30] - 2026-01-13 - Always-On Cycle Defense

### Added

- **Always-On Cycle Detection**: Cycle detection now runs in all modes, not just "health mode"
  - Removed health-mode guard from `LocalSimplificationTransformer`
  - Detects ping-pong patterns (A↔B) and self-loops in any phase

- **Per-Phase Reset**: `CycleDetector` resets when `SimplifyPhase` changes
  - Prevents false positives across phase boundaries
  - New field `cycle_phase: Option<SimplifyPhase>` tracks active phase

- **Blocklist for Cycle Prevention**: `HashSet<(u64, String)>` tracks `(fingerprint, rule_name)` pairs
  - Prevents re-entry into states known to cause cycles
  - Persists across phases (conservative approach)

- **BlockedHint Emission**: On first cycle detection, emits pedagogical hint
  - Suggestion: "cycle detected; consider disabling heuristic rules or tightening budget"
  - Deduplicated by (fingerprint, rule) - only one hint per pattern

- **Regression Tests** (`cycle_defense_tests.rs`):
  - `test_cycle_defense_fractional_powers_terminates`
  - `test_cycle_defense_terminates_not_hangs`
  - `test_cycle_detection_phase_reset`

### Changed

- `BlockedHint.rule` changed from `&'static str` to `String` to support dynamic rule names
- Updated affected files: `domain.rs`, `logarithms.rs`, `isolation.rs`, `repl.rs`

---

## [2.1.0] - 2026-01-05 - Education-First Polish

### Added

- **Issue #1: Output polishing** — `otherwise:` without "if" prefix (✅)
  - Natural reading in console: `otherwise: ...`
  - LaTeX uses `\text{otherwise}` cleanly

- **Issue #2: REPL Snapshot Tests** — 8 iconic regression tests (✅)
  - Deterministic snapshots for solver behavior
  - Tests include: quadratic, linear, conditional, equal bases

- **Issue #3: Explain Mode** — Compact assumption/hint summary (✅)
  - `explain on|off` toggle
  - "Assumptions used" + "Blocked simplifications" blocks

- **Issue #4: Stable API** — `cas_engine::api` module (✅)
  - Exports: `SolveResult`, `SolutionSet`, `Case`, `ConditionSet`
  - 10 compile tests in `public_api_contract.rs`

- **Issue #5: `solve --check`** — Solution verification (✅)
  - One-shot: `solve --check x+2=5, x`
  - Toggle: `semantics set solve check on|off`
  - Verifies by substitution in Strict mode
  - Shows ✓/⚠/ℹ for verified/unverifiable/not-checkable

- **Issue #10: Denominator Guards** — Structured safety (✅)
  - Clear denominators with `x ≠ value` guards
  - Conditional output for rational equations
  - 3 contract tests in `domain_guard_contract_tests.rs`

### Changed

- `EvalOptions` now includes `check_solutions: bool`
- Solver prepass uses `SolveSafety` classification

---

## [1.3.7] - 2026-01-05 - SolveSafety Architecture

### Added

- **SolveSafety Rule Classification System**: Protects solver from applying simplifications that could corrupt solution sets
  - `SolveSafety` enum: `Always`, `NeedsCondition(ConditionClass)`, `Never`
  - `SimplifyPurpose` enum: `Eval`, `SolvePrepass`, `SolveTactic`
  - New `solve_safety:` parameter for `define_rule!` macro
  - New macro variant for `targets + solve_safety`

- **Solver Pre-pass Integration**: 
  - `simplifier.simplify_for_solve()` method using `SolvePrepass` purpose
  - Filter logic in `LocalSimplificationTransformer::apply_rules()`
  - `SimplifyPurpose` propagation through orchestrator

- **13 Dangerous Rules Marked**:
  - **Definability (6)**: `CancelCommonFactorsRule`, `SimplifyFractionRule`, `QuotientOfPowersRule`, `IdentityPowerRule`, `MulZeroRule`, `DivZeroRule`
  - **Analytic (7)**: `LogExpansionRule`, `ExponentialLogRule`, `LogInversePowerRule`, `SplitLogExponentsRule`, `PowerPowerRule`, `HyperbolicCompositionRule`, `TrigInverseExpansionRule`

- **Contract Tests** (`solve_safety_contract_tests.rs`):
  - Rule marking verification
  - Prepass blocking validation
  - Solver correctness tests (`0^x=0`, `a^x=a`, linear equations)
  - Guardrail tests to prevent unmarked dangerous rules

### Changed

- `SEMANTICS_POLICY.md` updated to V1.3.7 with SolveSafety section
- `SOLVER_SIMPLIFY_POLICY.md` completely rewritten with implementation details

---

## [1.3.3] - 2026-01-01 - DomainMode Transparency & Pedagogical Hints

### Added

- **Condition-class gating** (Definability vs Analytic) with complementary `DomainMode`s:
  - `Strict`: only applies rewrites when required conditions are proven
  - `Generic`: allows *definability holes* (e.g., `x ≠ 0`) and records them
  - `Assume`: allows analytic assumptions (e.g., `x > 0`) and records them
- **Pedagogical blocked hints** in the REPL when a rewrite is blocked by an unproven condition (deduplicated, grouped by rule)
- **Hint verbosity toggle**: `semantics set hints on|off`
- **Timeline transparency**: Any rewrite applied under an unproven condition records it in the timeline as **Assumptions (assumed)**
- **Transparency Invariant** documented in `SEMANTICS_POLICY.md`: "No assumptions without timeline record"
- **Contract tests** for assumption tracking:
  - `step_tracks_assumed_nonzero_in_generic` — Definability (x ≠ 0)
  - `step_tracks_assumed_positive_in_assume` — Analytic (x > 0)

### Changed

- `Generic` no longer performs transformations requiring analytic assumptions (e.g., positivity for log expansions). Such rewrites are now exclusive to `Assume` unless the condition is proven
- `SimplifyFractionRule` (4 call sites) now uses centralized `assumption_events()` helper
- Definability hints extended to all 8 `can_cancel_factor` call sites

### Examples

- **Generic mode**:
  - `x/x → 1` with `Assumed: x ≠ 0`
  - `ln(x*y)` does **not** expand; REPL shows hint: `use domain assume`
- **Assume mode**:
  - `exp(ln(x)) → x` with `Assumed: x > 0`

---

## [1.3.2] - 2025-12-31 - Hint UX Improvements

### Added

- **Hint Grouping**: Multiple blocked hints for the same rule are now grouped into a single line (e.g., `requires x > 0, y > 0 [Log Expansion]`)
- **Verbosity Control**: New `semantics set hints on|off` command to toggle hint display
- **Contextual Suggestions**: Hints now suggest appropriate action based on current mode:
  - Strict → suggests `domain generic` or `domain assume`
  - Generic → suggests `domain assume`
- **Snapshot Tests**: Added contract tests for hint emission (V1.3.1 stability)

### Changed

- `hints_enabled` option in `EvalOptions` (default: `true`)
- Hints display respects current `DomainMode` for targeted suggestions

---

## [1.3.1] - 2025-12-31 - Blocked Hints

### Added

- **Pedagogical Blocked Hints**: When Generic mode blocks simplifications requiring analytic assumptions (e.g., `x > 0`), the REPL now displays actionable hints:
  ```
  ℹ️  Blocked in Generic: requires x > 0 [Exponential-Log Inverse]
     use `semantics set domain assume` to allow analytic assumptions
  ```

- **Thread-local Hint Collector**: `register_blocked_hint()`, `take_blocked_hints()`, `clear_blocked_hints()` in `domain.rs`
- **Rich Gate Function**: `can_apply_analytic_with_hint()` emits structured hints when Generic blocks Analytic conditions
- **Hint Propagation**: `Simplifier.extend_blocked_hints()` for context transfer in `Engine.eval()`

### Changed

- `BlockedHint` struct now includes `expr_id` for pretty-printing expression names in hints
- `ExponentialLogRule` now uses `can_apply_analytic_with_hint()` to emit pedagogical hints

### Technical

- Hints are deduplicated by `(rule, AssumptionKey)` preserving first-occurrence order
- REPL displays hints using `DisplayExpr` for consistent formatting with result output

---

## [Unreleased]

### Added - Pattern Detection Infrastructure (2025-12-07)

#### New Features

- **Context-Aware Pattern Detection System** ★
  - Pre-analysis scanner (`PatternScanner`) that detects mathematical patterns before simplification
  - Pattern marks (`PatternMarks`) to protect expressions from premature transformations
  - Parent context threading (`ParentContext`) to propagate pattern information through AST traversal
  - Pattern detection helpers for common trigonometric patterns

- **Pythagorean Identity Rules**
  - `SecTanPythagoreanRule`: Simplifies `sec²(x) - tan²(x) → 1`
  - `CscCotPythagoreanRule`: Simplifies `csc²(x) - cot²(x) → 1`
  - `TanToSinCosRule` with guard: Prevents premature `tan(x) → sin(x)/cos(x)` conversion when part of Pythagorean pattern

#### Core Changes

- **Modified `Rule` trait signature**: Added `parent_ctx: &ParentContext` parameter to all rules
  - Backward compatible: rules can ignore the parameter if context-awareness not needed
  - Enables rules to make decisions based on expression ancestry and pattern marks

- **Enhanced `Simplifier::apply_rules_loop`**: Now accepts `&PatternMarks` parameter
  - Threads pattern marks through transformation via `ParentContext`
  - Creates initial parent context with marks before recursive simplification

- **Updated `Orchestrator::simplify`**:
  - Performs pattern scanning before applying rules
  - Passes pattern marks to simplification engine

#### Key Discovery

- **AST Normalization Insight** ⭐: Discovered that `a - b` is internally represented as `Add(a, Neg(b))`, NOT `Sub(a, b)`
  - Critical for implementing pattern matching on subtraction expressions
  - Documented in ARCHITECTURE.md for future developers

#### Documentation

- **ARCHITECTURE.md** (+616 lines)
  - New section 2.5: "Pattern Detection Infrastructure" with complete architecture
  - Data flow diagrams showing pattern mark propagation
  - Implementation details, metrics, and extensibility guides
  - Updated index, rule listings, and future improvements section

- **MAINTENANCE.md** (+379 lines)
  - Complete Pattern Detection debugging guide
  - Extension guide for adding new pattern families
  - Performance considerations and optimization tips
  - Troubleshooting checklist (9 verification steps)
  - Updated Rule creation examples with new `ParentContext` parameter
  - Corrected debug system documentation (removed obsolete `enable_debug()`, documented `tracing` correctly)

- **DEBUG_SYSTEM.md** (+326 lines)
  - Component-by-component debugging techniques
  - 3 common troubleshooting scenarios with solutions
  - End-to-end trace guide (10 instrumentation points)
  - Testing commands and debug output examples
  - Roadmap for tracing integration

- **README.md** (+5 lines)
  - Added Pattern Detection as highlighted feature
  - Example: Pythagorean identity simplification
  - Link to ARCHITECTURE.md for technical details

#### Tests

- **New Tests**:
  - `pythagorean_variants_test.rs`: Integration tests for Pythagorean identities
  - `debug_sec_tan.rs`: Debug test for pattern detection verification
  - Unit tests in `pattern_scanner.rs` for pattern detection logic

- **Test Results**: All 102 tests passing ✅
  - Including new Pythagorean identity tests
  - Fixed `parent_context::tests::test_extend` (ancestor ordering)

#### Metrics

- **Code Added**: ~750 lines of pattern detection infrastructure
  - `pattern_marks.rs`: ~50 lines
  - `pattern_scanner.rs`: ~150 lines
  - `parent_context.rs`: ~100 lines
  - `pattern_detection.rs`: ~200 lines
  - Rule implementations: ~250 lines

- **Documentation Added**: +1,326 lines across 4 documents

- **Development Time**: ~10+ hours of implementation + 2 hours of documentation

#### Performance

- **Pattern Scanner**: O(n) complexity, one-time cost per `simplify()` call
- **Overhead**: ~5-10µs for typical expressions (100-500 nodes)
- **No performance regression**: All existing tests maintain same performance

### Fixed

- **Compiler Warnings**: Reduced from 31 to 25 warnings in `cas_engine`
  - Removed unused import `std::cmp::Ordering` from `reciprocal_trig.rs`
  - Fixed unused variables in pattern matching
  - Added `#[allow(dead_code)]` to helper functions and indirect-use fields
  - Remaining warnings are benign doc comments in macros

- **Test Compatibility**: Updated `debug_sec_tan.rs` to pass `pattern_marks` to `apply_rules_loop`

- **Parent Context Ordering**: Fixed `ParentContext::extend()` to maintain correct ancestor ordering

### Changed

- **Rule Application**: All rules now receive `ParentContext` instead of operating in isolation
  - Enables context-aware decision making
  - Preserves backward compatibility (parameter can be ignored)

- **Simplification Flow**: Added pre-analysis step
  1. Parse expression
  2. **NEW**: Scan for patterns, create marks
  3. Apply rules with pattern awareness
  4. Return result

### Developer Notes

#### Breaking Changes

- **Rule trait signature change**: Existing custom rules must add `_parent_ctx: &ParentContext` parameter
  - Can be safely ignored with: `let _ = parent_ctx;`
  - See MAINTENANCE.md for migration guide

#### Backward Compatibility

- **Macro-generated rules**: Automatically updated via `define_rule!` macro
- **Manual rules**: Require one-line signature update

#### Future Enhancements

Areas for potential extension (documented in ARCHITECTURE.md):

- Additional pattern families (sum-to-product, half-angle, etc.)
- Pattern mark categories for different transformation types
- Integration with `tracing` for pattern detection debugging
- Performance optimization via early-exit in scanner

---

## Previous Releases

_(Historical changes will be documented here as the project evolves)_

