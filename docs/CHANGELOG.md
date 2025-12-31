# Changelog

All notable changes to ExpliCAS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.1] - 2025-12-31 - Blocked Hints

### Added

- **Pedagogical Blocked Hints**: When Generic mode blocks simplifications requiring analytic assumptions (e.g., `x > 0`), the REPL now displays actionable hints:
  ```
  ℹ️  Blocked in Generic: requires x > 0 [Exponential-Log Inverse]
     use `domain assume` to allow analytic assumptions
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

