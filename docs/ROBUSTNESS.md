# Robustness & Safety Guarantees

This document describes the defensive engineering measures in ExpliCAS to ensure:
- **No `panic!` escapes to end users in production**, and
- **Future regressions are caught early** via CI and a golden corpus.

## Goals

1. **Production stability**
   - Avoid panics and uncontrolled crashes in production code paths.
   - Convert failures into structured errors (`CasError`) or safe residual outputs.

2. **Defense in depth**
   - Even if an internal bug or regression triggers a panic, the REPL remains alive and the user sees a stable error message.

3. **Regression detection**
   - A golden corpus ensures common workflows do not panic and remain stable across refactors.

---

## CI Enforcement: No panics in production

We enforce "no `panic!` in production code" with a dedicated make target.

### Command

```bash
make lint-no-panic-prod
```

### Policy

* The check runs over **production targets** (library/binaries/examples/benches), excluding `#[cfg(test)]`.
* Intentional panics are allowed only in explicitly documented places (see below).
* This is a **hard gate** in CI.

---

## Production panic strategy

### 1) Builders: safe fallback + debug assertions

Some internal expression builders can be called with empty input in corner cases.
Instead of panicking, we:

* return a mathematically correct **identity**
* keep a **debug-only assertion** to catch unintended invariant breaks during development

Examples:

* `build_balanced_add([])` → returns `0` + `debug_assert!`
* `build_balanced_mul([])` → returns `1` + `debug_assert!`

This prevents production crashes while still flagging suspicious internal states in debug builds.

### 2) Intentional panics (documented + scoped allow)

Some panics are intentional because they represent hard safety limits or debug-only invariants.

* `recursion_guard`: intentional panic when recursion depth is exceeded.
  * This is a safety mechanism to avoid runaway recursion.
  * Panics are explicitly allowed in this module with a scoped lint allowance.

* `engine` debug invariants: duplicate rule names / internal sanity checks.
  * Panics (if any) are scoped to debug-only checks.
  * Production stability remains guaranteed.

---

## Panic fence in REPL: catch_unwind + error_id

Even with production panics removed, future regressions can introduce a panic (e.g. via a new `unwrap()` path).
To ensure end-user stability, the REPL dispatch layer catches panics:

* REPL stays alive.
* User receives a stable error message with an **Error ID**.
* Developers can correlate the Error ID with stderr logs.

### User-facing behavior

When a panic is caught:

```
✖ Internal error (id: A3F8B2): <panic message>

The session is still active. You can continue working.
Please report this issue with the error id if it persists.
```

### Developer logging

Logging details can be enabled with:

```bash
EXPLICAS_PANIC_REPORT=1
```

When enabled, output includes:

* error id
* command line that triggered the panic
* ExpliCAS version
* panic message

> Note: `catch_unwind` requires panic strategy `unwind`. If production builds use `panic=abort`, the fence will not catch panics.

---

## Error taxonomy: stable external errors

Public API and user-facing layers return structured errors via `CasError`:

* `kind()` provides a stable classification (user input vs unsupported vs internal vs budget, etc.)
* `code()` provides a stable error code for clients

Conversions exist for common sources:

* `From<BudgetExceeded>`
* `From<ParseError>`

We also use `ensure_invariant!` for safe, explicit invariant checks when needed.

---

## Golden corpus tests

We maintain a golden corpus that executes representative user commands to ensure:

* no panics occur in common workflows
* behavior stays stable across refactors

### Minimal corpus

* File: `crates/cas_engine/tests/corpus/basic.txt` (~80 commands)
* Coverage:
  * arithmetic, algebra
  * solve
  * calculus
  * edge cases

### Tests

Run:

```bash
cargo test --test golden_corpus_tests
```

Included tests:

* `corpus_basic_no_panic`: executes all commands in `basic.txt`
* `corpus_solve_commands_no_panic`: executes solve-related commands

### Extending the corpus

Add commands to `tests/corpus/*.txt` and corresponding tests for:

* polynomials (gcd/factor/modp)
* limits and asymptotics
* logarithms/exponentials
* rationalization and domain constraints
* large expressions / budget behavior

Prefer:

* stable commands (deterministic outputs)
* normalization rules for non-deterministic fields (timing, ids)

---

## Developer workflow

Recommended local checks:

```bash
# Full workspace tests
cargo test --workspace

# Lint and warnings
cargo clippy --workspace -- -D warnings

# Enforce no panics in production
make lint-no-panic-prod

# Golden corpus
cargo test --test golden_corpus_tests

# Full CI
make ci
```

---

## Stringly-typed IR enforcement

We enforce type-safe patterns for internal IR nodes like `__hold` and `poly_result`.

### Command

```bash
make lint-no-stringly-ir
```

### Policy

| Node | Status | Baseline | Helpers |
|------|--------|----------|---------|
| `poly_result` | **Enforced** | 0 | `is_poly_result`, `parse_poly_result_id`, `wrap_poly_result` |
| `__hold` | **Enforced** | 0 | `is_hold`, `unwrap_hold`, `strip_all_holds` |

* `poly_result`: **hard gate** in CI. Any string comparison (e.g., `name == "poly_result"`) outside `poly_result.rs` fails the build.
* `__hold`: **hard gate** in CI. ✅ Enforced at 0 violations (Jan 2026). All access goes through canonical helpers in `cas_ast::hold`.

### Canonical modules

* `cas_ast::hold` — all `__hold` helpers
* `cas_engine::poly_result` — all `poly_result` helpers

---

## Summary: current guarantees

| Guarantee | Status |
|-----------|--------|
| Production targets enforce no `panic!` via CI | ✅ |
| Known intentional panics are documented and scoped | ✅ |
| REPL has a panic fence with error_id reporting | ✅ |
| Golden corpus prevents regressions in common workflows | ✅ |
| `poly_result` stringly-typed checks enforced to 0 | ✅ |
| `__hold` stringly-typed checks capped at baseline | ✅ |
| Multinomial expansion capped by 6-layer guard system (output_nodes ≤ 350) | ✅ |
| `budget_exempt` usage enforced via CI allowlist | ✅ |
