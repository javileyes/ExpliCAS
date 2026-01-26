# N4: AST Variants for Hold and PolyResult

## Status
- **Document only** (no implementation yet).
- This design is intended to be executed incrementally with strong backwards compatibility guarantees.

## Motivation

ExpliCAS currently represents some internal wrappers as function calls in the AST:
- `__hold(x)` used as a barrier / control wrapper.
- `poly_result(id)` used as a lightweight reference into PolyStore.

Even after eliminating stringly-typed checks (`BuiltinFn::Hold` + helpers, `poly_result.rs` helpers, and enforced lints),
the representation is still "function-shaped". This has costs:
- Wrappers are not exhaustively handled by `match` (easy to forget in new code).
- Boilerplate: repeated extraction/validation patterns.
- Performance: some overhead around function dispatch/matching even when treated specially.

### Goal
Introduce dedicated AST variants:
- `Expr::Hold(ExprId)`
- `Expr::PolyResult(PolyId)` (or an equivalent typed id)

This enables:
- Exhaustive handling by `match`.
- Cleaner, safer code with fewer ad-hoc patterns.
- Faster and more direct access to wrapper semantics.

## Non-Goals
- Changing user-facing syntax. Users will still write `__hold(x)` and see it printed similarly.
- Changing the semantics of hold barriers or PolyStore references.
- A big-bang refactor. This must be incremental.

## Compatibility Requirements

### Parser compatibility
- Input `__hold(x)` must continue to parse.
- Input `poly_result(123)` must continue to parse (even if this is internal-facing).
- During transition, parser may emit the old function-shape or the new variants depending on phase.

### Display/LaTeX compatibility
- Default pretty printing should remain stable:
  - `Expr::Hold(x)` prints as `__hold(x)` (or an equivalent canonical form).
  - `Expr::PolyResult(id)` prints as `poly_result(id)`.

### Serialization compatibility
If AST is serialized anywhere (JSON, logs, snapshots):
- Provide a stable representation.
- During transition, accept both representations when deserializing:
  - Function-shaped wrappers
  - Variant-shaped wrappers

## Proposed AST Changes

### New variants
Add to `Expr` enum:
```rust
Hold(ExprId),
PolyResult(PolyId),
```

### Canonicalization rule
After parsing or during evaluation, wrappers should be normalized to the new variants.
The old function-shaped wrappers remain supported only as an input/compatibility layer.

## Phase Plan

### Phase 1 — Introduce variants + bridging helpers (no behavioral change)
- Add new `Expr` variants.
- Add canonical helpers (single source of truth):
  - `hold::is_hold(...)`, `hold::unwrap_hold(...)`, `hold::wrap_hold(...)`
  - `poly_result::is_poly_result(...)`, `poly_result::parse_poly_result_id(...)`, `poly_result::wrap_poly_result(...)`
- Extend helpers to support both representations:
  - If expression is function-shaped `__hold(x)`, treat it as Hold.
  - If expression is `Expr::Hold(x)`, treat it as Hold.
  - Same for PolyResult.

**Exit criteria:**
- All existing tests + golden corpus pass with no output changes.
- Lints still enforce "no stringly typed" checks.

### Phase 2 — Normalize production creation paths to variants
- Modify `wrap_hold` and `wrap_poly_result` to produce the new variants.
- Ensure no other code constructs function-shaped wrappers directly (enforced via lints).
- If necessary, keep a compatibility function to construct legacy wrappers only for specific contexts (debug/testing).

**Exit criteria:**
- `__hold` and `poly_result` wrappers in runtime AST are variants in steady state.
- Golden corpus unchanged.

### Phase 3 — Migrate match sites to handle variants explicitly
- Update key modules that pattern-match `Expr::Function` for wrapper detection to instead match:
  - `Expr::Hold(_)`
  - `Expr::PolyResult(_)`
- Keep bridging inside helpers, but reduce reliance on function-shaped matching.

**Exit criteria:**
- No production logic checks for `Expr::Function("__hold", ...)` or `"poly_result"` anywhere.
- Helpers remain the only compatibility layer.

### Phase 4 — Parser emits only variants (optional)
- Parser lowers `__hold(x)` directly into `Expr::Hold(x)`.
- Parser lowers `poly_result(n)` into `Expr::PolyResult(PolyId::from(n))` if valid, else into a normal function (or error), depending on policy.

**Exit criteria:**
- Function-shaped wrappers are only accepted as legacy input if needed.
- Consider removing legacy acceptance if not required.

### Phase 5 — Remove legacy representations (optional)
- Remove function-shaped acceptance paths if they are truly unused.
- Tighten lints to ensure variants are used everywhere.

## Key Design Questions (to decide before implementation)

1) **Where does PolyId live?**
   - If `PolyId` is engine-specific, consider a small id type shared via a common crate or newtype in AST.

2) **Should PolyResult accept only typed ids?**
   - Recommended: `Expr::PolyResult(PolyId)` only.
   - Legacy parse helper may accept `poly_result(Number(n))` and convert to `PolyId`.

3) **Hold semantics**
   - Confirm that Hold is purely a barrier wrapper and does not affect mathematical equivalence.
   - Ensure rule pipelines respect this barrier.

4) **Serialization format**
   - Decide whether to serialize variants as:
     - `{"Function": {"name": "__hold", "args": [...]}}` (legacy shape), or
     - `{"Hold": ...}` / `{"PolyResult": ...}` (new shape)
   - Recommended: accept both; emit a stable format (likely legacy for compatibility unless you want a format version bump).

5) **User-facing `hold()` vs internal `__hold`**
   - `__hold(expr)`: Internal barrier, transparent in display, stripped before output.
   - `hold(expr)`: User-facing function with HoldAll semantics, VISIBLE in output.
   - N4 `Expr::Hold` should represent ONLY `__hold` (internal).
   - User `hold()` should remain as `Expr::Function("hold", ...)` with HoldAll attribute.
   - Alternative: Introduce `Expr::UserHold(ExprId)` if you want variant-exhaustive matching,
     but this adds complexity. Recommended: keep user `hold` as a function with attribute flag.

## Testing & Verification

- Full workspace test suite.
- Golden corpus must remain stable (no panic + stable outputs).
- Add focused unit tests:
  - `Hold` round-trip: `wrap_hold` then `unwrap_hold` returns original.
  - `PolyResult` parse/print round-trip.
  - Ensure helpers behave identically for function-shaped and variant-shaped wrappers during transition.

## Risks & Mitigations

### Risk: output drift
- Mitigation: golden corpus + stable printers + normalization rules.

### Risk: serde breaking changes
- Mitigation: accept both shapes; optionally bump a format version if you have one.

### Risk: PolyStore/thread-local coupling
- Mitigation: N4 does not require changing PolyStore. It only makes references typed in AST.
  Further store refactors can remain separate.

## Success Criteria
- Internal wrappers are represented with dedicated variants.
- No string comparisons for internal wrappers anywhere in production code.
- Golden corpus remains stable across refactor.
- New contributors have a clear model: wrappers are variants, not "magic functions".

## Related Work
- [builtin_guidelines.md](builtin_guidelines.md) - BuiltinFn API reference
- N3: `BuiltinFn::Hold` stabilization (completed)
- `make lint-no-stringly-ir` enforces 0 violations for both `__hold` and `poly_result`
