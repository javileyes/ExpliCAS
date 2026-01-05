# Roadmap V2.1 → V2.4: Education-First Solver Evolution

> **Primary user**: Professor (reliable, explainable, presentable outputs with cases)  
> **Secondary**: Student (understands "why", clear guards, verification, steps)  
> **Tertiary**: Integrator (stable API, serialization, contracts/regression)

---

## V2.1 — API/UX Stability & Polish

### Issue #1: Output polishing — `otherwise:` without "if" ✅
**Labels**: `ux`, `printer`

Natural reading in console/timeline.

**Done when**:
- [x] REPL prints `otherwise: ...` (no `if` prefix)
- [x] LaTeX uses `\text{otherwise}` without `if` prefix
- [ ] 1 snapshot test updated

---

### Issue #2: REPL "iconic" snapshots (UX regression) ✅
**Labels**: `tests`, `ux`

Detect experience breakage instantly.

**Implemented set (8 tests)**:
1. `solve a^x=a, x` with budget=2 (Conditional 3 cases) ✅
2. `solve a^x=a, x` with budget=1 (fallback to {1}) ✅
3. `solve 0^x=0, x` (interval x>0) ✅
4. `solve 2^x=8, x` (x = ln(8)/ln(2)) ✅
5. `solve x^2-4=0, x` (quadratic: {-2, 2}) ✅
6. `solve x+2=5, x` (simple linear: {3}) ✅
7. `solve 2*x=10, x` (linear mult: {5}) ✅
8. `solve a^x=a^2, x` (equal bases: {2}) ✅

**Done when**:
- [x] Deterministic snapshots (no IDs/noise)
- [x] `cargo test` runs them
- [x] Docs in test file: "how to update snapshots"

---

### Issue #3: "Explain mode" for solve (compact summary) ✅
**Labels**: `edu`, `ux`

Already have assumptions/hints/timeline; need a "professor mode" compact view.

**Done when**:
- [x] `explain on|off` command toggles explain mode
- [x] On `solve`, prints:
  - "Assumptions used" (if any) - dedup, stable order
  - "Blocked simplifications" (if any) - with contextual tip
- [x] Correct dedup, stable order

---

### Issue #4: Stable results API (for integrators) ✅
**Labels**: `api`, `ffi`

What you show in console = what you export.

**Done when**:
- [x] `cas_engine::api` module re-exports stable types:
  - `SolveResult`, `SolutionSet`, `Case`, `ConditionSet`, `ConditionPredicate`
  - `SolveBudget`, `SolverOptions`
  - `solve`, `solve_with_options`
  - `DisplayExpr`, `LaTeXExpr`
- [x] Stable API documented in api.rs
- [x] 10 compile tests in `public_api_contract.rs`

---

## V2.3 — Solution Verification (Educational Gold)

### Issue #5: `solve --check` (solution verification)
**Labels**: `solver`, `edu`

Confidence + didactic tool.

**Minimal design**: Verify by substitution in original equation, simplify with `Strict`, evaluate if result is `True/0` or residual.

**Done when**:
- [ ] Command: `solve --check ...` or `semantics set solve check on`
- [ ] For each solution/case:
  - Prints `✓ verified` or `⚠ unverifiable` + reason
- [ ] Respects guards: verifies "under guard" using guard env

---

### Issue #6: Verification for intervals/sets
**Labels**: `solver`, `edu`

Solver already returns intervals (`(0,∞)`), etc.

**Done when**:
- [ ] When verifying `x ∈ (0,∞)`, prints:
  - "verified symbolically under guard" if possible
  - or "requires numeric sampling" if not ready yet
- [ ] Doesn't lie: if it can't verify, marks as unverifiable

---

### Issue #7: "Counterexample hint" basic (only on failure)
**Labels**: `edu`, `solver`

Super educational: "this solution fails if…"

**Done when**:
- [ ] If a solution doesn't verify in Strict, attempts to find simple counterexample (only typical literals: 0,1,2,-1 if applicable) and shows it
- [ ] If none found, doesn't invent one

---

## V2.2 — Expressive Predicates & Guards

### Issue #8: Add `NeZero/NeOne` + basic simplification
**Labels**: `solver`, `guards`

**Done when**:
- [ ] `ConditionPredicate` includes `NeZero/NeOne`
- [ ] `ConditionSet::simplify()`:
  - `EqZero` + `NeZero` ⇒ contradiction
  - `EqOne` + `NeOne` ⇒ contradiction
  - `NeOne` doesn't eliminate anything yet (but allows clean guards)
- [ ] Snapshots updated if prints change

---

### Issue #9: Cheap extra implications (guard quality)
**Labels**: `solver`, `guards`

**Done when**:
- [ ] Safe rules (RealOnly):
  - `EqOne(x) ⇒ Positive(x)` (if deciding to assume 1>0 as numeric)
  - `Positive(x) ⇒ NeZero(x)` (redundancy)
- [ ] All with simplification tests

---

## V2.4 — New Solve Strategies (with Safety Net)

### Issue #10: Didactic strategy — clear denominators with guards ✅
**Labels**: `solver`, `edu`

Typical classroom rational equations.

Example: `(x^2-1)/(x-1)=0` → guard `x≠1` and solve `x+1=0`.
Example: `(x*y)/x=0, x` → guard `x≠0` and result is AllReals (when y=0).

**Done when**:
- [x] Strategy "clear denominators":
  - Extracts denominators containing solve variable
  - Produces Conditional with guards `den≠0`
  - Doesn't lose exclusions
- [ ] 2 iconic tests (pending - basic functionality works)

---

## Focus Notes

- In education, most important is **output stability** and **honesty** (if can't verify, say so)
- V2.4 only after `--check`, so each new strategy comes with safety net
- Run `make ci` before any merge

---

## Suggested Order

1. **Issue #1** (5 min) — `otherwise:` polish
2. **Issue #2** (1-2h) — Snapshots infrastructure
3. **Issue #3** (1h) — Explain mode
4. **Issue #4** (30 min) — API stability
5. **Issue #5** (2-3h) — `--check` verification
6. **Issue #6** (1h) — Interval verification
7. **Issue #7** (1h) — Counterexample hints
8. **Issue #8** (1h) — `NeZero/NeOne`
9. **Issue #9** (30 min) — Implications
10. **Issue #10** (2h) — Clear denominators strategy
