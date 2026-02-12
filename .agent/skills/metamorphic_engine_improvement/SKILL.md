---
name: Metamorphic Engine Improvement
description: Analyze metamorphic benchmark results (timeouts, cycles, numeric-only) to identify and fix engine inefficiencies
---

# Metamorphic Engine Improvement Skill

This skill guides systematic analysis of metamorphic benchmark results to discover and fix engine weaknesses. It should be invoked after running the unified metamorphic benchmark.

## 1. Run the Benchmark

// turbo
```bash
cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_unified_benchmark -- --ignored --nocapture 2>&1 | tee /tmp/metatest_output.txt
```

Capture the full output. The benchmark takes ~5 minutes.

## 2. Parse the Summary Table

Look for the `UNIFIED METAMORPHIC REGRESSION BENCHMARK` table at the end of the output. Extract:

| Metric | What It Measures |
|--------|------------------|
| **T/O (Timeouts)** | Combinations that exceeded the 5s wall-clock budget — indicates performance bottlenecks |
| **Cycles** | Combinations where the simplifier oscillated between forms — indicates missing cycle-breaking guards |
| **Numeric-only** | Combinations where equivalence was proven numerically but not symbolically — indicates normalization gaps |
| **Failed** | Semantic mismatches — indicates bugs (MUST be 0) |

Record the counts per suite (add, sub, mul, div, ⇄sub) for comparison against previous runs.

## 3. Analyze Timeouts (⏱️ T/O lines)

### 3.1 Extract Timeout Lines
Search the output for lines matching `⏱️  T/O`. Each line has the format:
```
⏱️  T/O [suite] #N: [Family A] * [Family B]  →  (expression_A) * (expression_B)
```

### 3.2 Categorize by Root Cause
Group timeouts into categories:

| Category | Pattern | Typical Fix |
|----------|---------|-------------|
| **Complex constant × polynomial** | Variable-free radical/surd multiplied by `(x+1)^n` | Add gate in `DistributeRule` (`polynomial/mod.rs`) to block distribution of complex irrationals |
| **Rationalization × polynomial** | `1/(1+x^(1/3)) * poly` | Investigate rationalization rule cost; add step budget or guard |
| **Trig resolution cascade** | `cos(π/N) * poly` where N is non-standard | Check if `values.rs` handles the angle; if so, the issue is post-resolution distribution |
| **Multi-variable product expansion** | `(x²+y²)(a²+b²)` or similar | Check expansion budget limits in `expansion.rs` |
| **Solver expression × anything** | Quadratic formula `(-b+√(...)/(2a)) * expr` | Multi-variable expressions are inherently expensive; may need solver-specific budget |

### 3.3 Investigation Workflow
For each category with >3 timeouts:

1. **Reproduce in REPL**: Try simplifying the expression directly:
   ```bash
   echo "simplify cos(pi/10) * (x+1)^4" | cargo run --release -p cas_cli
   ```
   If the REPL hangs, the timeout is confirmed as a simplifier bottleneck.

2. **Identify the expensive rule**: Use `METATEST_VERBOSE=1` to get per-rule breakdowns, or add temporary logging to the simplifier loop.

3. **Determine fix strategy**:
   - **Distribution guard**: If the issue is `constant * polynomial` expansion → add a gate in `DistributeRule` (see `is_complex_irrational_constant` pattern in `polynomial/mod.rs`)
   - **Step budget**: If the issue is rule oscillation → tighten the step budget in `budget.rs`
   - **Rule ordering**: If a cheaper rule should fire first → adjust rule priority in the relevant `register()` function

4. **Implement and verify**: Make the fix, run `cargo check`, run unit tests, then re-run the benchmark.

## 4. Analyze Cycles (🔄 lines)

Cycles indicate the simplifier is oscillating between equivalent forms (e.g., `a+b` → `b+a` → `a+b`).

### 4.1 Extract Cycle Info
Run with `METATEST_VERBOSE=1` to get per-rule cycle breakdowns:
```bash
METATEST_VERBOSE=1 cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_unified_benchmark -- --ignored --nocapture 2>&1 | tee /tmp/metatest_verbose.txt
```

Look for `🔄 Cycle Events Summary` and the per-rule breakdown.

### 4.2 Common Cycle Patterns

| Pattern | Cause | Fix |
|---------|-------|-----|
| Distribute ↔ Factor | Distribution undoes factoring and vice versa | Ensure `DistributeRule` and factor rules run in different phases (`PhaseMask`) |
| Ordering ↔ Rewrite | Canonical ordering triggers a rule that changes order | Add `compare_expr` guard to skip rewrites that don't change structure |
| Expand ↔ Collect | Expansion creates terms that get re-collected | Add phase gates or "already expanded" markers |

### 4.3 Fix Strategy
1. Identify the two conflicting rules from the verbose output
2. Check their `PhaseMask` configurations in the relevant `register()` function
3. Either separate them into different phases or add a mutual exclusion guard

## 5. Analyze Numeric-Only Results (🌡️)

Numeric-only means the engine can't symbolically prove equivalence — a normalization gap.

### 5.1 Get Detailed Report
```bash
METATEST_VERBOSE=1 cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_unified_benchmark -- --ignored --nocapture
```

Look for:
- **Family Classifier**: Groups numeric-only by function families (sec/csc, ln/log, sqrt/roots)
- **Top-N Shape Analysis**: Dominant residual patterns (NEG_EXP, DIV)
- **Residual LaTeX**: The symbolic difference `simplify(LHS - RHS)` for each case

### 5.2 Prioritize by Family
Focus on families with the highest numeric-only count. Common gaps:

| Family | Typical Gap | Where to Fix |
|--------|------------|--------------|
| **sec/csc** | Missing `sec(x) = 1/cos(x)` canonicalization | `trigonometry/identities/` |
| **sqrt/roots** | Missing `√a·√b = √(ab)` or radical denesting | `exponents/simplification.rs` |
| **ln/log** | Missing log rules (`ln(ab) = ln(a)+ln(b)`) | `rules/logarithm/` |
| **Polynomial** | Incomplete like-term collection after distribution | `polynomial/mod.rs` CombineLikeTermsRule |

### 5.3 Fix Strategy
1. Pick the residual LaTeX from the most common numeric-only family
2. Simplify it manually to identify the missing transformation
3. Check if a rule exists but isn't firing (wrong phase? wrong guard?)
4. If no rule exists, implement it following the existing `define_rule!` pattern

## 6. Recording Progress

After each analysis cycle, update these metrics:

```markdown
| Date | Timeouts | Cycles | Numeric-only | Failed | Notes |
|------|----------|--------|--------------|--------|-------|
| YYYY-MM-DD | N | N | N | 0 | What was fixed |
```

### Ratchet Policy
- **Timeouts**: Should decrease or stay constant after each fix
- **Failed**: MUST always be 0 — any increase is a regression
- **Numeric-only**: Should decrease as normalization gaps are closed
- **Cycles**: Should decrease as phase conflicts are resolved

## 7. Key Files Reference

| File | Purpose |
|------|---------|
| `crates/cas_engine/src/rules/polynomial/mod.rs` | Distribution rules and guards |
| `crates/cas_engine/src/rules/trigonometry/values.rs` | Trig angle lookup table |
| `crates/cas_engine/src/rules/trigonometry/evaluation.rs` | Trig evaluation rule |
| `crates/cas_engine/src/rules/exponents/simplification.rs` | Radical simplification |
| `crates/cas_engine/src/budget.rs` | Step budget configuration |
| `crates/cas_engine/tests/metamorphic_simplification_tests.rs` | The benchmark itself |
| `crates/cas_engine/tests/identity_pairs.csv` | Identity catalog |
