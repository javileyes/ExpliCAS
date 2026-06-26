# Soundness Audit — Round 5 (2026-06-26)

Fifth multi-axis adversarial soundness audit (ultracode), run AFTER the Cardano cubic-solver
cycles (1–4) and the fraction-base-power printer fix landed this session. Baseline commit `f88f627aa`.

## Method

- **16 fronts** hunted in parallel via a `Workflow` pipeline (solve poly/transcendental/radical-abs,
  inequalities, simplify, factor/gcd, diff, integrate, limits, series/sum, matrices, complex/abs/domain,
  trig, exp/log, **printer re-parse round-trip**, exact rational arithmetic), plus a completeness critic
  that spawned gap fronts. Each candidate was piped to a **default-reject adversarial verifier** that
  reproduced it on the release CLI and re-derived the correct real-domain answer EXACTLY (sympy/numpy,
  substitution, domain sampling, printed-form re-parse) — never float alone.
- **64 agents, ~1.5M tokens.** Read-only (CLI + python3; no git/file mutation).
- **NOTE on overlap:** agents were NOT fed the Rounds 1–4 defect lists, so they independently
  RE-DISCOVERED several already-logged open clusters (inequality operator-drop, symbolic power-law folds,
  matrix-shape mismatch, atanh-domain over-restriction). This round therefore doubles as a **regression
  check**: those clusters remain OPEN. The recent cubic solver + printer work was confirmed SOUND
  (no new defects; the cubic solver is clean except the already-documented factor-behind-irreducible-
  quadratic peldaño).
- **Operator spot-check:** 11 headline P0/P1 claims were independently re-run by hand and ALL reproduce
  exactly (e.g. `solve(sqrt(x)=-sqrt(x),x)`→`All real numbers`; `x^3+x+1>0` and `<0` byte-identical;
  `(x^a)^b`→`x^(a·b)` no condition; `(-1)^(1/3)` complex →`-1`; `x^5-5x^3+x^2-5=0`→`-1`;
  matrix shape-mismatch → `ok:true` echo). The report is faithful to engine behaviour.

## Headline

**40 candidates → 38 confirmed unsound** (reproduced AND independently adjudicated): **8 P0, 19 P1,
9 P2, 2 P3**. Categories: 14 lost-domain-condition, 11 dishonest-residual, 5 wrong-value,
4 missing-root, 3 display-ambiguity, 1 other.

# CAS-Engine Soundness Audit Report

## 1. Executive Summary

**Soundness is NOT broadly guaranteed.** The engine is strong and honest across large swaths of its surface (arithmetic, derivatives, limits, series/sums, matrix arithmetic on compatible shapes, most polynomial/transcendental/radical solving, exact-value trig), but it has **systematic, reproducible soundness holes** concentrated in five areas: (a) **inequality solving of irreducible polynomials** (operator silently dropped), (b) **symbolic-exponent power-law folds** (sign/parity guard dropped → wrong value), (c) **complex principal-branch powers of negative bases** (odd-root convention leaks into complex mode), (d) **plain-text formatter dropping domain conditions** on `solve` identities, and (e) **shape-incompatible matrix ops** dishonestly reported `ok:true`.

**Confirmed defect counts by severity:**

| Severity | Count | Nature |
|---|---|---|
| **P0** | 8 | Silent wrong value / wrong-kind result, trusted at the API boundary |
| **P1** | 18 | Missing roots, lost domain conditions, dishonest residuals (incl. one matrix family counted per-input) |
| **P2** | 7 | Display/honesty defects with structured-channel mitigation, or single-point domain loss |
| **P3** | 2 | Incompleteness / text-only condition drop, value-correct |
| **Total** | **35** | All reproduced AND independently adjudicated unsound |

Two candidates were **refuted** (not defects): the `ln(a)+ln(b)→ln(a·b)` strict-mode merge (conditions honestly surfaced; only a contract-consistency nit) and `((-8)^2)^(1/6)→2` (correct; the two nested/flat forms are genuinely different expressions and the engine does not collapse them).

The single most alarming pattern is **`ok:true` paired with a wrong or undefined result and empty `warnings`/`required_conditions`** — a consumer trusting the success flag is misled. This appears in the matrix-shape family, the quartic/abs circular residuals, and the inequality operator-drop.

---

## 2. Table of Confirmed Defects (by severity)

| Sev | Front | Input | Observed | True answer | Category | Why unsound (one line) |
|---|---|---|---|---|---|---|
| P0 | radicals | `solve(sqrt(x)=-sqrt(x), x)` | `All real numbers` | `{0}` | wrong_value | Sign-cancel bug yields ℝ; engine's own `sqrt(4)-(-sqrt(4))=4≠0` refutes it. |
| P0 | inequalities | `x^3+x+1>0` | `{ (1/6·(-sqrt(31/3)-3))^(1/3)+… }` (root set) | `(-0.68233, ∞)` | lost_domain_condition | `>` dropped; rewritten to `Equal(p,0)`, returns equation root set. |
| P0 | inequalities | `x^3+x+1<0` | same set as `>0` | `(-∞, -0.68233)` | wrong_value | `<` and `>` give byte-identical output; operator ignored. |
| P0 | inequalities | `x^3-3x+1>0` | trig root set (both operators) | `(-1.8794,0.3473)∪(1.5321,∞)` | wrong_value | Irreducible cubic ineq → equation root set; wrong-kind object. |
| P0 | simplify (powers) | `(x^a)^b` | `x^(a·b)`, no conditions | requires `x≥0` (else `|x|`) | lost_domain_condition | Symbolic-exp fold drops base-nonnegativity; `a=2,b=½,x=-3`→-3 vs true 3. |
| P0 | simplify (powers) | `((-2)^a)^b` | `(-2)^(a·b)` | not equal to fold; sign lost | wrong_value | Unconditional fold over negative literal base; engine's own eval gives 2 vs folded -2. |
| P0 | simplify (powers) | `(x^n)^(1/n)` | `x` (only `n≠0`) | `|x|` for even `n` | lost_domain_condition | Symbolic-`n` path drops parity guard; `n=2,x=-3`→-3 vs true 3. |
| P0 | complex | `(-1)^(1/3)` `--value-domain complex` | `-1` | `½ + (√3/2)i` | wrong_value | Real odd-root convention leaks into principal-branch complex mode. |
| P1 | polynomial solve | `x^5-5*x^3+x^2-5=0` | `{ -1 }` | `{-√5, -1, √5}` | missing_root | Expanded quintic loses `x²-5` factor; standalone solves fine. |
| P1 | quartic solve | `x^4-8*x^2+15=0` | circular `solve(x-(8x²-15)^(1/4)=0)` | `{±√3, ±√5}` | missing_root | Broken "isolate xⁿ, take nth root" fallback; 0 of 4 roots, `ok:true`. |
| P1 | transcendental solve | `solve(tan(x)=tan(x), x)` | `All real numbers` | `ℝ \ {π/2+nπ}` | lost_domain_condition | Implicit tan pole-domain dropped entirely (even from JSON). |
| P1 | abs solve | `solve(abs(x^2-2*x)=3, x)` | leaked `Solve: solve(…)=0` residual | `{-1, 3}` | missing_root | abs-branch needing a further squaring bails, emits unparseable residual, 0 roots. |
| P1 | inequalities | `sqrt(x-1)+sqrt(x-2)<3` | `Solve: solve(…)=0` dump | `[2, 34/9)` | dishonest_residual | Internal solver dump as result; inner piece solvable to `34/9`. |
| P1 | inequalities | `x^4-x-1>0` | `Solve: solve(x-(x+1)^(1/4)=0)=0` | `(-∞,r₀)∪(r₁,∞)` | dishonest_residual | Garbled substitution dump; loses elementary two-interval answer. |
| P1 | integrate | `integrate(1/(1-x^2), x)` | `atanh(x)`, requires `-1<x<1` | `½ln|（1+x)/(1-x)|`, only `x≠±1` | lost_domain_condition | Spurious positivity condition forbids valid |x|>1; output non-real there. |
| P1 | integrate | `integrate(1/(4-x^2), x)` | `½atanh(x/2)`, requires `-2<x<2` | `¼ln|(x+2)/(x-2)|`, only `x≠±2` | lost_domain_condition | Same atanh-family over-restriction across `c²-x²`. |
| P1 | logs solve | `solve(ln(x^2)=2*ln(x), x)` | `All real numbers` | `x>0` | lost_domain_condition | Plain text drops computed `x>0` (present in JSON). |
| P1 | logs solve | `solve(2*ln(x)=ln(x^2), x)` | `All real numbers` | `x>0` | lost_domain_condition | Same; sides swapped. |
| P1 | logs solve | `solve(ln(2*x)=ln(x)+ln(2), x)` | `All real numbers` | `x>0` | lost_domain_condition | Same; plain-text condition drop. |
| P1 | logs solve | `solve(e^(ln(x))=x, x)` | `All real numbers` | `x>0` | lost_domain_condition | Same; computed `x>0` not surfaced in text. |
| P1 | radicals solve | `solve(sqrt(x)^2=x, x)` | `All real numbers` | `x≥0` | lost_domain_condition | Same plain-text drop of `x≥0`. |
| P1 | matrix shape | `[[1,2],[3,4]] + [[1,2,3],[4,5,6]]` | echoed sum, `ok:true` | UNDEFINED (ShapeError) | dishonest_residual | Shape mismatch reported as success, no warning. |
| P1 | matrix shape | `[[1,2],[3,4]] * [[1,2,3]]` | echoed product, `ok:true` | UNDEFINED (ShapeError) | dishonest_residual | Incompatible product as clean residual. |
| P1 | matrix shape | `[[1,2],[3,4]] - [[1,2,3]]` | echoed diff, `ok:true` | UNDEFINED (ShapeError) | dishonest_residual | Same; plus lossy row/col display round-trip. |
| P1 | matrix shape | `[[1,2,3],[4,5,6]]^2` | echoed `^2` residual, `ok:true` | UNDEFINED (non-square) | dishonest_residual | `x*x` shape-blind rule generates the undefined power. |
| P1 | matrix shape | `([[1,2],[3,4]]+[[1,2,3],[4,5,6]])*[[1,2],[3,4]]` | distributed undefined terms, `ok:true` | UNDEFINED | dishonest_residual | Distributes an undefined sum, manufacturing a second undefined op. |
| P1 | parametric solve | `solve(a*x = a, x)` | `{ 1 }`, all channels empty | `{1}` if `a≠0`; ℝ if `a=0` | lost_domain_condition | Single-symbol cancel drops `a≠0` guard and `a=0` branch. |
| P2 | polynomial solve | `(x^2+1)*(x^3-5*x+1)=0` | restated product, no roots | 3 real cubic roots | missing_root | Cubic-beside-irreducible-quadratic silently drops all real roots. |
| P2 | logs solve | `solve(e^(ln(x)) = x, x)` | `All real numbers` (text) | `x>0` | lost_domain_condition | Text drops `x>0`; structured channels honest (hence P2). |
| P2 | factor | `factor(x^12-1)` | `factor(x^12-1)` echoed, `ok:true` | 6 cyclotomic factors | dishonest_residual | Unevaluated meta-call presented as answer; engine can factor it. |
| P2 | factor | `factor(x^11+x^10+1)` | unchanged, `ok:true` | `(x²+x+1)(x⁹-…+1)` | dishonest_residual | Reducible poly asserted irreducible. |
| P2 | complex | `log(-1)` `--value-domain complex` | `undefined` | `iπ` | dishonest_residual | Defined principal value reported with the does-not-exist sentinel. |
| P2 | logs solve | `solve(ln(x^2)=2*ln(abs(x)), x)` | `All real numbers` | ℝ\{0} | lost_domain_condition | Text drops single-point `x≠0`; JSON honest. |
| P2 | cubic display | `solve(x^3+x+1=0,x)` | Cardano cube-root sum | real root `-0.68233` | display_ambiguity | Printed form denotes a different value under declared principal branch. |
| P2 | display | `(1/2)!` | `1/2!` | `√π/2 ≈ 0.886` | display_ambiguity | Re-parses to `1/(2!)=0.5`; missing parens. |
| P2 | matrix shape | `[[1,2,3]] + 5` | `[1,2,3] + 5` echoed, `ok:true` | UNDEFINED (no broadcast convention) | dishonest_residual | matrix+scalar unevaluated yet `ok:true`, no warning. |
| P3 | factor | `factor(x^9-1)` | `(x-1)(x⁸+…+1)` | `(x-1)(x²+x+1)(x⁶+x³+1)` | other | Under-factors (value-correct round-trip); incompleteness. |
| P3 | parametric solve | `solve(a*x = b, x)` | `{ b/a }` (text) | `b/a`, requires `a≠0` | display_ambiguity | `a≠0` present in JSON/REPL/envelope, dropped only from `eval --format text`. |

> Borderline notes: **`solve(x^3+x+1=0,x)`** (P2) and **`solve(a*x=b,x)`** (P3) are genuinely borderline — the engine's *internal value* is the correct real root / `b/a`, and the issue is purely how the surface re-parses (cube-root convention) or which output channel carries the condition. **`solve(e^(ln(x))=x,x)`** and the matrix `[[1,2,3]]+5` case are also mitigated by honest structured channels. They are real but lower-impact than the P0/P1 wrong-value cases.

---

## 3. P0 / P1 Reproductions and Root-Cause Hypotheses

### P0-1 — `solve(sqrt(x)=-sqrt(x), x)` → `All real numbers` (true `{0}`)
`target/release/cas_cli eval "solve(sqrt(x)=-sqrt(x), x)"` prints `All real numbers` (JSON `ok:true`, `result_latex=\mathbb{R}`). The engine's own arithmetic refutes it: `sqrt(4)-(-sqrt(4))=4≠0`, and the JSON's own `required_conditions` list both `x≥0` and `-sqrt(x)≥0`, which jointly force `sqrt(x)=0 ⇒ x=0` — internally incoherent. **Root cause:** a "subtract both sides" normalizer mis-cancels `sqrt(x)-(-sqrt(x))` to `0`, collapsing the equation to the tautology `0=0 ⇒ ℝ` instead of `2·sqrt(x)=0`. Close analogues `2*sqrt(x)=0`, `sqrt(x)+sqrt(x)=0`, `sqrt(x)=-2*sqrt(x)` all correctly return `{0}`; only the unit-coefficient mirror collapses.

### P0-2/3/4 — Irreducible polynomial inequalities drop the operator
`x^3+x+1>0`, `x^3+x+1<0`, `x^3-3x+1>0`/`<0` all return the **equation's root set**, byte-identical across `>`, `<`, `>=`, `=`. The JSON exposes the smoking gun: `input_latex = \text{Equal}(p, 0)` — the parser/normalizer rewrites the strict inequality into an `Equal` relation, discarding the operator, then hands it to the root solver. `<0` and `>0` returning identical sets is self-contradictory (their solution sets must be disjoint). **Root cause:** the inequality-solving path is only wired for inequalities whose roots are *rational/factorable*; the moment roots are Cardano/trig/radical forms, the input is routed to the equation solver and the operator is lost. Control `x^2-1>0 → (-∞,-1)∪(1,∞)` proves the interval machinery exists and is operator-sensitive for factorable inputs.

### P0-5/6/7 — Symbolic-exponent power-law folds drop sign/parity guards
`(x^a)^b → x^(a·b)`, `((-2)^a)^b → (-2)^(a·b)`, `(x^n)^(1/n) → x` (only `n≠0`), all with empty `required_conditions`/`warnings`. Witnesses: `(x^a)^b` at `a=2,b=½,x=-3` folds to `x¹=-3` but the true value `((-3)²)^(½)=3`; `(x^n)^(1/n)` at `n=2,x=-3` likewise gives `-3` vs true `3`. **Root cause:** the parity/sign domain reasoning is carried correctly when exponents are **concrete literals** (the engine emits `x≥0`, or returns `|x|` for `(x^2)^(1/2)`), but is **dropped entirely on the symbolic-exponent path**. The engine literally knows the guard — for `x^(1/2)·x^(1/2)→x` it emits `x≥0`, and `(x^n)^(1/n)`'s own `--steps` trace admits rule "Root Power Cancel" requires `x>0` — yet that guard never reaches `required_conditions` while the result is still folded to `x`. This is the highest-leverage cluster: one fix (propagate the concrete-exponent guard to the symbolic path) likely closes all three.

### P0-8 — `(-1)^(1/3)` `--value-domain complex` → `-1` (true `½+√3/2 i`)
Under `--value-domain complex` (stated contract: "principal branch"), odd-denominator fractional powers of negative bases return the **real odd-root** value. `(-1)^(1/3)→-1`, `(-8)^(1/3)→-2`, `(-1)^(2/3)→1`, all wrong by the principal-branch definition `z^w=exp(w·Log z)`. Decisive inconsistency: the **even-root** case `(-4)^(1/2)→2i` IS correctly principal. **Root cause:** the rational-power evaluator dispatches negative-base-with-odd-denominator through the real-odd-root code path even when `value_domain=complex`, instead of `exp((p/q)·Log(base))`. The branch label is mislabeled "principal" for these cases.

### P1 — Missing roots in `solve`
- **`x^5-5*x^3+x^2-5=0` → `{-1}`** (true `{-√5,-1,√5}`). Factor is `(x+1)(x²-5)(x²-x+1)`. The engine solves `x²-5=0` standalone and even the explicitly-factored product correctly; it only loses `±√5` on the *expanded* quintic. **Root cause:** a lost-factor regression in the higher-degree factorizer — the `x²-5` factor is dropped during factorization of the expanded form.
- **`x^4-8*x^2+15=0` → circular `solve(x-(8x²-15)^(1/4)=0)`**, `ok:true`, 0 of 4 roots. Systematic across middle-term quartics lacking rational roots (`x^4-3x^2+1`, `x^4-6x^2+7`, `x^4-2x^2-1`, …). **Root cause:** the "isolate xⁿ then take nth root" fallback fires on biquadratics without rational roots, producing a circular pseudo-form with `x` on both sides and emitting it as the result.
- **`(x^2+1)*(x^3-5*x+1)=0` → restated product, no roots** (P2; cubic-beside-irreducible-quadratic). The standalone cubic returns 3 trig-form roots correctly; the cubic factor *beside* an irreducible quadratic triggers a silent drop.

### P1 — `solve(abs(x^2-2*x)=3, x)` → leaked residual (true `{-1,3}`)
Output `Solve: solve(x-(2x+3)^(1/2)=0, x)=0 if (2x+3)^(1/2)>=0`, `ok:true`, 0 roots; the string does not even re-parse. Whole family `abs(quadratic-with-linear-term)=expr` affected. **Root cause:** when an abs-branch reduces to `x=sqrt(linear)` requiring a further squaring step, the solver bails and emits the unsolved sub-problem as the result. Boundary cases without a linear term (`abs(x^2+x)=2→{-2,1}`) succeed.

### P1 — Inequality dishonest residuals (`sqrt(x-1)+sqrt(x-2)<3`, `x^4-x-1>0`)
Both emit raw `Solve: solve(…)=0` internal dumps that fail to re-parse (`Parse error at 5..6`), while `ok:true`. For the radical case the engine can solve the inner equation (`=3 → {34/9}`) and produces honest intervals for `x^2<4`; for `x^4+x+1>0` it loses the trivially-true "All real numbers". **Root cause:** same inequality-path failure as P0-2/3/4 — non-factorable inequality routed to the equation/residual path, here surfacing the literal solver dump rather than a root set.

### P1 — Integrate atanh domain over-restriction (`1/(1-x^2)`, `1/(4-x^2)`)
`integrate(1/(1-x^2),x) → atanh(x)` with a **false** `required_condition Positive(1-x²)` → `-1<x<1`, and `value_domain=real` though `atanh(x)` is non-real for `|x|>1`. The integrand is real/finite for `|x|>1` (e.g. `∫₂³ = -0.20273`). The sign-flipped control `1/(x^2-1) → ½ln|(x-1)/(x+1)|` correctly requires only `x≠±1`. **Root cause:** the `c²-x²`/`1-x²` antiderivative path emits `atanh` plus a spurious positivity guard instead of the domain-complete `½ln|…|` form the engine already uses for `x²-c²`.

### P1 — Log/sqrt `solve` identities drop domain condition in plain text
`solve(ln(x^2)=2*ln(x))`, `solve(2*ln(x)=ln(x^2))`, `solve(ln(2*x)=ln(x)+ln(2))`, `solve(e^(ln(x))=x)`, `solve(sqrt(x)^2=x)` all print bare `All real numbers` while JSON carries the correct `required_display` (`x>0` or `x≥0`). **Root cause:** the **plain-text solve formatter** prints only the bare `result` string and does not inline `required_display`, whereas the cases-form path (`1/x=1/x → "All real numbers if x != 0"`) does. The condition is computed and stored; only the text serializer drops it. (The `e^(ln(x))=x` and `ln(x^2)=2*ln(abs(x))` variants are P2 because every structured channel stays honest.)

### P1 — Matrix shape-mismatch dishonest residuals (5 inputs)
`2x2+2x3`, `2x2*1x3`, `2x2-1x3`, `2x3^2`, and the distributed `(2x2+2x3)*2x2` all return echoed/transformed residuals with `ok:true`, `warnings=[]`, `required_conditions=[]`. The engine *is* shape-aware (compatible ops compute correctly: `2x2*2x3→[[9,12,15],[19,26,33]]`) and *has* an `undefined` sentinel (used for `1/0`), so this is a genuine miss, not a symbolic blob. Worse, the **simplify path actively manufactures** undefined ops: `2x3*2x3` is rewritten to `^2` by a shape-blind `x*x` rule, and `(A+B)*A` is distributed into `A²+B·A` creating a *second* undefined operation. **Root cause:** matrix add/sub/mul/power rewrite rules lack a shape-compatibility gate before firing; undefined results are passed through as ordinary residuals instead of triggering the `undefined` sentinel or `ok:false`.

### P1 — `solve(a*x = a, x)` drops degenerate branch
Returns unconditional `{ 1 }`, all channels empty. True: `{1}` if `a≠0`, **ℝ if `a=0`**. The structurally identical compound form `solve((a-1)*x=a-1,x)` emits the **full** branch `{ 1 } if a-1≠0; All real numbers if a-1=0`, and the seed `solve(a*x=b,x)` carries `NonZero(a)`. **Root cause:** a special-case single-symbol-coefficient cancellation path computes `a/a=1` and short-circuits, bypassing the branch-emitting logic that the compound path uses.

---

## 4. Verified Sound (what was checked and held)

Fronts probed hard and confirmed clean (no wrong values, no spurious/missed roots, no false domain claims, honest residuals):

- **diff() / derivatives** (~80 inputs): chain/product/quotient, power towers (right-assoc verified), `a^x`, abs/sign with correct `x≠0` conditions, inverse trig/hyperbolic, higher-order, partials. All re-derived in sympy and numerically checked. **Clean.**
- **limit()** (~130 inputs): notable 0/0, higher-order Taylor, ∞−∞, 0·∞ squeeze, indeterminate powers (`e`, `e²`, `e^{-1/6}`…), radical sign-branches at −∞, one-sided poles with correct sign flips, signed-infinity even-power poles, honest punts on oscillating/two-sided-finite cases. **Clean.**
- **series()/taylor()/sum()/product()** (~70 inputs): finite sums exact, **integer-pole detection** (`undefined` only at integer poles, non-integer poles correctly pass), geometric convergence boundary exact, divergent-oscillating left unevaluated, telescoping/partial-fraction sums, Taylor coefficients all match sympy, products. **Clean.**
- **matrix operations** (~60 inputs, compatible shapes): det/inverse/transpose/trace/multiply/linear-systems all exact vs sympy; **singular handling sound** (exact rational + `is_provably_zero` gate, never a wrong inverse); honest residuals/errors for unsupported (rank, eigenvalues, 4×4 det). **Clean** (the *incompatible-shape* sub-front is the defect cluster above).
- **trig & inverse trig** (100+ inputs): full quadrant sweeps (432 cases, 0 mismatches), special angles, undefined poles, inverse-trig domain residuals, branch cuts (`arcsin(sin(3π/4))→π/4`, not naive), n-angle, identities, `sqrt(sin²)→|sin|`. **Clean.**
- **exact rational/number arithmetic** (~30+ inputs): big ints/rationals, gcd/lcm/mod, decimals as exact rationals, **no float leaks in keep/drop gates** (`2^53+1-2^53=1`). **Clean** (only defect: the `(1/2)!` display paren-loss, P2).
- **solve() polynomial core** (~35): cubic solver sound (rational, single-real, casus irreducibilis with/without x² term), quadratics (real/double/complex/nonunit). Defect confined to the **quartic biquadratic fallback** and the **lost-factor** cases above.
- **solve() transcendental core** (~70): `e^x=k`, `ln(x)=k`, `a^x=b`, exp-quadratic substitutions (correctly drop negative `u`-roots), log-domain extraneous-root rejection. Defects confined to identity-shaped inputs (tan pole-domain, plain-text condition drop).
- **solve() radicals & abs** (~40): extraneous-root handling excellent, abs equation/interval cases, domain-conditional results. Defects: `sqrt(x)=-sqrt(x)` (P0) and the abs-with-linear-term family (P1).
- **inequalities** (~50): factorable polynomial, degenerate quadratics, rationals with pole exclusion (more correct than sympy on removable holes), radicals, abs — all sound and operator-sensitive. The **memory-note irrational-constant case is now FIXED** (`x-x+pi>4 → No solution`). Defect: irreducible-polynomial inequalities (P0).
- **simplify() algebra**: removable-singularity folds with correct `≠` conditions, `0^0→undefined`, `sqrt(x²)→|x|`, log/exp inverses with correct guards, **concrete-exponent** power folds sound. Defects confined to the **symbolic-exponent** power folds (P0).
- **integrate()** (~45): rational/partial-fraction, by-parts, trig powers, radicals, honest non-elementary residuals (`e^{x²}` unevaluated). Defect: the `c²−x²` atanh family (P1).
- **diff of non-smooth functions** (~30): the brief's premise (silent drop of undefined-at-0) **did not reproduce** — the engine carries `x≠0` in `required_conditions`/wire; values sound. **Clean.**
- **nonlinear systems** (~35): genuinely nonlinear systems are **honestly rejected**, never mis-solved; linear path exact; consistency classification matches sympy. **Clean.**
- **display/printer re-parse honesty** (~180): fraction-base-powers, power towers, unary-minus-vs-power, surd arithmetic, sign distribution all round-trip; `result` vs `result_latex` agree numerically. Defects: composite-negative cube-root print (P2) and `(1/2)!` (P2).
- **exponentials & logs** (~30): even/odd-power abs handling, tower associativity, base-first `log(a,b)` convention (self-consistent), negative-base real-odd-root convention. Defect: plain-text solve-formatter condition drop (P1).
- **real-domain semantics**: default real mode is **clean** (imaginary-usage warnings, inert symbolic forms, no fabricated reals). The two complex defects (P0 odd-root, P2 `log(-1)`) require opt-in `--value-domain complex`.

**Refuted candidates (checked, NOT defects):**
- **`ln(a)+ln(b) → ln(a·b)` (`--domain strict`):** equivalence-preserving given the honestly-surfaced `a>0, b>0` conditions; no wrong value, no silent drop. Remaining issue is only a strict/generic-mode *contract-consistency* nit (introduces analytic conditions where docs reserve that for `assume`) — a P3 display/contract matter, **not a soundness bug**.
- **`((-8)^2)^(1/6) → 2`:** correct. `((-8)^2)^(1/6)` (square first → +2) and `(-8)^(2/6)=(-8)^(1/3)` (→ −2) are genuinely different expressions; the engine keeps them distinct (`((-8)^2)^(1/6) - (-8)^(2/6) → 4`) and never applies the unsound `(a^m)^n=a^{mn}` over negative bases. The symbolic rule `(x²)^(1/6)→|x|^(1/3)` is the correct identity.

---

## 5. Prioritized Fix Order

1. **Inequality operator-drop on irreducible polynomials (P0×3 + 2 P1 dumps).** Highest blast radius: silent wrong-kind result with `ok:true`, plus the dishonest `Solve: …=0` dumps. Fix the inequality path so the comparison operator is never rewritten to `Equal`; do sign analysis over the (closed-form) real roots and return interval unions. One fix retires `x^3+x+1>0/<0`, `x^3-3x+1>0/<0`, `sqrt(x-1)+sqrt(x-2)<3`, `x^4-x-1>0`.

2. **Symbolic-exponent power-law guard propagation (P0×3).** `(x^a)^b`, `((-2)^a)^b`, `(x^n)^(1/n)`: propagate the base-nonnegativity/parity guard the engine *already* computes on the concrete-exponent path (and already names in `--steps`) to the symbolic path. Until proven, do not fold (or emit `|x|`/the `x≥0` condition). Closes three wrong-value P0s, likely one shared code path.

3. **Matrix shape-compatibility gate (P1×5 + P2×1).** Add a shape check before add/sub/mul/power rewrite rules fire; on mismatch emit the existing `undefined` sentinel or set `ok:false` with a `required_condition`/warning. Also gate the `x*x→x^2` and distribution rules so simplify cannot *manufacture* undefined ops. Closes the entire incompatible-shape family.

4. **Complex principal-branch odd-root (P0×1).** Route negative-base odd-denominator rational powers through `exp((p/q)·Log(base))` when `value_domain=complex`, matching the already-correct even-root path. Closes `(-1)^(1/3)` and the `(-8)^(1/3)`, `(-1)^(2/3)`, `(-32)^(3/5)` family.

5. **`solve` plain-text formatter: inline `required_display` (P1×5 + P2×2).** Make the text serializer emit `"… if <conditions>"` (as the cases-path already does) so `x>0`/`x≥0`/`x≠0` are not dropped from the default surface. Closes all the log/sqrt identity text-drops at once.

6. **`solve` missing-roots / lost-factor (P1×2 + P2×1).** Fix the expanded-polynomial factorizer to not drop the `x²-5`-type factor (quintic case), and replace the broken biquadratic "isolate xⁿ, take nth root" fallback with a real quartic/biquadratic solver. Closes `x^5-5x^3+x^2-5`, the middle-term quartic family, and `(x^2+1)*(x^3-5x+1)`.

7. **`solve` abs-with-linear-term residual leak (P1×1) and `a*x=a` degenerate branch (P1×1).** For the abs family, complete the squaring step instead of emitting the unsolved sub-problem. For `a*x=a`, route the single-symbol cancellation through the same branch-emitting logic the compound `(a-1)` path already uses.

8. **P2/P3 honesty/incompleteness (factor echo & under-factoring, `log(-1)` complex, cube-root print, `(1/2)!` parens, `a*x=b` text).** Lower urgency: value-correct or structured-channel-mitigated. Fix `factor` to either complete the cyclotomic split or honestly mark incompleteness; return `iπ` for `log(-1)` in complex mode; print the trig form (or a real-cube-root head) for casus-irreducibilis cubics; add parens around `(rational)!` in the text serializer; inline `a≠0` in `solve … --format text`.

---

**Bottom line:** The engine is honest and correct across most of its surface, but soundness is **not** guaranteed — 8 P0 and 18 P1 confirmed defects (silent wrong values, dropped operators/domains, and `ok:true` over undefined results) mean a consumer trusting the success flag can receive a mathematically wrong answer, so fix the inequality-operator-drop, symbolic power-law, matrix-shape, and complex-odd-root clusters before relying on the engine for unattended use.

## Appendix — confirmed defects (structured)

| Sev | Cat | Input | Observed (short) | True answer (short) |
|---|---|---|---|---|
| P0 | wrong_value | `solve(sqrt(x)=-sqrt(x), x)` |  | { 0 }  (the only real solution is x=0) |
| P0 | lost_domain_condition | `x^3+x+1>0` |  | The strict inequality x^3+x+1 > 0 has solution = the op… |
| P0 | wrong_value | `x^3+x+1<0` |  | (-∞, -0.6823278038...) — the open ray of all x less tha… |
| P0 | wrong_value | `x^3-3x+1>0` |  | >0: (-1.8794, 0.3473) U (1.5321, +inf); <0: (-inf, -1.8… |
| P0 | lost_domain_condition | `(x^a)^b` |  | x^(a·b) is only valid under the guard x ≥ 0 (nonnegativ… |
| P0 | wrong_value | `simplify(((-2)^a)^b)   (equivalently bare eval ((-…` |  | ((-2)^a)^b is NOT equal to (-2)^(a·b) in general. The p… |
| P0 | lost_domain_condition | `(x^n)^(1/n)` |  | In the generic/real domain (x^n)^(1/n) must NOT be fold… |
| P0 | wrong_value | `(-1)^(1/3)   [flags: --value-domain complex]` |  | Principal-branch value of (-1)^(1/3) = e^(i*pi/3) = 1/2… |
| P1 | missing_root | `x^5-5*x^3+x^2-5=0` |  | Real solution set is { -sqrt(5), -1, sqrt(5) } (numeric… |
| P1 | missing_root | `x^4-8*x^2+15=0` |  | x in {-sqrt(5), -sqrt(3), sqrt(3), sqrt(5)} (approx -2.… |
| P1 | lost_domain_condition | `solve(tan(x) = tan(x), x)` |  | All real numbers EXCEPT where tan(x) is undefined, i.e.… |
| P1 | missing_root | `solve(abs(x^2-2*x)=3, x)` |  | {-1, 3} (both verified: \|(-1)^2-2(-1)\|=\|1+2\|=3 and … |
| P1 | dishonest_residual | `sqrt(x-1)+sqrt(x-2)<3` |  | [2, 34/9)  (i.e. 2 <= x < 34/9 ≈ 3.7778). Domain requir… |
| P1 | dishonest_residual | `x^4-x-1>0` |  | x^4-x-1>0  =>  (-inf, r0) U (r1, +inf) where r0 = CRoot… |
| P1 | lost_domain_condition | `integrate(1/(1-x^2), x)` |  | A real antiderivative valid on all three intervals is (… |
| P1 | lost_domain_condition | `integrate(1/(4-x^2), x)` |  | A correct real antiderivative is (1/4)·ln\|(x+2)/(x-2)\… |
| P1 | lost_domain_condition | `solve(ln(x^2)=2*ln(x), x)` |  | x > 0, i.e. the interval (0, inf). The identity ln(x^2)… |
| P1 | lost_domain_condition | `solve(2*ln(x)=ln(x^2), x)` |  | x > 0  (the open interval (0, ∞)). On x>0 the equation … |
| P1 | lost_domain_condition | `solve(ln(2*x)=ln(x)+ln(2), x)` |  | x > 0  (the real solution set is the open interval (0, … |
| P1 | lost_domain_condition | `solve(e^(ln(x))=x, x)` |  | x > 0  (equivalently the interval (0, infinity)). The e… |
| P1 | lost_domain_condition | `solve(sqrt(x)^2=x, x)` |  | x ≥ 0  (real solution set = [0, ∞)). Under the binary's… |
| P1 | dishonest_residual | `[[1,2],[3,4]] + [[1,2,3],[4,5,6]]` |  | UNDEFINED / shape error. Adding a 2x2 matrix to a 2x3 m… |
| P1 | dishonest_residual | `[[1,2],[3,4]] * [[1,2,3]]` |  | undefined (the matrix product is undefined). A = [[1,2]… |
| P1 | dishonest_residual | `[[1,2],[3,4]] - [[1,2,3]]` |  | Undefined / shape error. Subtracting a 1x3 matrix from … |
| P1 | dishonest_residual | `[[1,2,3],[4,5,6]]^2` |  | Undefined. A is a 2x3 matrix, so A squared equals A tim… |
| P1 | dishonest_residual | `([[1,2],[3,4]] + [[1,2,3],[4,5,6]]) * [[1,2],[3,4]…` |  | UNDEFINED (ShapeError). The inner sum [[1,2],[3,4]] + [… |
| P1 | lost_domain_condition | `solve(a*x = a, x)` |  | { 1 } if a != 0; All real numbers if a = 0. The engine … |
| P2 | missing_root | `(x^2+1)*(x^3-5*x+1)=0` |  | Three real roots (from the cubic factor x^3-5x+1; the x… |
| P2 | lost_domain_condition | `solve(e^(ln(x)) = x, x)` |  | x > 0 (open interval (0, infinity)). In the real value … |
| P2 | dishonest_residual | `factor(x^12-1)` |  | (x - 1)(x + 1)(x^2 + 1)(x^2 - x + 1)(x^2 + x + 1)(x^4 -… |
| P2 | dishonest_residual | `factor(x^11+x^10+1)` |  | (x^2 + x + 1)*(x^9 - x^7 + x^6 - x^4 + x^3 - x + 1) |
| P2 | dishonest_residual | `log(-1)  (flags: --value-domain complex)` |  | Principal Log(-1) = i*pi (well-defined in C). Engine re… |
| P2 | lost_domain_condition | `solve(ln(x^2)=2*ln(abs(x)), x)` |  | All real numbers except x = 0 (i.e. "All real numbers i… |
| P2 | display_ambiguity | `solve(x^3+x+1=0,x)` |  | The single real root of x^3+x+1=0 is x = -0.68232780382… |
| P2 | display_ambiguity | `(1/2)!` |  | (1/2)! = Gamma(3/2) = sqrt(pi)/2 ~= 0.8862269. The engi… |
| P2 | dishonest_residual | `[[1,2,3]] + 5` |  | Undefined in pure-CAS semantics. sympy raises TypeError… |
| P3 | other | `factor(x^9-1)` |  | Over Q, factor(x^9-1) = (x - 1)·(x^2 + x + 1)·(x^6 + x^… |
| P3 | display_ambiguity | `solve(a*x = b, x)` |  | x = b/a, but ONLY under the required condition a ≠ 0. I… |
