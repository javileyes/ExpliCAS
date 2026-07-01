# Auditoría P0 soundness 2026-07-01 (workflow cas-frontier-audit)

94 confirmed findings, 20 P0 wrong-answers.

I'll synthesize the report directly from the provided findings. No tool calls needed — the data is complete and my task is analysis and deduplication.

# ExpliCAS Audit Backlog — Ranked & Deduplicated

## Executive summary

52 CONFIRMED findings collapse to **~12 distinct fixes**. There are **20 P0 wrong-answers** (soundness) that cluster into **6 root-cause families** — all are real defects where the engine confidently returns an incorrect set/value. The remaining findings are under-answers (residual echoes, `no definida` errors, unfolded results) that collapse into a handful of high-ROI reduction families. The dominant recurring shape (seen ~15×) is unchanged from memory: **a reduction covers the rational/named/bare case but drops the surd/negated/wrapped/reciprocal sibling.**

---

## 1. P0 Wrong-Answers (soundness — HIGHEST priority)

Grouped by root cause; every one returns a confident, incorrect closed-form set/value.

### P0-A · Surd-RHS trig equations drop the second base root AND all periodicity (6 inputs)
| Input | Engine output | Correct |
|---|---|---|
| `solve(2cos(x)-sqrt(3)=0)` | `{π/6}` | `{π/6+2kπ, 11π/6+2kπ}` |
| `solve(sin(x)-sqrt(3)/2=0)` | `{π/3}` | `{π/3+2kπ, 2π/3+2kπ}` |
| `solve(tan(x)-sqrt(3)=0)` | `{π/3}` | `{π/3+kπ}` |
| `solve(cos(x)+sqrt(2)/2=0)` | `{arccos(-√2/2)}` | `{3π/4+2kπ, 5π/4+2kπ}` |
| `solve(sin(2x)-sqrt(3)/2=0)` | `{π/6}` | `{π/6+kπ, π/3+kπ}` |
| `solve(2sin(x)+sqrt(3)=0)` | `{arcsin(-√3/2)}` | `{4π/3+2kπ, 5π/3+2kπ}` |

**Root cause:** when a surd constant sits in the *LHS* (`=0` form), the trig solver bypasses the periodic-family builder that the surd-on-RHS and rational-`=0` forms use, emitting only the bare principal inverse value. Rational-RHS and direct-surd-RHS forms are correct, isolating the defect to LHS surd normalization.

### P0-B · Surd-RHS `|quadratic|` equations/inequalities leak complex roots & malformed intervals (6 inputs)
| Input | Engine output | Correct |
|---|---|---|
| `solve(abs(x^2-1)=sqrt(2))` | includes `±(1-√2)^(1/2)` (imaginary) | `{±√(1+√2)}` |
| `solve(abs(x^2+2)=sqrt(2))` | `±(√2-2)^(1/2)` (imaginary) | No solution |
| `solve(abs(x^2-1)=sqrt(3))` | includes `±(1-√3)^(1/2)` | `{±√(1+√3)}` |
| `solve(abs(x^2-1)<sqrt(2))` | `(√(1-√2), …)` (imaginary bound) | `(-√(1+√2), √(1+√2))` |
| `solve(abs(x^2-4)>sqrt(2))` | 2-piece, mis-signed, drops middle | 3-piece union |

**Root cause:** the `|f|=c` / `|f|≷c` split dispatches each branch `x²=k` to a sqrt atom, but the real-domain non-negativity guard on `k` only fires for numerically-decidable rationals. Surd `k` (e.g. `1-√2<0`) skips the guard → emits imaginary roots/bounds as real. Rational-RHS analogs are all correct.

### P0-C · Constant-numerator over surd/irrational pole → degenerate/holed interval (3 inputs)
| Input | Engine output | Correct |
|---|---|---|
| `solve(1/(x-sqrt(2))>0)` | `(√2+∞, ∞)` (degenerate) | `(√2, ∞)` |
| `solve(1/(x-pi)>0)` | `(∞, ∞)`; operator dropped to `=0` | `(π, ∞)` |
| `solve(1/(x+sqrt(2))>0)` | `(-√2,√2)∪(√2,∞)` (spurious hole) | `(-√2, ∞)` |

**Root cause:** the single-fraction sign-analysis injects a phantom `+∞` sentinel or both `±surd` copies into the pole/breakpoint set for the constant-numerator + irrational-pole case; the `>` operator is also dropped to `=0`. Numerator-in-x forms and rational poles are correct.

### P0-D · `abs(factored product)=0` returns only the first factor's root (3 inputs)
| Input | Engine output | Correct |
|---|---|---|
| `solve(abs(x*(x-2))=0)` | `{0}` | `{0, 2}` |
| `solve(abs((x-1)*(x-3))=0)` | `{1}` | `{1, 3}` |
| `solve(abs(x*(x-1)*(x-2))=0)` | `{0}` | `{0, 1, 2}` |

**Root cause:** the `abs=0` branch strips `|·|` correctly (`|E|=0 ⇔ E=0`) but then extracts only the first Mul factor's root instead of the zero-set of every factor. Expanded-argument and non-abs forms are correct.

### P0-E · Quadratic-in-substituted-variable drops one root / picks the wrong branch (4 inputs)
| Input | Engine output | Correct |
|---|---|---|
| `e^x+e^(-x)=4` | `{ln(2-√3)}` (one root, always-true guard) | `{ln(2-√3), ln(2+√3)}` |
| `e^x-e^(-x)=2` | `{ln(1-√2)}` if `1-√2>0` (never) → empty | `{ln(1+√2)}` |
| `2^x+2^(-x)=4` | `{log(2, 2-√3)}` | `{log₂(2±√3)}` |
| `solve(x^(1/3)=1-x)` | non-real Cardano branch `1.34+1.16j` | `≈0.31767` (real root) |

**Root cause:** after `u=e^x`/`u=a^x` substitution → quadratic in u, only one u-root is back-substituted (for `e^x-e^(-x)` the *negative* one is kept behind a false positivity guard); the surd-discriminant branch never emits the second/valid root. Rational-discriminant cases keep both roots. The cubic sibling `x^(1/3)=1-x` cubes to a surrogate and selects a non-real Cardano branch with no real-root verification.

### P0-F · Domain-filter escapes & sign-drops in transcendental (in)equalities (7 inputs)
| Input | Engine output | Correct |
|---|---|---|
| `ln(x)-ln(x+1)=1/3` | negative out-of-domain root `-3.53` | No solution |
| `ln(x)-ln(x+3)=1/4` | negative out-of-domain root | No solution |
| `ln(x)-ln(x+1)=3/2` | `-1.29` (violates its own `x>0`) | No solution |
| `sqrt(x)<-sqrt(2)` | `[0,2)` | No solution |
| `sqrt(x)>-sqrt(2)` | `(2,∞)` | `[0,∞)` |
| `x^(2/3)>-sqrt(2)` | interval with complex endpoint | All reals |
| `e^x>-1/2` | `(undefined,∞) if -1/2>0` | All reals |
| `log(1/2,x)>3` | `(1/8,∞)` (exact complement) | `(0,1/8)` |
| `abs(ln(x))<2` | No solution | `(exp(-2),exp(2))` |

**Root cause (three sub-mechanisms, all "guard fires only for rationals / one sign"):**
- **Log equations:** candidate roots not filtered against the already-collected `x>0` domain (rational-RHS siblings like `=1/2` filter correctly; `1/3`, `1/4`, `3/2` bypass).
- **Radical/exp/power inequalities:** negative surd/non-integer-rational RHS squared/inverted without a sign guard → solves the wrong (positive-RHS) problem or produces `undefined`/complex endpoints. Rational-RHS analogs correct.
- **`log(base<1)`:** operator not flipped for decreasing log; `x>0` lower bound dropped. Base>1 correct.
- **`abs(ln(x))`:** spuriously imposes `ln(x)>0` (log *value* positivity) instead of `x>0` (argument positivity), killing the lower branch.

### P0-G · Simplifier truncation corrupts 2nd derivatives of trig quotients (2 inputs)
| Input | Engine output | Correct |
|---|---|---|
| `diff(sin(x)*tan(x),x,2)` | `(…)/cos³` off by `2sin²/cos` | `(sin⁴+cos²+1)/cos³` |
| `diff(x/sin(x),x,2)` | numerator has `x·sin` not `x·sin²` | `(x(cos²+1)-sin(2x))/sin³` |

**Root cause:** on the 2nd pass the Core simplifier hits `depth_overflow` (depth 51) and returns a **numerically-inequivalent truncated tree** (drops one power of cos/sin, flips sign terms). 1st derivatives are correct. This is the worst class — a soundness invariant violation (simplify must never return a non-equivalent result on truncation).

> **P0 honesty note:** these are 20 genuine wrong-answers across 7 mechanisms — this is *not* a clean bill of health. However, all 7 share the same meta-shape (a guard/family-builder that only fires for the rational/named case), so the fix surface is smaller than the count suggests. Notably the surd families (P0-A, P0-B, P0-C, P0-F) may share plumbing: a single "surd constants are decidable real constants, not symbolic coefficients" upgrade to the sign/guard layer could close a large fraction.

---

## 2. P1 Under-Answers — grouped by FAMILY (ranked by ROI)

### F1 · Widen quadratic/poly-in-u substitution to surd coefficients & both roots ★ highest under-answer ROI
**Inputs:** `e^x±e^(-x)=2√2 / √5 / 3√2`, `2^x+2^(-x)=2√2`, `cosh(x)=2`, `2cosh(x)=5`, `e^x-e^(-x)=3`, `2^(x+1)*3^x=12`, `ln(x)^2-2√2·ln(x)+2=0`, `log(x,2)^2-3log(x,2)+2=0`, `x^6-5x^3+6=0`, `x^6-6x^4+12x^2-8=0`, `x^6+3x^4+3x^2+1=0` (→No solution), reciprocal-sum rational equations `(x+2)/(x-2)+(x-2)/(x+2)=4`, `(x+1)/(x-1)+(x-1)/(x+1)=10/3`, `(x+1)^2/(x-1)=4`.
**Missing reduction:** after substituting `u=e^x / a^x / ln(x) / log(a,b) / x^n`, solve the polynomial-in-u with **surd/algebraic (not just rational) coefficients**, keep every real root satisfying the domain (`u>0`, etc.), back-substitute, and verify. Also: register `cosh`/`2cosh(x)=c` as an alias reducing to `e^x+e^(-x)=c`; combine distinct-base exponential products `2^(x+1)·3^x → 2·6^x`; and clear reciprocal-sum rationals to the **polynomial** form (not a lossy radical isolation).
**ROI:** ~25 inputs; unlocks cosh/exp/log-quadratic/biquadratic/reciprocal-sum — all standard textbook. Note: the integer-RHS `e^x+e^(-x)=4` case is P0-E (drops a root); this family fixes both.

### F2 · Widen inequality/limit "numeric coefficient" gate to accept surd/irrational constants ★
**Inputs:** `x^2-2√2·x+2>0`, `abs(x^2+1)<sqrt(2)`, `x^2<sqrt(2)`; limits `(2√2·x²+x)/(x²-1)→2√2`, `(√2·x²-x)/(x²+1)→√2`, `(√2·x+1)/(x-1)→√2`, `sin(√2·x)/x→√2`, `(e^x-1)/(√3·x)→√3/3`, and removable holes at surd points `(x²-3)/(x-√3)@√3`, `(x-2√2)/(x²-8)@2√2`.
**Missing reduction:** treat a constant irrational/surd/π expression as a **concrete numeric constant** (sign & value decidable), not a "symbolic coefficient". For rational-function-at-∞ limits form the leading-coefficient ratio symbolically; for finite-point limits accept a symbolic approach point and cancel `(x−a)` treating `a` as opaque; for inequalities compute discriminant/endpoints as exact algebraic numbers.
**ROI:** ~14 inputs; the rational-coefficient siblings all work, so this is purely a gate widening. Overlaps mechanically with P0-B/C/F (surds mis-classed).

### F3 · Radical-equation fixpoint: recursively re-dispatch the reduced equation
**Inputs:** `x+sqrt(x-1)=3`, `sqrt(x+sqrt(x+11))=3`, `sqrt(x+sqrt(x)+1)=1`, `sqrt(x)+1/sqrt(x)=2`, `sqrt(x+3)-sqrt(7-x)=sqrt(2x-8)`, `x+sqrt(x+11)=9`, plus the reciprocal-sum radical residuals.
**Missing reduction:** make the radical reducer a **fixpoint loop** — after each isolate-and-square, re-detect remaining radical atoms and recurse; substitute `u=sqrt(ax+b)` for affine radicands (not just bare `x`); when radical-free, dispatch the polynomial, then filter by radicand domains **and** RHS≥0 range guard, verifying against the original. Currently a single squaring pass echoes `solve(...)=0`.
**ROI:** ~8 inputs; nested/sum-of-radicals are common textbook forms; the isolated forms already solve.

### F4 · Register reciprocal/alt-spelling functions in the solver dispatch
**Inputs:** `sec(x)=2`, `csc(x)=2`, `cot(x)=1`, `cot(x)=0`, `1/cos(x)=2`, `1/sin(x)=2`, `cbrt(x)=2`, `nthroot(x,3)=2`, `log(10,x)+log(10,x-3)=1` (log10 lowering), `arg(1+i)`, `re/im/conj`, `abs(3+4*i)`, `min`/`max` (diff), `exp(i*pi)`.
**Missing reduction:** pre-solve canonicalization of reciprocal/alias atoms to their invertible partner **before** dispatch: `sec(u)=c→cos(u)=1/c`, `csc→sin`, `cot→tan` (with `c=0` special-case), `cbrt/nthroot(a,n)→a^(1/n)`, `log10/log2→` generic base-b, `min/max→(a±b−|a−b|)/2→abs` (already supported), Gaussian `arg/re/im/conj/abs→` polar/modulus atom, `exp(iθ)→cos θ+i sin θ`. All target atoms already solve — pure dispatch/registration gap.
**ROI:** ~16 inputs across trig/radical/log/complex/calc fronts; each already has a working equivalent form, so low-effort high-breadth.

### F5 · Periodic-trig INEQUALITIES → interval-family SolutionSet
**Inputs:** `sin(x)>1/2`, `2*sin(x)>=2`, `cos(x)>1/2`, `tan(x)>1`, `cos(x)<-1/2`, `sin(2x)>0`, `1/x^(1/3)>2` (reciprocal-power sibling), and the whole documented periodic-inequality backlog.
**Missing reduction:** detect `a·trig(αx+β) ⋈ c` (normalize the coefficient/affine wrapper away first — `2sin(x)≥2 → sin(x)≥1`), solve the boundary for two principal roots, select the sub-interval by monotonicity/range, emit a **Periodic interval family** `{r₁+kT < x < r₂+kT}`. Currently echoes `solve(trig=c)=0`.
**ROI:** ~10 inputs; this is the interval-family analogue of the already-landed periodic-equation fix. Matches memory backlog.

### F6 · Re-run the final simplify/const-fold pass on definite-integral (FTC) output & normalize reciprocal powers
**Inputs (unfolded FTC):** `integrate(tan(x),x,0,pi/3)→ln(2)`, `integrate(arctan(x),x,0,1)→π/4-½ln2`, cot form. **Inputs (reciprocal-power normalization):** `integrate(1/x^(1/3),x)`, `integrate(1/x^(2/3),x)`, definite versions, `integrate(1/(e^x+e^(-x)),x)→arctan(e^x)` (strip constant from `1/(2cosh)`).
**Missing reduction:** (a) route the FTC `F(b)−F(a)` bound-substitution result through the same const-fold/trig-special-angle simplifier used at top level; (b) collapse same-base powers `x^(2/3)·x^(-1)→x^(-1/3)` / canonicalize `1/x^(p/q)→x^(-p/q)` before the power-rule integrator; (c) strip a constant factor from `1/(k·cosh)` before the sech rule.
**ROI:** ~9 inputs; values are already correct (pure cosmetic/normalization), so these are safe, high-visibility wins.

### F7 · Multivariate factor-by-grouping + difference-of-squares; series linearity; telescoping-of-squares; apart re-entry; abs product/sign
**Inputs:** `factor(a*x+a*y+b*x+b*y)→(a+b)(x+y)`, `factor(x^2*y^2-1)`, `sum((1/2)^k+(1/3)^k,0,oo)→7/2` (+ difference/3-term/single-fraction variants), `sum((2k+1)/(k²(k+1)²),1,oo)→1`, `apart((x^3+1)/(x^2-1))→x+1/(x-1)`, `solve(x/abs(x)=x)→{-1,1}`, `solve(abs(x*sqrt(3))=abs(x-1))`.
**Missing reductions (independent small fixes):** factor Add-of-≥4-terms by grouping + multivariate `A²−B²`; **linearity split over Add summands** in the infinite-series planner (single-term geometric already works); generalize telescoping to reciprocal-power bases `1/k^m`; re-dispatch `apart` after the simplifier cancels a common factor; factor `x·(1−|x|)` and intersect with `x≠0`; square `|A|=|B|→A²−B²=0` keeping surd coefficients.
**ROI:** ~12 inputs; several are extremely common (factor-by-grouping, sum of two geometric series).

---

## 3. P2 Under-Answers (niche)

- `solve(cbrt(x)=2)`, `nthroot(x,3)=2` — non-standard spellings (covered by F4; `x^(1/3)` already works).
- `diff(x^2,x,0)` — missing order-0 base case (identity); echoes the call.
- `log(x,10)+log(x-1,10)=log(6,10)` — transcendental, no elementary closed form; correct behavior is an honest decline, but the engine emits a malformed self-referential echo (guard: reject rewrites whose "solved" side still contains the solve variable).

---

## 4. Ranked family table (ROI = wrong-answers first, then breadth)

| Rank | Family | Kind | ~Inputs | Effort | Notes |
|---|---|---|---|---|---|
| 1 | **P0-G** simplifier truncation returns non-equivalent tree | wrong | 2+aliases | med | soundness *invariant* violation — fix first |
| 2 | **P0-A** surd-LHS trig drops 2nd root+periodicity | wrong | 6 | low-med | family-builder bypass |
| 3 | **P0-B** surd-RHS `\|quadratic\|` leaks complex roots | wrong | 6 | low | sign guard on surd `k` |
| 4 | **P0-F** transcendental domain-filter/sign escapes | wrong | 9 | med | 3 sub-mechanisms, shared "surd/one-sign" shape |
| 5 | **P0-E** quadratic-in-u drops root/wrong branch | wrong | 4 | low | ⊂ F1 fix |
| 6 | **P0-C** const-num/surd-pole degenerate interval | wrong | 3 | low | phantom `+∞` sentinel |
| 7 | **P0-D** `abs(product)=0` first-factor only | wrong | 3 | low | dispatch stripped arg to product solver |
| 8 | **F1** surd-coefficient poly-in-u + both roots | under | ~25 | med | biggest under-answer ROI; overlaps P0-E |
| 9 | **F2** widen numeric-coefficient gate (ineq+limits) | under | ~14 | low-med | overlaps P0-B/C/F |
| 10 | **F4** register reciprocal/alias/complex functions | under | ~16 | low | pure dispatch, all targets already solve |
| 11 | **F3** radical-equation fixpoint recursion | under | ~8 | med | |
| 12 | **F5** periodic-trig inequality families | under | ~10 | med | matches backlog |
| 13 | **F6** FTC re-simplify + reciprocal-power norm | under | ~9 | low | values already correct |
| 14 | **F7** factor/series/apart/abs grab-bag | under | ~12 | low each | independent |

---

## 5. Fix sketches — top 5

**① P0-G (simplifier truncation, `diff(sin(x)*tan(x),x,2)`)** — In the Core simplifier's depth-overflow handler: on abort, **return the last known-valid subexpression, never a partially-rewritten tree** (a soundness invariant: `simplify` must be numerically-equivalence-preserving even on truncation). Additionally, dispatch `d/dx` of `P(sin,cos)/cos^n` through a canonical quotient-rule atom that folds `sin²+cos²→1` and combines like power terms **before** recursing, keeping the 2nd pass under the depth cap. Guard: numeric spot-check the derivative against the original at a sample point before rendering. Likely file: Core simplify loop + the diff quotient-rule assembly.

**② P0-A (surd-LHS trig, `solve(2cos(x)-sqrt(3)=0)`)** — In `cas_engine/src/solve/solve_backend_local.rs` trig dispatch: normalize `a·trig(x)+b=0` uniformly to the canonical atom `trig(x)=c/a` **regardless of whether the surd arrived in the LHS or RHS** (do not let the surd-in-LHS path bypass the working surd-RHS branch). Then for `|c/a|<1` emit **both** families (`{arccos(c/a)+2kπ, 2π−arccos(c/a)+2kπ}`, analogous sin/tan); reserve a single family only for the true tangent `|c/a|=1`. Verify roots against the original before returning.

**③ P0-B (surd-RHS `|quadratic|`, `solve(abs(x^2-1)=sqrt(2))`)** — Reduce `|f|=c` to `{f=c, f=−c}`, solve each as the atom `x²=k`, and **gate each atom on `sign(k)` via a real-decidable comparator that handles surds** (`1−√2 < 0` must drop the branch under `value_domain=real`). Equivalently, verify each produced root with an `is_real` gate against the original over ℝ. Same comparator fixes the `<`/`>` inequality siblings (drop vacuous lower constraints, keep symmetric intervals). This is the same "surds are decidable constants" upgrade F2 needs.

**④ P0-F (log-equation domain escape, `ln(x)-ln(x+1)=1/3`)** — After producing each candidate root, **test it against the already-collected `required_conditions` (`x>0`) with exact arithmetic** and drop violators, returning No solution if none survive. The filter code path already exists (the `=1/2` sibling uses it); the `1/3`/`1/4`/`3/2` branches bypass it. Canonical: solve the invertible atom `x/(x+A)=e^c`, back-substitute, then verify against real-domain constraints. Pairs with the `log(base<1)` operator-flip fix and the `abs(ln(x))` argument-vs-value positivity fix in the same transcendental-inequality dispatch.

**⑤ F1 (surd-coefficient poly-in-u, `e^x+e^(-x)=2*sqrt(2)` / `cosh(x)=2`)** — In `try_solve_exponential_reciprocal_polynomial` (solve_backend_local.rs:2567): widen the coefficient carrier in `collect_exp_laurent_terms`/`exp_laurent_leaf` from `BigRational` to a **symbolic/ExprId-coefficient map** (or special-case the reciprocal-pair span-2 form to build `u²−RHS·u+1=0` with the raw RHS ExprId). Dispatch the quadratic-in-u to the **surd-capable** algebraic solver, iterate over **all** real roots keeping `u>0`, back-substitute `x=ln(u)/ln(base)`, and verify each against the original. Register `cosh`/`2cosh(x)=c` as an alias reducing to `e^x+e^(-x)=c` so they route through the same solver instead of `función [cosh] no definida`. This one change also closes P0-E's integer-RHS root-drop.
---

## FIX STATUS (2026-07-01, same day)

The audit's meta-hypothesis ("a 'surd constants are decidable real constants' upgrade to the sign/guard
layer closes a large fraction") was CONFIRMED — two sign-layer upgrades (`prove_positive`/
`prove_nonnegative`, and `is_known_negative`) each closed multiple families.

- **P0-D** — FIXED (`be76f3d00`): `|E| = 0 ⟺ E = 0` dispatched to the full solver (all factor roots).
- **P0-A** — FIXED (`7d7b3ec94`): the `A·trig + B = 0` normalization now keeps a SURD offset symbolically.
- **P0-E** — FIXED (`6ed3ad19e`): `prove_positive`/`prove_nonnegative` decide linear-surd signs exactly
  (`root_forms::provable_sign_vs_zero`). Also fixed `e^x = 2−√3`, `e^x+e^(−x)=3`, and the general
  `e^x = surd` / log-of-surd family, and the P0-F case `e^x > −1/2` → All reals.
- **P0-B** — PARTIAL (`538f0a902`): `is_known_negative` decides linear-surd signs, fixing the abs-split
  cases (`|x²−1| = √2`, `|x²−1| = √3`). REMAINING: the direct `x² = 1−√2` and all-imaginary
  `|x²+2| = √2` still leak — a separate even-root-of-negative-surd guard in the `x²=c` power inversion.

### REMAINING (next batch)
- **P0-B residual**: `x² = neg-surd` power inversion (the `abs(base)=rhs^(1/2)` even-root path) needs the
  surd non-negativity guard on the transformed RHS.
- **P0-C**: `1/(x − surd) > 0` — the single-fraction sign-analysis injects a phantom `+∞` sentinel /
  both `±surd` copies into the pole set (interval-endpoint arithmetic bug, not a sign-decision bug).
- **P0-F**: log-equation domain filter (`ln(x)−ln(x+1)=1/3` keeps a negative out-of-domain root);
  radical/power inequality with negative-surd RHS (`√x < −√2` → wrong ray); `log(base<1)` operator flip;
  `abs(ln(x)) < 2` (imposes `ln(x)>0` instead of argument `x>0`).
- **P0-G**: 2nd-derivative simplifier depth_overflow returns a NON-EQUIVALENT truncated tree
  (`diff(sin(x)·tan(x),x,2)`) — a simplify soundness-invariant violation (hardest; separate effort).
- **P1 families** F1–F4 (surd-coefficient poly-in-u; surd limits; radical fixpoint; sec/csc/cot/cbrt
  registration).
