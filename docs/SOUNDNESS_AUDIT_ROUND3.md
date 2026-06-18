# Soundness Audit — Round 3 (2026-06-16)

Third multi-axis adversarial soundness audit (ultracode), after the Round-1
(Clusters C/B/D/A) and Round-2 (R1–R6, R3-2, R4-3/4/5/6, exact special-function-value
oracle) fixes landed. Baseline commit: `983f8b8c6`.

- **~20 fronts** hunted in parallel (incl. 2 regression axes for the Round-2 fixes);
  every candidate independently re-verified by a **default-reject skeptic** with
  real-domain numeric ground truth (the engine's own evaluator at concrete points +
  Python `math`/`cmath` cross-checks); deduped against the fix ledger. **49 agents,
  28 candidates.**
- **24 NEW confirmed defects:** 14 honesty-violation, 6 wrong-value,
  3 dropped-condition, 1 sign-wrong.
- **Round-2 regression axes BOTH HELD** — `acosh∘cosh` (generic/strict/default) and
  the R3-family non-finite/pole cancellation re-verified clean; no Round-2 fix
  regressed; the B-2/A-2 power-merge deferrals remain correctly scoped (not broken).

**Fix progress (as of 2026-06-17).** Cluster **A** ✅ (`c26c608f5`), Cluster **B** ✅
(`c55b1879a`), Cluster **C inverse-trig half** ✅ (`b97329291`), Cluster **C extraneous-root
filter** ✅ (`6461f0c238923bfe6d296b036c2add2571c1ca43`, exact single-quadratic-surd sign check; 2 adversarial workflows /
940 probes / zero false-drops), Cluster **E** ✅ (`eaa02ecc8`) — all graduated, `make ci`
green, guardrail+pressure fingerprints byte-identical. Cluster **C** is now fully fixed —
its `ln(x)=ln(-x)→ℝ` contradictory-domain collapse landed (`8aaafaa7e76d3ac134a2402cf3537eea41bbd121`: an `AllReals`
carrying contradictory `Positive(a)∧Positive(-a)` conditions collapses to `Empty`). **Remaining:**
Cluster **D** (`acosh` in `--domain assume` — SCOPED + DEFERRED: the strip to `x` is by design; the
narrow gap is that the `x>0` assumption is dropped on the steps-off path, and the robust fix needs a
new thread-local assumption-event sink mirroring `blocked_hint_store` — disproportionate for a
steps-off-only defect that already surfaces correctly under `--steps on`; see Cluster D §3).
Cluster **F** (interval strictness) ✅ (`5b08d258658c41362c2f9c65e1f610844e133d4f`, eval-output renderer now
respects open/closed bounds). The six landed fixes retired ~21 of the 24 probes (the entire
complex-leak family A+B, the worst solve fabrications, all empty sum/product wrong values, and
the inequality-strictness membership error).

## 1. Methodology

This round hunted defects across ~20 independent axes (solve, powers-radicals, abs-sign-floor, hyperbolic, complex-leakage, series-sum, equation-systems-ineq, plus regression axes for the Round-2 fixes). Each axis was hunted in parallel by a dedicated prober that generated candidate failures, after which every candidate was handed to an independent **default-reject skeptic** who re-derived the truth from **numeric real-domain ground truth** (the engine's own evaluator at concrete points, plus Python `math`/`cmath` cross-checks) and discarded anything explainable as a sound symbolic residual, a capability gap, or a known-deferred item. Survivors were de-duplicated against the Round-2 fix ledger (R1 inverse-composition domain, R2 `acosh∘cosh`, R3-family non-finite/pole cancellation, R4-family zero-denominator, the exact special-value oracle) so that nothing already fixed or already scoped-out was double-counted. **24 defects** survived both the hunt and the adversarial verify.

## 2. Headline Counts

**Total confirmed: 24.**

| Severity | Count | Probes |
|---|---|---|
| honesty-violation | 14 | the `sqrt(-n)` / `arccos(2)` / `arcsin(2)` / `abs(sqrt(-4))` fabrications and complex-leakage products/quotients |
| wrong-value | 6 | `sum(k,k,6,3)`, `sum(1,k,5,1)`, `product(k,k,6,3)`, `acosh(cosh(x))-x`, `x>1`, `abs(x)>2` |
| dropped-condition | 3 | `solve(ln(x)+ln(x+5)=0,x)`, `solve(sqrt(x)*sqrt(x-1)=2,x)`, `acosh(cosh(x))` |
| sign-wrong | 1 | `acosh(cosh(2*x))` (assume mode) |

(The `acosh` family straddles labels — the JSON tags them dropped-condition / wrong-value / sign-wrong respectively; counted once each as labeled.)

### Round-2 regression status

Two axes were regression checks of Round-2 fixes. **Both held.**

- **`acosh∘cosh` (R2):** `acosh(cosh(x))` in **generic/default/strict** mode correctly returns `|x|`. The fix is intact on the paths it covered. The three new `acosh` defects below are confined to **`--domain assume`** mode, which the R2 fix did not touch — a *new* unsound surface, not a regression of the old one.
- **Pole / non-finite cancellation (R3 family):** no new infinity-arithmetic or pole-cancellation failures surfaced. The probers explicitly confirmed `sum(k,k,1,inf)-sum(...)` style indeterminates are still handled, and `sqrt(x)^2 - x -> 0` (the deferred R4-4 symbolic-variable case) remains the scoped-out behavior, distinct from the new *constant* `sqrt(-2)^2` defect.

No Round-2 fix regressed.

## 3. Defect Clusters by Root Cause

### Cluster A — `sqrt` of a negative literal fabricates a real value (the `i²=-1` leak)
**Severity: honesty-violation. 9 probes. Largest cluster. — FIXED (commit `c26c608f5d00f373accaabab5a0aefebe8516a62`).**

**Fix (commit `c26c608f5d00f373accaabab5a0aefebe8516a62`):** rather than guard each combining rule, the fix
extends the exact undefined-over-ℝ detector `is_structurally_undefined_over_reals`
(consumed by both `expr_carries_*` predicates, hence the R3/R4-5/R4-6 universal
filter) to recognize an **even root of a provably-negative base** as undefined:
a `Pow(b, p/q)` with `b` numerically negative and `p/q` a non-integer rational with
EVEN denominator (`(-2)^(1/2)`, `(-1)^(3/2)`, `(-4)^(1/4)`), plus the `Sqrt(neg)`
builtin spelling the parser emits. The universal filter then reverts every merge
that would drop it to a finite value: `sqrt(-2)^2`, `sqrt(-2)*sqrt(-2)`,
`sqrt(-4)*sqrt(-9)`, `sqrt(-8)/sqrt(-2)`, `sqrt(-9)/sqrt(-4)`, `(-1)^(1/2)*(-1)^(1/2)`
all stay symbolic (and `sqrt(-2)-sqrt(-2) → undefined` as a bonus). Exact and
conservative — ODD roots stay real (`(-8)^(1/3)=-2`, `(-8)^(2/3)=4`), positive bases
and integer powers are untouched (`(-2)^3=-8`, `sqrt(9)/sqrt(4)=3/2`), symbolic
`sqrt(-2)*sqrt(-3)` stays symbolic, and `sqrt(-n)` standalone is unchanged. No
cancel-rule edit needed; guardrail+pressure fingerprints BYTE-IDENTICAL; adversarial
(3-lens) clean. The `abs(sqrt(-4))^2 → -4` half is fixed (no longer `-4`); the
abs-strip itself (`abs(sqrt(-4)) → 2·(-1)^(1/2)`) is Cluster B, separate.

| Probe | actual | expected |
|---|---|---|
| `sqrt(-2)^2` | `-2` | undefined over R |
| `sqrt(-1)^2` | `-1` | undefined over R |
| `sqrt(-2)*sqrt(-2)` | `-2` | undefined / stay symbolic |
| `sqrt(-1)*sqrt(-1)` | `-1` | undefined |
| `sqrt(-4)*sqrt(-9)` | `-6` | undefined |
| `sqrt(-12)*sqrt(-3)` | `-6` | undefined |
| `sqrt(-8)/sqrt(-2)` | `2` | undefined |
| `sqrt(-9)/sqrt(-4)` | `3/2` | undefined |
| `abs(sqrt(-4))^2` (also `abs((-1)^(1/2))^2`) | `-4` / `-1` | `4` / `1` (or symbolic) |

**Root cause.** A *single* shared mechanism, not three: the engine soundly keeps `sqrt(-n)` symbolic as `n^(1/2)·(-1)^(1/2)` standalone (and even attaches an Imaginary Usage Warning), but the **power/product-combining rules treat `(-1)^(1/2)` as an ordinary base** and merge exponents. `(-1)^(1/2)·(-1)^(1/2) → (-1)^1 = -1` ("N-ary Mul Combine Powers"), `(sqrt(-1))^2 → -1` ("Deshacer raíz y potencia"), and `(3·sqrt(-1))/(2·sqrt(-1)) → 3/2` ("Reconocer un cociente notable" cancelling an undefined common factor). The sign tell is decisive: `sqrt(-4)*sqrt(-9) → -6`, not the naive-real `+6`, so the value can *only* come from `i²=-1` — a complex branch leaking into a `value_domain=real, branch_mode=strict` evaluation. The diagnostic siblings confirm the boundary: `sqrt(-2)*sqrt(-3)` stays symbolic (radicand product not a perfect square / exponents don't merge to an integer), so the leak fires precisely when the exponent arithmetic collapses to an integer power of `(-1)^(1/2)`.

**Fix direction.** In the multiplicative power-combining and quotient-cancellation rules, **guard exponent-merging on a negative (or non-provably-nonnegative) base when `value_domain=real`**: refuse to combine `b^p·b^q` / cancel `b^p/b^q` unless `b ≥ 0` is established, leaving the product symbolic instead. Likely owner: the n-ary `Mul` power-combine simplifier and the "cociente notable" / undo-root-power rules in the `cas_simplify`/powers module. This one guard collapses 8 of the 9 probes; the 9th (`abs(sqrt(-4))^2`) additionally needs Cluster C's abs fix.

### Cluster B — `abs` non-negativity classifier accepts `(-1)^(1/2)` as ≥ 0
**Severity: honesty-violation. 3 probes (axis abs-sign-floor). — FIXED (commit `c55b1879a4d07651a5cf2febd5ebf19f964f51bb`; the `^2 → -4 / -1` teeth were already retired by Cluster A's filter, this cycle closes the bare abs-strip).**

| Probe | actual | expected |
|---|---|---|
| `abs(sqrt(-4))` | `2·(-1)^(1/2)` | symbolic / abs retained (true value 2) |
| `abs(sqrt(-4))^2` | `-4` | `4` |
| `abs((-1)^(1/2))^2` | `-1` | `1` |

**Root cause.** The abs simplifier's non-negativity test (rule "Quitar valor absoluto de una expresión no negativa") **misclassifies `(-1)^(1/2)` as a non-negative real** and strips the bars — also silently dropping the Imaginary Usage Warning. The bug is spelling-specific: the identical quantity spelled `i` is handled honestly (`abs(2*i) → 2·|i|`, `abs(i)^2 → i^2`), and sister functions are sound (`sign(sqrt(-1)) → sign((-1)^(1/2))`, `floor(sqrt(-1)) → floor((-1)^(1/2))`). So `|x|²` escaping to a *negative* value is impossible-by-definition and uniquely traces to this classifier. Note Cluster A's leak then squares the un-barred `2·(-1)^(1/2)` into `-4`, so the two clusters compound here.

**Fix (commit `c55b1879a4d07651a5cf2febd5ebf19f964f51bb`).** Two abs-strip surfaces in `crates/cas_math/src/abs_support.rs` independently assumed every square root is `≥ 0`: the non-negativity classifier `is_sum_of_nonnegative` (its `Pow` sqrt-form arm and its `Sqrt` function arm) and the `Abs Of Sqrt` identity extractor `try_extract_abs_sqrt_like_arg`. Both now reject a square-root form whose radicand is a **provably negative rational** (`as_rational_const(radicand).is_negative()`) — an even root of a negative is imaginary, not a non-negative real, so the bars stay. Result: `abs(sqrt(-4)) → 2·|(-1)^(1/2)|`, `abs((-1)^(1/2)) → |(-1)^(1/2)|`, `abs(sqrt(-4)+sqrt(-9)) → 5·|(-1)^(1/2)|` — all keep the bars on the imaginary part. The `^2`/`^4` cases stay symbolic-undefined (Cluster A's filter blocks the collapse). Real radicands are untouched (`abs(sqrt(x)) → sqrt(x)`, `abs(sqrt(2)) → sqrt(2)`, `sqrt(x^2) → |x|`), and the odd-root real case is correct (`abs((-8)^(1/3)) → 2`). Residual (honest, not a false value): a *symbolic* always-negative radicand whose negativity needs sign analysis (`abs(sqrt(-x^2-1)) → (-x^2-1)^(1/2)`) still strips, but the result is itself undefined-over-ℝ everywhere — no finite false value is produced; closing it needs a symbolic-sign oracle, deferred.

### Cluster C — `solve` emits non-real / extraneous roots (no domain filtering)
**Severity: honesty-violation (3) + dropped-condition (2). 5 probes (axis solve). — FIXED (all three sub-mechanisms): inverse-trig out-of-range (commit `b97329291c00b60f8dc673239914c681038f22da`), the extraneous-root filter (commit `6461f0c238923bfe6d296b036c2add2571c1ca43`), and the `ln(x)=ln(-x)→ℝ` contradictory-domain collapse (commit `8aaafaa7e76d3ac134a2402cf3537eea41bbd121`). See Fixes below.**

**Fix — inverse-trig out-of-range (commit `b97329291c00b60f8dc673239914c681038f22da`).** `arcsin(c)`/`arccos(c)` with `|c|>1` is undefined over ℝ (their real domain is `[-1,1]`), so a solve root carrying such a term is not a real solution. The final real-solution filter `solution_contains_nonfinite` in `crates/cas_solver/src/solve_backend_local.rs` — already applied to every solve result and already collapsing an emptied discrete set to `SolutionSet::Empty` ("No solution") — now also treats `arcsin/arccos/asin/acos(c)` with a provably-rational `|c|>1` as non-real (recursively, so a containing expression is non-real too). Result: `solve(cos(x)=2,x)`, `solve(sin(x)=2,x)`, `solve(sin(x)=-5,x)`, `solve(2*cos(x)=3,x)`, `solve(x=arcsin(2),x)` → "No solution"; bonus `solve(sin(x)^2=2,x)` → "No solution". Exact and conservative: boundary `|c|=1` is kept (`sin(x)=-1 → {arcsin(-1)}`, `cos(x)=-1 → {arccos(-1)}`), in-range cases are untouched (`cos(x)=1/2 → {π/3}`, `cos(x)=1 → {0}`), `tan`/`arctan` (no range limit) is unaffected, and ordinary algebraic/exp/log solves are unchanged. Guardrail + pressure fingerprints byte-identical.

**Fix — extraneous-root filter (commit `6461f0c238923bfe6d296b036c2add2571c1ca43`).** The roots reach `filter_real_solutions` as a bare `Discrete` set, and `check_root` only classifies *rational* roots (these are irrational, e.g. `½(-√29-5)`), so the extraneous roots survived. The fix wires the solver's already-computed `required_conditions` (`ctx.required_conditions() : Vec<ImplicitCondition>` — `Positive`/`NonNegative`/`LowerBound`/`NonZero`) into the filter and drops a root **only on an EXACT proof of violation**: a new cas_math helper `provable_sign_vs_zero` substitutes the root into the condition target and decides its sign over a single quadratic surd `A + B·√n` (exact rationals; `as_linear_surd` reduces rationals, a `sqrt(≥0)` surd, the rational-linear closure, and half-integer powers `b^(k/2)` so the reciprocal-surd spelling `-N·N^(-1/2)=-√N` the solver emits is handled). `None` (sign not provable — nested radicals, two surds, transcendental radicands) ⇒ KEEP, so a valid root can never be dropped. Results: `solve(ln(x)+ln(x+5)=0,x) → {½(√29-5)}`, `solve(sqrt(x)*sqrt(x-1)=2,x) → {½(√17+1)}`, `solve(sqrt(x-2)*sqrt(x+2)=3,x) → {√13}`, `solve(ln(x-1)+ln(x+1)=0,x) → {√2}`. **EXACT, not float** — the adversarially-found killer `solve((x^2-2)/(x-131836323/93222358)=0,x)` correctly keeps BOTH `±√2` (the denominator `93222358·√2−131836323` is irrational, provably ≠ 0), where an f64 gate would have dropped `√2` (it rounds to `0.0`). Valid roots kept: `ln((x-2)·(x+1))=0` keeps both, `sqrt(x)=0 → {0}` (NonNegative boundary), `sqrt(-x)=2 → {-4}`. Verified by 2 adversarial workflows (940 probes, **zero false-drops**). Guardrail + pressure byte-identical. **Residual (honest under-fix, never drops a valid root):** an extraneous root whose substituted condition target is NOT a single quadratic surd — a nested-power/un-normalized form like `sqrt(x^2+3x)=sqrt(2x^2-1)` (target arrives as `Pow(root,2)+3·root`, expanded `10-3√13` would prove `<0`), or a transcendental radicand like `ln(x)+ln(x+3)=1` (radicand `9+4e`) — is conservatively kept; closing it needs simplify-before-extract or a broader algebraic-sign oracle.

**Fix — `ln(x)=ln(-x)→ℝ` collapse (commit `8aaafaa7e76d3ac134a2402cf3537eea41bbd121`).** `solve(ln(x)=ln(-x),x)` returned "All real numbers": it collapses to an identity but the engine *does* record the required conditions `Positive(x)` (from `ln(x)`) and `Positive(-x)` (from `ln(-x)`) — i.e. `x>0 ∧ x<0`, a contradictory domain whose real solution set is EMPTY. An `AllReals` result means "every real satisfying the required conditions", so when those conditions are mutually unsatisfiable it must collapse to `Empty`. The final filter `filter_real_solutions` (which already receives `ctx.required_conditions()`) now adds an `AllReals` arm that returns `Empty` when `required_conditions_are_contradictory`: a `Positive(a) ∧ Positive(b)` pair whose targets are negations (`a == -b`, via `cas_math::poly_compare::poly_relation` → `Negated`). Result: `solve(ln(x)=ln(-x),x)`, `solve(ln(-x)=ln(x),x)`, `solve(log(x)=log(-x),x)`, `solve(ln(x-1)=ln(1-x),x)`, `solve(ln(x)+ln(-x)=0,x)` → "No solution". Strictly conservative — the check is on STRICT `Positive` only: a genuine identity on a non-empty domain stays `AllReals` (`ln(x)=ln(x)`, `x+1=x+1`, `sin²+cos²=1`), and a `NonNegative(a) ∧ NonNegative(-a)` pair meets at `0` (`sqrt(x)=sqrt(-x) → {0}`) so it is NOT collapsed. Guardrail+pressure byte-identical; full `cargo test --workspace` green. (Note: the user-facing eval/CLI path resolves the var-eliminated residual to `AllReals` before this filter and is now correct; the bare `cas_solver::solve` API returns the unresolved `Residual` for these — honest, not a false `AllReals` — so the contradiction collapse is covered by a direct unit test on `required_conditions_are_contradictory`.) (Separately, verification surfaced a PRE-EXISTING bug unrelated to this filter — now **FIXED (commit `1a0806509ff3ee3c2226af5cfa6f19e88e0b5138`)**: `solve((x^2-2x-1)/(x-1+sqrt(2))=0,x)` returned "No solution" and `solve((x^2-2x-1)/(x-1)=0,x)` returned a malformed `solve(x = sqrt(2x+1))` residual. Root cause: the Div-isolation route re-solved the numerator via the isolation-only reentry, which √-isolates a quadratic-with-linear-term (`x^2-2x-1=0 → x=√(2x+1)`, recursive depth-overflow) instead of using the Quadratic strategy — even though the same numerator solves correctly standalone. Fix: in `crates/cas_solver_core/src/solve_runtime_flow_isolation_custom_pow.rs`, the Div route's numerator re-solve now routes a degree-2 *equality* equation — an expanded `x^2-…` polynomial OR a factored product of ≥2 var-bearing factors `(x-a)(x-b)` — through the full strategy-selecting solve (the callback the Mul route already uses), so the Quadratic strategy claims it; the denominator-zero root is excluded by the usual NonZero machinery (`1-√2` for `den=x-1+√2` ⇒ `{1+√2}`). Gated to `op == Eq` so the inequality interval routes are untouched. Now: `(x^2-2x-1)/(x-1)=0 → {1-√2, 1+√2}`, `(x^2-2x-1)/(x-1+√2)=0 → {1+√2}`, `(x^2-4x+1)/(x-3)=0 → {2±√3}`. Verified: factorable/constant-numerator/reciprocal/linear cases unchanged, michaelis-menten step text preserved, full `cargo test --workspace` green, guardrail+pressure byte-identical.)

| Probe | actual | expected |
|---|---|---|
| `solve(cos(x)=2,x)` | `{ arccos(2) }` | No solution |
| `solve(sin(x)=2,x)` | `{ arcsin(2) }` | No solution |
| `solve(ln(x)=ln(-x),x)` | `All real numbers` (R) | No solution (empty) |
| `solve(ln(x)+ln(x+5)=0,x)` | both roots `½(±√29−5)` | `½(√29−5)` only |
| `solve(sqrt(x)*sqrt(x-1)=2,x)` | both roots `½(1±√17)` | `½(1+√17)` only |

**Root cause.** Two sub-mechanisms, same theme — **the final solution set is not validated against the real domain**:
1. *Out-of-range inverse-trig:* `cos(x)=2` is inverted to `arccos(2)` and emitted as a real root even though the engine itself refuses to give `arccos(2)` a real value (`cos(arccos(2))` does not round-trip). The engine *has* the "No solution" branch and uses it for `x²=-1`, `exp(x)=-1`, `sqrt(x)=-1` — it just isn't reached on the `|rhs|>1` inverse-trig path.
2. *Extraneous roots / canceled variable:* the log and radical equations produce candidate roots (or, for `ln(x)=ln(-x)`, collapse to "undefined = 0" and declare **R**) and the engine **prints the correct required conditions** (`x>0`; `x≥0 ∧ x≥1`; `{x<0 ∧ x>0}`) but does not filter the roots against them. The negative roots violate the engine's own stated conditions; `ln(x)=ln(-x)` even has the *true* answer (empty = the conjunction `x<0 ∧ x>0`) sitting in its own condition list while the headline says R.

**Fix direction.** Add a **post-solve filter**: every candidate root must satisfy the already-collected `required_conditions` (numeric/sign check) before entering the result set; an empty surviving set → "No solution". For `|rhs|>1` route inverse-trig through the existing No-solution branch. For the `undefined = 0` collapse, treat a vacuous/contradictory residual as empty, not as R. Likely owner: the solver's root-collection / condition-enforcement stage in `cas_solver` (the `solve` pipeline that already computes `required_conditions` — wire it into result filtering).

### Cluster D — `acosh(cosh(x))` drops the `|x|` restriction in `--domain assume`
**Severity: dropped-condition / wrong-value / sign-wrong. 3 probes (axis hyperbolic).**

| Probe | actual (assume mode) | expected |
|---|---|---|
| `acosh(cosh(x))` | `x` (cond only `cosh(x) ≥ 1`) | `|x|` (or `x` with `x≥0`) |
| `acosh(cosh(x)) - x` | `0` | `|x| - x` |
| `acosh(cosh(2*x))` | `2·x` (cond only `cosh(2x) ≥ 1`) | `2·|x|` |

**Root cause.** This is the R2 identity surviving only on the **assume-mode** path, which the R2 fix did not cover. The assume path cancels `acosh∘cosh → |·|` then applies "Abs Under Positivity" to strip the abs under an `x>0` assumption — but in the **default (non-`--steps`) output the assumption is never surfaced**: `assumptions_used` is absent and the only emitted condition, `cosh(x) ≥ 1`, is **vacuous** (cosh ≥ 1 for all real x). Numeric falsifier: at `x=-3` the engine's own evaluator gives `acosh(cosh(-3)) = 3`, but the simplified `x` gives `-3`. The assumption *does* surface under `--steps on` (via `collect_output_assumptions_used`, which reads only from collected steps), so the surfacing channel exists but is bypassed on the default path.

**Fix direction (scoped + attempted 2026-06-18; left as an honest residual — the correct fix is disproportionate machinery for a steps-off-only gap).** A 5-agent scoping workflow established the *intended* contract definitively: the strip `acosh(cosh(x)) → x` in assume mode is **correct and by design** (consistent with the whole abs-family — `abs(x)`, `integrate(sqrt(x^2))`, `sqrt(abs(x))^2` — and with `try_plan_abs_positive_rewrite`, which in the Assume branch *always* returns `PositiveAssume + Some(Positive)`; `docs/Requires_vs_assumed.md` names `abs(x)->x ⇒ assume x≥0` as the canonical Assumed example). So option A (keep `|x|`) is **wrong**, and routing `x≥0` into `required_conditions` is **also wrong** (it violates the documented "Requires never mixes with Assumed" invariant and breaks the passing `integrate(sqrt(x^2))` contrast test that asserts `required_conditions` stays empty). The real defect is narrow: the `AbsPositiveSimplifyRule` correctly emits an `AssumptionEvent::positive_assumed(x>0)` classified `HeuristicAssumption`, and it surfaces correctly **with `--steps on`** (`collect_output_assumptions_used` reads it from the steps), but on the **steps-off** default path it is dropped.

The correct fix is to give that assumption event a **steps-independent sink** that survives to `collect_output_assumptions_used`. An attempt (2026-06-18) added a per-pass accumulator on `LocalSimplificationTransformer` hoisted to `Simplifier.last_assumption_events` (mirroring `required_conditions`), pushed it into `diagnostics.assumed` at the `build_eval_diagnostics` boundary, and merged it in `collect_output_assumptions_used` — but it **does not work**: the multi-pass simplify pipeline clears/overwrites the per-pass/per-field accumulator (`local_simplify_with_phase`'s `=` hoist clobbers it on later empty passes, and the eval-start clear wipes it again). The robust mechanism is the one `blocked_hints` already uses: a **thread-local sink in `cas_solver_core` (mirror `blocked_hint_store`: `register_assumption_event` / `take_assumption_events` / `clear_assumption_events`)**, cleared once at the orchestration eval-start (next to `clear_blocked_hints`, orchestration.rs:179) and drained once after the full pipeline (next to `self.last_blocked_hints = take_blocked_hints()`, orchestration.rs:191). That's a new cross-crate store (cas_solver_core + cas_engine wiring + the `collect` merge) — disproportionate for a defect that is already correct under `--steps on`, so it is deferred to a dedicated cycle. Owner: a new `cas_solver_core::assumption_event_store` + `crates/cas_engine/src/engine/orchestration.rs` (clear/drain) + `eval/diagnostics.rs` (push into `diagnostics.assumed`) + `crates/cas_solver/src/eval_output_presentation_conditions.rs` (`collect_output_assumptions_used` reads the steps-independent list, deduped against steps by `(kind, expr_canonical, display)`).

### Cluster E — empty `sum`/`product` (reversed bounds) returns wrong finite value
**Severity: wrong-value. 3 probes (axis series-sum). — FIXED (commit `eaa02ecc89a766f94425f9bd3b5e16ee32ee9f4f`).**

| Probe | actual | expected |
|---|---|---|
| `sum(k,k,6,3)` | `-9` | `0` |
| `sum(1,k,5,1)` | `-3` | `0` |
| `product(k,k,6,3)` | `1/20` | `1` |

**Root cause.** The closed-form anti-difference/Faulhaber and product formulas are applied **without an emptiness guard** (`lower ≤ upper`). With lower > upper the formula extrapolates: `sum(k,k,6,3) = F(3)−F(5) = 6−15 = −9`; constant case uses term-count `n−m+1` → negative; product divides past the boundary → `1/20`. The off-by-one `a=b+1` returns the correct `0`/`1` only coincidentally. (Note: `sum(2^k,k,5,1) → -28` too — the geometric form is equally unguarded, slightly wider than the original write-up suggested.) Forward sums/products are all correct, so the formulas themselves are right; only the empty case is unsound.

**Fix (commit `eaa02ecc89a766f94425f9bd3b5e16ee32ee9f4f`).** Both planners `try_plan_finite_sum_evaluation` / `try_plan_finite_product_evaluation` in `crates/cas_math/src/summation_support.rs` now check `lower > upper` (both bounds exact integers) **before any closed form** and return the empty-aggregate identity via the existing direct builder — `start..=end` is an empty Rust range, so `build_finite_sum_substitution` yields `0` and `build_finite_product_substitution` yields `1`. The guard had to precede the closed-form attempts because `SumOfFirstIntegers` is tried first and was the source of `-9`. Result: `sum(k,k,6,3)`, `sum(1,k,5,1)`, `sum(k^2,k,10,1)`, `sum(k,k,0,-3)` → `0`; `product(k,k,6,3)`, `product(k+1,k,5,2)` → `1`. Forward ranges and single-element ranges (`sum(k,k,3,3) → 3`) are unchanged; guardrail + pressure fingerprints byte-identical (no fixture exercises a reversed range, so reusing the `FiniteDirect` kind caused no classification drift).

### Cluster F — inequality solver always emits closed brackets (strictness dropped)
**Severity: wrong-value. 2 probes (axis equation-systems-ineq). — FIXED (commit `5b08d258658c41362c2f9c65e1f610844e133d4f`).**

| Probe | actual | expected |
|---|---|---|
| `x>1` | `[1, infinity]` | `(1, infinity)` (open at 1) |
| `abs(x)>2` | `[-infinity, -2] U [2, infinity]` | `(-infinity,-2) U (2,infinity)` (open) |

**Root cause (corrected on fix).** NOT architectural — the original "no open-bracket representation" guess was wrong. The `Interval` type already carries per-endpoint `BoundType::{Open, Closed}` (`cas_ast/src/domain.rs`), and the solver's `isolated_var_solution` (`cas_solver_core/src/solution_set.rs`) already builds them correctly (`x>1 → (rhs, ∞)` open, `x≥1 → [rhs, ∞)`). The bug was purely in ONE display renderer: `eval_output_presentation_solution_display/interval.rs::format_output_interval` hard-coded `"[{}, {}]"`, discarding the bound types — so every interval, strict or not, printed with closed brackets (and ∞ wrongly closed). The sibling renderer `solution_display/interval.rs` was already correct; only the eval-output path was broken.

**Fix (commit `5b08d258658c41362c2f9c65e1f610844e133d4f`).** `format_output_interval` now takes the whole `&Interval` and renders `(`/`)` for `Open` and `[`/`]` for `Closed` per endpoint (mirroring the correct sibling renderer). No solver/type change. Result: `solve(x>1) → (1, infinity)`, `solve(x>=1) → [1, infinity)`, `solve(x<=1) → (-infinity, 1]`, `solve(x^2>4) → (-infinity, -2) U (2, infinity)`, `solve(x!=3) → (-infinity, 3) U (3, infinity)` — strict vs non-strict now distinct, ∞ always open. Guardrail + pressure byte-identical (no scorecard lane renders inequality intervals); no test asserted the old closed-bracket format. A renderer unit test (`renders_open_and_closed_bounds_distinctly`) locks it in.

### Axis areas with ZERO confirmed defects (positive results)

- **`acosh∘cosh` generic/strict/default paths (R2 regression):** clean — `|x|` returned correctly. Held.
- **R3-family non-finite / pole / infinity-arithmetic (regression):** clean — no new indeterminate-arithmetic failures.
- The probers found **no** new defects in the B-2/A-2 negative-base power-merge family (`(-8)^(1/3))^3 → -8` correct; `(-2)^x·(-2)^x` stays symbolic), confirming those remain correctly scoped/deferred rather than newly broken.

## 4. Prioritized Next Steps

**P0 — soundness, fix first (honesty violations, fabricated real values):**
1. ~~**Cluster A** — guard exponent-merge/quotient-cancellation on negative bases under `value_domain=real`.~~ **DONE** (`c26c608f5`): fixed at the value level via `is_structurally_undefined_over_reals` + the universal non-finite-drop filter, not per-rule.
2. ~~**Cluster B** — fix the abs non-negativity classifier to reject `(-1)^(1/2)`.~~ **DONE** (`c55b1879a4d07651a5cf2febd5ebf19f964f51bb`): both abs-strip surfaces (`is_sum_of_nonnegative` + the `Abs Of Sqrt` identity) now reject a provably-negative radicand; the `^2 → -4` teeth were already retired by Cluster A.
3. ~~**Cluster C**~~ — **DONE** (all three sub-mechanisms): inverse-trig out-of-range (`b97329291`), the extraneous-root filter (`6461f0c2`, exact single-quadratic-surd sign check, adversarially verified zero false-drops), and the `ln(x)=ln(-x)` contradictory-domain collapse (`8aaafaa7e76d3ac134a2402cf3537eea41bbd121`, `AllReals` + contradictory `Positive(a)∧Positive(-a)` ⇒ `Empty`).

**P1 — wrong/sign-wrong values:**
4. ~~**Cluster E** — add the `lower ≤ upper` empty-sum/product guard.~~ **DONE** (`eaa02ecc89a766f94425f9bd3b5e16ee32ee9f4f`): both finite-aggregate planners short-circuit a reversed range to the empty identity (`0` / `1`) before any closed form.
5. **Cluster D** — close the assume-mode `acosh∘cosh` abs-drop (extend R2 fix to the assume path or fix `assumptions_used` surfacing). Quick-ish; touches the presentation-conditions module.

**P2 — representational:**
6. ~~**Cluster F** — open/closed interval endpoints for strict inequalities.~~ **DONE** (`5b08d258658c41362c2f9c65e1f610844e133d4f`): it was plumbing, not architectural — the eval-output renderer ignored the (already-correct) `BoundType` flags; fixed to render open/closed per endpoint.

## 5. What Is Now Well-Covered — Confidence Statement

The engine is **largely sound and the Round-2 fixes held** — both regression axes (`acosh∘cosh`, R3 non-finite/pole cancellation) re-verified clean, and the B-2/A-2 power-merge deferrals are correctly scoped rather than broken. The 24 confirmed defects are **not a broad collapse**: they concentrate in **six tight clusters**, and **two single mechanisms account for half of them** — the `(-1)^(1/2)` complex leak in exponent/quotient combining (Cluster A, 9 probes incl. the compounded abs case) and the missing empty-sum/product guard (Cluster E). Notably, the engine *already knows the right answers internally* in most cases (it keeps `sqrt(-n)` symbolic standalone, prints the correct `required_conditions` in solve, returns `|x|` in generic mode, and returns the correct numeric value at substituted points) — the defects are **failures to enforce knowledge the engine already has**, not missing mathematics. That makes them mostly **quick family-cycles** rather than deep rewrites; the only candidate for architectural work is the open/closed interval representation (Cluster F). Net: real-domain core is in good shape, with a well-bounded P0 list whose top two fixes should retire roughly two-thirds of the round.
