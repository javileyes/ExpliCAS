# Soundness Audit — Round 4 (2026-06-18)

Fourth multi-axis adversarial soundness audit (ultracode), after Rounds 1–3 and the log-equation-family hardening landed this session. Baseline commit: `07acd585d`.

## Method

- **18 fronts** hunted in parallel via a `Workflow` pipeline: each axis's hunter ran the release CLI on concrete probes, then piped its candidates straight to a **default-reject skeptic** that re-derived the correct real-domain answer **exactly** (sympy 1.14 `solveset`/`integrate`/`limit`, by hand, and the engine's own evaluator at concrete points) — never float alone.

- Every confirmed bug then faced an **independent second refuter** that had to reproduce it on the CLI and produce a concrete witness, or it was dropped. A synthesizer clustered the survivors by root cause and a completeness critic probed under-covered angles.

- **108 agents, ~3.0M tokens.** Agents were given the full list of Rounds 1–3 + this-session fixes and the known-open residuals, and instructed to dedup against them (report regressions only, classify known items as `known_already_logged`).

- **Read-only**: agents were forbidden any git/file mutation; CLI + python3 only.

- **Accuracy spot-check (operator):** 9 representative claims across the clusters were re-run by hand and **all reproduce exactly** — e.g. `equiv(x,y)` → `true`, `solve(log(2,x)+log(2,x+2)=3,x)` → `Solve: solve(x = 2^3 / (x + 2), x) = 0`, `integrate(cos(x)/sin(x)^2,x,0,2*pi)` → `infinity`, `tan(pi/2)-tan(pi/2)` → `0`, `sum(k,k,1,inf)` → `1/2·infinity^2`, `solve(x/0=5,x)` → `All real numbers`, `((x^2)^(1/3))^(3/2)` → `x`. The report is faithful to engine behaviour, not agent narration.

## Headline

**77 findings, 74 unique confirmed, 64 survived the second refuter** → **64 twice-verified unsound bugs** in **15 clusters (A–O)**.


| Axis | confirmed findings |
|---|---|
| solve-log-exp | 7 |
| solve-radical | 4 |
| solve-trig-inverse | 7 |
| solve-rational-poly | 4 |
| inequalities | 9 |
| systems | 0 |
| simplify-undefined-leak | 6 |
| powers-radicals | 6 |
| abs-sign-floor | 5 |
| limits-finite | 0 |
| limits-infinity | 0 |
| integration-definite | 4 |
| integration-indefinite | 0 |
| differentiation | 1 |
| series-sums-products | 8 |
| trig-hyperbolic-identities | 3 |
| special-values-constants | 6 |
| numeric-float-leakage | 7 |

## Overall assessment

Real-domain soundness is NOT broadly guaranteed. The evaluator core (eval of concrete scalars) is largely sound -- it independently returns "undefined" for x/0, ln(-5), tan(pi/2), and folds (x^2)^(1/2) to |x| -- but every higher layer built on top of it leaks. The 64 confirmed bugs concentrate in five structural weak spots, each a place where a "finishing"/closed-form path commits to a confident answer without re-checking the domain invariants the evaluator already knows:

1. The SOLVE pipeline is the single largest hazard (Clusters A, B, D, E, F, J: 33 of 64 bugs). It (a) leaks an internal unfinished "Solve: solve(...) = 0" token as an ok=true success on log-product, log-quotient, and |f|=-f reductions that the engine can in fact finish (the reduced quadratic solves correctly in isolation); (b) drops parameter-side sign conditions when inverting even roots with a symbolic RHS (sqrt(x)=a -> {a^2} unconditionally); (c) collapses provable-division-by-zero equations to "All real numbers"; (d) drops the denominator-sign case split for rational inequalities, returning sets disjoint from the truth; and (e) fails to intersect the argument-domain into the solution set for sqrt/ln/log inequalities, including the entire negative axis where the predicate is undefined. The default text output shows only the wrong interval with no caveat, so an educational user is actively misled.

2. The infinite-series / infinite-product closed-form machinery (Cluster L, 6 bugs) blindly applies finite Faulhaber/geometric/factorial identities at an infinite upper bound, emitting literal "infinity" tokens inside the expression tree (1/2*infinity^2, 2^infinity-1, infinity!) for divergent sums/products -- the exact malformed-garbage class the charter names, and worse, infinity! folds as a finite atom so 0*product(k,k,1,inf) = 0.

3. Definite integration over intervals with endpoint poles (Cluster K, 4 bugs) short-circuits its zero-scan at the first boundary zero, misses the interior/second pole, and fabricates a signed +/-infinity for genuinely indeterminate (inf-inf) integrals -- including a sign-wrong, critical-severity case over [0,2pi] with a non-integrable interior pole.

4. The additive like-term collector (Cluster G, 4 bugs) cancels coefficients to zero BEFORE checking definedness, fabricating finite values (0, 1) for expressions containing undefined-over-R subterms (tan(pi/2)-tan(pi/2)=0, ln(-5)-ln(-5)=0), contradicting the evaluator's own verdict on every other arithmetic path.

5. equiv (Cluster O, 7 bugs) is the most systemically dangerous primitive: a single shared numeric probe (all variables = 1.23456789) plus an absolute f64 epsilon (1e-9) certifies arbitrarily different expressions as equal whenever their residual vanishes on the diagonal x=y or at that one point. This is a textbook violation of the project's own MEMORY note 'Soundness gates must be exact -- never f64 for drop/keep decisions'. equiv(x,y)=true, equiv(x*y,x^2)=true, equiv(x/1e12,0)=true.

The remaining clusters (C: surd/transcendental trig RHS bypassing the |c|<=1 range check, returning non-real arcsin/arccos as real roots; H: imaginary warning gated on input not result; I: rational-exponent power towers dropping abs; M: principal inv-trig asserting a provably-false range membership; N: choose/perm guard-ordering fabricating 0 for negative n) are narrower but all share the same meta-cause: a fast/closed-form path that does not re-validate against the domain facts the engine already possesses, frequently proven by the engine contradicting ITSELF (concrete-RHS solve says No solution while symbolic-RHS asserts a root; concrete eval gives |x| while symbolic gives x). Soundness is reachable in almost every case -- the answers are within the engine's grasp -- which makes these honesty violations rather than honest capability gaps.

## Clusters by root cause


### K. Definite-integral endpoint-pole zero-scan short-circuits at the first boundary zero, missing the second/interior pole and fabricating a signed +/-infinity for indeterminate integrals

**Severity: critical · honesty-violation · 4 probes. — FIXED (commit `c73315a2045b4b21c054cb3c54fab08c40b38d80`).** Two changes in `crates/cas_engine/src/rules/calculus/definite_integration.rs`: (1) `trig_nonzero_on_interval` now scans EVERY zero in the closed interval (accumulating `interior_pole` / `touch_lower` / `touch_upper` / `undecidable`) and decides only after the full scan, instead of returning on the first zero — so the interior pole at `π` in `[0,2π]` and the second endpoint pole at `π` in `[0,π]` are no longer missed. (2) `boundary_touch_evaluation`'s both-endpoints-infinite arm `(Some(us), Some(ls))` now returns `undefined` when the two one-sided antiderivative limits share a sign (an `inf − inf` indeterminate → the doubly-improper integral diverges) and a definite `±∞` when they oppose, instead of falling through to an unevaluated residual. Now `integrate(cos(x)/sin(x)^2, x, 0, π)`, `…, 0, 2π)`, `…, π, 2π)`, `…, 0, 3π/2)`, `…, 0, 5π/2)`, and `integrate(csc(x)*cot(x), x, 0, π)` all return **`undefined`** (were `±infinity`). The single-endpoint-pole improper convention is preserved (`integrate(cos(x)/sin(x)^2, x, π/2, π) → -infinity`), and the `IntervalCertificate::Undefined` controls (`1/x²` over `[-1,1]`, `cos/sin²` over `[π/2,3π/2]`) plus all convergent integrals are unchanged. Guardrail+pressure byte-identical except the one new test's count; new unit test `poles_at_both_endpoints_and_interior_are_undefined`. **A 45-probe adversarial sweep confirmed ZERO false positives** (no convergent/convergent-improper/removable integral wrongly rejected — the key regression risk). It surfaced one PRE-EXISTING low-severity convention imprecision unrelated to this fix: `integrate(1/x², x, -1, 1)` returns `undefined` though the true value is a definite `+∞` (both one-sided pieces diverge to `+∞`); `1/x²` is a *polynomial* denominator handled by the untouched interior-pole `IntervalCertificate::Undefined` path (the audit itself listed it as a working control), and the engine blanket-classifies every interior pole as `undefined` even when it is a definite `±∞` (correct for `1/x`, which is genuinely indeterminate, but imprecise for `1/x²`). Sound (rejects as non-finite); refining it to distinguish definite-`±∞` from `inf−inf` at interior poles is a separate residual.


**Root cause.** trig_nonzero_on_interval (definite_integration.rs ~lines 2059-2095) returns at the FIRST zero of the denominator it finds. For sin(x) on [0,pi], it hits x=0 (k=0), returns BoundaryTouch{lower:true,upper:false}, and never iterates to the second zero at pi. The un-detected pole is then evaluated by plain Newton-Leibniz substitution (infinite_sign=None), driving the (None,Some(sign)) arm to build a one-sided signed infinity instead of the (Some,Some)->None indeterminate arm. Over [0,2pi] the interior pole at pi (where the integrand blows up to -inf from BOTH sides) is missed entirely, yielding a sign-wrong +infinity for a non-integrable interior pole. The controls integrate(1/x^2,-1,1) and integrate(.../sin^2, pi/2, 3pi/2) correctly return 'undefined', proving the IntervalCertificate::Undefined path exists but is suppressed by the short-circuit when endpoint zeros are also present.


| probe | expected |
|---|---|

| `integrate(cos(x)/sin(x)^2, x, 0, pi)` | undefined / divergent |

| `integrate(csc(x)*cot(x), x, 0, pi)` | undefined / divergent |

| `integrate(cos(x)/sin(x)^2, x, 0, 2*pi)` | undefined / divergent (interior pole at pi) |

| `integrate(cos(x)/sin(x)^2, x, pi, 2*pi)` | undefined / divergent |


**Suggested fix.** Make trig_nonzero_on_interval enumerate ALL zeros of the denominator within [lower, upper] (closed), not just the first: continue the k-loop across the whole interval and record every boundary-touch and every interior pole. If two or more poles are found, or any interior pole exists, route to the IntervalCertificate::Undefined / (Some,Some)->None indeterminate arm (the path 1/x^2 over [-1,1] already uses). When both endpoints are poles, the antiderivative's two one-sided limits must be combined and, if they give inf-inf, return undefined -- never a single signed infinity. The Cauchy principal value is not the improper-integral value and must not be returned as one.


### A. Solve log-product / log-quotient finishing path leaks an internal 'solve(...) = 0' token as ok=true

**Severity: high · honesty-violation · 5 probes. — FIXED (commit `cddbc719b9ee1977ff869af6068d688b95758ea3`).** Root cause is not log-specific: `solve(x = 8/(x+2), x)` leaks the same residual standalone — the isolated-variable resolver reaches `var = N/D(var)` (variable on both sides via the denominator) and gives up. Implemented NON-recursively at the same chokepoint `resolve_isolated_variable_outcome` (Cluster J's site): for `op == Eq`, when the RHS is rational with `var` in a denominator, cross-multiply `var·D − N = 0` (`combine_fractions_deterministic` + `Polynomial::from_expr`, which expands) and solve the polynomial — rational roots exactly via `rational_sqrt`, irrational roots as `(−b ± √Δ)/(2a)` (only for a LINEAR denominator, which can never vanish at an irrational point). Spurious denominator roots are dropped exactly (`Polynomial::eval` on `D`), so `solve(x=(x²-4)/(x-2),x) → No solution` (drops the cross-multiply root `x=2`). The log-product reductions land here, and the recorded domain (`x>0`) is intersected downstream: `solve(log(2,x)+log(2,x+2)=3,x) → {2}`, `solve(log(3,x)+log(3,x-2)=1,x) → {3}`, `solve(log(5,x)+log(5,x-4)=1,x) → {5}`, `solve(log(2,x·(x-2))=3,x) → {-2,4}`; `solve(x=8/(x+2),x) → {-4,2}`. Exact (BigRational/multipoly, no f64); guardrail+pressure byte-identical; new unit test `resolve_isolated_variable_outcome_cross_multiplies_rational`. **A 53-probe adversarial sweep confirmed the fix introduces ZERO spurious roots, ZERO wrong intervals, and ZERO regressions** (every recovered root substitution-verified; denominator-zero exclusion sound). It surfaced PRE-EXISTING gaps in the same family that this fix does NOT extend to (all `is_regression=false` — they leaked identically before): **(a) degree-≥2 denominators** — `solve(x=8/x²,x)`, `solve(x=16/x³,x)` cross-multiply to a *cubic+* polynomial (`x³-8`, `x⁴-16`) that the recovery (degree ≤ 2 only) bails on, so they still leak the `Solve: solve(…)=0` token; closing this needs the engine's polynomial strategy (a cubic/quartic solver, i.e. the recursive re-solve the non-recursive path avoided). **(b) `x` in the numerator with higher structure** (`solve(x=2x²/(x+1),x)`) — same leak. **(c) a base-10-specific crash** `solve(log(10,x)+log(10,x+3)=1,x) → "Solver error: función [log10] no definida"` (drops the valid root `x=2`), while bases 2/5/7 solve correctly — the log-product *combination* path injects an unregistered `log10` symbol; distinct from this fix. **Cosmetic residuals:** irrational roots render unsimplified (`(2−√8)/2` instead of `1−√2`) — the quadratic *strategy* path produces the clean form; `solve(ln(x)/ln(x-1)=1,x)` reduces to `x = e^(ln(x-1))` (an exp∘ln shape, not `N/D`) and still residuals (sound under-fix).


**Root cause.** After reducing log_b(f)+log_b(g)=k (or =1) to the rational equation x = b^k/(g) via the log-sum-to-product rewrite, the solver fails to recurse into / finalize the resulting equation and instead serializes the unevaluated inner solve(...) node, wrapping it in a 'Solve: ... = 0' display template. ok=true is asserted with no error/warning even though no solution set was produced. The engine CAN finish the reduced quadratic (solve(x*(x+2)=8,x) -> {-4,2}) and CAN evaluate the equation at the root, so the answer is fully reachable; only the log-substitution finishing step is broken. The valid in-domain root (and the recorded x>0 / x>k domain condition) is silently dropped.


| probe | expected |
|---|---|

| `solve(log(2,x)+log(2,x+2)=3,x)` | {2} (x=-4 rejected by x>0) |

| `solve(log(3,x)+log(3,x-2)=1,x)` | {3} (x=-1 rejected by x>2) |

| `solve(log(5,x)+log(5,x-4)=1,x)` | {5} (x=-1 rejected by x>4) |

| `solve(log(2,x*(x-2))=3,x)` | {-2, 4} (both in-domain) |

| `solve(ln(x)/ln(x-1)=1,x)` | No solution / EmptySet |


**Suggested fix.** In the log-product/log-quotient solve branch, after producing the reduced polynomial/rational equation, recursively call the solver on it (the same code path that already handles solve(x*(x+2)=8,x)), then intersect the returned root set with the recorded domain conditions (each log argument > 0). If the recursion returns no finished set, do NOT emit the inner solve(...) node as a result: either return the honest 'cannot solve' error (ok=false) or keep a clearly-marked symbolic residual. Never serialize an unevaluated nested solve(...) wrapped in 'Solve: ... = 0' as an ok=true result. Cross-check the ln/ln quotient form against the ln(x)=ln(x-1) reduction which already correctly returns No solution.


### B. Symbolic-RHS even-root solve drops the parameter sign condition (a >= 0)

**Severity: high · dropped-condition · 4 probes. — FIXED (commit `bb1074c72e00638e8acd73b6fc766a7ba2ba1db3`).** `solve(sqrt(x)=a,x) → {a²}` recorded only the radicand domain `x ≥ 0` (from `infer_implicit_domain`) but dropped the even-root RANGE condition `a ≥ 0` (an even root has range `[0,∞)`, so `sqrt(x)=a` has a real solution only when `a ≥ 0`). Added a pure helper `even_root_range_conditions(ctx, lhs, rhs)` in `crates/cas_solver_core/src/domain_inference.rs`: when EXACTLY ONE side is a positive even root (`sqrt(·)` via `extract_sqrt_argument_view`, or `Pow(·, p/q)` with `q` even and `p > 0` via `is_even_root_exponent` — so `x^(1/2)`, `x^(1/4)`, `x^(3/4)` qualify; odd roots and `1/√` do not), it emits `NonNegative(other_side)`. It is applied at the preflight derive-closure (`solve_runtime_pipeline_preflight_context_bound_runtime.rs`) **gated to `op == RelOp::Eq`** — a scoping workflow proved the condition deriver also runs for inequalities, where `a ≥ 0` would be UNSOUND (`sqrt(x) > a` holds for `a < 0`). This avoids threading `op` through `derive_requires_from_equation`'s 15+ call sites. Now `solve(sqrt(x)=a,x)`, `solve(x^(1/2)=a,x)`, `solve(x^(1/4)=a,x)`, `solve(sqrt(x-1)=a,x)`, `solve(sqrt(x)=2a,x)` all carry **`a ≥ 0`** in `required_display`; numeric cases are unchanged (`sqrt(x)=2 → {4}`, `sqrt(x)=-2 → No solution` via the existing collapse, `sqrt(x)=0 → {0}`); the odd root `solve(x^(1/3)=a,x) → {a³}` stays unconditional; and inequalities (`sqrt(x)>a`, `<a`, `>=a`) correctly get NO `a ≥ 0` (op-gated). The condition flows through the same sink as the radicand `x ≥ 0` (display for symbolic, contradiction-collapse for numeric-negative, dropped-as-trivial for numeric-nonneg). Guardrail+pressure structurally byte-identical (state/passed/failed + all scalar counters unchanged; no fixture solves a symbolic-RHS even-root equation); new unit test `even_root_range_conditions_constrain_the_non_radical_side`. **A 183-probe / 4-lens adversarial sweep (primary lens: an op-gating LEAK adding `a ≥ 0` to an inequality, plus a false No-solution) found ZERO op-gating leaks and ZERO over-conditions.** It surfaced 3 pre-existing coverage gaps (bisect-verified BYTE-IDENTICAL pre/post the fix commit at `e932385c5`, so NOT regressions): a COEFFICIENT/SIGN on the radical was not recognised (`3·√x = a`, `4·x^(1/4) = a` dropped `a ≥ 0`; `-√x = a` dropped `a ≤ 0`). The POSITIVE-coefficient forms were then CLOSED (follow-up) by generalising the detector to `is_nonnegative_even_root_form` (a bare even root OR a positive rational multiple/quotient of one: `c·√x`, `√x/c` with `c > 0`) — now `solve(3·√x=a)`, `solve(2·√x=a)`, `solve(4·x^(1/4)=a)`, `solve(√x/3=a)`, `solve(3·x^(1/6)=a)` all carry `a ≥ 0`. **Residual:** a NEGATIVE coefficient on the radical (`-√x = a`, `-2·√x = a`) needs the OTHER side `≤ 0` (`NonNegative(-a)`), which cannot be built from the immutable `&Context` at the preflight derive-closure; it emits NOTHING (a sound under-answer, never a WRONG `a ≥ 0`) and stays an honest residual.

**Severity: high · dropped-condition · 4 probes.**


**Root cause.** When solving root(x)=a (or 2*sqrt(x)=a, x^(1/4)=a, sqrt(x-1)=a) for x with a SYMBOLIC right-hand side, the solver squares/raises both sides to get x = a^2 (etc.) and emits only the radicand-nonnegativity guard (x >= 0), which is vacuous because a^2 >= 0 always. The mandatory parameter condition a >= 0 (the principal even root has range [0,inf)) is never generated on this path. The engine demonstrably has the machinery -- solve(x^2=a,x) correctly emits 'a >= 0' and solve(sqrt(x)=-3,x) correctly returns No solution -- but the radical-equals-parameter inversion path omits it. Odd roots (x^(1/3)=a) are correctly unconditional, confirming the defect is specific to even-index radicals.


| probe | expected |
|---|---|

| `solve(sqrt(x)=a,x)` | { a^2 } guarded by a >= 0 (No solution for a<0) |

| `solve(x^(1/4)=a,x)` | { a^4 } guarded by a >= 0 |

| `solve(sqrt(x-1)=a,x)` | { a^2 + 1 } guarded by a >= 0 |

| `solve(2*sqrt(x)=a,x)` | { a^2/4 } guarded by a >= 0 |


**Suggested fix.** In the radical-equals-RHS inversion, when the radical index is even, after computing x = RHS^index attach the parameter condition RHS >= 0 (i.e. require the original RHS to be nonnegative) to required_conditions, exactly as the inverse problem solve(x^2=a,x) already does. For a leading coefficient c (e.g. 2*sqrt(x)=a), the condition is a/c >= 0 with c's sign accounted for. Reuse the existing condition-emission machinery from the x^2=a path; do not rely on the radicand guard x>=0, which is vacuous for a squared RHS.


### C. Trig solve range check |c|<=1 bypassed for non-rational (surd/transcendental) RHS, returns non-real inverse-trig as a real root

**Severity: high · honesty-violation · 6 probes. — FIXED (commit `55be6ea3e9bf2d45593aa453ac3d79234c940740`).** The out-of-range drop in `solution_contains_nonfinite` (`crates/cas_solver/src/solve_backend_local.rs`) decided `|c| > 1` only via `as_rational_const`, so a SURD argument (`√2 ≈ 1.41`) returned `None` and slipped through — `solve(sin(x)=√2,x)` leaked `{arcsin(√2)}`. Replaced the rational-only test with a new `inv_trig_arg_provably_out_of_range` that reduces `c` to the single quadratic-surd normal form `A + B·√n` (`cas_math::root_forms::as_linear_surd`) and decides `|c| > 1 ⟺ c−1 > 0 ∨ c+1 < 0` with the EXACT surd-sign logic of `provable_sign_vs_zero` (`sign(A+1±… )` from `B²n` vs `A²`) — never f64 (the project rule: a float gate could drop a valid root at `c = √2`). It subsumes the rational case. Now `solve(sin(x)=√2,x)`, `solve(cos(x)=√2,x)`, `solve(sin(x)=√3,x)`, `solve(cos(x)=√5,x)`, `solve(sin(x)=2√2,x)`, `solve(sin(x)=√8,x)` all return **No solution**, while in-range surds/rationals are unchanged: `solve(sin(x)=√2/2,x) → {π/4}`, `solve(cos(x)=√3/2,x) → {π/6}`, `solve(sin(x)=1/2,x) → {π/6}`, `solve(sin(x)=1,x) → {π/2}`, `solve(cos(x)=-1,x) → {arccos(-1)}`, and the rational out-of-range controls (`sin(x)=2`, `cos(x)=3/2`) still give No solution. Guardrail+pressure structurally byte-identical (no fixture solves a surd-RHS trig equation); extended unit test `out_of_range_inverse_trig_root_is_not_real` with surd cases. **A 245-probe / 4-lens adversarial sweep (primary lens: a FALSE "No solution" over-drop — an IN-RANGE surd wrongly dropped) found ZERO over-drops** (every `|c| ≤ 1` surd, incl. near-boundary `√99/10 ≈ 0.995`, still solves; boundary `|c| = 1` kept). The 4 findings it surfaced are all the SAME pre-existing residual (bisect-verified BYTE-IDENTICAL pre/post the fix commit at `3222cc610`, so NOT regressions): a NAMED algebraic constant that the engine normalizes BEFORE the filter — `(1+√5)/2 → phi` — escapes the gate (`as_linear_surd(phi) = None`), so `solve(sin(x)=(1+√5)/2,x) → {arcsin(φ)}` and `solve(sin(x)=2·φ,x) → {arcsin(2·φ)}` still leak, even though the LITERAL surd is now dropped (`solve(sin(x)=1+√5,x) → No solution`). **Residual (same opaque-constant class for all of `π`, `e`, `φ`):** a recognized constant that `as_linear_surd` cannot decode is KEPT (a sound under-drop, never a false No-solution). Closing `φ` is cleanly possible (it is exactly `1/2 + (1/2)√5`) but means teaching the SHARED `as_linear_surd` about `φ` — which has real huella risk (`φ` appears in solve outputs such as `solve(x²-x-1=0) → {φ, …}`, and newly-decidable surd signs could drop phi-roots in `root_violates_required_condition` elsewhere), so it belongs to a separate, separately-validated step. Dropping `π`/`e`/`φ` on a float estimate would violate the exact-gate rule.

**Severity: high · honesty-violation · 6 probes.**


**Root cause.** solve(sin(x)=c) / solve(cos(x)=c) gates the |c|<=1 emptiness check on c folding to a rational/decidable literal. For c a surd (sqrt(2), sqrt(3), 1+sqrt(2)) or a transcendental constant (pi/2, e, pi, ln(10)), the comparison c-1>0 / c+1<0 is left symbolic/undecided, so the range gate is skipped and the solver returns { arcsin(c) } / { arccos(c) } (or +/- and pi - variants). Over the reals these inverse-trig values are non-real (complex) for |c|>1, so a non-real value is presented as a real solution in a real-domain-only engine. The rational/integer controls (sin(x)=2, cos(x)=3/2) correctly return No solution, proving the gate exists but is not fired for non-rational RHS.


| probe | expected |
|---|---|

| `solve(sin(x)=sqrt(2),x)` | No solution (sqrt(2)~1.414 > 1) |

| `solve(cos(x)=sqrt(3),x)` | No solution (sqrt(3)~1.732 > 1) |

| `solve(sin(x)=pi/2,x)` | No solution (pi/2~1.571 > 1) |

| `solve(sin(x)=-sqrt(2),x)` | No solution (-sqrt(2) < -1) |

| `solve(cos(x)=-sqrt(3),x)` | No solution (-sqrt(3) < -1) |

| `solve(cos(x)=e,x)` | No solution (e~2.718 > 1) |


**Suggested fix.** Before constructing arcsin(c)/arccos(c) in the trig solve path, prove |c| <= 1 using the EXACT comparison machinery (not f64): decide c-1 and c+1 signs symbolically (the engine already evaluates sqrt(2)>1, e>1 elsewhere). If |c|<=1 cannot be proven, and especially if |c|>1 CAN be proven, return No solution / EmptySet. As a defense-in-depth backstop, refuse to admit arcsin/arccos of an argument the engine itself leaves unevaluated as out-of-domain (the engine already declines to fold arcsin(sqrt(2))). This is the same range gate already working for rational RHS, extended to surd/transcendental RHS.


### D. Solve with provable division-by-zero collapses to identity -> 'All real numbers' instead of empty set

**Severity: high · honesty-violation · 4 probes. — FIXED (commit `42781b6031e33cff092b44d17698850b9abac3ef`).** A new early guard `equation_has_identically_zero_denominator` in `crates/cas_solver/src/solve_backend_local.rs` (`LocalSolveBackend::solve_with_ctx_and_options`, sitting right beside the existing `equation_is_nonzero_const_over_polynomial` short-circuit) walks BOTH sides of the equation for any `Div(_, D)` whose denominator `D` simplifies to the EXACT rational constant `0` and returns `SolutionSet::Empty` BEFORE the isolation logic runs — so the spurious `All real numbers` (and the impossible-conditioned `ℝ if 0 ≠ 0`) is never fabricated. The decision is exact (`as_rational_const`, never f64): a denominator that merely vanishes at isolated points (`x` in `3/x`, `x-1` in `1/(x-1)`) or is nonzero (`x-x+1 → 1`) does NOT match, so legitimate excluded-point and identity solves are untouched. Now `solve(x/0=5,x)`, `solve(x/(x-x)=0,x)`, `solve(x=1/0,x)`, `solve(x/0=0,x)`, `solve(x/0=x,x)`, `solve(5/0=x,x)` all return **No solution**, while controls are unchanged: `solve(0*x=0,x) → ℝ` (genuine identity), `solve(3/x=0,x) → No solution`, `solve(1/x=2,x) → {1/2}`, `solve(x/(x-1)=2,x) → {2}`, `solve(1/(x-x+1)=1,x) → ℝ`. Guardrail+pressure structurally byte-identical (no fixture exercises a zero-denominator solve); new unit test `identically_zero_denominator_makes_equation_unsolvable`. **Residual (sound under-fix):** a denominator that is identically zero but does NOT fold under the simplifier keeps prior behaviour (never a false "No solution"). **A 267-probe / 5-lens adversarial sweep (hunting the only soundness risk — a FALSE "No solution" — across point-vanishing denominators, identities/AllReals, genuine zero-denominators, excluded-point cross-multiplies, and no-division regressions) found ZERO regressions: all 11 confirmed findings were bisect-verified BYTE-IDENTICAL pre/post the fix commit (worktree at `10528f6bb`), so the fix introduces no false empties.** The 11 are a PRE-EXISTING residual class outside this fix's scope: rational equations that combine to `x = polynomial(x)` of degree ≥ 2 (sum/difference of fractions, `1/(x-1)+1/(x+1)=1 → {1±√2}`; cross-multiply-with-excluded-point, `1/(x-2)=x-2 → {1,3}`) still leak the garbled `Solve: solve(…)=0` token with `ok=true` and no roots — the same recursive-solve leak as Cluster A, which fixed only the single-fraction `x = N/D(x)` shape. Closing it needs the engine's polynomial strategy to finish a `x = deg≥2 poly(x)` reorientation (the recursive re-solve the non-recursive path avoided); the highest-ROI next step in the solve family.

**FOLLOW-UP — `x = deg-2 poly(x)` leak closed (commit `8f96cf7d8550b591ff1e9f3a2fe577cd9d88f9c4`).** `try_cross_multiply_rational` (`crates/cas_solver_core/src/solve_outcome.rs`) now handles a CONSTANT (var-free) denominator, not just a var-bearing one: when `var - rhs` is itself a polynomial (its denominator has no variable), it is built DIRECTLY from `rhs` — NOT via `combine_fractions_deterministic`, whose `add_terms_no_sign` flatten DROPS the signs of subtracted terms (`x^2-2` would become `x^2+2`, fabricating a wrong discriminant / a false "No solution") — and the degree-2 case is solved by the shared `solve_rational_quadratic`. The quadratic is canonicalized to a primitive integer triple with `a > 0` and surd roots are emitted in canonical `rat ± coeff·√m` form with a SQUAREFREE radicand (`split_square_factor`), so they render as `1-√2` / `2-√5` (matching the direct-`solve` path) rather than `(2-√8)/2`. Now `solve(1/(x-1)+1/(x+1)=1,x) → {1-√2, √2+1}`, `solve(1/(x-2)+1/(x+2)=1,x) → {1-√5, √5+1}`, `solve(1/(x-1)+1/(x+1)=1/2,x) → {2-√5, √5+2}`, `solve(1/(x-2)=x-2,x) → {1,3}`, `solve(1/(x-1)=x-1,x) → {0,2}`, `solve(x=x^2-2,x) → {-1,2}`, `solve(log(7,x)+log(7,x-1)=1,x) → {(1+√29)/2}` (domain-filtered; sympy itself returns only a `ConditionSet` here). Sign-correctness verified: `solve(x=x^2+2,x) → No solution` (genuine `disc<0`, not a fabricated empty). A LINEAR result is left to the linear-collect path, a cubic+ stays an honest residual, and `from_expr` returns `Err` on any variable-in-denominator (so the direct path can never misread a rational function — `solve(x=8/(x+2),x) → {-4,2}` and all Cluster A cases are unchanged). Guardrail+pressure structurally byte-identical; new unit tests `resolve_isolated_variable_outcome_solves_constant_denominator_quadratic`, `split_square_factor_and_normalize_quadratic_coeffs_canonicalize`. **Residuals (still leak, honest):** reorientations that do NOT land as a clean rational `x = N/D` — `solve(2/(x-1)+3/(x+1)=1,x)` (combines to a nested continued fraction) and `solve((x+1)/(x-1)+(x-1)/(x+1)=4,x)` (reorients through a `√`); the var-denominator-with-`x²`-numerator routing (`solve(x=2x²/(x+1),x)`); and the cubic+ denominator cases (`solve(x=8/x²,x)`).

**Root cause.** The solver's identity/clear-denominator branch multiplies both sides by a denominator D and, when the equation reduces to a tautology, returns AllReals (R). When D is a PROVABLE zero (literal 0, or x-x which folds to 0, or 1/0), the guard D != 0 is either dropped entirely or canonicalized to the impossible NonZero(0) ('0 != 0'); either way the conditional case should collapse to Empty, but the inner AllReals is surfaced as the headline. The evaluator itself returns 'undefined' for x/0 and 1/0, so the solve path contradicts the engine's own division semantics. This is a regression of the Rounds 1-3 'div-by-provable-zero -> undefined' fix that did not cover the solve path. The control solve(0*x=5,x) (genuine zero coefficient) correctly returns No solution, isolating the broken division-by-provable-zero branch.


| probe | expected |
|---|---|

| `solve(x/0=5,x)` | No solution (empty set) |

| `solve(x/(x-x)=0,x)` | No solution (x-x folds to 0) |

| `solve(x=1/0,x)` | No solution (1/0 undefined) |

| `solve(x/0=0,x)` | No solution (empty set) |


**Suggested fix.** In the solve clear-denominator / identity path, before emitting AllReals under a guard D != 0, evaluate D with the existing const-fold/zero-prover (the same one that makes eval x/0 -> undefined and folds x-x -> 0). If D is provably zero (or the NonZero guard canonicalizes to '0 != 0'), collapse the conditional case to SolutionSet::Empty. Likewise, if either side of the equation evaluates to undefined (k/0, 1/(1-1)), the equation has no real solution -> Empty. Do not let an undefined side propagate into a tautology.


### E. Rational inequality solver drops the denominator-sign case split (and demotes the inequality to an equation)

**Severity: high · honesty-violation · 5 probes. — FIXED (commit `7dab4b5a2`).** A scoping workflow refined the root cause: the denominator-sign split is NOT missing — it is fully wired and CORRECT for the variable-in-numerator route (`(x-1)/(x-3) >= 0 -> (-inf,1] U (3,inf)`). The gap is `derive_div_isolation_route` (`solve_outcome.rs`) keying purely on the numerator: a CONSTANT numerator over a var-bearing denominator (`1/(x-2) > 1`) routes to `VariableInDenominator`, which does plain denominator isolation and emits a single ray, dropping the pole. The fix (`crates/cas_solver_core/src/isolation_arithmetic.rs`, at the `derive_div_isolation_route` seam): for an INEQUALITY with the variable only in a LINEAR denominator and a RHS that resolves to a NON-ZERO RATIONAL (the gate tests the exact value, so a folded zero like `1-1` is excluded and a non-rational RHS like `sqrt(2)` stays on the legacy path — the fold is only validated for rational constants), fold the RHS into one fraction — `f/g (op) c  ==>  (f - c*g)/g (op) 0` — so the combined numerator `p = f - c*g` carries the variable; the split's branch equation is then `p (op) 0` with the variable on the LHS (the raw `f (op) c*g` cross-multiply leaves it on the RHS, which the low-level isolate cannot solve). Route that through the numerator pipeline, which owns the sign split + finalize. Two subtleties handled: (1) the RHS is normalized to a canonical `Number` first — a negative/fractional literal arrives as a raw `Div`/`Neg` node (`-1/2` as `Div(-1,2)`) that otherwise mis-signs the polynomial fold (every `-1/2` case was inverted before this); (2) `should_split_division_denominator_sign_cases` (`isolation_utils.rs`) dropped its redundant `contains_var(numerator)` check (always true on the numerator route, so behaviour-preserving there) to license the folded constant-numerator case. The pole stays excluded by the split's strict open domain guards `g>0`/`g<0`, so it is open even for `<=`/`>=`. Now `1/(x-2)>1 -> (2,3)`, `1/(x-2)<1 -> (-inf,2) U (3,inf)`, `1/(x+1)>2 -> (-1,-1/2)`, `2/(x-1)>1 -> (1,3)`, `1/(x-2)<-1 -> (1,2)`; non-strict endpoints are exact (`1/(x-2)>=1 -> (2,3]`, pole open / root closed); negative-fraction RHS is sign-correct (`1/(x-2)>-1/2 -> (-inf,0) U (2,inf)`). **A 1584-probe membership sweep** (constant numerators 1..5 over nine linear denominators incl. negative-leading `3-x`/`5-2x`, all four ops, RHS in {±1,±2,±3,±1/2,±3/2,5/2,...}, checked point-by-point against sympy 1.14) found **ZERO mismatches**. Guardrail+pressure structurally byte-identical (no fixture solves a rational inequality); workspace green; new CLI tests `rational_inequality_cluster_e_tests.rs` (5 audit probes + pole/endpoint + negative-fraction + zero-RHS guards + numerator-route regression) and an extended `should_split` unit test. **Follow-up (commit `b606671ed`).** Generalized the fold to ANY linear-denominator rational inequality and rebuilt it on EXACT polynomial arithmetic. `p = f - c*g` is now formed via `Polynomial` (`f_poly.sub(&c_poly.mul(&g_poly))`) rather than `Sub(f, c*g)` + simplify — the latter left a var-minus-var shape (`(3/2)*(2-x) + x`) uncollected and degenerated to a spurious "No solution". The dispatch on the folded numerator: `p == 0` (`f == c*g`, an identity off the pole) → legacy; `deg p == 1` → the numerator sign split; `deg p == 0` (variable cancelled, nonzero `c'`) → reduce `c'/g (op) 0` to a single STRICT inequality `g (op') 0` (the value is never zero so `>=`/`<=` collapse to strict, pole excluded); `deg p >= 2` → legacy residual. Dropping the nonzero-RHS gate makes the SAME path subsume the `c = 0` linear cases. This closes the variable-in-numerator gap and the `c = 0` linear residual: `(x+1)/(x-2)>3 -> (2,7/2)`, `x/(x-2)>2 -> (2,4)`, `x/(x-2)>3/2 -> (2,6)`, `(x-1)/(x-3)>=1 -> (3,inf)` (variable cancels, `p=2`), and the zero-RHS forms `1/(x-2)>0 -> (2,inf)`, `2/(x-3)>=0 -> (3,inf)` plus the pre-existing **sign bug** `-1/x>0 -> (-inf,0)` (was `(0,inf)`). **A 2840-probe membership sweep** (eleven numerators incl. constant/linear/negative-leading over seven linear denominators, all four ops, RHS in {0,±1,±2,±3,±1/2,±3/2}, point-by-point vs sympy 1.14) found **ZERO mismatches**; all twelve apparent misses in an earlier run were bisect-confirmed PRE-EXISTING (identical output at `7dab4b5a2`). Guardrail+pressure byte-identical; new CLI tests `cluster_e_variable_in_numerator_nonzero_rhs` / `cluster_e_zero_rhs_is_sign_reduced`.

**Residuals (honest, out of scope):** (a) NON-LINEAR denominators — product/power/quadratic `1/((x-1)(x-3))>0`, `1/(x-2)^2>0`, `1/(x^2-1)>1` — have multiple poles and need a full critical-point partition; the two-way sign split captures only one sign change, so they stay an honest residual on the legacy path (still the malformed-token output); (b) the variable-in-numerator degenerate `(x-1)/(x-1)>=1 -> All real numbers` is a SEPARATE pre-existing pole-collapse bug in the union/finalize path (the fraction cancels to `1` before the solver, so it never reaches this fold); (c) a non-rational RHS (`sqrt(2)`) stays on the legacy single-ray path (the fold is only validated for rational constants); (d) a QUADRATIC-or-higher folded numerator (`(x^2-1)/(x-2)>3`) is left to the legacy path (`deg p >= 2`); (e) the CONSTANT-VALUE case where the numerator is a scalar multiple of the denominator (`(2x-4)/(x-2) = 2` off the pole, so `(2x-4)/(x-2)>2 -> All real numbers`, truth `EmptySet`) is a SEPARATE pre-existing bug — the fraction cancels to a constant (losing the `x != pole` domain) UPSTREAM of the solver, so it never reaches this fold. The 3-lens adversarial workflow flagged it as caused by the fold, but a worktree bisect against `7dab4b5a2` proved it BYTE-IDENTICAL before the follow-up (the upstream cancellation + constant-relation handling owns it, not this fold).

**Residuals (b) and (e) NOW FIXED (commit `4a7f97887`).** A scoping workflow corrected the framing: the cancellation does NOT lose the domain — it RECORDS a `NonZero` condition (`(2x-4)/(x-2) -> 2` with `NonZero(x-2)`); the bug was in the solver's variable-eliminated resolver. Two independent defects, both in `crates/cas_solver_core/src/solve_analysis.rs` (`resolve_var_eliminated_residual_with_exclusions`): **RC-A** — for an inequality the constant relation's truth value `diff (op) 0` was never evaluated (the pipeline used equation semantics `diff==0 => identity => AllReals`), so a FALSE relation like `0 > 0` wrongly became AllReals; a new `const_relation_truth` (exact `BigRational` only, `Eq` and non-rational `diff` fall back) maps true→AllReals / false→Empty. **RC-B** — the `NonZero` pole exclusions were applied to only ONE of the three terminal arms; now applied to ALL, so `AllReals` over a canceled fraction becomes `R \ {pole}`. This also fixed the equation path (`(2x-4)/(x-2)=2 -> R\{2}`, previously plain AllReals). Now `x-x>0 -> Empty`, `x-x>=0 -> AllReals`, `(2x-4)/(x-2)>2 -> Empty`, `(2x-4)/(x-2)>=2 -> R\{2}`, `(x-1)/(x-1)>=1 -> R\{1}`. The `op` is threaded down the var-eliminated resolver chain (3 functions); `Eq` semantics are byte-for-byte unchanged. Validation: equation sweep + var-cancel + constant-value fraction membership sweeps vs sympy 1.14 all ZERO mismatch; guardrail+pressure huella byte-identical; new CLI tests `constant_relation_solve_tests.rs`. **Remaining residual (pre-existing soundness hole, separate follow-up).** An IRRATIONAL/symbolic `diff` has no sign oracle, so `const_relation_truth` returns `None` and the relation BAILs to the prior equation-semantics classification — which for a not-provably-nonzero symbolic constant yields `AllReals`. So a FALSE irrational inequality like `x-x+pi > 4` (`pi > 4` is false) wrongly returns "All real numbers" instead of Empty (and `x-x+ln(2) > 0`, true, wrongly returns "No solution"). A 3-lens adversarial workflow surfaced this; a worktree bisect against `78bd3df11` proved it BYTE-IDENTICAL before this fix, so it is PRE-EXISTING (this fix neither introduces nor worsens it — it only adds the exact-rational truth path, leaving irrationals on the legacy path). Closing it needs a symbolic-constant SIGN oracle using EXACT rational interval bounds for `pi`/`e`/`ln`/`sqrt` (never f64); that is a separate, larger capability. The pure variable-free `solve(2>2,x)` likewise stays a `VariableNotFound` error (a different routing layer, deliberately out of scope).

**Irrational-constant residual NOW CLOSED for algebraic constants (commit `128be3ee3`).** Built `cas_math::const_sign` — an EXACT rational sign oracle. It computes verified rational value bounds `[lo, hi]` for a constant expression over rationals, `pi`/`e` (hand-verified to 50 decimal places), `phi` (derived from the arbitrary-precision `sqrt(5)`), `sqrt` (Newton-from-above + perfect-square fast path), and `+ - * /` / integer powers, plus cheap exact sign rules for bare `ln`/`log` (sign = `arg` vs 1) and `exp` (always positive). `const_relation_truth` now calls `provable_const_sign(diff)` instead of `as_rational_const` only, so `x-x+pi>4 -> Empty`, `x-x+pi>3 -> AllReals`, `2*pi<6 -> Empty`, `e<2 -> Empty`, `sqrt(2)>2 -> Empty`, `ln(2)>0 -> AllReals`, and the equation arm overrides for a provably-nonzero constant (`x-x+pi=4 -> Empty`). Everything is `BigRational`; the oracle returns `None` (BAIL) when it cannot prove the sign — never a float guess. **An adversarial workflow caught that the first cut used only 8-decimal bounds, so a near-boundary threshold within ~1e-8 (e.g. `e - 2.71828182 = +8.5e-9`) made the interval straddle zero and the oracle BAIL to the legacy AllReals fallback** (a pre-existing wrong verdict, NOT introduced by the oracle, which correctly bails); widening to 50 decimals (and `phi` via `sqrt(5)`) closed every one. **Validation: a 1,134-case near-boundary sweep with mpmath at 60 digits (thresholds truncated to 1e-12 of each constant) found ZERO unsound verdicts**, plus the original 1,740-case algebraic sweep vs sympy; guardrail+pressure huella byte-identical; `const_sign` unit tests incl. `near_zero_thresholds_are_decided` + CLI `irrational_constant_relations_use_the_exact_sign_oracle`. **Undecidable-constant SOUNDNESS now closed via an honest conditional (commit `630cc166a`).** The deeper hole the adversarial workflow exposed was not the oracle (which correctly bails) but the var-eliminated resolver's DEFAULT: when the oracle cannot prove the sign of a variable-free constant, the equation-semantics fallback still committed to a definite `AllReals`/`Empty` (so `x-x+sin(1)>2` wrongly returned "All real numbers", `x-x+ln(2)<1` wrongly "No solution"). Now, for an INEQUALITY whose residual is a variable-FREE constant the oracle could not decide, `resolve_var_eliminated_residual_with_exclusions` returns an honest `Conditional[<relation> -> AllReals, else -> Empty]` instead — e.g. `solve(x-x+sin(1)>2,x) -> "All real numbers if sin(1) - 2 > 0"`. This NEVER asserts an unjustified verdict (sound); `Conditional::simplify` still collapses to a definite set when a downstream prover can decide the predicate. The `Eq` identity path and the PARAMETRIC case (a free variable, e.g. `a > 0`) keep the legacy classification; oracle-decidable constants stay definite (no over-hedging). Guardrail+pressure huella byte-identical; new CLI test `undecidable_constant_inequality_is_honest_conditional_not_a_wrong_verdict`. **Remaining residuals (COMPLETENESS, no longer SOUNDNESS):** deciding `ln`/`log`/`exp` VALUE comparisons (`ln(2) < 1`) needs ln/exp value bounds (Taylor/atanh series error terms) and `sin`/`cos`/`tan` need range-reduced bounds — until then those return the honest conditional rather than a definite verdict; a threshold within 1e-50 of a `pi`/`e` value also yields the conditional. The PARAMETRIC var-eliminated inequality (`x-x+a>0 -> AllReals`) is a separate, untouched legacy behavior. **The EQUATION case is NOW also closed via an exact EqZero prover (commit `9b82a0f46`).** Previously `x-x+sin(1)=1/2` wrongly returned "All real numbers" (the `Constraint -> AllReals` default); the hazard to fixing it was that a GENUINE but un-foldable identity (`log2(8) - 3 = 0`, since `log2(8)` does not fold to `3`) would have regressed from a correct `AllReals` to a hedged conditional. The dual `is_provably_zero_constant` (`solve_outcome.rs`) — mirror of `is_provably_nonzero_constant`, recognising a folded rational `0`, anything the sign oracle pins to zero (`sqrt(4)-2`), the rational-base log identity `k*log_b(c)+s=0 <=> c=b^m` via the exact `b^p == c^q` test (`rational_power_provably_equal`), and `log_b(1)=0` — lets the resolver's undecidable EQUATION arm give a definite `AllReals` for a proven identity and the honest conditional ONLY for a genuinely-undecidable equality. So `log2(8)-3=0 -> AllReals` (identity), `sin(1)=1/2 -> "All real numbers if sin(1) - 1/2 = 0"` (honest), and the provably-irrational `ln(2)-1=0 -> Empty` is preserved (no regression). **A 720-case transcendental sweep (sin/cos/tan/ln/log/exp/pi/e × all six ops, incl. `Eq`) found ZERO wrong definite verdicts (was 54, all `Eq`).** With this, the variable-cancellation constants zone is SOUND for every operator THAT THE PROVERS CAN DECIDE: decidable constants get the correct verdict, genuine identities (rational, sqrt, log) `AllReals`, provably-nonzero `Empty`, and anything the oracle/EqZero prover leaves genuinely undecidable an honest conditional. Guardrail+pressure huella byte-identical; new unit test `is_provably_zero_constant_separates_identity_from_false_and_irrational` + CLI `undecidable_constant_equation_separates_identity_from_false_equality`. The PARAMETRIC var-eliminated case (`x-x+a>0`) remains separate, untouched legacy behaviour.

**An adversarial sweep of the EqZero prover (53+45+33 constants; ground truth sympy `minimal_polynomial`/`is_zero` + mpmath to 130–200 dps) confirmed the prover itself is exact and sound, but surfaced TWO PRE-EXISTING soundness holes upstream of it. Both are bisect-confirmed IDENTICAL at the parent commit `985a904a7`, so NEITHER was introduced by `9b82a0f46`:**
- **(H1) f64 log-ratio fold → false `AllReals` — NOW FIXED (commit `PENDING_HASH_H1`).** The simplifier folded `log(a)/log(b) − <decimal>` to a literal `0` on f64-closeness, so `eval "log(2)/log(3) - 0.6309297535714574"` returned `0` though the true residual is `+3.7e-17` (a transcendental ratio cannot equal a finite decimal); the exact EqZero prover then correctly reported `AllReals` for the literal `0` it was handed, so the unsoundness was the upstream fold, not the prover. Scope was log-ratio-specific because the only CALLERS are fraction simplifications (`pi-3.14159…`, `exp(1)-2.71828…`, `ln(2)-0.693…` correctly stayed symbolic — no `Div`, so the fraction path never ran). **Root cause (bisected to `c9a46c5e`):** `numeric_poly_zero_check` (`cas_math/src/numeric_eval.rs`), called by the fraction zero-numerator / denominator-equivalence checks, had — for a variable-free constant — a fallback `if let Some(v) = eval_f64(expr) { return v.abs() < 1e-10; }` after the exact `as_rational_const` path. That `1e-10` f64 gate declared a nonzero transcendental (whose f64 value lands within `1e-10` of zero because the decimal literal IS its f64 rounding) to be exactly zero — the textbook "soundness gates must be exact" violation. **Fix:** replace the f64 gate with an EXACT decision — `as_rational_const == 0`, else `provable_const_sign == Zero` (the exact rational sign oracle, which still confirms rational and perfect-square-surd zeros like `sqrt(4)-2` and bails otherwise, never guessing from a float). Now every H1 witness stays symbolic (`log(2)/log(3) - 0.6309297535714574 → ln(2)/ln(3) - 3154648767857287/5000000000000000`), surd-zeros still fold (`sqrt(2)·sqrt(2)-2 → 0` via the upstream exact sqrt rules + `const_sign`), and fraction combination is intact (`1/(x-2)+1/(x+2) → 2x/(x²-4)`). The only behavioural loss is a COMPLETENESS one: a genuine but un-evaluated rational-log identity (`log(4)/log(2) - 2`) no longer folds to `0` in a fraction context — acceptable and consistent, since the engine does not evaluate `log(4)/log(2)` to `2` standalone anyway, and no test/scorecard depended on it. Workspace green (313 ok-suites), clippy clean, guardrail+pressure huella structurally clean (only timing / non-deterministic list ordering differ — confirmed by a run-to-run identical-code re-generation), new unit test `numeric_poly_zero_check_is_exact_not_f64_for_constants`.
- **(H2) trig-product → false `Empty` — NOW FIXED (commit `8b8abfe56`).** The "false Empty" in `solve` was only a downstream symptom; the real bug was a SIMPLIFIER soundness fault affecting plain `eval`: a 2-factor product of trig functions at a NON-constructible angle (denominator 7, 9, 11, …) collapsed to a WRONG constant — `sin(pi/7)·cos(pi/7) → 0` (true ≈ 0.39), `cos(pi/7)·cos(2pi/7) → 1` (≈ 0.56), `sin(pi/7)·sin(pi/7) → 0` (≈ 0.19). Constructible angles (4, 5, 8 → `sqrt(5)/4`, etc.) were correct, and a leading coefficient (`2·sin·cos`) masked it. **Root cause (bisected to `6ed8d388`):** `parse_angle_from_expr` (`cas_math/src/trig_table.rs`) extracted the π-coefficient of `Mul(Number(k/n), pi)` via `n.to_integer().to_i32()`, which **TRUNCATES** the fraction — so `(2/7)·π` (how the simplifier stores `2·π/7`) parsed as the integer `0·π`, and the special-angle table returned `sin(0)=0, cos(0)=1`. The orchestrator's `root.mul.19.special_angle_exact_value_factor` / `root.mul.14.two_factor_direct_pair_anchor` shortcuts then multiplied those bogus per-factor values into a false product constant; the solver subsequently signed the corrupted constant and returned `Empty`. **Fix:** carry numerator AND denominator (`AngleSpec::new(n.numer(), n.denom())`) in both the `Mul` and `Div` arms, so a non-table angle correctly BAILS (returns `None`) instead of mis-evaluating to `0`. Now the products are preserved (`sin(pi/7)·cos(pi/7)` stays symbolic), the legitimate double-angle contraction still fires (`2·sin(pi/7)·cos(pi/7) → sin(2pi/7)`), the dyadic identity `8·cos(pi/9)·cos(2pi/9)·cos(4pi/9) → 1` is untouched, and the `solve` symptom now returns the honest conditional (the constant truly is `0`, but the engine cannot prove the trig-product identity, so it conditions rather than falsely asserting). Workspace green (313 ok-suites), guardrail+pressure huella structurally clean (only timing / non-deterministic list ordering differ — confirmed by a run-to-run identical-code re-generation), new unit test `parse_angle_keeps_fractional_pi_coefficient` + regression tests `test_non_constructible_trig_products_not_collapsed` / `test_double_angle_contraction_still_works_at_seventh`. **Adversarial verification:** a 40+-input sweep (trig products/singles/multi-angle/sums at constructible AND non-constructible angles, integer AND fractional π-coefficients) compared every DEFINITE engine result against `sympy`/`mpmath` at 50 digits — 0 wrong definite values (26 correctly left symbolic, 14 definite all exact), and an explicit hunt for SIBLING fractional-coefficient truncation (single values `sin(2pi/7)`/`cos(2pi/7)`/`tan(2pi/9)`, multi-angle expansions) found none. (Run deterministically by hand because the parallel-agent workflow hit transient API-overload errors; the by-hand sweep covers the same attack surface.)

Both H1 and H2 are now FIXED. The remaining items are COMPLETENESS, not soundness: deciding ln/exp/trig VALUES (so a genuine-but-undecidable constant comparison stays an honest conditional rather than a definite verdict), the genuine rational-log identity folding in fraction contexts that H1's sound fix dropped (recoverable later via an exact `a^q == b^p` log recogniser if wanted), and the parametric var-eliminated case (`x-x+a>0`).


**Root cause.** For f(x)/g(x) > c (or <, <=, >=), the solver finds the boundary root by solving the EQUATION f/g = c (visible in input_latex, which rewrites '1/(x-2)>1' to '1/(x-2)=1') and then emits a single ray above/below that root, never splitting on the sign of the denominator g(x) and never accounting for the pole. The result interval is frequently DISJOINT from the truth (e.g. (3,inf) instead of (2,3)), so every claimed membership is wrong. The engine's own exact arithmetic refutes its answer at concrete points. This is well-formed (no infinity-token garbage), so it is DISTINCT from the known-open malformed-token case solve(1/((x-1)*(x-3))>0,x).


| probe | expected |
|---|---|

| `solve(1/(x-2)>1,x)` | (2, 3) |

| `solve(1/(x-2)<1,x)` | (-inf, 2) U (3, inf) |

| `solve(1/(x+1)>2,x)` | (-1, -1/2) |

| `solve(2/(x-1)>1,x)` | (1, 3) |

| `solve(1/(x-2)<-1,x)` | (1, 2) |


**Suggested fix.** Implement the standard rational-inequality algorithm: move everything to one side (f/g - c), find the zeros of the numerator AND the poles (zeros of the denominator), partition the real line at all of these critical points, and determine the sign of the expression on each open subinterval by exact test-point evaluation. Take the union of intervals where the (strict/non-strict) inequality holds, excluding poles. Do NOT solve the demoted equation and emit a single monotone ray. The poles must always be excluded from the solution set. This same routine should subsume the known-open product-denominator case and eliminate the infinity-token output.


### F. Domain inequality (sqrt/ln/log) does not intersect the result with the argument-domain; negative/undefined axis included

**Severity: high · dropped-condition · 3 probes. — FIXED (commit `c787e992625cd190d6624457b30267f09236a73f`).** A new backend post-step `intersect_inequality_with_function_domain` (`crates/cas_solver/src/solve_backend_local.rs`, applied in `solve_with_ctx_and_options` after `filter_real_solutions`) intersects a monotonic-function inequality result with the function's real argument-domain, which the inversion (square/exponentiate) drops. It runs ONLY for the four inequality ops, ONLY when the LHS is `√(x)` / even-root `Pow` / `ln(x)` / `log(b,x)` over the BARE solve variable (detected by `detect_monotonic_lhs`). Domain: even root → `[0,∞)` (closed at 0); `ln`/`log` → `(0,∞)` (open). A scoping workflow surfaced TWO traps beyond the audit's framing, both handled: (1) **the even-root RANGE** — squaring the threshold `c` is invalid when `c` is on the wrong side of 0, so the helper folds it directly: `√<c≤0`/`√≤c<0 → ∅`, `√>c<0`/`√≥c≤0 → [0,∞)` (the audit only noted the `<` direction; `sqrt(x)>-1 → (1,∞)` was an UNLISTED invalid-squaring bug, now `[0,∞)`); (2) **a soundness hole** — the inverted bound reaches the result UNSIMPLIFIED (`2^2`), so the interval-validity gate would fall back to STRUCTURAL ordering (`compare(0, 0^2)=Less` wrongly keeps `sqrt(x)<0` non-empty); the helper SIMPLIFIES the bound (`simplify_solution_bounds`) before intersecting so the gate is an EXACT numeric comparison (the project's "soundness gates must be exact" rule). Now `solve(sqrt(x)<2,x) → [0,4)`, `solve(ln(x)<0,x) → (0,1)`, `solve(log(2,x)<3,x) → (0,8)`, `solve(sqrt(x)<-1,x) → No solution`, `solve(sqrt(x)<=0,x) → {0}`, `solve(sqrt(x)>-1,x) → [0,∞)`; the `>`/`≥` positive-RHS and EQUATION paths are unchanged (`solve(sqrt(x)>2,x) → (4,∞)`, `solve(sqrt(x)=-2,x) → No solution`). The accompanying LaTeX endpoint bug is also fixed (`render_continuous_interval` in `eval_output_presentation_solution_latex/intervals.rs` hardcoded `\left[ … \right]`; now respects each `BoundType`, so `[0,4)` renders `\left[0, 4\right)` and `-∞`/`∞` get open brackets). Guardrail+pressure structurally byte-identical (no fixture solves a sqrt/ln/log inequality); workspace green; new unit test `monotonic_inequality_intersects_argument_domain`. **A 194-probe / 4-lens adversarial sweep (membership-based — substitute concrete `x`; primary lens: over-narrowing / dropped solutions) found ZERO over-narrows and ZERO regressions** on the bare-variable forms. It surfaced 2 pre-existing gaps (bisect-verified at `65adb65d0`): (a) a COEFFICIENT on the radical was not detected (`2·√x < 6 → (-∞,9)`, missing the domain bound) — CLOSED (follow-up) by having `detect_monotonic_lhs` see THROUGH a positive rational multiplicative coefficient/divisor (`c·√x`, `√x/c`, `c>0`), which preserves the `[0,∞)` range so the range correction is unaffected: now `2·√x<6 → [0,9)`, `3·√x≤6 → [0,4]`, `2·√x<-6 → No solution`, `2·ln(x)<6 → (0,e³)`; (b) `log(1/2,x)<3 → (0,1/8)` is WRONG (truth `(1/8,∞)`) — a SEPARATE pre-existing **upstream inversion bug**: a base `< 1` log is DECREASING and the inversion fails to flip the inequality direction (this fix only narrows the already-wrong half-line to its correct domain `(0,∞)`; it does not address the direction). **Residuals (honest):** a COMPOUND argument (`sqrt(x-1)<2` stays `(-∞,5)`), a function on the RHS (`2>sqrt(x)`), an ADDITIVE shift (`√x+1<3`, shifts the range), a NEGATIVE coefficient (`-2·√x<6`, flips the range), and the base-`<1` log direction flip.

**Severity: high · dropped-condition · 3 probes.**


**Root cause.** For sqrt(x)<c, ln(x)<c, log(b,x)<c the solver inverts the function (square / exponentiate) to get the boundary and returns a half-line, but never intersects the solution set with the function's argument-domain (x>=0 for sqrt, x>0 for ln/log). The result therefore includes the entire region where the function is undefined over R (e.g. (-inf,4) for sqrt(x)<2, (-inf,1) for ln(x)<0). The x>0/x>=0 condition appears only in a separate required_display field that is NOT folded into the result and is absent from the default/only text output. The engine's equation contract DOES fold domain into the result (solve(sqrt(x)=-2,x)->No solution), so the inequality path violates that contract. The >/>= directions happen to be correct only because their true set already lies inside the domain.


| probe | expected |
|---|---|

| `solve(sqrt(x)<2,x)` | [0, 4) |

| `solve(ln(x)<0,x)` | (0, 1) |

| `solve(log(2,x)<3,x)` | (0, 8) |


**Suggested fix.** After solving the inverted inequality, intersect the resulting interval with the argument's real domain (sqrt: [0,inf); ln/log: (0,inf)) and emit THAT as the result set, mirroring the equation path that already returns No solution when the domain excludes the root. Use the exact closed/half-open endpoint at the domain boundary (e.g. [0,4) keeps x=0 for sqrt(x)<2 since sqrt(0)=0<2). Also fix the LaTeX endpoint rendering bug where -infinity is shown with a closed bracket '\left[-\infty'.


### G. Additive like-term collector cancels coefficients BEFORE checking definedness, fabricating finite values for undefined/non-real subterms

**Severity: high · honesty-violation · 4 probes. — FIXED (commit `ce1e8dbb5bef81b5a872b01a6dc8bae01b7aabe5`).** The additive-annihilation guard already declined to cancel a term carrying a LITERAL non-finite/undefined value (`1/0 - 1/0`, `inf - inf`, `undefined - undefined`) via `expr_carries_nonfinite_or_undefined` → `is_structurally_undefined_over_reals` (`crates/cas_math/src/arithmetic_cancel_support.rs`), but that structural check did not recognise a var-free FUNCTION call that evaluates to undefined. Extended `is_structurally_undefined_over_reals` with three exact, var-free rules: (1) `tan(kπ)` / `sec(kπ)` are undefined at the half-odd-integer multiples of π (`k = arg/π` has `2k` an odd integer); (2) `cot(kπ)` / `csc(kπ)` are undefined at every integer multiple of π (generalising the prior `cot(0)`/`csc(0)`); (3) `ln(c)` / `log(c)` of a provably-NEGATIVE rational `c` is undefined over ℝ. The π-multiple is read by a local recursive evaluator (`rational_pi_multiple_signed`) that resolves `Neg`/`Div`/`Mul` sign placement (`-π/2 = Div(Neg(π),2)`, `3π/2`, `(-1/2)·π`) WITHOUT touching the huella-sensitive shared `extract_rational_pi_multiple`. Because the backstop only blocks ADDITIVE drops (function *evaluations* on non-additive nodes are never blocked), the annihilation `u − u` is rejected so the terms survive and the evaluator folds them to `undefined`, while `tan(π/2)` etc. still evaluate normally standalone. Now `tan(pi/2)-tan(pi/2)`, `tan(-pi/2)-tan(-pi/2)`, `tan(3*pi/2)-…`, `sec(±pi/2)-…`, `cot(±pi)-…`, `cot(2*pi)-…`, `csc(pi)-…`, `ln(-5)-ln(-5)`, `ln(-1/2)-…`, and `tan(pi/2)-tan(pi/2)+5` all return **`undefined`**. Controls unchanged (cancel to `0`): `tan(pi/3)`, `tan(pi/4)`, `cot(pi/2)`, `cos(pi)`, `sin(-pi)`, `tan(0)`, `ln(5)`, `x-x`, `sin(x)-sin(x)`. Guardrail+pressure structurally byte-identical (state/passed/failed and all scalar counters unchanged; only timing-ranked diagnostic lists reshuffle); new unit test `trig_poles_and_negative_log_block_additive_cancellation`. **A 256-probe / 4-lens adversarial sweep (primary lens: a FALSE "undefined" over-block — a DEFINED trig/log value wrongly flagged) found ZERO over-blocks**, confirming no legitimate cancellation was turned undefined. It surfaced 25 PRE-EXISTING under-blocks (bisect-verified BYTE-IDENTICAL pre/post the fix commit at `d812410f0` — `→ 0` both before and after, so NOT regressions) of two kinds, both then CLOSED in the follow-up commit `d7f8eef8de6b63930b58cd079c54c15b24a38dc6`: (a) **sum/difference arguments** that normalize to a pole (`tan(π/4+π/4)`, `tan(π/2+π)`, `cot(π/2+π/2)`, `tan(π/2+0)`) — `rational_pi_multiple_signed` now evaluates `Add`/`Sub` (and `0 = 0·π`), resolving the π-multiple of a sum while still returning `None` for a symbolic `x+π/2` and leaving DEFINED normalizations alone (`tan(π/2+π/2)=tan(π)=0`, `cot(π/3+π/6)=cot(π/2)=0` still cancel to `0`); (b) **two-argument `log(base, value)` of a negative value** (`log(2,-8)`, `log(10,-3)`) — a new `args.len()==2` arm flags `args[1] < 0`. **Residual:** `ln(0)` is intentionally NOT flagged (it is −∞, non-finite, with `1/ln(0)=0`); `sqrt(-5)-sqrt(-5)` stays in the imaginary-form handling (Cluster H/I territory, not additive-cancel).

**Severity: high · honesty-violation · 4 probes.**


**Root cause.** The like-term collector sums integer coefficients of matching atoms (e.g. tan(pi/2): 1 + (-1) = 0) and discards the atom when the coefficient is 0, BEFORE the pole/undefined/non-real fold runs. So tan(pi/2)-tan(pi/2) -> 0, sec(pi/2)+1-sec(pi/2) -> 1, ln(0)-ln(0) -> 0, ln(-5)-ln(-5) -> 0, even in strict mode. Every OTHER arithmetic path correctly propagates undefined (tan(pi/2)*0, tan(pi/2)+tan(pi/2), 2*ln(0)-ln(0), ln(-5)/ln(-5) all -> undefined), and sqrt(-5)-sqrt(-5) stays symbolic -- proving the fold-to-finite is an internal contradiction, not a defensible f-f=0 convention (the engine commits ln(0) to -infinity and applies inf-inf=undefined everywhere else).


| probe | expected |
|---|---|

| `tan(pi/2)-tan(pi/2)` | undefined |

| `sec(pi/2)+1-sec(pi/2)` | undefined |

| `ln(0)-ln(0)` | undefined |

| `ln(-5)-ln(-5)` | undefined (ln(-5) non-real over R) |


**Suggested fix.** Before (or during) like-term coefficient collection, screen each term's atom for definedness over R: if any atom evaluates to undefined / a pole / a non-real value (using the same checks that already make tan(pi/2)*0 and ln(-5)/ln(-5) undefined), do NOT cancel it away to a zero coefficient -- propagate undefined for the whole sum (matching the extended-real inf-inf and (-inf)-(-inf) indeterminate rules the engine already applies on the matched-coefficient path 2*ln(0)-ln(0)). Equivalently, run the undefined/pole detection pass before the additive simplifier discards subterms.


### H. Imaginary-Usage-Warning gated on the INPUT containing i, not the RESULT, so perfect-square radicals fold to k*i silently

**Severity: high · honesty-violation · 4 probes. — FIXED (commit `942c05fe4f3f457540ebe6e77ae9a669d139129d`).** The warning site (`crates/cas_engine/src/eval/simplify_action.rs`) already scanned the RESULT (`resolved`), but with `contains_i`, which only recognises a base of EXACTLY `-1` (literal `(-1)^(1/2)` / `sqrt(-1)` / `I`). The engine stores `sqrt(-25)` as `(-25)^(1/2)` (the display factors out `5·(-1)^(1/2)`), so `contains_i` missed it. Added `expr_contains_imaginary` in `crates/cas_math/src/numeric_eval.rs` — `contains_i` PLUS an even root of a provably-NEGATIVE value (`(-n)^(1/2)`, `sqrt(neg)`, `(-2)^(3/2)`, `(-16)^(1/4)`; via `is_even_root_exponent` + a negative rational base) — and switched the warning gate to it. Kept SEPARATE from `contains_i` so the INPUT-gated `ComplexMode::Auto` resolution (and the orchestrator's `contains_i` use) are untouched: zero result/mode change, only the missing caveat. Now `sqrt(-4)+sqrt(-9)`, `sqrt(-16)`, `sqrt(-25)`, `(-4)^(1/2)` all carry the **Imaginary Usage Warning** (as `sqrt(-1)+sqrt(-1)` already did), and the family extends correctly: non-perfect-square `sqrt(-2)` and higher even roots `(-8)^(3/2)`, `(-16)^(1/4)` warn too. Crucially, REAL results are NOT flagged — an ODD root of a negative `(-8)^(1/3) = -2` and `(-27)^(1/3) = -3`, even roots of positives, `sqrt(2)`, `x^2`, and plain rationals get no warning. **A 144-case sweep** (`base^(p/q)` over `base ∈ [-27,27]` × nine exponents, plus sums/products) found **ZERO mismatches** against the exact rule "imaginary ⟺ negative base AND even-denominator exponent". Guardrail+pressure structurally byte-identical; new unit test `expr_contains_imaginary_detects_even_root_of_negative`. **Note:** consistent with the pre-existing `sqrt(-1)` behaviour, `value_domain` stays `real` in RealOnly mode and the WARNING is the honesty signal (left as-is to match the established convention); the expression is correctly kept symbolic.

**Severity: high · honesty-violation · 4 probes.**


**Root cause.** The warning is emitted only when contains_i scans the INPUT expression for literal sqrt(-1)/(-1)^(1/2)/I (numeric_eval.rs contains_i, gated at simplify_action.rs around line 5454). For sqrt(-4)+sqrt(-9) the input has no literal i, but the RESULT folds to 5*(-1)^(1/2) = 5i; the scan misses it, so value_domain is reported as 'real' with an EMPTY warnings array for an imaginary quantity. The structurally identical sqrt(-1)+sqrt(-1) (literal i in input) correctly warns. Keeping the expression symbolic is fine; the unsound part is the missing honesty caveat plus value_domain=real on an imaginary value.


| probe | expected |
|---|---|

| `sqrt(-4)+sqrt(-9)` | kept symbolic WITH Imaginary Usage Warning and value_domain != real (it is 5i) |

| `sqrt(-16)` | symbolic 4i WITH warning |

| `sqrt(-25)` | symbolic 5i WITH warning |

| `(-4)^(1/2)` | symbolic 2i WITH warning |


**Suggested fix.** Move the imaginary-content check from the INPUT to the canonicalized RESULT: scan the final simplified expression for (-1)^(1/2) / I / any non-real factor (reuse contains_i but run it on `resolved`/result, not the input). If present, emit the Imaginary Usage Warning and set value_domain accordingly. This makes the warning fire uniformly across the perfect-square-radical family that folds to k*i.


### I. Rational-exponent power-tower simplifier multiplies exponents and drops the absolute value (sign-wrong for x<0)

**Severity: high · honesty-violation · 5 probes. — FIXED (commit `3720fa40a`).** Both collapse paths in `cas_math::root_power_canonical_support` now carry the absolute value when an even-numerator inner exponent forces the base nonnegative, generalizing the existing even-INTEGER capability to even-NUMERATOR rationals (`x^(a/b) = (x^a)^(1/b) >= 0` when `a` is even; the reduced denominator is then necessarily odd, so `x^y` is real and nonnegative for every real `x`). (1) `try_rewrite_power_power_even_root_abs_expr` (the multiply-exponents path, fires for non-reciprocal towers like `(x^(2/3))^(9/2)`) now gates on `as_rational_const(inner).numer().is_even()` instead of an even integer, computing `P = (a/b)*n` as a rational and emitting `|x|^P` (deferring to the sign-safe plain power only when `P` itself has an even numerator). (2) `try_rewrite_powpow_cancel_reciprocal_expr_with` (the `(x^y)^(1/y)` reciprocal-cancel path, which fired FIRST for the product-1 probes and emitted bare `x` plus a silent `x>0` assumption) now, for an even-numerator `y`, returns `|x|` UNCONDITIONALLY (no assumption, in strict and assume modes alike — it is an identity over all reals), instead of dropping the `x<0` branch. Now `((x^2)^(1/3))^(3/2) → |x|`, `(x^(2/3))^(3/2) → |x|`, `(x^(2/5))^(5/2) → |x|`, `(x^(4/3))^(3/4) → |x|`, `(x^(2/3))^(9/2) → |x|^3` (= `|x|·x^2`); the symbolic-outer case `(x^(2/3))^y → |x|^(2/3·y)` is also covered. Odd-numerator reciprocals correctly stay bare `x` (`(x^3)^(1/3) → x`); a provably-nonnegative base collapses the abs (`((x^2)^(2/3))^(3/2) → x^2`). Validation: a 1764-pair value-preserving sweep (every `(x^(a/b))^(c/d)` for `a,c∈1..6, b,d∈1..5`, instantiated at `x∈{-2,-3,-1/2,2}`) shows ZERO cases where the simplified form's value diverges from the original's at negative `x`; guardrail+pressure huella byte-identical; new unit tests `powpow_cancel_even_numerator_reciprocal_yields_abs_over_all_reals` / `powpow_cancel_odd_numerator_reciprocal_keeps_plain_base` and extended `power_power_even_root_abs_rewrite_fires_only_when_abs_required` with rational-inner cases.


**Root cause.** ((x^p)^q) with rational p,q is simplified by multiplying exponents p*q and returning x^(p*q), discarding the sign information. Under the engine's own real even-root convention, an even-numerator inner exponent makes the base nonnegative (x^(2/3)=(x^2)^(1/3)>=0, x^(4/3)=(x^4)^(1/3)>=0), so the correct result carries an absolute value (|x| or |x|^k), but the simplifier returns bare x / x^3. The attached guard (e.g. x^(2/3) >= 0) is VACUOUS (true for all real x) so it excludes nothing. The engine's own concrete evaluator contradicts the symbolic answer (((-2)^2)^(1/3))^(3/2) -> 2 while symbolic 'x' -> -2; ((-8)^(2/3))^(9/2) -> 512 while x^3 -> -512). The sibling (x^2)^(1/2) correctly yields |x|, so the abs capability exists but is dropped here. Strict --domain mode is sound (keeps symbolic); the default generic/assume paths are unsound.


| probe | expected |
|---|---|

| `((x^2)^(1/3))^(3/2)` | |x| |

| `(x^(2/3))^(3/2)` | |x| |

| `(x^(2/5))^(5/2)` | |x| |

| `(x^(4/3))^(3/4)` | |x| |

| `(x^(2/3))^(9/2)` | |x|^3 |


**Suggested fix.** When collapsing (x^p)^q over the reals, detect when an intermediate even-index root forces the base nonnegative (even numerator in a rational exponent, or an even root applied to x). In that case the result must be |x|^(p*q) (= |x| when p*q=1), not x^(p*q). Reuse the exact rule already producing |x| for (x^2)^(1/2). Only drop the abs when x is provably nonnegative (a real, non-vacuous guard), and treat the emitted vacuous guard (which holds for all reals) as licensing nothing. Match the engine's own concrete-evaluation result, which already yields |x|.

**Residual (honest, out of scope of this fix).** The fix gates strictly on an EVEN-numerator inner exponent (the only case that forces the base nonnegative and so warrants |x|). Odd-numerator / even-denominator towers such as `(x^(3/5))^(5/2) -> x^(3/2)` and `(x^(1/3))^(3/2) -> x^(1/2)` are left as-is: those are domain-restricted (the original is already imaginary/undefined for x<0, matching the collapsed form, so they are value-consistent where both are defined), not a dropped-abs soundness hole. The fix neither closes nor worsens them. Adversarial audit (commit `3720fa40a`): 3/3 soundness lenses sound, 0 confirmed counterexamples across ~11.3k self-consistency checks + sympy ground truth.


### J. Solve |f(x)| = -f(x) (and |f|=f-mirror gaps) leaks the 'Solve: solve(...) = 0' template, collapsing an interval answer to a phantom '= 0'

**Severity: high · honesty-violation · 3 probes. — FIXED (commit `705f0c8bb50a695553f0a040498e9938b35458dd`).** Implemented per the scoped plan, but NON-recursively (no callback plumbing): the shared chokepoint `resolve_isolated_variable_outcome` (`crates/cas_solver_core/src/solve_outcome.rs`) — which receives every `x = f(x)` reorientation — now, for `op == Eq`, recognizes `x = α·|arg| + β` and recovers the identity `|f| = ±f` exactly. `|f| = f` holds on `{f ≥ 0}`, `|f| = -f` on `{f ≤ 0}` (a CLOSED half-line, boundary included); for a *linear* `arg = a·x+b` it isolates `x` (flipping the relation when `a < 0`) and emits the interval via the existing `isolated_var_solution`. Now `solve(abs(x-1)=1-x,x) → (-∞, 1]`, `solve(abs(2x)=-2x,x) → (-∞, 0]`, `solve(x+abs(x)=0,x) → (-∞, 0]`, and the orientation leak `solve(x=abs(x),x) → [0, ∞)`. Exact (`expr_domain::exprs_equivalent` over multipoly, no f64); a non-linear `arg` falls through to the honest residual. Numeric-RHS abs equations (`abs(2x)=6 → {3,-3}`, `abs(2x)<8 → (-4,4)`) and the `|f|=f` Conditional path are unchanged; guardrail+pressure byte-identical; new unit test `resolve_isolated_variable_outcome_recovers_abs_self_equation`.


**Root cause.** The absolute-value solve branch handles |f|=f (returns a conditional AllReals) but has no case for |f| = -f(x); on that input it falls through to the same unfinished-residual template as Cluster A, emitting 'Solve: solve(x = -|...|, x) = 0' with ok=true and no warning. The trailing '= 0' is a constant display-template artifact (it appears verbatim for unsolved transcendentals where x=0 is NOT a root), so a reader takes the answer to be x=0 while the true solution set is the half-interval (-inf, c]. The engine handles the mirror case |f|=f correctly, isolating |f|=-f as the unhandled hole.


| probe | expected |
|---|---|

| `solve(abs(x-1)=1-x,x)` | (-inf, 1] |

| `solve(abs(2*x)=-2*x,x)` | (-inf, 0] |

| `solve(x+abs(x)=0,x)` | (-inf, 0] |


**Suggested fix.** Add the |f(x)| = -f(x) case to the abs-equation solver: it holds exactly when f(x) <= 0, so return the solution set { x : f(x) <= 0 } (an interval/region), mirroring the existing |f|=f -> { f(x) >= 0 } handling. More generally, never emit the 'Solve: solve(...) = 0' template as an ok=true result: if a branch cannot finish, return an honest ok=false / cannot-solve, or a clearly-marked symbolic residual -- not a nested unevaluated solve(...) with a phantom '= 0'.


### L. Infinite sum/product applies finite closed-form identities at an infinite upper bound, leaking 'infinity' tokens into the expression tree for divergent series

**Severity: high · honesty-violation · 6 probes. — FIXED (commit `d6aadcdeb5d1d772908a4b1e2b3e0febaf364dd3`).** `try_plan_finite_sum_evaluation` / `try_plan_finite_product_evaluation` (`crates/cas_math/src/summation_support.rs`) now, after the telescoping pass and BEFORE the substituting closed-form builders, guard on an infinite upper bound (`is_positive_infinity`) and route to a divergence classifier instead of substituting `infinity` into a finite formula: `classify_infinite_sum` returns `Constant::Infinity` for a polynomial summand (degree ≥ 1, sign of the leading coefficient), a non-zero constant summand, or a geometric `r^k` with rational `r > 1`; `classify_infinite_product` returns `Infinity` for `∏ k` (start ≥ 1) or a constant factor `c > 1` (and `0`/`1` for `0<c<1` / `c=1`); both return `None` (leave UNEVALUATED) for convergent or unclassifiable cases. Crucially the result is a genuine `Constant::Infinity` (a new `DivergentInfinite` plan kind), not a folded `infinity!`/`pow(c,infinity)` atom, so the extended-real arithmetic resolves `0·sum(...)` and `sum(...)−sum(...)` to `undefined`. Now `sum(k,k,1,∞)`, `sum(k^3,…)`, `sum(k^2,…)`, `sum(2k+1,…)`, `sum(2^k,…)`, `sum(3^k,…)`, `product(k,k,1,∞)`, `product(2,…)` → `infinity`; `0*product(k,k,1,∞)` and `sum(k,k,1,∞)−sum(k,k,1,∞)` → `undefined`; convergent siblings (`sum((1/2)^k,…)`, `sum(1/k^2,…)`) stay unevaluated and telescoping (`sum(1/(k(k+1)),1,∞)→1`) and all finite sums/products are unchanged. Guardrail+pressure byte-identical; new unit test `infinite_bound_classifies_divergence_instead_of_finite_formula`. **Verified across 16+ probes:** convergent series stay unevaluated (`sum(1/k^2)`, `sum(k/2^k)`), oscillating/alternating are not claimed (`sum((-1)^k)`, `sum((-2)^k)` → unevaluated), signs are correct (`sum(-k^3) → -infinity`, `sum(5-k) → -infinity`), finite sums/products unchanged. **Residuals (sound under-answers — never a false claim):** slow-divergent series the classifier does not recognise stay unevaluated (`sum(1/k)` harmonic; `product(k^2,..)` — only `∏ k` and constant factors are classified). **Separate pre-existing gap exposed (not worsened):** `infinity/infinity → 1` and `infinity*infinity → infinity^2` (the `X/X→1` / `X*X→X^2` simplifier rules do not guard `X = infinity`, unlike `inf−inf` and `0*inf` which correctly give `undefined`); since divergent products now fold to a genuine `Constant::Infinity`, `product(k,..)/product(k,..)` surfaces this as `1` (it was `infinity!/infinity! → 1` before — same result, not a regression). Fixing the extended-real `/` and `*` rules is a distinct follow-up.


**Root cause.** The Faulhaber (sum of k^p), geometric (sum of r^k), and factorial/product closed-form builders (summation_support.rs: try_build_geometric_power_sum ~line 447, try_build_product_of_first_integers ~line 608, try_build_product_of_powers ~line 635) substitute the upper bound n into the finite closed form WITHOUT checking that n is finite. With n=infinity they emit 1/2*infinity^2, 2^infinity-1, infinity!, 2^infinity, etc. -- divergent series presented as ok=true closed forms with no divergence warning. Worse, infinity! and 2^infinity are folded as finite atoms, so 0*product(k,k,1,inf) -> 0 and sum(k,k,1,inf)-sum(k,k,1,inf) -> 0 (true values are indeterminate 0*inf / inf-inf). The engine HAS an honest path -- it leaves sum(1/k^2,...), sum((1/2)^k,...), k^4/k^5/k^6 Faulhaber, and product(1/k^2,...) unevaluated -- so the closed-form path is the regression.


| probe | expected |
|---|---|

| `sum(k, k, 1, inf)` | divergent (infinity) or unevaluated residual |

| `sum(k^3, k, 1, inf)` | divergent (infinity) or unevaluated residual |

| `sum(2^k, k, 0, inf)` | divergent (infinity) or unevaluated residual |

| `sum(3^k, k, 1, inf)` | divergent (infinity) or unevaluated residual |

| `product(k, k, 1, inf)` | divergent (infinity) or unevaluated residual |

| `product(2, k, 1, inf)` | divergent (infinity) or unevaluated residual |


**Suggested fix.** Guard every finite closed-form builder (Faulhaber, geometric, factorial/product) with a finiteness check on the upper bound: if end is infinity, do NOT substitute it into the finite formula. Instead apply a convergence test -- polynomial sum(k^p) with p>=0 and end=inf diverges; geometric sum(r^k) diverges iff |r|>=1; product(k,...) and product(c,...) with c>1 and infinite bound diverge -- and return the engine's first-class Constant::Infinity (the honest divergence convention used by sum(1,k,1,inf)) or keep the operator unevaluated (as the convergent siblings already do). Never produce fact(infinity) or pow(c, infinity) closed-form nodes.


### O. equiv uses a single shared numeric probe (all vars = 1.23456789) plus an absolute f64 epsilon, certifying non-equivalent expressions as equal

**Severity: high · honesty-violation · 7 probes. — FIXED (commit `d2956f24b0bd47f7ceaa5dd547aab001cdc5e704`).** The user-facing `equiv` path (`crates/cas_engine/src/eval/actions.rs` `eval_equiv`) now disables numeric confirmation (`allow_numerical_verification = false`) so a numeric probe can never CONFIRM equivalence, and recovers genuine identities that the bare simplifier leaves in a non-cancelling form by checking whether the FULL evaluator reduces `a − b` to exactly `Number(0)` (`equiv_difference_evaluates_to_zero`, an exact symbolic zero — not numeric). Result: all 7 probes (`equiv(x,y)`, `equiv(x*y,x^2)`, `equiv(exp(x),exp(y))`, `equiv(sin(x),sin(1.23456789))`, `equiv(x/1e12,0)`, `equiv(x^2,1.23456789*x)`, and the scaled quadratic) now return `false`; genuine equivalences stay `true` (`sin²+cos²=1`, `(x+1)²=x²+2x+1`, `x/x=1`, and the `d/dx √(sec x)` / `√(csc x)` derivative-identity family, which previously relied on the unsound numeric confirmation and is now recovered via the exact eval-level zero). Internal `are_equivalent` callers (solve verification/substitution) are untouched. Guardrail+pressure byte-identical; new wire test `test_eval_json_equiv_rejects_numeric_only_false_equivalences`. Directly enforces the project rule that soundness gates must be exact.


**Root cause.** The CLI equiv path (actions.rs:445 eval_equiv -> Simplifier::are_equivalent, the BOOLEAN variant) falls back to a numeric check (engine/equivalence.rs:72-83) that builds default_equiv_probe_map (cas_solver_core/equivalence.rs:35-42) assigning the SAME value DEFAULT_EQUIV_NUMERIC_PROBE=1.23456789 to EVERY free variable, evaluates the residual once, and returns is_numeric_equiv_zero (|v|<1e-9, an ABSOLUTE float epsilon) DIRECTLY as the boolean truth value (allow_numerical_verification defaults true). Two failure modes: (1) all variables collapse to the diagonal x=y, so any residual vanishing on x=y is declared zero (equiv(x,y), equiv(x*y,x^2), equiv(sin(x),sin(y))); (2) any residual that happens to be small/zero at the single point x=1.23456789 (perfect squares tangent there, expressions scaled below 1e-9 like x/1e12) is declared zero. This is exactly the project MEMORY anti-pattern 'never f64 for drop/keep decisions'. The tri-state are_equivalent_extended path correctly returns Unknown, but the CLI does not use it.


| probe | expected |
|---|---|

| `equiv(x, y)` | false |

| `equiv(x*y, x^2)` | false |

| `equiv(exp(x), exp(y))` | false |

| `equiv(sin(x), sin(1.23456789))` | false |

| `equiv(x/1000000000000, 0)` | false |

| `equiv(x^2, 1.23456789*x)` | false |

| `equiv(x^2, 2.46913578*x - 1.5241578750190521)` | false |


**Suggested fix.** Three combined fixes: (1) assign DISTINCT probe values to distinct free variables (e.g. a deterministic sequence of mutually-incommensurate values), never the same value to all, so the diagonal x=y is not the only sample. (2) Sample at MULTIPLE points and require the residual to vanish at all of them; a single sample can never establish a function identity. (3) Replace the absolute f64 epsilon keep/drop gate with an exact symbolic zero test as the authority -- the numeric probe may only REFUTE equivalence (nonzero -> false), never CONFIRM it; if the symbolic simplifier cannot prove the residual is identically zero, route to the tri-state Unknown (use are_equivalent_extended from the CLI path) rather than returning a definitive true. This directly enforces the MEMORY rule that soundness gates must be exact.


### M. Principal-branch inverse-trig folds arcfn(fn(u))->u without a range check, asserting a provably-false range-membership 'assumption'

**Severity: medium · honesty-violation · 2 probes. — FIXED (commit `e810d18b36caeba616751a0eef7a30d4e21a03c6`).** `try_plan_principal_branch_inverse_trig_expr` (`crates/cas_math/src/inverse_trig_composition_support.rs`) now gates each `arcfn(fn(u)) → u` fold behind a new `principal_fold_is_sound(ctx, u, kind)` check that only folds when `u` is PROVABLY in the inverse function's principal range: for a rational multiple `k·π` (read by a local `signed_pi_multiple`) the membership is EXACT (`arcsin·sin`: `|k| ≤ 1/2`; `arccos·cos`: `0 ≤ k ≤ 1`; `arctan·tan`: `|k| < 1/2`); for a bare rational `v` it is decided against tight rational bounds on `π/2` / `π` (PROVABLY-in only); a free symbol keeps the assumption-gated fold (the membership is genuinely unknown, an honest conditional). An out-of-range or undecidable literal NO LONGER folds (it stays symbolic) — never asserting a provably-false membership. Now `asin(sin(3)) --inv-trig principal → arcsin(sin(3))` and `acos(cos(10)) → arccos(cos(10))` (not the wrong bare `3` / `10`), and the previously-unlisted `atan(tan(3π/4)) → arctan(tan(3π/4))` (not `3π/4`); in-range folds are preserved (`asin(sin(1))→1`, `acos(cos(3))→3` since `3<π`, `asin(sin(π/4))→π/4`, `atan(tan(π/6))→π/6`, `asin(sin(x))→x`), and default strict mode is unchanged. **A 93-case sweep (bare rationals `-12..12` + `k·π` multiples × the three folds) found ZERO unsound (out-of-range) folds.** Guardrail+pressure structurally byte-identical; new unit test `principal_branch_fold_gated_by_provable_range_membership`; the `--branch principal` CLI contract test was retargeted from the (out-of-range, unsound) `arctan(tan(2))=2` to the in-range `arctan(tan(1))=1`. **Residual:** an out-of-range argument stays SYMBOLIC rather than being range-reduced to its true principal value (`asin(sin(3))` could fold to `π-3`); the reduction for bare (non-`π`-multiple) literals needs exact `π` range-reduction — a separate, narrow step.


**Root cause.** try_plan_principal_branch_inverse_trig_expr (inverse_trig_composition_support.rs ~lines 1164-1176) matches arcsin(sin(u))/acos(cos(u)), extracts u, and unconditionally returns u with a templated assumption string 'u in <fn> principal range' -- it never tests whether u lies in the principal range. For a numeric LITERAL u the membership is fully decidable and PROVABLY FALSE (3 is outside [-pi/2,pi/2]; 10 is outside [0,pi]), yet the engine returns the bare argument (a concrete wrong real number, and for acos one outside the codomain [0,pi]) and asserts the false range-membership as a fact. Gated behind opt-in --inv-trig principal; default strict mode correctly keeps it symbolic.


| probe | expected |
|---|---|

| `asin(sin(3)) --inv-trig principal` |  |

| `acos(cos(10)) --inv-trig principal` |  |


**Suggested fix.** In the principal-branch inverse-trig planner, when the argument u is a decidable numeric literal, evaluate whether u is actually in the principal range. If it is, fold to u; if not, compute the correct reduced value (arcsin(sin(u)) = the unique value in [-pi/2,pi/2] congruent under sin; acos(cos(u)) = the unique value in [0,pi]) via range reduction, or keep symbolic. Only emit the 'u in principal range' assumption when u is a free symbol whose membership is genuinely unknown -- never assert it for a literal where it is provably false. Optionally clamp acos/asin results to their real codomains as a backstop.


### N. choose/perm guard ordering: the 'k > n -> 0' short-circuit fires before the 'k == 0 -> 1' boundary, fabricating 0 for negative first argument

**Severity: medium · wrong-value · 5 probes. — FIXED (commit `42a8d4551118dc3a2c69aaa77f64f3a767f4ee9d`).** `compute_choose_expr` / `compute_perm_expr` (`crates/cas_math/src/number_theory_support.rs`) reordered: `k < 0 → 0` and `k = 0 → 1` (for ANY `n`) are checked FIRST; the `k > n → 0` short-circuit (and, for `choose`, the `C(n,k)=C(n,n-k)` symmetry) is now restricted to `n ≥ 0`. For `n < 0` the code falls through to the GENERALIZED binomial `C(n,k) = ff(n,k)/k!` (and `perm` to the falling factorial `ff(n,k)`), exactly matching `sympy.binomial` / `sympy.ff` — never a fabricated 0. Now all 5 probes are correct: `choose(-5,0)=1`, `choose(-1,1)=-1`, `choose(-1,2)=1`, `perm(-3,0)=1`, `choose(-1,0)=1`, plus the generalized values `choose(-2,3)=-4`, `choose(-1,3)=-1`, `perm(-3,2)=12`. Positive cases are unchanged (`choose(5,2)=10`, `choose(2,5)=0`, `choose(0,0)=1`, `perm(5,2)=20`, `perm(2,5)=0`) and `k<0` stays `0`. **A 255-cell grid cross-check against sympy** (`n ∈ [-6,12]`, `k ∈ [-2,10]`, plus large/boundary spot checks `choose(1000,998)`, `choose(-100,5)`, `choose(50,50)`) found **ZERO mismatches** for both functions. Guardrail+pressure structurally byte-identical; new unit test `choose_and_perm_handle_negative_n_via_generalized_binomial`.

**Severity: medium · wrong-value · 5 probes.**


**Root cause.** In compute_choose_expr (number_theory_support.rs lines 570-571) and compute_perm_expr (lines 601-602) the guard `if val_k.is_negative() || val_k > val_n { return 0; }` is evaluated BEFORE the `if val_k.is_zero() || val_k == val_n { return 1; }` boundary (line 573/604). For negative n, the comparison k > n (e.g. 0 > -5, 1 > -1, 2 > -1) is true, so the function returns 0 before reaching the k==0 branch. choose(n,0)=1 holds under EVERY convention (combinatorial empty-selection and generalized binomial), so the 0 is wrong with no exception; choose(-1,1)=-1 and choose(-1,2)=1 under the generalized binomial. The engine is self-inconsistent (choose(0,0)=choose(5,0)=1) and these all return ok=true with no warning, indistinguishable from a genuine choose(2,5)=0.


| probe | expected |
|---|---|

| `choose(-5,0)` | 1 |

| `choose(-1,1)` | -1 |

| `choose(-1,2)` | 1 |

| `perm(-3,0)` | 1 |

| `choose(-1,0)` | 1 |


**Suggested fix.** Reorder the guards so the k==0 (and k==n) boundary returning 1 is checked BEFORE the k>n -> 0 short-circuit, and restrict the 'k > n -> 0' rule to the case n >= 0 only. For negative n, either implement the generalized binomial via the falling factorial C(n,k)=ff(n,k)/k! (matching sympy.binomial) or, if the engine intends combinatorial-only semantics, return an honest error/symbolic residual for negative n -- never a fabricated 0. Apply the same fix to perm. Add tests covering choose/perm with negative n and k=0.


## Recommended fix priority (best ROI first)

1. O. equiv single-shared-probe + f64 epsilon (7 bugs, one shared root cause in cas_solver_core/equivalence.rs + engine/equivalence.rs; equiv is a core educational soundness primitive that silently certifies arbitrary inequalities as true; fix is localized: distinct multi-point probes + exact symbolic authority + route CLI to the existing tri-state are_equivalent_extended)

2. A+J. Solve finishing-path 'Solve: solve(...) = 0' token leak (8 bugs: log-product/log-quotient + |f|=-f; one display/finishing mechanism; recursing into the reduced equation and refusing to serialize an unfinished nested solve as ok=true removes leaked garbage AND recovers the correct sets; the engine already solves the reductions in isolation)

3. K. Endpoint-pole definite-integral zero-scan short-circuit (4 bugs incl. one CRITICAL sign-wrong interior-pole case; single fix in trig_nonzero_on_interval to enumerate ALL poles and route multi-pole/interior-pole to the existing Undefined certificate; high correctness stakes for a calculus engine)

4. L. Infinite sum/product closed-form-at-infinity infinity-token leak (6 bugs; add a finite-upper-bound guard + convergence test in summation_support.rs builders; honest divergence path already exists for sibling cases)

5. D. Solve div-by-provable-zero -> AllReals (4 bugs; reuse the evaluator's zero-prover in the solve clear-denominator branch to collapse to Empty; completes the Rounds 1-3 div-by-zero fix on the solve path)

6. E+F. Rational and domain inequality solver (8 bugs; implement the standard critical-point sign-chart for rational inequalities and intersect sqrt/ln/log inequality results with the argument-domain; also subsumes the known-open infinity-token product-inequality case)

7. C. Trig solve range-check bypass for surd/transcendental RHS (6 bugs; extend the existing |c|<=1 exact gate from rational RHS to surds/transcendentals; backstop by refusing arcsin/arccos of an argument the engine itself leaves unevaluated)

8. B. Symbolic-RHS even-root solve dropped a>=0 condition (4 bugs; emit the parameter sign condition on the even-radical inversion path, reusing the x^2=a machinery)

9. G. Additive like-term cancellation drops undefined subterms (4 bugs; run the undefined/pole/non-real screen before the additive collector discards zero-coefficient atoms)

10. I. Rational-exponent power-tower abs-drop (5 bugs; preserve |x| when an even-index root forces a nonnegative base, reusing the (x^2)^(1/2)->|x| rule)

11. N. choose/perm guard ordering for negative n (5 bugs; reorder the k==0 boundary before the k>n short-circuit and restrict k>n->0 to n>=0; localized 2-line ordering fix in number_theory_support.rs)

12. H. Imaginary-usage warning gated on input not result (1 bug; scan the result for (-1)^(1/2) instead of the input)

13. M. Principal inverse-trig false range-membership assertion (2 bugs; range-check/range-reduce for literal arguments; lowest priority since gated behind opt-in --inv-trig principal and not the default)


## Completeness critic

Gaps / under-covered (mostly honest capability gaps, not unsoundness):

- Matrix operations beyond det: inverse([[...]]) returns an honest 'función no definida' error (capability gap, sound); could not exercise matrix algebra soundness because inverse/eigenvalue/rank verbs are not implemented.

- System solving: solve([eq1,eq2],[x,y]), solve(eq1 and eq2,...), and solve(eq1,eq2) all return honest Parse errors (capability gap). The 2x2 linear-system soundness axis remains unprobed because the syntax is unsupported.

- Complex value-domain mode: --complex on still returns sqrt(-4) as 2·(-1)^(1/2) with value_domain=real rather than 2i; this is an under-evaluation (conservative residual), not unsound, but the complex-mode evaluation path is effectively untested for actual i-folding.

- floor/ceil do not const-fold even with --const-fold safe (floor(-0.5) stays floor(-1/2)); conservative non-fold, sound, but means floor/ceil numeric-correctness on concrete inputs is hard to stress.

- Very large numbers (factorial(100), 2^1000, exp(100000)) compute exactly/symbolically with no overflow or wrong value observed; no defect found, axis appears robust.

- Big-number symbolic comparison (2^1000 > 3^600) could not be evaluated as a standalone predicate (the engine treated it as containing variable x); boolean-comparison soundness for large literals remains unprobed via this surface.


New witness found by the critic:

- **[medium]** `integrate(cos(x)/sin(x)^2, x, 0, 3*pi/2)  [also 0..5pi/2 -> infinity, and pi..5pi/2 -> -infinity]` → engine `ok=true, result="infinity", required_display=["sin(x) ≠ 0"], exit code 0. Variant [0,5pi/2] -> "infinity"; variant [pi,5pi/2] -> "-infinity". All presented as definite signed-infinity definite results.`; correct: undefined / divergent (indeterminate, sympy returns nan for all three). The integrand cos(x)/sin(x)^2 has antiderivative -1/sin(x). On [0,3pi/2] the singular points are the LOWER ENDPOINT x=0 and an INTERIOR pole at x=pi (3pi/2 is regular, sin=-1). The piece near x=0 diverges to +inf (integral_0^(pi/2) = -1 - (-inf) = +inf) while the piece around the interior pole x=pi diverges to -inf (integral_(pi/2)^pi = -inf), giving an indeterminate +inf + (-inf) => divergent/undefined. Numeric shrink-cut around pi (eps=1e-3,1e-4) confirms the left and right tails blow up to -inf while the near-0 tail is +inf. So the correct verdict is undefined; the reported signed +infinity (and -infinity for [pi,5pi/2]) is both wrong-as-a-value and wrong-in-sign.
  Genuine honesty-violation: the engine asserts a definite signed infinity (ok=true, no warning beyond the vacuous sin(x)≠0 note) for an improper integral that is actually a divergent ∞−∞ indeterminate (sympy=nan). This is the SAME root-cause family as already-logged findings #41-44 (the trig_nonzero_on_interval certifier short-circuits on the first endpoint zero), but these are NEW witnesses on genuinely different interval geometries: the logged cases all had poles at BOTH ENDPOINTS, whereas here the lower endpoint x=0 is a pole AND there is an INTERIOR pole at x=pi that is never scanned because the endpoint-zero short-circuit fires first. The decisive control proving this is a live distinct manifestation: integrate(cos/sin^2, pi/2, 3pi/2) (same interior pole at pi but REGULAR endpoints) correctly returns 'undefined', while integrate(cos/sin^2, 0, 3pi/2) (add an endpoint pole at 0) wrongly returns 'infinity' -- the only difference is the endpoint zero, which suppresses interior-pole detection. Marked medium (not high) because it shares root cause with already-logged high-severity items and is therefore an extension witness rather than a new defect class; it is not in the KNOWN-OPEN list and is not a sound residual or capability gap (the engine commits to a definite wrong signed value rather than 'undefined' or an honest error).

## A+J implementation plan (scoped 2026-06-18, read-only workflow)

A 4-agent read-only scoping workflow mapped Clusters A and J precisely. Captured here so the fix is resumable.

**Single shared leak site.** Both clusters reach the same fall-through: `solve_outcome.rs:193`, the `ContainsTargetVariable` arm of `solve_isolated_variable_lhs_with_resolver_with_state`. When isolation reorients to a bare variable (`x = f(x)`), `try_linear_collect` / `try_linear_collect_v2` both fail (nonlinear/abs RHS) and it falls to `residual_solution` → `mk_residual_solve` (`isolation_utils.rs:7`) → `SolutionSet::Residual(solve(x = f(x), x))`, rendered `Solve: … = 0` at `render.rs:47` with `ok=true`. Confirmed leaks: A → `solve(x = 8/(x+2), x)`, `solve(x = (x+1)/(x-1), x)`; J → `solve(x = -|x|, x)`, `solve(x = 1-|x-1|, x)` (and `solve(x = |x|, x)` — orientation alone triggers it).

**Plumbing prerequisite (shared, main cost).** The isolated-variable entry does NOT receive a recursive-solve callback. The general callbacks (`solve_split_case_with_var` / `solve_equation_with_var`) live in `solve_runtime_flow_isolation_default_routes.rs:64/87`; the dispatch (`isolation_dispatch.rs:280-380`) only has shape-specific `on_div`/`on_mul`/…. So a new `try_recover` closure must be threaded: `default_routes` → `execute_isolation_dispatch_…_for_var_with_state` → `execute_isolated_variable_entry_…with_state` (`isolation_dispatch.rs:171/226`) → `solve_isolated_variable_lhs_with_resolver_with_state` (`solve_outcome.rs:156`), running before line 193. The non-state twin (`solve_outcome.rs:103`) must stay consistent.

**Cluster J (do first — self-contained recognizer).** Orientation-independent: for `(side_abs, side_other)` in `[(rhs, var), (var, rhs)]`, if `extract_abs_argument_view(side_abs) = Some(arg)`: same-sign `|arg|=arg` (`exprs_equivalent(arg, side_other)`) → recurse `solve(arg >= 0)`; opposite-sign `|arg|=-arg` (`exprs_equivalent_up_to_sign && !exprs_equivalent`) → recurse `solve(arg <= 0)` (the missing case). Gate strictly `op == Eq`. Exact (`expr_domain` over multipoly, no f64). `{f<=0}` is a CLOSED interval (boundary included). Building blocks all exist (`cas_math::expr_extract::extract_abs_argument_view`, `cas_math::expr_domain::exprs_equivalent{,_up_to_sign}`).

**Cluster A (second).** Mirror the proven Div-route fix `1a0806509`. Gate `op == Eq`, single var-denominator. `combine_fractions_deterministic(lhs - rhs)` → `(num, den)`, **then `expand_algebraic(num)`** (critical: the quadratic extractor needs `x^2+2x-8`, not `Sub(Mul, Number)`), build `expanded_num = 0` (no top-level Div → cannot re-enter Div route, Quadratic precedes Isolation), re-solve via the threaded callback. Denominator-zero exclusion (`x+2≠0`) and log-origin domain (`x>0`) are applied for free by the existing `guard_solved_result_with_exclusions` / preflight `required_conditions`, so `{-4,2}∩{x>0} → {2}`.

**Guardrail: none.** A presentation-layer "reject nested-solve residuals" net is UNSOUND — genuine transcendentals are structurally identical (`solve(x+e^x=0,x)` → `solve(x = -(e^x), x)`), and the wildcard residual contract tests assert the residual contains `solve`. The fix-at-source IS the only sound guardrail.

**Huella.** No fixture pins the leaked `Solve: solve(…)=0` text (grep = 0 matches). Expected delta: IMPROVE (residual → real answer), never regress. Must stay green: `solver_wildcard_residual_contract_tests`, `solver_isolation_scope_contract_tests`, the michaelis pedagogical tests (untouched — Div-on-LHS, var-free RHS, never reaches this entry), and the abs-split unit tests. Regenerate scorecards if any A/J input is in the corpus.

**Status: SCOPED, not yet implemented** — the shared callback plumbing is a multi-file generic-closure change and is the bulk of the work.
