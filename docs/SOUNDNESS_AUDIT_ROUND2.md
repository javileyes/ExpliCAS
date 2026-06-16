# Soundness Audit — Round 2 (2026-06-15)

Second multi-axis adversarial soundness audit, run via a multi-agent workflow
(ultracode), after the Round-1 fixes (Clusters C/B/D/A) landed. Baseline commit:
`f5bdce689`.

- **19 fronts** hunted in parallel; every candidate independently re-verified by
  a skeptic with a refutation lens (default-reject, real-domain numeric truth,
  real-root convention, no complex-branch ground truth).
- **306 probes**, **87 agents**.
- **52 NEW confirmed defects** + **12 rediscovered known-deferred** (B-2 / A-2 /
  `(x^a)^b`-of-negatives — expected, not new).
- Severity of the 52 new: **5 sign-wrong, 20 wrong-value, 23 honesty-violation,
  4 dropped-condition.**

## Regression check — Round-1 fixes held

The audit re-covered the Round-1 territory (abs/sign, cancellation, powers/roots,
trig, inverse-trig, differentiation). **None of the C/B/D/A fixed cells were
re-flagged.** Confirmed still-correct in passing: `diff(arcsin−arccos)`,
`(a^2)^y=|a|^(2y)`, `arctan(x)+arctan(1/x)=(π/2)sign(x)`, `sqrt(-2)*sqrt(-3)`
symbolic. The 12 rediscovered defects are all in the explicitly-deferred B-2/A-2
families.

## The systemic theme

Most of the 52 defects share **one root cause**: the engine applies an algebraic
identity, cancellation, or function-inverse simplification **without checking its
operands are defined and finite over ℝ**. `sin(arcsin(2))→2`, `inf−inf→0`,
`(1²−1)/(1−1)→0`, `acosh(cosh(x))→x`, `solve(3/x=0)→{∞}` are all instances of
"simplify first, never ask whether the input has a real value." Round 1 found the
*sign* failures of this theme; Round 2 finds the *definedness/finiteness* failures.

## Confirmed new defects — 6 root-cause clusters

### R1 — Inverse-function composition collapses past the inverse's domain (HONESTY, ~14) — FIXED (commit `261f1de28`)
`f(f⁻¹(x)) → x` rewrites fired without gating on the inverse's domain, fabricating
a real value for an undefined input:
- `sin(arcsin(2)) → 2`, `cos(arccos(5)) → 5`, `tan(arcsin(2)) → 2/√(-3)`,
  `cos(arcsin(2)) → √(-3)`, `sin(arccos(2)) → √(-3)` (need `|x|≤1`).
- `tanh(atanh(2)) → 2` (need `|x|<1`).
- `sec(asec(0.5)) → 1/2`, `csc(acsc(0.5)) → 1/2` (need `|x|≥1`).
- `cosh(acosh(0)) → 0`, `cosh(acosh(-3)) → -3` (need `x≥1`).
**Fix (commit `261f1de28`):** the defect spanned **four** rule families — the
composition planner + n-angle recurrence (`inverse_trig_composition_support.rs`,
`inv_trig_n_angle_support.rs`), the hyperbolic compositions
(`hyperbolic_core_support.rs`), the trig expansion forms
(`trig_inverse_expansion_support.rs`: `tan/cos(arcsin)`, `sin(arccos)`, …), and the
reciprocal-trig forms (`trig_reciprocal_eval_support.rs`: `csc/sec(arccsc/arcsec)`).
Each now declines when the inner inverse's argument is a literal provably outside
its domain (`arcsin/arccos`: |x|≤1; `atanh`: |x|<1; `acosh`: x≥1; `arcsec/arccsc`:
|x|≥1; `arctan/arccot`/`asinh`: all of ℝ, never gated). The adversarial sweep found
the 3rd and 4th families after the first two were fixed; a re-run (133 probes) is
clean — every out-of-domain literal stays symbolic, every in-domain case (incl.
boundary `±1`, `n=2` multiples, and all `arctan` forms) still simplifies, no
over-firing. Guardrail+pressure fingerprints byte-identical.

### R2 — `acosh(cosh(x)) → x` should be `|x|` (SIGN-WRONG, ~5) — FIXED (commit `d22eec10e`)
`acosh` has range `[0,∞)`, so `acosh(cosh(x)) = |x|`, not `x`:
- `acosh(cosh(x)) → x`, `acosh(cosh(2*x)) → 2*x` (true `2|x|`),
  `acosh(cosh(-x)) → x`.
- `acosh(cosh(x)) - x → 0` (true `|x|−x`, nonzero for `x<0`).
- `diff(acosh(cosh(x))) → 1` (true `sign(x)`).
- The attached condition `cosh(x) ≥ 1` is **vacuous** (always true) — it does not
  encode the real restriction.
**Fix (commit `d22eec10e`):** `try_rewrite_hyperbolic_composition` now emits
`Abs(x)` for the `acosh∘cosh` arm only (the other five compositions are genuine
identities and stay `x`). `diff(acosh(cosh(x))) → sign(x)` follows automatically
through the Round-1 `diff(|x|)=sign(x)` work. Verified: `acosh(cosh(x))→|x|`,
`acosh(cosh(2x))→2|x|`, `acosh(cosh(-x))→|x|`, `acosh(cosh(-5))→5`; the genuine
identities (`asinh(sinh)`, `tanh(atanh)`, `sinh(asinh)`, `cosh(acosh)`) unchanged.
Adversarial 2-lens / 29 probes: clean; guardrail+pressure fingerprints
byte-identical.

### R3 — Cancellation of identical UNDEFINED / INFINITE operands `X − X → 0` (HONESTY/WRONG, ~11)
The additive like-term / cancellation machinery (the Cluster-C family) fires even
when an operand has no real finite value:
- `inf − inf → 0` (indeterminate), `2*inf − inf → 0` and `3*inf − inf → 0`
  (true `+inf`, **wrong value** not just honesty), `undefined − undefined → 0`.
- `(0/0) − (0/0) → 0`, `(1/0) − (1/0) → 0`, `tan(π/2) − tan(π/2) → 0`.
- `factorial(-2)*0 → 0` (`∞·0` indeterminate), `0^0 − 1 → 0`, `0^0 − 0^0 → 0`.
- `sum(k, k, 1, ∞) − sum(k, k, 1, ∞) → 0` (both divergent).
**FIXED (commit `PENDING_HASH`) for literal non-finite/undefined operands.**
The "this additive combination is zero / these terms cancel" conclusion is reached
by a LARGE family of independent rules and orchestrator shortcuts (`Annihilation`,
`Subtraction Self-Cancel`, `Add Inverse`, `Combine Like Terms`/collect,
`Polynomial Identity`, `Collapse Common-Scale Equivalent Difference`, `Collapse
Exact Zero Additive Subexpression`, …). Gating them one-by-one was whack-a-mole —
the adversarial sweep kept surfacing new producers (function-wrapped `sqrt(inf)`,
multi-pair `1/0-1/0+2/0-2/0`, …). The fix has two layers:
- a shared predicate `cas_math::arithmetic_cancel_support::expr_carries_nonfinite_or_undefined`
  (Infinity/Undefined constant, or division by a provably-zero denominator, anywhere
  in the tree), used to make the cas_math cancellation primitives (annihilation,
  sub-self, add-inverse, collect-like-terms) DECLINE so 2-term forms fold to
  `undefined`;
- a UNIVERSAL post-filter `rewrite_unsoundly_drops_nonfinite` applied at the two
  simplifier chokepoints (`transform_expr_recursive` per node at any depth, and the
  `simplify_pipeline` shortcut dispatcher): no rewrite may turn a non-finite/undefined
  Add/Sub into a result that no longer carries the non-finite. Function/quotient
  *evaluations* (`atan(inf) → π/2`, `1/inf → 0`) operate on non-additive nodes and
  are never blocked.

Now: `inf − inf`, `x/0 − x/0`, `(1/0) − (1/0)`, `undefined − undefined` → `undefined`;
`sqrt(inf) − sqrt(inf)`, `ln(inf) − ln(inf) + 7`, `1/0 − 1/0 + 2/0 − 2/0`,
`sin(undefined) − sin(undefined)` stay symbolic (NOT `0`/finite). Two adversarial
sweeps (≈725 probes, ~50 confirmed leaks in the first, 0 in the second) drove the
universal-filter design; guardrail+pressure fingerprints BYTE-IDENTICAL.

**R3-2 (deferred):** *semantic* indeterminates that look finite syntactically still
fold: `tan(π/2) − tan(π/2) → 0` (the cancellation fires before `tan(π/2)` folds to
`undefined`), `0^0 − 0^0 → 0`, `0^0 − 1 → 0` (the `0^0 = 1` convention applied in an
additive context), `factorial(−2)·0 → 0`, `2·inf − inf → 0` and `sum(k,k,1,∞) −
sum(k,k,1,∞) → 0`. These are *indeterminate-arithmetic / semantic-pole* defects,
distinct from the structural "non-finite term never cancels" fix; they need a pole/
indeterminate oracle (or `2·inf − inf` is a true `+inf`, a wrong-VALUE not honesty).

**R3-3 (deferred — PRE-EXISTING, overlaps R4):** a denominator that is *provably*
but not *literally* zero still cancels: `1/(x−x) − 1/(x−x) → 0`, `1/(0·x) − 1/(0·x)
→ 0`, `1/(x²−x²) − 1/(x²−x²) → 0`, `sin(x)/(x−x) − sin(x)/(x−x) → 0`. The shared
predicate only flags a `Div` whose denominator is a *literal* zero constant
(`as_rational_const(den).is_zero()`); `x−x`, `0·x`, `x²−x²` are zero-VALUED but
symbolic, so they slip the gate, and the `A − A` cancellation fires before the
denominator simplifies to `0`. The engine even emits a self-contradictory
`required_condition "0 ≠ 0"` on these, so it *knows* the denominator is zero. Closing
this needs a provably-zero oracle in the predicate (or eager denominator
simplification before cancellation); it overlaps R4 (provable `0/0`). Verified
identical on HEAD — NOT a regression of this fix.

**Not regressions (verified byte-identical on HEAD):** the adversarial flagged
`5·a·b·c − a·b·c → 5·a·b·c − a·b·c` (collect fails for ≥3-factor products with an
implicit-1 coefficient) and `cos(x) + cos(x) → 2·cos(0)·cos(x)` (a spurious unit
`cos(0)` factor). Both pre-date this fix and involve no non-finite term, so the R3
guards never touch them — a separate pre-existing collect-normalization defect.

### R4 — Numeric `0/0` folds to a finite value (WRONG/HONESTY, 3)
- `(1²−1)/(1−1) → 0`, `(2²−4)/(2−2) → 0`: the `0/denominator → 0` fast path does
  not check `denominator ≠ 0`.
- `(1³−1)/(1−1) → 1+1+1` (=3): a literal-zero factor is cancelled.
The engine **knows** this is undefined — with `--steps on` it emits "Zero Property
of Division: 0/0 → undefined", and bare `0/0` is kept symbolic — but the default
(steps-off) path short-circuits. The audit doc's "no interior pole produced a false
finite value" invariant covered *symbolic* poles; this all-numeric `0/0` slips through.

**INVESTIGATED — deferred to its own cycle (needs simplifier instrumentation).**
Two obvious fix sites were tried and are NOT the default-mode path:
`DivZeroRule` (`arithmetic.rs`) was extended to treat a *provably-zero* (not just
literal-`0`) denominator as `0/0 → undefined`, and `const_fold`'s `Div` arm was
given the same `0/0` guard. Both correctly fix the `--steps on` path, but with
`eprintln` instrumentation **neither fires** in the default path — yet
`(1*0)/(1-1)` and `(1²-1)/(1-1)` still fold to `0`. The trigger is a numerator
containing a `Mul`/`Pow` (`(1*0)/(1-1) → 0` but the structurally-identical
`(0)/(1-1) → 0/(1-1)` stays symbolic): const_fold rebuilds the numerator and the
*rebuilt* `Div` is re-simplified to `0` by a THIRD, unidentified rule that bypasses
`DivZeroRule`. Pinning that rule needs deeper instrumentation. NOT YET FIXED
(changes reverted to keep the tree clean).

### R5 — `solve` returns spurious / non-existent roots (WRONG, 12)
- **R5a — abs equations don't filter extraneous roots — FIXED (commit `4d07aaee6`)
  for RATIONAL roots:** both branch roots were returned with only a *set-level* `≥0`
  guard, not a per-root back-substitution. `solve(|x| = x−1) → {1/2}` (extraneous),
  `solve(|2x+3| = x−5) → {−8, 2/3}`, `solve(|x−2| = 2x+1) → {−3, 1/3}`,
  `solve(|x| = 2x−6) → {6, 2}`. **Fix:** the solve backend now back-substitutes each
  candidate root into the original equation (numeric, real domain) and drops the ones
  that fail; a conditional whose roots are all classified collapses to an
  unconditional set. Now: `solve(|x|=x-1) → No solution`, `solve(|x-2|=2x+1) → {1/3}`,
  `solve(|2x+3|=x-5) → No solution`; valid roots (`{3,-3}`) and irrational roots
  preserved. **CONSERVATIVE:** only RATIONAL, bounded-magnitude roots are checked —
  an adversarial sweep proved that f64 back-substitution of an IRRATIONAL root
  (`500000 − 127·sqrt(15500031)`, the small root of `x²−10⁶x+1`) suffers catastrophic
  cancellation and would wrongly DROP a valid root, so irrational roots are kept.
- **R5a-2 — irrational/transcendental extraneous roots still survive (NEW, ~11,
  surfaced by the R5a sweep):** because R5a only checks rational roots, extraneous
  roots that are irrational slip through: `solve(|x| = 2−e) → {2−e, e−2}` (|x| can't
  be negative → no solution), `solve(|x| = ln(1/2))`, `solve(|x+5| = 3−π)`, and
  log-domain cases `solve(ln(x)+ln(x−3)=1)` (one root violates `x>3`). These need an
  EXACT/symbolic back-substitution (the engine's own equality checker), which f64
  cannot do robustly given the catastrophic-cancellation tradeoff. Own cycle.
- **R5b — `c/poly = 0` returns `{∞}` — FIXED (commit `14a471e1d`):** a nonzero
  constant over a polynomial is never zero → no solution, but the solver isolated
  the denominator (`poly = c/0 = ∞`) and returned `{∞}` (`solve(3/x=0)`) or, for an
  irreducible quadratic with a linear term, a malformed nested
  `solve(x = ∞ − x², x)` (`solve(7/(x²+x+1)=0)`). **Fix:** (1) short-circuit
  `c/poly = 0` (simplified `lhs−rhs` is a fraction with a nonzero-constant
  numerator) to `Empty` *before* the isolation divides by zero; (2) a defensive
  final filter drops any `∞`/undefined entry from the solution set. Both
  manifestations now return "No solution"; genuine roots
  (`solve((x−2)/(x+3)=0) → {2}`) preserved. Adversarial 2-round / 9+ probes:
  the `c/poly=0` class is clean.
- **R5c — out-of-range transcendental (1):** `solve(sin(x)=3) → {arcsin(3)}`
  (**no real solution**). (Rediscovered: `solve(cos(x)=2) → {arccos(2)}`.)
**Fix:** back-substitute candidate roots into the original equation (real-domain
definedness check) before returning; treat `nonzero/poly = 0` as no-solution.

### R5d — Rational-equation isolation fabricates malformed nested solves, DROPPING valid roots (WRONG, ~10 — NEW, surfaced by the R5b adversarial sweep)
A pre-existing, broader sibling of R5b (NOT caused by, nor fixed by, the R5b fix):
for several rational equations the isolation strategy emits an unevaluated,
malformed nested `solve(x = poly ± …, x) = 0` instead of the root set — silently
**dropping genuine finite real roots**:
- `solve(7/(x²+x+1) = 7) → solve(x = −x², x) = 0` (true `{0, −1}`),
  `solve(1/(x²+x+1) = 1)` (true `{0, −1}`) — `c/poly = nonzero`.
- `solve(x + 1/x = 2) → solve(x = (2x−1)^(1/2), x) = 0` (true `{1}`).
- `solve((x²−2x+1)/(x−5) = 0)` (true `{1}`), `solve((x²−4x+4)/(x−9)=0)` (true `{2}`)
  — perfect-square numerator over a non-constant denominator.
- The trigger is the solver reaching a form like `x = ±√(poly)` / `x = c − x²` and
  failing to recurse into the inner solve (the inner solve *alone* works:
  `solve(x = −x², x) → {−1, 0}`). Root cause is in the isolation/reciprocal path.
- **Plus a hard crash:** `solve(1/sin(x)=0)` (and `1/cos`, `1/tan`) →
  `InternalError: función [csc] no definida` — the solver rewrites `1/sin → csc`
  and hits an unimplemented function. Should be "No solution".
This is higher-severity than R5b (it drops *correct* roots / crashes) but needs a
deeper isolation-strategy fix; own cycle. NOT YET FIXED.

### R6 — Dropped domain conditions & misc (COND-DROP/WRONG, ~4)
- `(a*b)^x → a^x·b^x` with **no** `a>0 ∧ b>0` condition (the split is invalid for
  negative `a,b` and real `x`).
- `diff(arccot(x))` — the `arccot(x)=arctan(1/x)` convention's `x≠0` discontinuity
  is not surfaced (same arccot gap noted in Round-1 Cluster D).
- `sum(0, k, 1, ∞) → undefined` — a zero summand sums to **0**, not undefined.

## Priority sequence (by severity × tractability)

1. **R2** — `acosh(cosh(x)) = |x|`. Sign-wrong, bounded, reuses the round-1
   abs/sign machinery. Highest value-per-risk.
2. **R5b** — `solve(c/poly = 0)` → no solution. FIXED (commit `14a471e1d`).
   The sweep surfaced **R5d** (malformed nested solves dropping valid roots +
   `csc` crash) — broader, higher-severity, own cycle.
3. **R4** — numeric `0/0` fold. INVESTIGATED, deferred: the `--steps on` path is
   fixable via `DivZeroRule`, but the default-mode fold is a third, unidentified
   rule (neither `DivZeroRule` nor `const_fold`) — needs simplifier instrumentation.
4. **R5a** — `solve` abs extraneous-root filtering. FIXED (commit `4d07aaee6`)
   for rational roots; irrational extraneous (R5a-2) needs exact verification.
5. **R1** — gate `f(f⁻¹(x)) = x` by the inverse's domain. FIXED (commit `261f1de28`)
   across four rule families.
6. **R3** — block cancellation/like-term folding on non-finite/undefined operands.
   FIXED (commit `PENDING_HASH`): shared predicate + universal post-filter at the two
   simplifier chokepoints. R3-2 (semantic indeterminates / infinity-arithmetic) deferred.
7. **R6** — dropped conditions (`(a*b)^x`, arccot, zero-summand sum). Lower severity.
8. **R5c** — out-of-range transcendental solves (folds into R5/R1 domain work).

## Known-deferred, rediscovered (12 — not new)

All in the explicitly-deferred families, confirming Round-1's scoping:
- **B-2** (symbolic-even-inner even root): `(x^(2k))^(1/2) → x^k`,
  `diff((x^(2k))^(1/2)) → k·x^(k−1)`, `(x^(2k))^(1/(2k)) → x`.
- **A-2 / `(x^a)^b`-of-negatives**: `((-2)^x)^y → (-2)^(x·y)`,
  `((-2)^a)^(1/a) → -2`, `((-2)^(1/6))^2 → -(2^(1/3))`, `(x^a)^(1/a) → x`,
  `integrate((x^a)^(1/a)) → x²/2`.
- **Inverse-trig identity out of domain**: `asin(2)+acos(2) → π/2`,
  `solve(cos(x)=2) → {arccos(2)}`, `diff(arcsec(x)+arccsc(x)) → 0` (these overlap
  R1 and could be folded into the R1 inverse-domain gate).

## Status

- [x] R2 — `acosh(cosh(x)) = |x|` (sign-wrong, bounded) *(FIXED 2026-06-15, commit `d22eec10e`)*
- [x] R5b — `solve(c/poly=0)` no-solution *(FIXED 2026-06-15, commit `14a471e1d`)*
- [ ] R5d — rational-equation isolation fabricates malformed nested solves (drops valid roots) + `csc/sec/cot` solver crash (NEW)
- [ ] R4 — numeric `0/0` fold guard *(investigated; default-mode path is a third unidentified rule — own cycle w/ instrumentation)*
- [x] R5a — `solve` abs extraneous-root filter *(FIXED 2026-06-15, commit `4d07aaee6`, rational roots; irrational extraneous split to R5a-2)*
- [ ] R5a-2 — irrational/transcendental extraneous roots (e.g. `solve(|x|=2-e)`) need exact/symbolic back-substitution
- [x] R1 — inverse-composition domain gate (`f(f⁻¹(x))`) *(FIXED 2026-06-16, commit `261f1de28`, four rule families)*
- [x] R3 — non-finite/undefined operand cancellation guard *(FIXED 2026-06-16, commit `PENDING_HASH`, shared predicate + universal post-filter at the two simplifier chokepoints; literal ∞/undefined/`c÷0` no longer cancel to 0)*
- [ ] R3-2 — *semantic* indeterminates (`tan(π/2)−tan(π/2)`, `0^0−0^0`, `factorial(−2)·0`) and infinity-arithmetic (`2·inf−inf` → true `+inf`) need a pole/indeterminate oracle
- [ ] R3-3 — *provably*-but-not-*literally*-zero denominators (`1/(x−x)`, `1/(0·x)`, `1/(x²−x²)`) cancel; needs a provably-zero oracle in the predicate (PRE-EXISTING, overlaps R4)
- [ ] R6 — dropped conditions (`(a*b)^x`, arccot, zero-summand sum)
