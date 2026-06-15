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

### R1 — Inverse-function composition collapses past the inverse's domain (HONESTY, ~14)
`f(f⁻¹(x)) → x` rewrites fire without gating on the inverse's domain, fabricating
a real value for an undefined input:
- `sin(arcsin(2)) → 2`, `sin(arcsin(3/2)) → 3/2`, `cos(arccos(5)) → 5`,
  `cos(arccos(2)) → 2` (need `|x|≤1`).
- `tanh(atanh(2)) → 2` (need `|x|<1`).
- `sec(asec(0.5)) → 1/2`, `csc(acsc(0.5)) → 1/2` (need `|x|≥1`).
- `cosh(acosh(0)) → 0`, `cosh(acosh(-3)) → -3` (need `x≥1`).
- (Rediscovered same family: `asin(2)+acos(2) → π/2`.)
**Fix:** gate each `f(f⁻¹(x)) = x` rewrite by the inverse's domain condition; for a
provably out-of-domain literal, keep symbolic / mark undefined.

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
**Fix:** cancellation / like-term combination must not fire when an operand is
provably non-finite or undefined (∞, `0/0`, `1/0`, `tan(π/2)`, divergent sum,
`factorial(neg)`). This touches the foundational cancellation path (higher huella)
— scope carefully.

### R4 — Numeric `0/0` folds to a finite value (WRONG/HONESTY, 3)
- `(1²−1)/(1−1) → 0`, `(2²−4)/(2−2) → 0`: the `0/denominator → 0` fast path does
  not check `denominator ≠ 0`.
- `(1³−1)/(1−1) → 1+1+1` (=3): a literal-zero factor is cancelled.
The engine **knows** this is undefined — with `--steps on` it emits "Zero Property
of Division: 0/0 → undefined", and bare `0/0` is kept symbolic — but the default
fast path short-circuits. The audit doc's "no interior pole produced a false finite
value" invariant covered *symbolic* poles; this all-numeric `0/0` slips through.
**Fix:** guard the `0/x → 0` fold and the zero-factor cancellation on `x ≠ 0`.

### R5 — `solve` returns spurious / non-existent roots (WRONG, 12)
- **R5a — abs equations don't filter extraneous roots (6):** both branch roots are
  returned with only a *set-level* `≥0` condition, not a per-root back-substitution.
  `solve(|x| = x−1) → {1/2}` (extraneous: `|1/2| = 1/2 ≠ −1/2`; **no solution**),
  `solve(|2x+3| = x−5) → {−8, 2/3}` (**no solution**),
  `solve(|x−2| = 2x+1) → {−3, 1/3}` (true `{1/3}`),
  `solve(|x| = 2x−6) → {6, 2}` (true `{6}`), and similar.
- **R5b — `c/poly = 0` returns `{∞}` (5):** a nonzero constant over a polynomial is
  never zero → **no solution**, but the engine returns infinity:
  `solve(3/x=0) → {∞}`, `solve(1/(x−3)=0) → {∞}`, `solve(2/(x+1)=0) → {∞}`,
  `solve((x+1)/(x²−1)=0) → {∞}`, `solve(1/(x²+1)=0) → {∞^(1/2), −∞^(1/2)}`.
- **R5c — out-of-range transcendental (1):** `solve(sin(x)=3) → {arcsin(3)}`
  (**no real solution**). (Rediscovered: `solve(cos(x)=2) → {arccos(2)}`.)
**Fix:** back-substitute candidate roots into the original equation (real-domain
definedness check) before returning; treat `nonzero/poly = 0` as no-solution.

### R6 — Dropped domain conditions & misc (COND-DROP/WRONG, ~4)
- `(a*b)^x → a^x·b^x` with **no** `a>0 ∧ b>0` condition (the split is invalid for
  negative `a,b` and real `x`).
- `diff(arccot(x))` — the `arccot(x)=arctan(1/x)` convention's `x≠0` discontinuity
  is not surfaced (same arccot gap noted in Round-1 Cluster D).
- `sum(0, k, 1, ∞) → undefined` — a zero summand sums to **0**, not undefined.

## Priority sequence (by severity × tractability)

1. **R2** — `acosh(cosh(x)) = |x|`. Sign-wrong, bounded, reuses the round-1
   abs/sign machinery. Highest value-per-risk.
2. **R5b** — `solve(c/poly = 0)` → no solution. Wrong "solution"; clear fix.
3. **R4** — numeric `0/0` fold. Guard `0/x → 0` on `x≠0`; the engine already knows
   the answer on the slow path.
4. **R5a** — `solve` abs extraneous-root filtering (back-substitution).
5. **R1** — gate `f(f⁻¹(x)) = x` by the inverse's domain (broad but mechanical;
   ~14 defects, one rule family).
6. **R3** — block cancellation/like-term folding on non-finite/undefined operands
   (foundational cancellation path; scope carefully — high huella).
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
- [ ] R5b — `solve(c/poly=0)` no-solution
- [ ] R4 — numeric `0/0` fold guard
- [ ] R5a — `solve` abs extraneous-root filter
- [ ] R1 — inverse-composition domain gate (`f(f⁻¹(x))`)
- [ ] R3 — non-finite/undefined operand cancellation guard
- [ ] R6 — dropped conditions (`(a*b)^x`, arccot, zero-summand sum)
