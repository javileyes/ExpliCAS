# Soundness Audit — multi-axis (2026-06-15)

Baseline commit at audit time: `f60c8970a`.

Multi-axis adversarial soundness audit of the CAS engine, run via a multi-agent
workflow (ultracode): 10 axes hunting in parallel, every candidate defect
independently re-verified by a skeptic (to reject sympy complex-branch artifacts,
sound over-conditioning, and the real-root convention). Ground truth is always
**real-domain** (numeric at both signs / `sympy.real_root` / by hand).

- **622 probes**, 21 agents.
- **7 of 10 axes completely clean.**
- **11 confirmed defects in 4 root-cause clusters.**

## Clean axes (0 defects) — confirm the recent hardening holds

`abs_sign` (62), `cancellation_domains` (65), `integration_roundtrip` (85),
`integration_honesty` (31), `definite_integrals` (62), `limits` (110), and the
symbolic half of `powers_roots`.

Notably solid: every removable cancellation keeps its `≠0` condition; **no**
interior pole ever produced a false finite value (the classic unsound failure);
honesty residuals (`∫e^(-x²)`, `∫sin x/x`, divergent `∫₀ˣ ln(t)/t → undefined`)
stay honest; DNE / two-sided-divergent limits are not folded to wrong values; the
`(x^m)^n` family with **literal** exponent, `∫|x|=x|x|/2`, `d/dx|x|=sign(x)`, and
`x²/|x|→|x|` are all correct.

## Confirmed defects (4 root-cause clusters)

### Cluster C — arcsin/arccos derivative cancellation (`differentiation`)
- `diff(arcsin(x) - arccos(x), x)` → `0` ; correct `2/sqrt(1-x^2)` on `-1<x<1`.
  **WRONG VALUE.** Symmetric: `diff(arccos(x) - arcsin(x), x)` → `0` (true `-2/sqrt(1-x^2)`).
- `diff(arcsin(x) + arccos(x), x)` → `0` but **drops** the required `-1<x<1`
  condition.

**ROOT CAUSE — FOUND AND FIXED (2026-06-15).** Pinned with `eprintln!` instrumentation
on the minimal reproducer. It is NOT a canonicalization rule and NOT the cancellation
rule. The defect is in `extract_fraction_pair` (`cas_math::fraction_pair_support`):
it **double-counts the sign**. It calls `FractionParts::to_num_den`, which *bakes*
the fraction's sign into the numerator (so for the subtrahend `Neg(Div(-1,y))` = +1/y
the numerator becomes `+1`), AND it *also* returns `fp.sign = -1` separately. Any
consumer reading both `n` and `sign` as `sign·(n/d)` then sees `-1·(+1/y) = -1/y` for
a term that is actually `+1/y` — violating the `FractionPair` contract (value = sign·n/d).

- This makes `1/y + (+1/y)` look like an opposite pair `1/y + (-1/y)` → cancels to 0.
- The fraction add/sub rules were unaffected (they use the baked numerator alone, which
  has the correct value). Only the two cancellation **detectors** —
  `should_defer_exact_opposite_fraction_pair_to_additive_cancellation` and
  `fractions_match_same_value` — used both `n` and `sign`, so only they misfired. Both
  run exclusively on the baked path (gated by `parts.is_frac1 && parts.is_frac2`).
- **Fix:** in both detectors, re-derive the sign purely from the baked numerator
  (pass sign-base `1`, let `normalize_fraction_numerator_sign` extract it). Surgical,
  scoped to the two detectors, behaviour-identical for the common `fp.sign=+1` case
  (the minus inside the numerator); only the outer-`Neg` double-negative case changes.
  `crates/cas_engine/src/rules/algebra/fractions/addition_rules.rs`.
- **Verified:** `1/y - (-1)/y → 2/y` ✓, `1/y - (-1/y) → 2/y` ✓, while genuine
  cancellations still hold: `1/y - 1/y → 0` ✓, `1/y + (-1/y) → 0` ✓; non-opposites
  unaffected: `1/y + 1/y → 2/y` ✓, `a/y - (-1)/y → (a+1)/y` ✓. And the diff defects:
  `diff(arcsin−arccos) → 2/sqrt(1-x²)` ✓, `diff(arccos−arcsin) → -2/sqrt(1-x²)` ✓.
- **Remaining (reclassified as P3-educational, separate cycle):** `diff(arcsin+arccos) → 0`
  is value-correct (π/2) but does not surface the `-1<x<1` domain condition. This is
  **systemic, not cancellation-specific**: even a single `diff(arcsin(x)) = 1/sqrt(1-x²)`
  reports `required_conditions: []` — the engine carries the domain *implicitly* via the
  `sqrt(1-x²)` denominator, which vanishes on cancellation. Surfacing explicit domain
  conditions for inverse-trig derivatives is a separate educational gap, not a wrong value.

### Cluster B — `(a^even)^y` with a **symbolic** outer exponent drops `|a|` (`logs_exps`)
- `(a^2)^y` → `a^(2·y)` ; correct `|a|^(2·y)`. At `a=-2, y=1/2`: returns `-2`,
  correct `2`. **WRONG VALUE.** Also `(a^4)^y → a^(4·y)` and
  `((-2)^x)^y → (-2)^(x·y)`.
- Root cause: this is the **symbolic-outer-exponent gap of the same `(x^m)^n`
  family fixed earlier for the literal case** (`(a^2)^(1/2)=|a|` is correct; only
  the symbolic outer exponent still drops the abs). Natural follow-up to that cycle.

**FIXED (2026-06-15, commit `5b13a9baa`) — the literal-even-inner half.**
`try_rewrite_power_power_even_root_abs_expr` bailed to the sign-unsafe
`MultiplyExponents` branch whenever the outer exponent was not a rational
constant. For an even **literal** inner exponent `m`, `(x^m)^y = |x|^(m·y)` holds
unconditionally over the reals, so for a symbolic `y` (where `m·y` parity is
undecidable) the abs is now kept. Verified: `(a^2)^y → |a|^(2·y)`,
`(a^4)^y → |a|^(4·y)`, `(a^6)^y → |a|^(6·y)`, `(a^2)^(y/2) → |a|^y`,
`(a^2)^(2*y) → |a|^(4·y)`; literal cases unchanged (`(a^2)^(1/2)=|a|`,
`(a^2)^3=a^6`, `(a^2)^(2/3)=a^(4/3)`). Adversarial 2-lens / 325 probes: clean
for this scope; full validation green; guardrail/pressure fingerprints
unchanged (only `filtered_out +1` from the new unit test).

### Cluster B-2 — `(a^(2k))^(1/2)` drops `|a|` for a **symbolic even** inner exponent (NEW, found by the Cluster B adversarial sweep)
- `(a^(2*k))^(1/2)` → `a^k` ; correct `|a|^k` (real principal root is
  non-negative). At `a=-2,k=1`: returns `-2`, correct `2`. **WRONG VALUE.**
  Also `(a^(6*k))^(1/2) → a^(3·k)` (true `|a|^(3·k)`),
  `(a^(2*k))^(3/2) → a^(3·k)`, `(a^(2*k))^(1/(2*k))`. Fires exactly when halving
  the symbolic even inner coefficient yields an **odd** coefficient (`2k→k`,
  `6k→3k`); `(a^(4k))^(1/2)→a^(2k)` and `(a^(8k))^(1/2)→a^(4k)` stay even and are
  correct. The same family as `((-2)^x)^y → (-2)^(x·y)` from Cluster B.
- DISTINCT code path from Cluster B's fix: the inner exponent `2*k` is **not** a
  literal even integer, so it declines the `inner_int` gate and falls through the
  `MultiplyExponents` cancellation. PRE-EXISTING (untouched by the Cluster B fix).
- Fix under the engine's already-tracked `a^(2k) ≥ 0` condition:
  `sqrt(a^(2k)) = |a^k| = |a|^k`. Enabler verified: the engine folds `|a^even| →
  a^even` but keeps `|a^k|`/`|a^(2k)|`, so wrapping the cancelled result in abs is
  self-cleaning.
- **GATE IS SUBTLE (why this needs its own scoped cycle, not a blanket wrap):** the
  abs is needed for even-root outer ONLY when the inner exponent is *not provably
  odd*. `(a^3)^(1/2) → a^(3/2)` is already correct — odd inner ⟹ the realness
  condition reduces to `a ≥ 0` (on the BASE), forcing the result non-negative. Even
  / unknown-parity inner (`2k`, `k`, `3k`) attach `a^E ≥ 0` (not on the base), so the
  result can be negative ⟹ abs required. A blanket "wrap on even outer denominator"
  would regress `(a^3)^(1/2)` to a redundant `|a^(3/2)|`. So the fix must detect
  provably-odd inner exponents (literal odd / `2j+1` form) and decline there.
  NOT YET FIXED.

### Cluster D — `arctan(x)+arctan(1/x)` / `arctan+arccot` → `π/2` unconditionally (`trig_invtrig`)
- `arctan(x)+arctan(1/x)` → `π/2` (only `x≠0`) ; correct `(π/2)·sign(x)` →
  **`-π/2` for x<0**. **WRONG VALUE.**
- `arctan(x)+arccot(x)` → `π/2` ; under the engine's own `arccot(x)=arctan(1/x)`
  convention the true value is `-π/2` for x<0. **WRONG VALUE** (internally
  inconsistent). Bounded fix: gate the identity by `sign(x)`.

**FIXED (2026-06-15, commit `PENDING_HASH`).** The reciprocal-pair planner
(`try_plan_inverse_atan_reciprocal_pair_expr` in
`cas_math/src/inverse_trig_composition_support.rs`) returned `π/2`
unconditionally; it now returns `(π/2)·sign(x)`. `x` and `1/x` share a sign, so
the non-fraction argument is used for a clean `sign(x)`; for a literal argument
`EvaluateSignRule` folds `sign(c) → ±1`, recovering `arctan(2)+arctan(1/2)=π/2`
and `arctan(-2)+arctan(-1/2)=-π/2`. Verified numerically at both signs; the
metamorphic identity suite (which samples `x>0`) is unaffected since
`(π/2)·sign(x)=π/2` there. Updated the `nary_pattern_matching` contract +
regression test. *(Minor pre-existing follow-up, not introduced here: the
`arctan(x)+arccot(x)` form does not surface the `x≠0` condition — a
condition-completeness gap in the arccot→arctan conversion, P3.)*

### Cluster A — even-index root of a negative base (`powers_roots`)
The fallacy `(z^(1/even))^even = z` / `sqrt(a)·sqrt(b)=sqrt(ab)` applied to
negatives, where the even root has no real value:
- `sqrt(-2)*sqrt(-3)` → `sqrt(6)` ; undefined over ℝ (and even the complex
  principal value is **−√6**, opposite sign). **WRONG VALUE.**
- `(sqrt(-2))^2` → `-2` ; undefined. Honesty violation.
- `sqrt(-2)*sqrt(-2)` → `-2` ; undefined (its own symbolic path knows `(√x)²=x`
  needs `x≥0`, dropped on the negative literal). 
- `((-2)^(1/4))^4` → `-2` ; undefined.
- Inconsistent with the honest siblings (`sqrt(-1)` warns); no warning here.

## Priority sequence

1. **Cluster C** — wrong value across the whole domain. *(FIXED 2026-06-15: root
   cause was `extract_fraction_pair` double-counting the fraction sign; the two
   cancellation detectors now re-derive the sign from the baked numerator. The
   wrong-value defect is resolved; the arcsin+arccos domain-condition drop is
   reclassified P3-educational. See the root-cause section above.)*
2. **Cluster B** — wrong value; FIXED the literal-even-inner half (commit
   `5b13a9baa`). The symbolic-even-inner half is split out as Cluster B-2.
3. **Cluster B-2** — wrong value (NEW, surfaced by the Cluster B adversarial
   sweep); same `(x^even)^outer` family but a distinct sign-unaware cancellation
   path. Bounded fix under the existing `a^(2k) ≥ 0` condition.
4. **Cluster D** — wrong branch value; FIXED (commit `PENDING_HASH`) by gating
   the arctan reciprocal identity with `sign(x)`.
5. **Cluster A** — real-domain honesty; the sign-wrong `sqrt(-2)*sqrt(-3)=+√6` is
   the most urgent within the cluster.

## Status

- [x] Cluster C — arcsin/arccos derivative cancellation *(wrong value FIXED 2026-06-15, commit `810c0a6db`; arcsin+arccos condition-drop reclassified P3-educational)*
- [x] Cluster B — `(a^even)^symbolic` drops `|a|` *(literal-even-inner half FIXED 2026-06-15, commit `5b13a9baa`; symbolic-even-inner half split to Cluster B-2)*
- [ ] Cluster B-2 — `(a^(2k))^(1/2)` drops `|a|` for symbolic even inner exponent
- [x] Cluster D — `arctan(x)+arctan(1/x)` branch *(FIXED 2026-06-15, commit `PENDING_HASH`: gated by `sign(x)`)*
- [ ] Cluster A — even-root of negative base
