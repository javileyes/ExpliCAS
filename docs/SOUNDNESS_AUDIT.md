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
  **WRONG VALUE.** A "Cancel Opposite Fractions" rule (visible in `--steps`)
  cancels two **identical** `+1/sqrt(1-x^2)` terms (the `-arccos` derivative is
  `+1/sqrt(1-x^2)`, so it is a sum of equal terms, not opposite). The standalone
  simplifier gives `2/sqrt(1-x^2)`, so the fault is specific to the diff path.
  Symmetric: `diff(arccos(x) - arcsin(x), x)` → `0` (true `-2/sqrt(1-x^2)`).
- `diff(arcsin(x) + arccos(x), x)` → `0` but **drops** the required `-1<x<1`
  condition (the analogous `ln`/`√`/`1/x` cancellations correctly keep theirs).

### Cluster B — `(a^even)^y` with a **symbolic** outer exponent drops `|a|` (`logs_exps`)
- `(a^2)^y` → `a^(2·y)` ; correct `|a|^(2·y)`. At `a=-2, y=1/2`: returns `-2`,
  correct `2`. **WRONG VALUE.** Also `(a^4)^y → a^(4·y)` and
  `((-2)^x)^y → (-2)^(x·y)`.
- Root cause: this is the **symbolic-outer-exponent gap of the same `(x^m)^n`
  family fixed earlier for the literal case** (`(a^2)^(1/2)=|a|` is correct; only
  the symbolic outer exponent still drops the abs). Natural follow-up to that cycle.

### Cluster D — `arctan(x)+arctan(1/x)` / `arctan+arccot` → `π/2` unconditionally (`trig_invtrig`)
- `arctan(x)+arctan(1/x)` → `π/2` (only `x≠0`) ; correct `(π/2)·sign(x)` →
  **`-π/2` for x<0**. **WRONG VALUE.**
- `arctan(x)+arccot(x)` → `π/2` ; under the engine's own `arccot(x)=arctan(1/x)`
  convention the true value is `-π/2` for x<0. **WRONG VALUE** (internally
  inconsistent). Bounded fix: gate the identity by `sign(x)`.

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

1. **Cluster C** — wrong value across the whole domain; the "Cancel Opposite
   Fractions" rule cancelling *equal* terms may affect other expressions too.
   *(in progress)*
2. **Cluster B** — wrong value; closes the symbolic-exponent half of the
   `(x^m)^n` soundness work already started this session.
3. **Cluster D** — wrong branch value; bounded fix (gate by `sign(x)`).
4. **Cluster A** — real-domain honesty; the sign-wrong `sqrt(-2)*sqrt(-3)=+√6` is
   the most urgent within the cluster.

## Status

- [ ] Cluster C — arcsin/arccos derivative cancellation
- [ ] Cluster B — `(a^even)^symbolic` drops `|a|`
- [ ] Cluster D — `arctan(x)+arctan(1/x)` branch
- [ ] Cluster A — even-root of negative base
