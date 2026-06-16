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
operands are defined and finite over ℝ**. `sin(a rcsin(2))→2`, `inf−inf→0`,
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
**FIXED (commit `7b6297fca`) for literal non-finite/undefined operands.**
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
sum(k,k,1,∞) → 0`. The R4-5 adversarial pass added more instances of the same
family: `sec(π/2) − sec(π/2)`, `cot(0) − cot(0)`, `csc(0) − csc(0)`,
`ln(0) − ln(0)` (all → `0`). These are *indeterminate-arithmetic / semantic-pole*
defects, distinct from the structural "non-finite term never cancels" fix: the
operand is `undefined`/non-finite only AFTER evaluation, but syntactically it is a
plain `tan(…)`/`0^0`/`ln(0)` that the structural `A − A → 0` cancellation collapses
first — so `arithmetic_cancel_support::expr_carries_undefined` (purely structural)
cannot see it. They leak under wrappers too (`simplify(tan(π/2) − tan(π/2)) → 0`,
`expand(0^0 − 0^0) → 0`) for the same root reason; R4-5 fixes only the SYNTACTIC
`c/0`/`Undefined` class. Closing R3-2 needs a pole/indeterminate oracle that
evaluates the operand's definedness before the cancellation (or `2·inf − inf` is a
true `+inf`, a wrong-VALUE not honesty).

**R3-3 — FIXED (commit `750f0f185`), together with R4 via a shared provably-zero
oracle.** A denominator that is *provably* but not *literally* zero used to cancel:
`1/(x−x) − 1/(x−x) → 0`, `1/(0·x) − 1/(0·x) → 0`, `1/(x²−x²) − 1/(x²−x²) → 0`. The
shared predicate `expr_carries_nonfinite_or_undefined` only flagged a `Div` with a
*literal* zero denominator (`as_rational_const(den).is_zero()`); `x−x`, `0·x`,
`x²−x²`, `1²−1` slipped it, so the `A − A` cancellation fired first. **Fix:** a new
exact `is_provably_zero(ctx, e)` oracle — numeric folding INCLUDING integer-exponent
powers (`1²−1`, `2²−4`), structural additive cancellation (`x−x`, `x²−x²`,
telescoping), and a product with a provably-zero factor (`0·x`) — now drives the
`Div` arm. It is EXACT (no float/probing), so it never false-positives a nonzero
denominator. Now all those differences stay symbolic instead of folding to `0`.
A universal backstop (`rewrite_unsoundly_drops_nonfinite`, the R3 filter) was also
extended to reject any rewrite whose `before` is a `Div` with a provably-zero
denominator and whose `after` is finite — closing the shortcut paths that bypass
`transform_div` (`(x²−x²)/(x−x) → x+x`, `(3x−3x)/(x−x) → 3`). `1/inf → 0` is NOT
blocked (`inf` is not *zero*), so legitimate evaluations are untouched.

**R4-2 — FIXED (commit `134c351fa`) for POLYNOMIAL-identity zero denominators.**
A denominator zero only after algebra used to slip the exact-SYNTACTIC oracle:
`1/(x*x − x²) − … → 0` (`x*x` vs `x²`), `1/(2x − x − x) − … → 0` (coefficient),
`1/((x−1)(x+1) − (x²−1)) − … → 0` (expansion). **Fix:** `is_provably_zero` now also
converts an additive node to a multivariate polynomial (`MultiPoly`, exact rational
coefficients, `PolyBudget`-bounded) and returns true when it normalizes to the ZERO
polynomial — which means zero for ALL values, so the denominator is `c/0` everywhere.
A non-polynomial sub-term (a function, a division, an over-budget power) converts to
an error and falls through, so it is EXACT (no float, no probing) and never
false-positives a nonzero denominator (`x*x − x² + 1 → 1`, `2x − x → x`, `(x+1)^4 −
x^4` stay divisible). The oracle also peels a power of a zero polynomial (`0^n = 0`
for `n > 0`: `(x*x − x²)^2`, `(2x−x−x)^3`). Three adversarial rounds drove it to
convergence: round 1 raised `PolyBudget.max_pow_exp` (cube/quartic identities), round 2
added the `0^n` arm, round 3 is clean (94 probes, 0 leaks). engine-fast clean (no
slow/timeout from the hot-path conversion); guardrail+pressure BYTE-IDENTICAL.
**R4-3 — FIXED (commit `fb1e7b2394223de1de376b0f7d22dc54848269cf`) for the PYTHAGOREAN family.** Transcendental-
identity zero denominators (`1/(sin²x + cos²x − 1) − … → 0`) are NOT polynomial
identities (`s² + c² − 1` is not the zero polynomial in atoms `s,c`), so the MultiPoly
check declines them. The complete oracle would be the engine's own simplifier (it
reduces each to 0 standalone), but calling it from `is_provably_zero` (cas_math) is
blocked by the crate layering, and a sub-simplification per additive-transcendental
denominator is a hot-path perf hazard (a fresh `Simplifier` allocates the full rule
registry). **Fix:** an EXACT Pythagorean detector `is_pythagorean_identity_zero` —
collects the additive terms' rational coefficients of `f(arg)²` (via `mul_leaves`/
`AddView`, no float/probing) and recognizes `k·sin²+k·cos²−k`, `k·cosh²−k·sinh²−k`,
`k·sec²−k·tan²−k`, `k·csc²−k·cot²−k` for any `k`, argument, sign and order. It requires
EXACTLY the two squared terms (same argument) + a numeric constant, so it never
false-positives a nonzero denominator (`sin²+cos²` → 1, `sin²−cos²`, `sin²+cos²+1` →
1/2, different-argument or `cos⁴` combinations all stay divisible). Now the four
Pythagorean families stay symbolic/undefined in cancellation; legitimate trig
divisions are unchanged. guardrail+pressure BYTE-IDENTICAL. **Scope = the `eval`
path** (the primary surface, matching how R3/R4/R4-2 graduated): a 119-probe
adversarial round confirmed 0 bare-`eval` leaks across all four families and all
spellings (`f²`, `f·f`, fractional `k`, reordered, hyperbolic/sec/csc). The
`simplify(…)`/`expand(…)` COMMAND wrappers (and any strict wrapper) were a
separate broader gap, since fixed by R4-5.

**R4-5 — FIXED (commit `e2aafdc741045fd51c7c6495c25259cbc10375ac`).** The universal filter
`rewrite_unsoundly_drops_nonfinite` only triggered when its `before` node was an
`Add`/`Sub`/`Div`. A genuinely-undefined value wrapped in any STRICT operator —
a function (`simplify`/`expand`/`factor`/`abs`/`sin`/`ln`), a power
(`(1/(x−x)−1/(x−x))^2`), a product (`0·(1/(x−x))`) or a negation — fell through
to `_ => false`, so the inner additive cancellation `1/D − 1/D → 0` was not
reverted at the wrapper and `f(0)` produced a finite value. This leaked **R3, R4,
R4-2 AND R4-3 alike** on the command surface (`simplify(1/(x−x) − 1/(x−x)) → 0`,
`expand(1/(sin²+cos²−1) − …) → 0`). **Fix:** the filter now rejects ANY rewrite
whose `before` is genuinely UNDEFINED over ℝ (carries a `c/0` provably-zero
denominator or an `Undefined` constant) unless `after` is STILL undefined —
because real-domain functions/products/powers are strict (`f(undefined) =
undefined`). A new `expr_carries_undefined` predicate excludes pure `Infinity`
(so `1/inf → 0`, `tanh(inf)`, `atan(inf)` and other limit evaluations are never
blocked), and — caught by adversarial round 1 — `after` carrying mere `Infinity`
does NOT excuse the drop either (`ln(1/(x−x) − 1/(x−x)) → −inf` via `ln(0)` is
unsound: `ln(undefined) = undefined ≠ −inf`). The additive-`Infinity`
indeterminate (`inf − inf`) keeps the looser non-finite check (R3-2 still defers
its value-correctness). Wrappers of genuinely-undefined values now stay symbolic
(the command form retains its unevaluated wrapper, e.g.
`simplify(1/(x−x) − 1/(x−x))`), legitimate nonzero-denominator cancellation
(`simplify(1/(x+1) − 1/(x+1)) → 0`) and `Infinity` evaluations are unchanged.
guardrail+pressure fingerprints BYTE-IDENTICAL.

**R4-4 — exp/log inverse-composition FIXED (commit `44f2a272de9a494061e97965590e6ca0f7ddfbf1`); other
transcendental families still deferred.** These identities are not polynomial and
not Pythagorean, so the MultiPoly and Pythagorean arms of `is_provably_zero`
decline them; the *complete* oracle is the engine's own simplifier (it reduces each
to `0` standalone), but that is layering-blocked from `cas_math` and a hot-path
perf hazard. Following the R4-3 template, this cycle adds ONE more EXACT bounded
detector — `is_exp_log_inverse_identity_zero` — for the inverse-composition pair
`ln(e^f) − f` (≡ 0 for all real `f`) and `e^(ln f) − f` (≡ 0 for `f > 0`, undefined
elsewhere — either way `1/D` is never finite-nonzero, the same domain-hole
treatment already accepted for the pole-bearing Pythagorean `sec²−tan²−1`). It
accepts both exp spellings (`e^f`, `exp(f)`), both sign orders, and a COMPOUND
argument `f` (`ln(e^(2x+1)) − (2x+1)`): it locates the composed term, extracts `f`,
and multiset-matches the remaining `AddView` terms against `∓f`'s own
decomposition (no construction, stays `&Context`). The composed term must have
coefficient ±1, so it never flags a non-zero denominator (`ln(e^x) − x + 1 = 1`,
`ln(e^x) − 2x = −x`, different argument/function all decline). No cancel-rule edit
was needed: once `is_provably_zero(D)` is true, the R4-5 strict-wrapper filter +
the R3 universal backstop already keep `1/D − 1/D` undefined. Three adversarial
rounds drove convergence: round 1 (13 leaks) added the recursive `peel` (nesting
`ln(e^(ln(e^x)))`) and base-`e` log spellings (`log(e, ·)`); round 2 surfaced the
non-canonical atom-spelling residual (below); round 3 confirmed the in-scope
Div-spelled family clean. **Scope = the `1/D − 1/D` (Div-spelled) cancellation**;
the `D^(-1) − D^(-1)` Pow-reciprocal spelling is the separate cross-cutting R4-6
gap (below).

**Still deferred (the continuing R4-4 frontier, one family per future cycle, each
reusing this template):** `sin(2x) − 2 sin x cos x` and `cos(2x)` double-angle
forms (need argument-doubling + product matching), `tan x − sin x/cos x` (tan
definition), `sqrt(x)² − x` and `ln(x²) − 2 ln(x)` (algebraic, domain `f ≥ 0` /
`f > 0`), and coefficient multiples like `2 ln(e^x) − 2x` (the detector requires a
±1 coefficient on the composed term). Each is a `1/D − 1/D → 0` leak today.

**Structural-spelling residual (inherent to the exact-detector approach, NOT a
regression — these leaked identically before R4-4):** the detector matches the
*canonical* AST, so it misses NON-canonical spellings of the atoms it reads —
`e` written as `exp(1)` or `e^1` (`ln(exp(1)^x) − x`, `ln((e^1)^x) − x`), or an
un-normalized argument (`ln(e^(1·x)) − x`, where `1·x` ≠ `x` structurally). The
engine reduces all of these to `0` standalone (its simplifier normalizes the
atoms first), but the cancellation preorder sees the raw form. Closing the
spelling gap is exactly what the *complete* simplifier-oracle (the rejected
Approach B) would do; the same boundary applies to the R4-3 Pythagorean detector
(`sin(1·x)² + cos(1·x)² − 1`). Documented, low-impact (such spellings do not arise
from normal computation, which canonicalizes `exp(1) → e`, `1·x → x` early), and
left to the eventual simplifier-oracle cycle. (`ln(e^x · e^x) − 2x` — where
`e^x · e^x` must combine to `e^(2x)` before the peel applies — is the same
normalization-spelling class.)

**R4-6 — FIXED (commit `PENDING_HASH`).** The R4-4 round-3 adversarial surfaced a
pre-existing gap orthogonal to every identity family: a reciprocal written as
`Pow(D, −1)` (or `D^(−n)`) instead of `Div(1, D)` was NOT recognized as `c/0` when
`D` is provably zero, so `(x−x)^(-1) − (x−x)^(-1) → 0`, `(x·x−x²)^(-1) − … → 0`,
`(sin²+cos²−1)^(-1) − … → 0`, `(ln(e^x)−x)^(-1) − … → 0` all leaked — across R3-3,
R4-2, R4-3 AND R4-4 alike (the `Div` spelling of each was already correctly
`undefined`). Root cause: `expr_carries_undefined` /
`expr_carries_nonfinite_or_undefined` treated `D/0` (the `Div` arm) as undefined
but recursed blindly through `Pow(base, exp)`. **Fix:** a shared
`pow_is_reciprocal_of_provable_zero(base, exp)` — `is_provably_zero(base)` AND a
provably-NEGATIVE rational `exp` (via `exact_rational_value`) ⇒ `0^(neg) = 1/0`,
undefined — added to the `Pow` arm of BOTH predicates. One arm un-leaked the
Pow-reciprocal spelling across all four families (the standalone `D^(-n)` already
folds to `undefined`; only the `A − A` cancellation preempted it). It is exact and
conservative: a POSITIVE exponent (`0^n = 0`, defined — already handled by
`is_provably_zero`), the indeterminate `0^0` (exponent not negative — left to
R3-2), a non-provably-zero base (`x^(-1)`, `(x+1)^(-1)`, `2^(-1)`), and a
symbolic/sign-unknown exponent (`(x−x)^n`) all decline, so no defined value is ever
turned undefined and every legitimate reciprocal still cancels. guardrail+pressure
fingerprints BYTE-IDENTICAL. The adversarial (3-lens, ~70 probes) confirmed the
cancellation scope clean (0 leaks / 0 false-positives / 0 regressions) — the
`((D)^(-1))^(-1) − …` double-reciprocal cancellation is also handled (the predicate
recurses into the inner undefined). **Residual (out of scope, pre-existing):** the
STANDALONE double-nested reciprocal `((D)^(-1))^(-1)` (no cancellation) returns a
symbolic `(D)^(-1)^(-1)` in the steps-off path instead of `undefined` (steps-on
folds it correctly) — a `transform_pow` `(b^m)^n` evaluation gap, not the `A − A`
cancellation R4-6 targets, and a symbolic result rather than a false finite value.

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

**FIXED (commit `750f0f185`).** The prior investigation could not pin the
default-mode producer (it is neither `DivZeroRule` nor `const_fold`). Instrumenting
the rule loop (`RULE_TAP`) showed the fold bypasses `apply_rules` entirely — the
default-mode producers are the **fraction-simplification PREORDERS** in
`engine/transform/transform_helpers.rs::transform_div` (eval-mode fast paths that
cancel a common factor / fold a `0` numerator before `DivZeroRule` can intervene),
plus the **sum/difference-of-cubes quotient cancellation**
(`try_plan_sum_diff_of_cubes_in_num`, which gives `(1³−1)/(1−1) → 1+1+1`). **Fix
(two sites, both using the shared `is_provably_zero` oracle):** a guard at the TOP of
`transform_div` resolves a provably-zero denominator to `undefined` up front (so no
preorder cancels a zero factor); and the cubes-quotient planner declines when the
denominator is provably zero. Now `(1²−1)/(1−1)`, `(2²−4)/(2−2)`, `(1³−1)/(1−1)`,
`(1*0)/(1−1)`, `(x−x)/(x−x)`, `5/(x−x)` all → `undefined`; every legitimate quotient
cancellation with a nonzero denominator (`(x³−1)/(x−1) → x²+x+1`, `(a³−b³)/(a−b)`,
`1/(2²−1) → 1/3`) is unchanged. Bare `0/0`, `0/(1−1)`, `(3−3)/(5−5)` remain symbolic
(sound — they never folded to a finite value). guardrail+pressure BYTE-IDENTICAL.

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

### R6 — Dropped domain conditions & misc (COND-DROP/WRONG, ~4) — Fronts 1 & 3 FIXED (commit `fdade4506`)
- **Front 1 — FIXED:** `(a*b)^x → a^x·b^x` split unconditionally even for a symbolic
  (possibly non-integer) exponent, where the split is invalid for negative `a,b`
  over ℝ (`a^x`,`b^x` are individually complex). Both the default simplify path
  (`try_rewrite_power_product_distribution_expr`) AND the explicit `expand` path
  (`expand_ops::expand_pow` — the adversarial sweep caught this second bypass) now
  decline the split when the exponent is non-numeric/non-integer UNLESS both bases
  are provably non-negative (positive constant, even-integer power `y^(2k)`, `|·|`,
  `e`, or a product of such). The SAME gate was mirrored onto THREE producers the
  adversarial sweeps enumerated: the product split (`try_rewrite_power_product_distribution_expr`),
  the `expand` recursion (`expand_ops::expand_pow`), and the QUOTIENT split
  (`try_rewrite_power_quotient_expr` — `(a/b)^x → a^x/b^x` had the identical hole).
  Integer exponents stay universally safe; the `^(1/2)` paths are unchanged. Now
  `(a*b)^x`, `(x*y)^n`, `(a*b)^π`, `(a/b)^x`, `((-2)/b)^x`, `expand((a*b*c)^x)` stay
  `(…)^exp` (unsplit); `(a*b)^2 → a²·b²`, `(a/b)^2 → a²/b²`, `(x²·y²)^n →
  |x|^(2n)·|y|^(2n)` still split. Three adversarial sweeps (~770 probes) — the 1st
  caught the `expand` bypass, the 2nd the quotient sibling, the 3rd confirmed clean.
  (Residual, PRE-EXISTING, A-2 territory: the MERGE direction `(-2)^x·(-3)^x → 6^x`
  fabricates a real over negative bases — negative-base power family, untouched here.)
- **Front 3 — FIXED:** `sum(0, k, 1, ∞) → undefined` (it built `0 * (∞−1+1) = 0·∞`).
  `try_build_sum_of_constant` now returns `0` early when the summand is structurally
  zero, before computing the term count — so `sum(0, k, 1, ∞)` and `sum(k−k, k, 1, ∞)`
  are `0`; finite/symbolic non-zero sums are unchanged.
- **Front 2 — DEFERRED as R6-2 (convention decision + deep diff/domain surgery):**
  `diff(arccot(x)) → -1/(x²+1)` drops the `x≠0` that `arccot(x)→arctan(1/x)` and
  `diff(arctan(1/x))` surface. Diff conditions are inferred from the RESULT's
  structure (sqrt→radicand>0, div→denom≠0); arccot's derivative `-1/(x²+1)` has no
  such subterm, so x≠0 is lost. Surfacing it requires either declaring arccot's
  function-domain as `x≠0` (broad) or diff-pipeline surgery. CONVENTION FORK: the
  engine's arccot is the non-standard `arctan(1/x)` form (`arccot(0)=undefined`,
  range ≠ (0,π), discontinuous at 0) — under which x≠0 IS required; but the standard
  EDUCATIONAL arccot is CONTINUOUS on ℝ (`arccot(0)=π/2`, differentiable everywhere,
  derivative `-1/(1+x²)` with NO condition), under which the current result is
  CORRECT and `arccot(0)=undefined` is itself the bug. Needs a convention decision
  before fixing — not a bounded edit.

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
   FIXED (commit `7b6297fca`): shared predicate + universal post-filter at the two
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
- [x] R4 — numeric `0/0` fold guard *(FIXED 2026-06-16, commit `750f0f185`, shared `is_provably_zero` oracle: `transform_div` top-guard + cubes-quotient planner gate; `(1²−1)/(1−1)`, `(1³−1)/(1−1)` → undefined)*
- [x] R5a — `solve` abs extraneous-root filter *(FIXED 2026-06-15, commit `4d07aaee6`, rational roots; irrational extraneous split to R5a-2)*
- [ ] R5a-2 — irrational/transcendental extraneous roots (e.g. `solve(|x|=2-e)`) need exact/symbolic back-substitution
- [x] R1 — inverse-composition domain gate (`f(f⁻¹(x))`) *(FIXED 2026-06-16, commit `261f1de28`, four rule families)*
- [x] R3 — non-finite/undefined operand cancellation guard *(FIXED 2026-06-16, commit `7b6297fca`, shared predicate + universal post-filter at the two simplifier chokepoints; literal ∞/undefined/`c÷0` no longer cancel to 0)*
- [ ] R3-2 — *semantic* indeterminates (`tan(π/2)−tan(π/2)`, `sec(π/2)`/`cot(0)`/`csc(0)` poles, `ln(0)−ln(0)`, `0^0−0^0`, `factorial(−2)·0`) and infinity-arithmetic (`2·inf−inf` → true `+inf`) still fold (incl. under `simplify`/`expand` wrappers); need a pole/indeterminate oracle that evaluates operand-definedness before the structural `A−A` cancellation
- [x] R3-3 — *provably*-but-not-*literally*-zero denominators (`1/(x−x)`, `1/(0·x)`, `1/(x²−x²)`) cancel *(FIXED 2026-06-16, commit `750f0f185`, exact `is_provably_zero` oracle in the `Div` arm of the non-finite predicate)*
- [x] R4-2 — *polynomial-identity* zero denominators (`x*x−x²`, `2x−x−x`, `(x−1)(x+1)−(x²−1)`) *(FIXED 2026-06-16, commit `134c351fa`, exact `MultiPoly` normalization in `is_provably_zero`)*
- [x] R4-3 — *Pythagorean-identity* zero denominators (`sin²+cos²−1`, `cosh²−sinh²−1`, `sec²−tan²−1`, `csc²−cot²−1`) *(FIXED 2026-06-16, commit `fb1e7b2394223de1de376b0f7d22dc54848269cf`, exact `is_pythagorean_identity_zero` coefficient check)*
- [~] R4-4 — *non-Pythagorean* transcendental-identity zero denominators: exp/log **inverse-composition** (`ln(e^f)−f`, `e^(ln f)−f`, both spellings/orders/compound args) *(FIXED 2026-06-16, commit `44f2a272de9a494061e97965590e6ca0f7ddfbf1`, exact `is_exp_log_inverse_identity_zero` multiset detector)*; **still deferred** — `sin(2x)−2 sin x cos x`, `cos(2x)` forms, `tan x − sin x/cos x`, `sqrt(x)²−x`, `ln(x²)−2 ln x`, coefficient multiples (one family per future cycle, same template)
- [x] R4-5 — *strict-wrapper / command-surface* gap: a genuinely-undefined value (`c/0`, `undefined`) inside a function/power/product/neg (incl. `simplify`/`expand`/`factor`) collapsed to a finite value, leaking R3/R4/R4-2/R4-3 on the command surface *(FIXED 2026-06-16, commit `e2aafdc741045fd51c7c6495c25259cbc10375ac`, universal filter now rejects dropping a carried `undefined` under any strict node via `expr_carries_undefined`; Infinity-limit evals preserved; adversarial caught the `ln(0)→−inf` sub-case)*
- [x] R4-6 — *`D^(-1)` reciprocal spelling* (CROSS-CUTTING): `(D)^(-1) − (D)^(-1) → 0` for every provably-zero `D` (`x−x`, `x·x−x²`, `sin²+cos²−1`, `ln(e^x)−x`) leaked via the `Pow(zero, negative)` spelling, across R3-3/R4-2/R4-3/R4-4 alike *(FIXED 2026-06-16, commit `PENDING_HASH`, `pow_is_reciprocal_of_provable_zero` arm added to both `expr_carries_*` predicates — `0^(neg)=1/0`; positive-exp/`0^0`/non-zero-base/symbolic-exp all decline)*
- [x] R6 — dropped conditions: `(a*b)^x` split gated + `sum(0,…,∞)=0` *(FIXED 2026-06-16, commit `fdade4506`, Fronts 1 & 3)*
- [ ] R6-2 — `diff(arccot(x))` `x≠0`: needs an arccot convention decision (non-standard `arctan(1/x)` vs standard continuous arccot) + diff/domain surgery
