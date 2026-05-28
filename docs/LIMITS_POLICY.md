# Limits Policy

> **V1.30** — Pre-simplification contract, finite quotient policies, one-sided finite orientation/pole, log-endpoint, root-endpoint, finite inverse-trig/acosh endpoint policies, finite one-sided inverse-trig and atanh endpoint paths, finite one-sided domain-path residual policy, domain-bearing resolved-base log quotients with compact condition witnesses, natural/fixed-base/binary-log endpoint residuals, and enforcement

## 1. Purpose

The limit engine uses a **conservative policy**: never invent results.

To improve resolution rate without breaking this guarantee, V1.3 introduces a **pre-simplification layer** that is:

- **Optional**: `--presimplify off|safe` (default: off)
- **Bounded**: allowlist-only transforms
- **Auditable**: lint + contract tests

---

## 2. PreSimplify Safe Contract

### 2.1 Entry Point

```rust
// crates/cas_math/src/limits_support.rs
pub fn presimplify_safe_for_limit(ctx, expr) -> ExprId
```

Single point of entry ensures auditability and test coverage concentration.

### 2.2 Allowlist (Transforms Permitted)

| Transform | Example | Notes |
|-----------|---------|-------|
| Add zero | `a + 0 → a`, `0 + a → a` | Identity |
| Mul one | `a * 1 → a`, `1 * a → a` | Identity |
| Mul zero | `0 * a → 0`, `a * 0 → 0` | Absorbing |
| Sub zero | `a - 0 → a` | Identity |
| Sub self | `a - a → 0` | Structural equality only |
| Add neg | `a + (-a) → 0` | Structural equality only |
| Double neg | `-(-a) → a` | Involution |

### 2.3 Denylist (Explicitly Forbidden)

| Category | Examples | Why Forbidden |
|----------|----------|---------------|
| Division cancel | `a/a → 1` | Domain assumption (`a ≠ 0`) |
| Power zero | `a^0 → 1` | `0^0` undefined |
| Rationalization | `1/(1+√2) → ...` | Non-conservative |
| Expansion | `(a+b)^n → ...` | Term explosion |
| Polynomial ops | `gcd_*`, `poly_*` | Out of scope |
| General simplify | `simplify()`, `apply_rules_loop` | Bypass |

### 2.4 Non-goals (What PreSimplify Safe Does NOT Do)

> [!IMPORTANT]
> PRs attempting to add these capabilities to `presimplify_safe` should be rejected.

- **No rationalization**: Does not conjugate or clear radicals in denominators
- **No domain assumptions**: Does not assume `x ≠ 0` for cancellations like `x/x → 1`
- **No new canonical forms**: Only eliminates structural noise (`+0`, `*1`, `-a+a`)
- **No term expansion**: Does not expand `(a+b)^n` or distribute multiplication
- **No general simplification**: Does not call into the main simplifier pipeline

---

## 3. Enforcement

### 3.1 Lint: `lint_limit_presimplify.sh`

**HARD FAIL** if the isolated `presimplify_safe_for_limit` region contains:

```bash
DENY_PATTERNS=(
  "rationalize"
  "expand_with_stats"
  "expand::"
  "multinomial"
  "gcd_"
  "poly_"
  "simplify_with_stats"
  "apply_rules_loop"
  "domain_assumption"
  "Simplifier"
  "::simplify("
)
```

**Also enforced**: no local `is_zero`/`is_one` (use `crate::helpers::*`).

### 3.2 Contract Tests

| ID | Test Case | Contract Verified |
|----|-----------|-------------------|
| T1 | `(x-x)/x` → 0 with safe | Improvement |
| T2 | `x/x` → 1 (by limit rule, not presimplify) | No domain assumption |
| T3 | `1/(1+sqrt(2))` → residual | No rationalization |
| T4 | `x^2/x^2` same result off/safe | Stability |
| T5 | `0*x/x` → 0 | Mul zero works |
| T6 | Standard polynomial limit | No regression |
| T7 | `sqrt(x)+1` → ∞ | No expand irrationals |
| T8 | `((x+0)+0)/x` → 1 | Nested transforms |

File: `crates/cas_cli/tests/presimplify_contract_tests.rs`

---

## 4. Usage

### CLI

```bash
# Default (off): maximum conservatism
expli limit "x^2/x" --to infinity

# With safe pre-simplification
expli limit "(x-x)/x" --to infinity --presimplify=safe
```

### REPL

```
> limit x^2/x
lim_{x→+∞} = ∞

> limit (x-x)/x, x, infinity, safe
lim_{x→+∞} = 0
```

---

## 5. Evolution Path

### 5.1 Relationship To Calculus Strategy

Limits are one vertical slice of the broader calculus engine strategy:

- [CALCULUS_ENGINE_STRATEGY.md](/Users/javiergimenezmoya/developer/math/docs/CALCULUS_ENGINE_STRATEGY.md)

That broader strategy does not weaken this policy. It makes this policy the
model for calculus work:

- conservative first
- explicit domain and infinity assumptions
- single auditable entry points for risky pre-processing
- tests before promotion
- unsupported residuals instead of speculative answers

When an auto-improvement cycle chooses `calculus` and touches limits, it should
follow this file first, then use the calculus strategy to decide how the limit
work feeds simplification, equivalence, domains, and didactic traces.

Future enhancements should follow the same pattern:

1. **Define contract** (allowlist + denylist)
2. **Single entry point** (auditable)
3. **Add lint** (prevent regression)
4. **Add contract tests** (freeze behavior)
5. **Document in POLICY** (this file)

Candidates for similar treatment:
- Limits V2 (x→a point limits)
- Rationalization layer
- Algebraic cancellation (`x/x`, `(x²-1)/(x-1)`)

### 5.2 Finite-Point Residual Boundary

Inline eval syntax may accept finite-point notation such as
`limit(ln(x), x, -1)`, but this is only a residual contract until Limits V2 is
defined and tested beyond the narrow exception below.

Current contract:

- accepted finite points render as `limit(expr, var, point)` when unresolved
- finite-point eval syntax may carry an explicit side as
  `limit(f(x), x, 0+)`, `limit(f(x), x, 0-)`, or
  `limit(f(x), x, 0, right|left)`; this does not make ordinary finite limits
  directional, and unsupported one-sided families must remain residual with an
  explicit one-sided warning rather than falling through as internal parse
  errors
- expressions that do not depend on the limit variable may evaluate directly,
  preserving their own implicit domain requirements
- expressions that do not depend on the limit variable but have a statically
  empty real domain, such as `ln(0)`, `sqrt(-1)`, or `log(1, 2)`, may resolve
  to `undefined`; this does not relax residual policy for variable-dependent
  domain, branch, endpoint, or path cases
- the identity variable limit may evaluate directly to the finite point when
  that point does not depend on the limit variable
- polynomial expressions may evaluate at numeric rational finite points because
  real polynomials are total on the real line and require no continuity,
  branch, side, or domain assumption
- rational polynomial expressions may evaluate at numeric rational finite
  points only when the denominator evaluates explicitly to a nonzero value at
  that point, or when exact polynomial multiplicity checks prove a removable
  hole by repeatedly differentiating numerator and denominator until the
  denominator has a nonzero value at the point; finite poles such as
  `limit(1/x, x, 0)` and unresolved zero-denominator forms remain residual
  until a one-sided or infinite finite-point policy exists, and must retain
  the denominator nonzero requirement such as `x ≠ 0`
- `exp(p(x))`, `sin(p(x))`, `cos(p(x))`, `sinh(p(x))`, `cosh(p(x))`,
  `tanh(p(x))`, `atan(p(x))`/`arctan(p(x))`, `asinh(p(x))`, `cbrt(p(x))`,
  and `abs(p(x))` may evaluate at numeric rational finite points when `p(x)`
  is polynomial in the limit variable, because these functions are continuous
  and total on the real line; only exact special values such as a zero
  argument, table-backed `atan(1)`/`arctan(1)` to `pi/4`, exact rational cube
  roots, or exact rational absolute values may collapse to constants in this
  local rule
- `sqrt(p(x))`, `ln(p(x))`, `log2(p(x))`, and `log10(p(x))` may evaluate at
  numeric rational finite points only when `p(x)` is polynomial in the limit
  variable and `p(a)` is strictly positive; zero and negative argument values
  remain residual to avoid endpoint, side, branch, or domain-path assumptions
- natural, fixed-base, and binary-log endpoint boundaries such as
  `limit(ln(x), x, 0)`, `limit(log2(x), x, 0)`, and
  `limit(log(2, x), x, 0)` remain residual under the current two-sided
  finite-point policy, but must retain the positive-domain requirement
  (`x > 0`) and a residual explanation instead of silently returning a
  one-sided infinite result; for an explicit valid binary-log base such as `2`,
  no extra base-domain requirement should be displayed
- `asin(p(x))`/`arcsin(p(x))`, `acos(p(x))`/`arccos(p(x))`, `atanh(p(x))`,
  and `acosh(p(x))` may evaluate at numeric rational finite points only when
  the polynomial argument lands strictly inside the real domain interior:
  `-1 < p(a) < 1` for inverse trig and `atanh`, and `p(a) > 1` for `acosh`;
  endpoints, outside-domain values, and unresolved sublimits remain residual to
  avoid side, branch, or domain-path assumptions
- `tan(p(x))` and `sec(p(x))` may evaluate at numeric rational finite points
  when the polynomial argument evaluates exactly to `0`; this gives an explicit
  safe denominator witness through `cos(0) = 1`; nonzero numeric rational
  arguments remain residual unless a later composition rule has already
  produced a recognized special-angle table hit; `csc(p(x))` and `cot(p(x))`
  have no zero-safe numeric-rational exception
- arithmetic compositions of already-resolved safe finite sublimits may
  evaluate through `+`, `-`, `*`, unary negation, and division only when the
  computed denominator is either an explicit nonzero numeric value or is proven
  structurally positive by the existing sign prover; if any sublimit is
  unresolved or a denominator is not proven safe, the whole finite limit remains
  residual; after all operands are safe, this local rule may fold exact rational
  arithmetic and structural identities such as `g - g -> 0`, `0 + g -> g`,
  `1*g -> g`, and `g/g -> 1` only after the denominator has been proven
  nonzero
- the standard small-angle quotient may evaluate finite limits of
  `c*sin(p(x))/q(x)` when the approach point is rational, `p(x)` and `q(x)`
  are polynomials in the limit variable, `p(a) = 0`, and the exact removable
  rational-polynomial value of `c*p(x)/q(x)` at `a` is finite; this promotes
  cases such as `sin(x)/x -> 1`, `sin(2*x)/x -> 2`, and shifted polynomial
  variants, but leaves nonzero sine arguments, finite poles, non-polynomial
  denominators, and broader trigonometric quotient families residual
- the standard exponential zero quotient may evaluate finite limits of
  `c*(exp(p(x)) - 1)/q(x)` or `c*(e^p - 1)/q(x)` when the approach point is
  rational, `p(x)` and `q(x)` are polynomials in the limit variable,
  `p(a) = 0`, and the exact removable rational-polynomial value of
  `c*p(x)/q(x)` at `a` is finite; this promotes cases such as
  `(exp(x)-1)/x -> 1`, `(exp(2*x)-1)/x -> 2`, and shifted polynomial variants,
  while leaving nonzero exponent arguments, finite poles, and non-polynomial
  denominators residual
- the standard natural-log unit quotient may evaluate finite limits of
  `c*ln(g(x))/q(x)` when the approach point is rational, `g(x)` and `q(x)`
  are polynomials in the limit variable, `g(a) = 1`, and the exact removable
  rational-polynomial value of `c*(g(x)-1)/q(x)` at `a` is finite; this
  promotes cases such as `ln(1+x)/x -> 1`, `ln(1+2*x)/x -> 2`, and
  `ln(x)/(x-1) -> 1` at `x -> 1`; because `g(a)=1`, polynomial continuity
  gives a local positive-domain witness for the two-sided real limit, while
  non-unit log arguments, finite poles, non-polynomial arguments, and
  non-polynomial denominators remain residual
- the same unit-log quotient policy may evaluate fixed-base variants
  `c*log2(g(x))/q(x)` and `c*log10(g(x))/q(x)` under the identical rational
  point, polynomial, `g(a)=1`, and finite removable-ratio checks; the result is
  represented exactly as the rational removable value divided by `ln(2)` or
  `ln(10)`, preserving the base-change factor explicitly; broader binary
  `log(base, argument)` forms remain residual unless a separate policy proves
  base constancy, base positivity, `base != 1`, argument positivity, and the
  same removable unit-argument shape without hidden branch or domain assumptions
- binary unit-log quotients with a literal rational base may evaluate finite
  limits of `c*log(b, g(x))/q(x)` under the same rational point, polynomial,
  `g(a)=1`, and finite removable-ratio checks, but only when `b` is an explicit
  rational constant with `b > 0` and `b != 1`; the result is represented exactly
  as the rational removable value divided by `ln(b)`, for example
  `log(3, 1+2*x)/x -> 2/ln(3)` and
  `log(1/2, 1+2*x)/x -> 2/ln(1/2)`
- binary unit-log quotients with a variable base may evaluate finite limits of
  `c*log(b(x), g(x))/q(x)` under the same rational point, polynomial
  `g(a)=1`, and finite removable-ratio checks, but only when `b(x)` is either
  polynomial at the point or resolves through an already supported finite
  sublimit to an exact rational value `b(a) > 0` with `b(a) != 1`; the public
  command output must still surface the input's base positivity, `base != 1`,
  log-argument positivity, and quotient-denominator conditions unless one
  condition is already implied by a stronger displayed condition; the result is
  represented exactly as the rational removable value divided by `ln(b(a))`,
  for example `log(x+1/4, 1+2*x)/x -> 2/ln(1/4)` and
  `log(exp(x)+2, 1+2*x)/x -> 2/ln(3)` at `x -> 0`; if the resolved base has
  its own real-domain requirement, such as
  `log(sqrt(x+4)+1, 1+2*x)/x -> 2/ln(3)`, that condition must remain visible
  in command output alongside the base, argument, and quotient-denominator
  conditions unless it is already implied by a stronger displayed condition;
  internal `base - 1` witnesses should be compacted before display, so
  intrinsically nonzero gaps are suppressed and radical gaps are exposed as
  ordinary real-domain inequalities rather than as unevaluated arithmetic;
  non-rational resolved bases, unresolved bases, base sublimit `1`, nonpositive
  base sublimits, non-unit log arguments, and finite poles remain residual
- total-real continuous unary compositions already in the finite allowlist
  (`exp`, `sin`, `cos`, `sinh`, `cosh`, `tanh`, `atan`/`arctan`, `asinh`,
  `cbrt`, and `abs`) may evaluate when their argument has already resolved to a
  safe finite sublimit; this does not add support for discontinuous functions
  (`sign`, `floor`, `ceil`) or bypass the dedicated domain-partial checks;
  table-backed exact folds may reduce special total-real results such as
  `sin(pi/6)` to `1 / 2`, `cos(pi/3)` to `1 / 2`, `atan(1)`/`arctan(1)` to
  `pi/4`, and `atan(sqrt(3))`/`arctan(sqrt(3))` to `pi/3`
- orientation-sensitive quotients such as `limit(abs(x)/x, x, 0)` remain
  residual under the current two-sided finite-point policy, but must retain
  denominator definedness such as `x ≠ 0`; this avoids silently selecting a
  one-sided sign or rewriting to a discontinuous surrogate without an explicit
  direction policy
- explicit one-sided finite limits may evaluate narrow orientation and pole
  regimes when local polynomial sign evidence proves the side behavior:
  `limit(abs(x)/x, x, 0+) -> 1`, `limit(abs(x)/x, x, 0-) -> -1`,
  `limit(1/x, x, 0+) -> infinity`, and
  `limit(1/x, x, 0-) -> -infinity`; rational polynomial pole variants may also
  evaluate when exact local order and side-sign evidence prove the behavior,
  including shifted, scaled, and higher-order cases such as
  `limit(2/(x-1)^3, x, 1-) -> -infinity`; the public output must retain source
  definedness such as `x ≠ 0` or `x ≠ 1`. When removable local cancellation
  leaves other denominator factors in the source, those nonlocal pole
  conditions must remain visible too, for example
  `limit((x^2-1)/((x-1)*(x+3)), x, 1+) -> 1/2` must retain both `x ≠ 1` and
  `x ≠ -3`
- one-sided finite log endpoints may evaluate when local polynomial tail-sign
  evidence proves the log argument approaches `0` through positive values from
  the requested side: `limit(ln(x), x, 0+) -> -infinity`,
  `limit(log2(x), x, 0+) -> -infinity`, and
  `limit(log10(x), x, 0+) -> -infinity`; the same endpoint policy may use
  rational-polynomial arguments when the denominator is nonzero at the approach
  point and local order/sign evidence proves a positive zero-tail, for example
  `limit(ln((x-1)/(x+3)), x, 1+) -> -infinity`; reciprocal bases flip the
  infinity sign, including finite valid bases resolved from a variable base,
  for example `limit(log(1/2, x), x, 0+) -> infinity` and
  `limit(log(x-1/2, (x-1)/(x+3)), x, 1+) -> infinity`; the public output must
  retain positive-argument and valid-base domain requirements such as
  `x > 0`, `x < -3 or x > 1`, `x > 1/2`, and `x ≠ 3/2`; wrong-side paths such as
  `limit(ln(x), x, 0-)` and `limit(ln((x-1)/(x+3)), x, 1-)` remain residual,
  but now carry a `Limit Domain Path` warning when local polynomial or
  denominator-nonzero rational side evidence proves the requested path is
  outside the input domain. Variable bases approaching `1` are allowed only for
  explicit one-sided paths when local sign evidence proves which side of the
  unit base is used and the argument has a positive zero-tail; for example
  `limit(log(x, (x-1)/(x+3)), x, 1+) -> -infinity` while the wrong-side
  argument path remains residual with a domain-path warning. Rational variable
  bases may use the same unit-boundary policy when numerator-minus-denominator
  sign evidence is available; for example
  `limit(log((x+2)/(2*x+1), (x-1)/(x+3)), x, 1+) -> infinity` with readable
  real-domain requirements such as `x < -2 or x > -1/2` and
  `x < -3 or x > 1`, rather than exposing the internal `(base - 1) ≠ 0`
  witness
- one-sided finite square-root endpoints may evaluate to `0` when local
  polynomial tail-sign evidence proves the radicand approaches `0` through
  nonnegative values from the requested side: `limit(sqrt(x), x, 0+) -> 0`,
  `limit(sqrt(-x), x, 0-) -> 0`, and
  `limit(sqrt(x + 1), x, -1+) -> 0`; the same denominator-nonzero
  rational-polynomial positive zero-tail check may resolve cases such as
  `limit(sqrt((x-1)/(x+3)), x, 1+) -> 0`; the public output must retain
  nonnegative-radicand requirements such as `x ≥ 0` or
  `(x - 1)/(x + 3) ≥ 0` plus source denominator conditions; wrong-side paths
  such as `limit(sqrt(x), x, 0-)` and
  `limit(sqrt((x-1)/(x+3)), x, 1-)` remain residual with a
  `Limit Domain Path` warning when polynomial or denominator-nonzero rational
  side evidence proves the requested path violates the nonnegative-radicand
  domain; non-polynomial endpoint shapes such as `sqrt(abs(x))` remain residual
  until a separate orientation policy is promoted
- `ln(g(x))`, `log2(g(x))`, `log10(g(x))`, and `sqrt(g(x))` may evaluate when
  `g(x)` has already resolved to a safe finite sublimit that is explicitly
  numeric positive or is proven strictly positive by the existing sign prover;
  exact rational-power folds may reduce values such as `log2(8) -> 3` and
  `log10(100) -> 2`; domain-bearing sublimits must keep their public real-domain
  requirements visible, for example `limit(ln((x+2)/(x+3)), x, 0) -> ln(2/3)`
  requires `x < -3 or x > -2`; under the two-sided finite policy, zero,
  negative, or unproven positive sublimits remain residual, including
  endpoint-looking cases such as `sqrt(abs(x))` at `x -> 0`
- `asin(g(x))`/`arcsin(g(x))`, `acos(g(x))`/`arccos(g(x))`, `atanh(g(x))`,
  and `acosh(g(x))` may evaluate only when `g(x)` has already resolved to a
  numeric rational sublimit strictly inside the real domain interior; exact
  table-backed folds may reduce values such as `asin(0)`/`arcsin(0)` to `0`,
  `acos(0)`/`arccos(0)` to `pi/2`, `asin(1/2)`/`arcsin(1/2)` to `pi/6`,
  `acos(1/2)`/`arccos(1/2)` to `pi/3`, and `atanh(0)` to `0`, while
  endpoint-looking cases such as `asin(x)` at `x -> 1`, `atanh(x)` at
  `x -> 1`, and one-sided-only lower-bound cases such as `acosh(x)` at
  `x -> 1` remain residual
- finite two-sided upper-bound inverse-trig endpoint paths may evaluate
  `asin(p(x))`/`arcsin(p(x))` and `acos(p(x))`/`arccos(p(x))` when `p(x)` is
  polynomial, `p(a) = 1`, and local polynomial side evidence proves
  `1 - p(x) > 0` from both sides of the approach point, for example
  `limit(acos(1-x^2), x, 0) -> 0` and
  `limit(asin(1-x^2), x, 0) -> pi/2`; one-sided-only or empty-punctured-domain
  shapes such as `acos(x)`, `acos(1-x^3)`, or `acos(1+x^2)` at the upper
  endpoint remain residual
- finite two-sided lower-bound inverse-trig endpoint paths may evaluate
  `asin(p(x))`/`arcsin(p(x))` and `acos(p(x))`/`arccos(p(x))` when `p(x)` is
  polynomial, `p(a) = -1`, and local polynomial side evidence proves
  `p(x) + 1 > 0` from both sides of the approach point, for example
  `limit(acos(-1+x^2), x, 0) -> pi` and
  `limit(asin(-1+x^2), x, 0) -> -pi/2`; one-sided-only or
  empty-punctured-domain shapes such as `acos(x)`, `acos(-1+x^3)`, or
  `acos(-1-x^2)` at the lower endpoint remain residual
- finite one-sided inverse-trig endpoint paths may evaluate
  `asin(p(x))`/`arcsin(p(x))` and `acos(p(x))`/`arccos(p(x))` when `p(x)` is
  polynomial, `p(a) = 1` and `1 - p(x) > 0` from the requested side, or
  `p(a) = -1` and `p(x) + 1 > 0` from the requested side, for example
  `limit(acos(x), x, 1-) -> 0`,
  `limit(asin(x), x, 1-) -> pi/2`,
  `limit(acos(x), x, -1+) -> pi`, and
  `limit(asin(x), x, -1+) -> -pi/2`; wrong-side paths such as
  `limit(acos(x), x, 1+)`, empty-domain gaps such as
  `limit(acos(1+x^2), x, 0+)`, and non-polynomial endpoint arguments remain
  residual while preserving the interval requirement; when the requested
  one-sided path is proven outside the interval domain by local polynomial
  evidence, the residual must carry a `Limit Domain Path` warning
- finite one-sided `atanh(p(x))` endpoint paths may evaluate when `p(x)` is
  polynomial, `p(a) = 1` and `1 - p(x) > 0` from the requested side, or
  `p(a) = -1` and `p(x) + 1 > 0` from the requested side; the upper endpoint
  approached from inside the open interval resolves to `infinity`, for example
  `limit(atanh(x), x, 1-) -> infinity`, and the lower endpoint approached
  from inside resolves to `-infinity`, for example
  `limit(atanh(x), x, -1+) -> -infinity`; wrong-side paths such as
  `limit(atanh(x), x, 1+)`, empty-domain gaps such as
  `limit(atanh(1+x^2), x, 0+)`, and non-polynomial endpoint arguments remain
  residual while preserving the strict open-interval requirement; when the
  requested one-sided path is proven outside the open interval by local
  polynomial evidence, the residual must carry a `Limit Domain Path` warning
- finite two-sided lower-bound inverse-hyperbolic endpoint paths may evaluate
  `acosh(p(x)) -> 0` when `p(x)` is polynomial, `p(a) = 1`, and local
  polynomial side evidence proves `p(x) - 1 > 0` from both sides of the
  approach point, for example `limit(acosh(1+x^2), x, 0) -> 0`; odd-gap or
  negative-gap endpoint shapes such as `acosh(1+x^3)` or `acosh(1-x^2)` at
  `x -> 0` remain residual because the real-domain path is one-sided or empty
  in a punctured neighborhood
- finite one-sided lower-bound inverse-hyperbolic endpoint paths may evaluate
  `acosh(p(x)) -> 0` when `p(x)` is polynomial, `p(a) = 1`, and local
  polynomial side evidence proves the requested approach enters the real domain
  `p(x) > 1`, for example `limit(acosh(x), x, 1+) -> 0` and
  `limit(acosh(2-x), x, 1-) -> 0`; wrong-side paths such as
  `limit(acosh(x), x, 1-)` remain residual, preserve the required condition
  `x ≥ 1`, and carry an explicit `Limit Domain Path` warning, while
  non-polynomial endpoint arguments remain residual until a separate policy is
  promoted
- `tan(g(x))`, `sec(g(x))`, `csc(g(x))`, and `cot(g(x))` may evaluate when
  `g(x)` has already resolved to exact numeric zero where defined, or to a
  recognized special angle whose table value is defined, such as `tan(pi/4) ->
  1`, `sec(pi/3) -> 2`, `csc(pi/6) -> 2`, or `cot(pi/4) -> 1`;
  table-undefined pole cases such as `tan(pi/2)`, `sec(pi/2)`, `csc(pi)`, and
  `cot(pi)` remain residual, as do arbitrary nonzero rational arguments
- after those finite composition checks succeed, local exact presentation folds
  may reduce `ln(exp(g))` to `g`, `exp(ln(g))` to `g` only when `g` is
  explicitly or structurally proven strictly positive, and `abs(g)`/`abs(-g)`
  to `g` only when `g` is explicitly or structurally proven strictly positive;
  these are finite-limit result folds, not global simplification or
  pre-simplification rules
- binary `log(b(x), g(x))` may evaluate only when `b(x)` has already resolved
  to an explicit rational finite sublimit with `b > 0` and `b != 1`, and
  `g(x)` has already resolved to a safe strictly positive finite sublimit;
  exact rational-power folds may reduce values such as `log(2, 8) -> 3`,
  `log(1/2, 8) -> -3`, `log(4, 8) -> 3/2`, and `log(27, 9) -> 2/3`; invalid
  base sublimits, base sublimit `1`, non-rational or unresolved base sublimits,
  zero arguments, negative arguments, or unproven positive arguments remain
  residual
- integer powers `g(x)^n` may evaluate when `g(x)` has already resolved to a
  safe finite sublimit; positive integer exponents are total over real finite
  base sublimits, while zero and negative integer exponents require the base
  sublimit to be explicitly or structurally proven nonzero so the engine does
  not promote `0^0` or division by zero; exact rational square-root sublimits
  such as `sqrt(q)` with `q > 0` may present even integer powers as `q^k` or
  `1/q^k`, while odd powers and broader root algebra remain in the explicit
  root/power form; exact rational cube-root sublimits such as `cbrt(q)` may
  present integer powers that are multiples of three as `q^k` or `1/q^k`, with
  reciprocal forms allowed only when `q != 0`; non-multiple powers remain in
  explicit root/power form
- real one-third powers, written as `(p(x))^(1/3)`, follow the same finite
  cube-root rule as `cbrt(p(x))`; this does not promote arbitrary fractional
  powers or even roots over negative values
- unresolved finite-point limits carry a warning that finite point limits are
  not supported safely yet
- unresolved finite-point residuals must not surface literal-impossible public
  requirements such as `0 ≠ 0` from a collapsed denominator witness; suppressing
  that display artifact does not prove or promote the limit, and the residual
  result plus finite-limit warning remains the contract
- no finite-point answer is invented
- no side, path, continuity, branch, or domain assumption is inferred
- the CLI subcommand `limit --to` remains scoped to `infinity` and `-infinity`
  until it has a dedicated finite-point API contract

This boundary is intentional. A future Limits V2 implementation must still
define:

- point representation and side policy
- domain/path conditions at the approach point
- residual rendering for unresolved finite limits
- contract tests before any evaluation rule is promoted

---

## 6. PR Checklist

Before merging changes to `limits/presimplify.rs`:

- [ ] **Allowlist check**: Is the new transform in the allowlist table (§2.2)?
- [ ] **Domain neutral**: Does it assume any domain constraints (e.g., `x ≠ 0`)?
  - If yes → **REJECT** or move to a different mode
- [ ] **Contract test**: Is there a new contract test if behavior changes?
- [ ] **Lint passes**: Does `./scripts/lint_limit_presimplify.sh` pass?
- [ ] **Denylist untouched**: Was the lint denylist modified to "make it pass"?
  - If yes → **REJECT** (unless documented escalation)
- [ ] **LIMITS_POLICY.md updated**: Is this doc updated if allowlist/denylist changes?
