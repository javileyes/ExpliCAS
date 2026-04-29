# Calculus Engine Strategy

This document is part of the engine auto-improvement loop. It defines when and
how the campaign should expand from the mature pre-calculus core into calculus.

## Position

The current pre-calculus engine is a professional foundation for curated and
broad product-like traffic, but it is not a complete universal CAS.

That distinction matters:

- the simplification/equivalence core is strong enough to support calculus work
- normal-form convergence, quotient cancellation, domain frontiers, and didactic
  traces still need continued hardening
- waiting for "complete pre-calculus" before starting calculus would postpone a
  valuable source of pressure on the core
- starting calculus without discipline would bypass the same simplifier, domain,
  and explanation machinery that makes the engine trustworthy

Therefore the strategy is:

- start calculus now
- do it as bounded vertical slices
- keep improving pre-calculus in parallel
- treat calculus failures as structured feedback for simplification,
  equivalence, domains, and didactic trace quality

## Core Principle

Calculus is not a separate feature bolted on top of the symbolic engine.

It is a pressure generator for the pre-calculus core:

- differentiation stresses algebraic simplification, product factoring, powers,
  trigonometric identities, logs, and chain-rule step quality
- limits stress safe pre-simplification, cancellation policy, asymptotic forms,
  domain/range assumptions, and infinity semantics
- integration stresses pattern recognition, substitution traces, constant
  factors, inverse differentiation, and conservative non-goal handling

Every calculus improvement should answer four questions before promotion:

1. What user-visible calculus capability improves?
2. What pre-calculus capability does it reuse, expose, or harden?
3. What domain, branch, or constant-of-integration assumption is introduced or
   deliberately avoided?
4. What didactic trace should the user see, and where must the engine avoid a
   magical jump?

If a calculus candidate does not improve a real calculus surface or produce
useful pressure on the pre-calculus core, it is probably not high ROI.

## Investment Class

An auto-improvement cycle may choose `calculus` as its primary investment class
when the retained value is public calculus capability:

- `diff` / symbolic differentiation
- `limit` / conservative limit solving
- `integrate` / conservative table or substitution integration
- later: series, asymptotics, and related calculus commands

Use another class when the retained value is not calculus itself:

- use `coverage` when a calculus-discovered gap is really a reusable
  simplification or equivalence family
- use `robustness` when the primary retained value is avoiding timeout,
  overflow, panic, or nontermination on calculus-shaped inputs
- use `observability` when the next profitable calculus move is unclear and the
  needed work is measurement, diagnostics, or corpus visibility
- use `runtime` when the calculus path exposes broad hot-path cost that must be
  reduced before capability can grow safely
- use `combination` only for a documented local-win/global-fail calculus or
  pre-calculus hypothesis that now has a complementary fix

## Scope Order

### Phase 0. Baseline And Inventory

Keep an explicit inventory of what calculus commands already support, what they
delegate to the pre-calculus engine, and where the step trace becomes opaque.

Useful baseline questions:

- which derivative families are supported directly?
- which derivative results require simplification to become presentable?
- which integral families are table-supported versus unsupported?
- which limit families are intentionally conservative?
- which CLI/API contracts already expose calculus steps?

### Phase 1. Differentiation Vertical Slice

Prioritize differentiation first because it is the safest high-ROI calculus
surface and it feeds the simplifier constantly.

Good early targets:

- polynomial and rational derivatives
- product, quotient, and chain rule traces
- `exp`, `ln`, trig, and inverse-trig derivatives where the domain policy is
  clear
- simplification of derivative outputs without hiding rule applications
- didactic substeps for nontrivial compositions

Retention should require both:

- the derivative is mathematically correct
- the resulting expression and steps are presentable through existing symbolic
  simplification and didactic machinery

### Phase 2. Limits Vertical Slice

Limits should remain conservative.

Good early targets:

- polynomial and rational limits at infinity
- safe structural pre-simplification
- simple finite point limits only when domain assumptions are explicit
- documented residuals when the engine cannot solve safely

Do not relax the limits policy to chase isolated successes. Follow
[LIMITS_POLICY.md](/Users/javiergimenezmoya/developer/math/docs/LIMITS_POLICY.md)
for allowlist/denylist discipline.

### Phase 3. Integration Vertical Slice

Integration should start narrower than differentiation.

Good early targets:

- powers of `x` excluding singular exponent cases
- constant multiples and sums
- direct table forms for `exp`, `sin`, `cos`, and `1/x` with domain policy
- simple linear substitution where the substitution trace is explicit
- verification by differentiating supported antiderivatives when safe

Non-goals:

- no general integration prover
- no broad search over substitutions
- no fake antiderivatives without verification or clear unsupported fallback
- no hiding the integration constant policy

### Phase 4. Calculus And Didactic Quality

Calculus commands must not become "answer-only" shortcuts when the rest of the
engine is becoming increasingly explainable.

For nontrivial calculus results, prefer visible steps such as:

- apply product rule
- apply chain rule
- simplify derivative result
- apply supported integral table row
- verify antiderivative by differentiation when used as a guardrail
- apply conservative limit pre-simplification

Avoid noisy traces, but also avoid magical one-step transformations when the
user would naturally expect substeps.

### Phase 5. Series And Asymptotics Later

Series and asymptotics should wait until differentiation, limits, and the
domain policy are more stable. They are high value, but they multiply branch,
budget, and explanation complexity.

## Calculus ROI

For calculus candidates, extend the usual ROI estimate with:

```text
calculus_roi ~= calculus_value
              + precalculus_reuse_value
              + domain_safety_value
              + didactic_value
              + corpus_reuse_value
              - runtime_risk
              - unsoundness_risk
              - complexity_cost
```

Interpretation:

- `calculus_value`
  - new or stronger public behavior in `diff`, `limit`, or `integrate`
- `precalculus_reuse_value`
  - simplifier/equivalence/domain machinery improved or exercised by the change
- `domain_safety_value`
  - the change preserves or clarifies assumptions instead of silently erasing
    them
- `didactic_value`
  - the trace becomes more teachable, less magical, or easier to audit
- `corpus_reuse_value`
  - the retained case can feed both calculus and pre-calculus guardrails
- `runtime_risk`
  - broad cost added to common simplification paths
- `unsoundness_risk`
  - risk of returning a calculus result outside its valid domain or branch
- `complexity_cost`
  - implementation and maintenance cost

## Corpus Policy

Calculus corpus work should be vertical-slice oriented.

Preferred promotion path:

1. start with unit tests for the calculus family
2. add a CLI/API contract test if public behavior changes
3. add a didactic/highlight test when the step trace is the point
4. add a metamorphic or pressure case when the result stresses simplification
5. promote a minimal representative to live guardrails only after the family is
   stable

Good calculus corpus rows are small but structurally informative:

- `diff((x^2+1)^3, x)` exercises chain rule and power simplification
- `diff(x*ln(x), x)` exercises product rule and log domain awareness
- `limit((x^2-1)/(x-1), x, 1)` exercises cancellation and finite point policy
- `integrate(2*x*exp(x^2), x)` exercises substitution only if that route is
  explicitly supported

Large composed calculus expressions are useful as discovery pressure, not as
default live guardrail rows.

## Validation Policy

Use the narrowest validation that can prove retention.

Typical calculus validation:

- touched unit tests for differentiation, limits, or integration
- CLI/API contract tests when command behavior changes
- didactic audit or snapshot tests when steps/highlights change
- `make engine-fast` when the change interacts with simplification
- `make engine-scorecard` when a public calculus change depends on broad
  pre-calculus behavior
- `make engine-scorecard-pressure` when the change touches normal forms,
  orchestration, deep simplification traffic, or composed expressions
- `make ci` for broad public behavior changes, cross-crate wiring, or release
  closure

Do not treat a successful calculus unit test as sufficient when the change adds
new broad simplification behavior.

## Guardrails

Reject or defer a calculus candidate when it:

- returns a result by assuming a hidden domain condition
- broadens the simplifier in a way that slows embedded traffic without a clear
  retained win
- implements a one-off calculus shortcut that cannot be explained or tested as
  a family
- hides unsupported integration or limit work behind a generic simplify result
- produces a correct answer with misleading steps or broken highlights
- requires unbounded search, broad pattern matching, or non-deterministic route
  choice

Prefer recording a clear unsupported residual over returning a speculative
calculus result.

## Relationship To `derive`

The `derive` command is not the derivative command.

However, the same planning and didactic standards apply:

- calculus traces should learn from `derive` path quality, target awareness, and
  anti-magical-step policy
- successful calculus output simplification may reveal algebraic transformations
  that deserve a `derive` shadow case
- an unsupported `derive` bridge may reveal a reusable simplification capability
  needed by calculus results

Do not add derive rows for every calculus example. Add them only when the
calculus work exposes a reusable target-form algebraic transformation that a
user would plausibly ask `derive` to explain.

## Recommended Next Direction

The highest-ROI near-term direction is a differentiation-first vertical slice:

- broaden derivative family support conservatively
- improve simplification of derivative outputs
- make product/quotient/chain-rule traces visible and not magical
- feed reusable algebraic gaps back into pre-calculus coverage
- keep limits conservative and integration table-driven until their policies are
  equally strong

This lets the project start building calculus now while continuing to improve
the pre-calculus heart of the engine.
