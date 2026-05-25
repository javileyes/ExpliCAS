# Calculus Engine Strategy

This document is part of the engine auto-improvement loop. It defines how the
campaign should turn the mature pre-calculus core and the existing calculus
verticals into a serious real-domain calculus surface.

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

The project has now accumulated enough green `diff`, `limit`, and `integrate`
vertical slices that the priority should move beyond adding one more narrow
case at a time.

Therefore the strategy is now:

- generalize calculus from the supported slices already present
- prefer shared real-domain capability over isolated new cases
- build explicit calculus coverage matrices for `diff`, `limit`, and
  `integrate`
- keep improving pre-calculus in parallel
- treat calculus failures as structured feedback for simplification,
  equivalence, domains, and didactic trace quality

This is not a license to implement a universal CAS or open broad unbounded
search. It is a shift in default ROI: when several narrow calculus cases already
work, the next cycle should ask what shared rule, domain policy, presentation
layer, verification path, or didactic trace would make the family general.

## Core Principle

Calculus is not a separate feature bolted on top of the symbolic engine.

It is a pressure generator for the pre-calculus core:

- differentiation stresses algebraic simplification, product factoring, powers,
  trigonometric identities, logs, and chain-rule step quality
- limits stress safe pre-simplification, cancellation policy, asymptotic forms,
  domain/range assumptions, and infinity semantics
- integration stresses pattern recognition, substitution traces, constant
  factors, inverse differentiation, and conservative non-goal handling

Every calculus improvement should answer five questions before promotion:

1. What user-visible calculus capability improves?
2. What pre-calculus capability does it reuse, expose, or harden?
3. What domain, branch, or constant-of-integration assumption is introduced or
   deliberately avoided?
4. What didactic trace should the user see, and where must the engine avoid a
   magical jump?
5. What final presentation form should the user see, and is that form distinct
   from the internal canonical form used for matching, equivalence, and runtime?

If a calculus candidate does not improve a real calculus surface or produce
useful pressure on the pre-calculus core, it is probably not high ROI.

## Current Priority: Generalize Calculus

The near-term calculus campaign should optimize for generalization, not
isolated coverage.

Prefer candidates that do at least one of these:

- turn several existing derivative examples into one reusable derivative family
  with consistent domain and step behavior
- convert repeated integration table or substitution patterns into a shared
  recognizer, verifier, or didactic trace
- extend finite or infinite limits by a documented policy matrix rather than by
  a single answer shortcut
- add a calculus coverage matrix that makes supported, residual, and rejected
  regimes explicit
- improve post-calculus presentation for a whole result family while preserving
  canonical internal forms and required conditions
- expose a reusable pre-calculus normalization or equivalence gap that blocks
  multiple calculus results

Reject or defer candidates that only add one more syntactic variant unless the
variant reveals a new domain regime, rule interaction, didactic gap, runtime
cliff, or public behavior class.

## Block-Based Calculus Maturity Plan

The calculus campaign should now be planned by maturity blocks, not by an
unbounded list of examples. Each auto-improvement cycle should select one small
checkbox inside one block, then validate it with the normal ROI and guardrail
process.

A calculus block is not mature just because several cases pass. Treat a block as
retained only when its representative capability has:

- correct public behavior or a safe residual/undefined outcome
- explicit real-domain requirements, including poles, positive arguments,
  branch-sensitive intervals, infinity assumptions, or integration constants
  when applicable
- explainable steps or a documented reason why the command remains residual
- at least one minimal support-matrix representative for the new public regime
- verification by equivalence, limit policy, or differentiation of the
  antiderivative when applicable
- green `fast`, `guardrail`, and, for broad core changes, `pressure`

Use these blocks as the active checklist:

1. **Public calculus contracts and matrices**
   - keep `diff`, `limit`, and `integrate` matrices organized by command,
     family, argument regime, domain regime, trace regime, presentation regime,
     and verification/residual policy
   - make support, residual, undefined, and deliberately rejected regimes
     visible
   - avoid promoting rows that only differ syntactically from an existing cell

2. **Real-domain differentiation maturity**
   - complete reusable handling for elementary functions, products, quotients,
     chains, general powers, roots, absolute values, logs, trig, hyperbolic, and
     inverse families
   - preserve required conditions for log arguments, denominator poles, root
     domains, reciprocal trig poles, and branch-sensitive inverse functions
   - keep rule traces explicit enough to show `u`, `du`, product/quotient
     factors, and final post-calculus presentation when nontrivial

3. **Real-domain limits maturity**
   - generalize safe direct substitution, removable rational cancellation,
     finite point policies, infinity policies, and structural positivity
   - represent endpoint, one-sided, discontinuity, and empty-domain boundaries
     conservatively before returning a value
   - prefer a clear residual with domain hints over speculative continuity or
     side-limit behavior

4. **Base integration maturity**
   - keep linears, constants, powers, exponentials, direct trig/hyperbolic
     tables, `1/x`, and log-derivative forms verified and explainable
   - preserve `ln(|f|)` versus positive-log choices through explicit real-domain
     evidence
   - verify promoted antiderivatives by differentiating back to the integrand

5. **Generalized substitution**
   - detect `u` and `du` across affine, polynomial, scaled, root, and bounded
     rational inner forms only when the evidence is explicit
   - show the substitution evidence in steps instead of silently table-matching
   - keep unsupported substitutions residual rather than opening broad search

6. **Rational integration**
   - add polynomial division, proper rational forms, partial fractions over real
     linear factors, and irreducible quadratic regimes only when the domain and
     verification story is clear
   - distinguish logarithmic and arctangent primitives with explicit denominator
     and positivity evidence

7. **Trig and hyperbolic integration**
   - consolidate `tan/sec`, `cot/csc`, `sinh/cosh/tanh`, and reciprocal-square
     families through shared pole, sign, and presentation policies
   - use identities only under bounded route order and guardrail-visible
     runtime checks

8. **Radical and inverse-family calculus**
   - generalize root/inverse-trig/inverse-hyperbolic forms with interval
     orientation, symbolic parameter conditions, denominator scale, and external
     scale handled explicitly
   - preserve the selected primitive family when several antiderivatives are
     derivative-equivalent but domain evidence points to one form

9. **Residuals and non-goal policy**
   - make unsupported calculus behavior stable, educational, and domain-aware
   - classify residuals by missing method, empty real domain, unsafe branch,
     unverified antiderivative, or intentionally unsupported search
   - never return an unverified antiderivative or a limit value that depends on
     unrepresented assumptions

10. **Didactic trace maturity**
    - ensure promoted public calculus behavior has visible rule names, useful
      highlights, and no magical source-to-target jump
    - add derive shadow cases only when calculus exposes a reusable algebraic
      transition a user might ask to explain

11. **Architecture, observability, and runtime**
    - separate detection, domain reasoning, transformation, verification, and
      rendering when a file or route family becomes too opaque to generalize
    - improve observability before broadening matchers if the next calculus move
      cannot be localized safely
    - keep embedded and pressure lanes green; calculus maturity does not justify
      broad hot-path regressions

Selection rule:

- every calculus cycle should name its `calculus_maturity_block`
- every promoted calculus row should state which block gate it advances
- if a candidate does not advance a block gate, treat it as discovery pressure
  or reject it as a near-duplicate
- in ties, choose the candidate that moves an earlier incomplete block only
  when it also has high retention probability and does not hide a reusable
  pre-calculus blocker

### Family Sweep And Consolidation Policy

Recent cycles showed that calculus presentation work can accidentally repeat the
same local improvement across sibling inverse families. The loop should treat
that repetition as evidence for a shared policy, not as a reason to keep adding
near-duplicate variants.

Before accepting a calculus presentation or derivative-family candidate, run a
bounded sibling sweep over the nearest mathematical family cluster. This is not
open-ended search; it is a short probe set chosen from the same rule shape.

For root and scaled-root derivative families, the minimum sweep should usually
include:

- the direct family and its dual orientation, such as `arcsin/arccos` or
  `arctan/acot`
- the corresponding hyperbolic family when it shares the same argument parser,
  such as `asinh/atanh/acosh`
- positive scale, negative scale, and external constant scale
- exact-square content under the root, for example `sqrt(4*x)/3`
- unit and negative-unit scale cases, for example `sqrt(x)` and `-sqrt(x)`
- required-condition preservation and ordinary calculus trace preservation

If one cycle fixes a family and a later cycle finds the same structural pattern
in a sibling family, the following cycle should prefer consolidation over
another syntactic variant. Consolidation can mean extracting a small helper,
documenting a shared matrix policy, or promoting one minimal support-matrix row
that represents the family cluster.

A calculus candidate should remain local only when:

- the sibling sweep shows no analogous gap
- the sibling gap has different domain or branch semantics
- consolidation would change route ordering or broaden matcher traffic without
  measurable value
- the local fix is needed to keep `failed = 0` before a safer consolidation can
  be attempted

This policy is especially important for post-calculus presentation. Presentation
helpers often encode domain and orientation assumptions indirectly, so a compact
form must be checked against sign/orientation siblings before promotion.

## Investment Class

An auto-improvement cycle may choose `calculus` as its primary investment class
when the retained value is public calculus capability:

- `diff` / symbolic differentiation
- `limit` / conservative limit solving
- `integrate` / conservative table or substitution integration
- generalization of an existing `diff`, `limit`, or `integrate` family across
  polynomial degree, affine/polynomial arguments, orientation, sign, and
  domain regimes when the behavior remains explainable and testable
- post-calculus presentation for a `diff`, `limit`, or `integrate` result when
  the mathematical capability already exists but the public form is needlessly
  awkward
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

### Phase 1. Differentiation Generalization

Prioritize differentiation first because it is the safest high-ROI calculus
surface and it feeds the simplifier constantly.

The current goal is no longer just to add early derivative examples. The goal is
to make differentiation feel systematic over the real-domain elementary
surface.

Good generalization targets:

- polynomial and rational derivatives
- product, quotient, and chain rule traces
- `exp`, `ln`, trig, and inverse-trig derivatives where the domain policy is
  clear
- affine and polynomial inner arguments where the chain factor is explicit
- sign/orientation variants and reciprocal/root presentations for the same
  mathematical family
- simplification of derivative outputs without hiding rule applications
- didactic substeps for nontrivial compositions

Retention should require both:

- the derivative is mathematically correct
- the resulting expression and steps are presentable through existing symbolic
  simplification and didactic machinery

### Phase 2. Limits Generalization

Limits should remain conservative.

Good generalization targets:

- polynomial and rational limits at infinity
- safe structural pre-simplification
- finite point limits under an explicit policy matrix
- one-sided limits only after direction and domain behavior are represented
- asymptotic dominance families with documented residual boundaries
- documented residuals when the engine cannot solve safely

Do not relax the limits policy to chase isolated successes. Follow
[LIMITS_POLICY.md](/Users/javiergimenezmoya/developer/math/docs/LIMITS_POLICY.md)
for allowlist/denylist discipline. If a broader limits move requires new
continuity, cancellation, side-limit, or infinity semantics, write that policy
first and make unsupported residuals part of the contract.

### Phase 3. Integration Generalization

Integration should remain narrower than differentiation, but it should now
generalize the supported table, substitution, and verification patterns instead
of accumulating unrelated primitives.

Good generalization targets:

- powers of `x` excluding singular exponent cases
- constant multiples and sums
- direct table forms for `exp`, `sin`, `cos`, and `1/x` with domain policy
- simple linear substitution where the substitution trace is explicit
- verification by differentiating supported antiderivatives when safe
- polynomial-derivative substitution families where the `u` and `du` evidence
  can be shown to the user
- repeated table families that can share one domain and constant policy
- integration-by-parts families only when the residual path is bounded and the
  primitive is verified

Non-goals:

- no general integration prover
- no broad search over substitutions
- no fake antiderivatives without verification or clear unsupported fallback
- no hiding the integration constant policy

#### Antiderivative Family Selection

Integration contracts should remember that a real-domain antiderivative is not
canonical in the same sense as a simplified expression. Two valid primitives can
have identical derivatives while differing by a constant on the active domain
component.

This matters for inverse-hyperbolic reciprocal-root families. For example, a
kernel may admit both an `atanh(sqrt(c/g))` shaped primitive and an
`asinh(sqrt(c/(g-c)))` shaped primitive after a domain-preserving rewrite. Those
forms can be equally correct even though they are not structurally identical.

Policy:

- verify supported antiderivatives by differentiating the chosen primitive back
  to the integrand whenever the family is promoted
- keep required real-domain conditions visible; do not erase the condition that
  makes the selected primitive valid
- prefer the family whose denominator or positivity condition is directly
  witnessed by the integrand when the kernel is ambiguous
- preserve an unambiguous inverse-differentiation target only when the input
  syntactically carries enough evidence for that family
- do not rewrite a valid primitive from `asinh` to `atanh`, or the reverse,
  purely for aesthetics or string similarity
- promote ambiguous cases only when the contract states whether the behavior
  under test is derivative verification, domain retention, or deliberate family
  selection

This policy is intentionally narrower than a universal canonical primitive
policy. It is a guardrail for integration generalization, not a promise that the
engine will choose the same primitive family as another CAS.

### Phase 4. Post-Calculus Presentation And Didactic Quality

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

Post-calculus presentation is a separate concern from canonical simplification.

The internal canonical form should remain optimized for:

- matching
- equivalence
- deterministic normal-form convergence
- runtime and budget behavior

The public post-calculus form should be optimized for:

- mathematical readability
- compactness that preserves meaningful structure
- domain-safe notation choices
- compatibility with didactic steps and highlights

Do not make global simplification rules just to make one calculus answer look
prettier. A presentation improvement is valid only when it is local to a
calculus result, preserves semantic conditions, and can be verified by
equivalence or by differentiating an antiderivative when the source command is
`integrate`.

Good presentation targets:

- reciprocal fractional powers after differentiation:
  - internal acceptable form: `x^(-1/2)/(2*x + 2)`
  - preferred post-diff form: `1/(2*sqrt(x)*(x + 1))`
- denominator content that should remain factored when it improves readability:
  - prefer `2*(x + 1)` over `2*x + 2` in a final reciprocal denominator when
    the factored form is compact and domain-equivalent
- square-root notation for simple real-domain half powers when the required
  conditions are explicit
- preservation of compact antiderivative denominators after verification by
  differentiation
- undefined calculus outputs should not render literal-impossible requirements
  such as `0 ≠ 0` from a collapsed denominator witness; the public result can
  remain `undefined` without pretending there is a satisfiable requirement

Presentation work must still keep required conditions visible. For example,
`diff(arctan(sqrt(x)), x)` may prefer
`1/(2*sqrt(x)*(x + 1))`, but the derivative path must keep the real-domain
condition needed for the square-root derivative, such as `x > 0`.

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
              + presentation_value
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
- `presentation_value`
  - the final public calculus result becomes materially easier to read without
    changing the internal canonical route, hiding domain conditions, or adding
    broad simplifier cost
- `corpus_reuse_value`
  - the retained case can feed both calculus and pre-calculus guardrails
- `runtime_risk`
  - broad cost added to common simplification paths
- `unsoundness_risk`
  - risk of returning a calculus result outside its valid domain or branch
- `complexity_cost`
  - implementation and maintenance cost

## Corpus Policy

Calculus corpus work should now be matrix-oriented.

Vertical slices remain useful as discovery and first support probes, but the
promotion target should be a small coverage matrix that records:

- command: `diff`, `limit`, or `integrate`
- family: polynomial, rational, elementary, inverse, trig, hyperbolic, root,
  log, product, quotient, substitution, or by-parts
- argument regime: variable, affine, polynomial, nested elementary, or rejected
  unsupported form
- domain regime: unconditional, required condition, branch-sensitive,
  infinity-sensitive, integration-constant-sensitive, or residual
- trace regime: direct rule, product/quotient/chain rule, substitution,
  verification by differentiation, pre-simplification, or residual explanation
- presentation regime: canonical result, post-calculus presentation, or
  deliberately deferred display cleanup

Preferred promotion path:

1. start with unit tests for the calculus family or shared helper
2. add a CLI/API contract test if public behavior changes
3. add a didactic/highlight test when the step trace is the point
4. add a metamorphic or pressure case when the result stresses simplification
5. promote a minimal representative matrix row to live guardrails only after the
   family is stable

Good calculus corpus rows are small but structurally informative:

- `diff((x^2+1)^3, x)` exercises chain rule and power simplification
- `diff(x*ln(x), x)` exercises product rule and log domain awareness
- `diff(arctan(sqrt(x)), x)` exercises post-diff reciprocal-root
  presentation and domain retention
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
- presentation contract tests when only the final calculus display form changes
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
- changes the internal canonical form only to satisfy final display taste
- implements a one-off calculus shortcut that cannot be explained or tested as
  a family
- hides unsupported integration or limit work behind a generic simplify result
- produces a correct answer with misleading steps or broken highlights
- produces a prettier answer by expanding, cancelling, or refactoring across a
  domain boundary that is not represented in required conditions
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

The highest-ROI near-term direction is calculus generalization from existing
green slices:

- build a calculus support matrix for `diff`, `limit`, and `integrate`
- prioritize differentiation as the first generalization surface
- make product/quotient/chain-rule traces visible and consistent
- generalize integration only through table, substitution, by-parts, and
  verification families with explicit domain and constant policy
- generalize limits through policy-backed finite, infinity, side, and residual
  regimes
- feed reusable algebraic gaps back into pre-calculus coverage rather than
  adding calculus-only workarounds
- add calculus didactic audits for rule traces, highlights, residual
  explanations, and post-calculus presentation

The campaign should stop spending routine cycles on more isolated calculus
examples unless they reveal a new reusable axis. The default calculus cycle
should now ask: what makes this whole family coherent, explainable, domain-safe,
and verifiable?
