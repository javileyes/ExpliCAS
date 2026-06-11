# Calculus Engine Strategy

## North Star

The end goal of this track is one engine that is both:

- **universal**: systematic differential and integral calculus coverage over
  the real domain, not a curated demo set
- **educational**: every supported result can explain itself step by step,
  with honest domain conditions and honest residuals

The three maturity scopes used by the auto-improvement loop ("serious
educational", "generally mature", "CAS-style general integration") are stages
toward that single goal, not alternative end states. In particular, the hybrid
backend's `algorithmic_summary` trace level is transitional: a backend family
is not finished until it is either elevated to a step-by-step educational
derivation (see Phase 6, didactic elevation, in
GENERAL_INTEGRATION_BACKEND_ROADMAP.md) or explicitly documented as
summary-only with a reason.

The complex domain is deliberately deferred (see Deferred Horizons below), but
real-domain decisions should not paint it into a corner.

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
- treat architecture pressure as an active calculus risk when local helpers,
  presentation patches, domain-condition builders, and residual verifiers start
  repeating across nearby families
- keep improving pre-calculus in parallel
- treat calculus failures as structured feedback for simplification,
  equivalence, domains, and didactic trace quality

This is not a license to turn the current educational `integrate` route into
broad unbounded search. It is a shift in default ROI: when several narrow
calculus cases already work, the next cycle should ask what shared rule, domain
policy, presentation layer, verification path, or didactic trace would make the
family general.

The long-term direction is hybrid:

- keep a conservative, educational real-domain calculus surface as the public
  default
- add a separate algorithmic integration backend only behind an explicit
  boundary, with verification, domain/constant policy, and conservative
  fallback
- never let a broad integration heuristic silently bypass the educational
  route, required conditions, or didactic/residual policy

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

## Current Maturity Gap

The calculus surface is now credible for curated, real-domain educational
traffic, but it should not be treated as mature or general merely because the
current promoted matrices are green.

The practical status is:

- `diff` has a serious supported slice and is the closest command to systematic
  real-domain coverage, but it still needs broader uniformity across general
  powers, products, quotients, chains, absolute values, logs, trig,
  hyperbolic, inverse families, domain retention, and post-calculus
  presentation.
- `limit` is useful but intentionally conservative. It still needs more policy
  depth around one-sided behavior, endpoints, discontinuities, removable
  singularities, continuity evidence, infinity dominance, empty domains, and
  explanatory residuals.
- `integrate` has many verified table, substitution, rational, trig,
  hyperbolic, radical, and inverse-family slices, but it remains narrower than
  differentiation by design. Maturity means verified families with explicit
  domain and constant policy, not broad integration search.
- didactic quality is part of calculus maturity: nontrivial public results
  should show useful rule names, `u`/`du` evidence, domain or pole reasoning,
  final presentation choices, or a clear residual explanation.
- architecture remains a first-order risk. If detection, domain reasoning,
  transformation, verification, presentation, and steps keep being solved
  locally inside large files, the next capability cycle becomes less safe and
  less general.

Use these planning horizons as rough engineering heuristics, not commitments:

- a serious educational one-variable real-domain calculus surface likely needs
  dozens more retained ROI cycles, mostly expanding and consolidating existing
  families rather than adding isolated examples
- a mature elementary real-domain calculus surface likely needs on the order of
  one hundred or more retained cycles because the hard work is systematic
  coverage, domain policy, verification, didactic trace quality, and
  architecture
- a universal integration engine is not a bounded target for ordinary ROI
  cycles inside the existing educational route; if pursued, it should be a
  separate hybrid backend track that starts with architecture, verification,
  corpus, and policy boundaries before broad algorithms. Status (2026-06-10):
  this track exists — boundary, shared verifier, probe runner, observability
  lanes, and the first public family (Hermite positive quadratic) are live;
  see GENERAL_INTEGRATION_BACKEND_ROADMAP.md phase status.

This gap assessment should influence ROI selection. A new green row is valuable
only when it moves the engine toward a coherent block capability. If a proposed
row does not improve family generality, domain safety, verification, didactic
quality, or architecture, it should usually be rejected as another isolated
example.

## Hybrid General Integration Direction

The project should allow a future "route C": a conservative educational
calculus engine plus a more general algorithmic integration backend.
The detailed roadmap for that backend lives in
[GENERAL_INTEGRATION_BACKEND_ROADMAP.md](/Users/javiergimenezmoya/developer/math/docs/GENERAL_INTEGRATION_BACKEND_ROADMAP.md).

This does not mean replacing the current integrator with broad heuristic search.
It means creating a clear boundary where a general backend can be developed,
measured, and rejected safely when it cannot produce a verified real-domain
answer.

Required properties for this backend track:

- **Separate owner and entrypoint.** The algorithmic backend must not be hidden
  inside table lookup, post-calculus presentation, or residual cleanup.
- **Verification first.** A returned antiderivative must be checked by
  differentiating it back to the integrand when the family is promoted. If
  verification is unavailable or inconclusive, the educational route should
  keep a residual rather than trust the backend.
- **Domain and constant policy.** The backend must state whether its result is
  real-domain, principal-branch, component-local, or conditional. Integration
  constants and domain components must not be implicit.
- **Fallback discipline.** Unsupported or unverified outputs remain residuals.
  A more general backend is allowed to fail quietly; it is not allowed to invent
  public certainty.
- **Trace separation.** The educational route should keep detailed steps. The
  general backend may expose a summarized method trace, but must not pretend to
  have a full didactic derivation when it only has an algorithmic result.
- **Corpus separation.** General backend probes should start as stress or
  discovery cases. Promote to live/frozen only when they represent stable
  method families and have verification/domain contracts.
- **Runtime isolation.** Broad integration algorithms must not slow ordinary
  `integrate` traffic unless a measured capability gain justifies it and the
  relevant runtime guardrails stay green.

Early backend work should be classified by retained value:

- `observability` when mapping backend boundaries, method classes, or
  verification gaps
- `robustness` when preventing unbounded search, nontermination, or route churn
- `calculus` only when a verified algorithmic family becomes public behavior

Good initial backend milestones are:

1. define an integration method result type that can carry candidate primitive,
   method tag, assumptions, verification status, and residual reason
2. route one existing verified family through that boundary without changing
   public behavior
3. add discovery-only probes for algorithmic rational integration or simple
   heurisch-like substitutions, keeping unsupported cases residual
4. require `diff(candidate, x) ~ integrand` or a documented domain-aware
   verifier before promotion

Rejected backend moves:

- adding a broad substitution search directly to the current table integrator
- accepting unverified antiderivatives because they look plausible
- hiding branch or real-domain assumptions in display-only cleanup
- making the educational trace claim a derivation the backend did not produce
- slowing promoted calculus lanes for speculative coverage

## Current Risk: Architecture Pressure From Calculus Generalization

Recent retained and rejected calculus cycles show a new primary weakness: many
remaining failures are no longer missing formulas. They are pipeline-boundary
failures.

Signals that the next high-ROI move is architectural:

- a local presentation helper fixes one calculus row but changes stable sibling
  rows
- antiderivative verification enters `depth_overflow`, `cycle_detected`, or
  broad residual search even though the mathematical primitive is correct
- domain conditions are rebuilt locally, duplicated, ordered differently, or
  displayed through family-specific compaction paths
- the same detection or presentation pattern appears in adjacent inverse,
  root, trig, or hyperbolic families
- `crates/cas_engine/src/calculus_residual_support/mod.rs`,
  `symbolic_integration_support.rs`, or didactic step helpers absorb another
  route-specific shortcut without a clearer owner
- a candidate can be described more naturally as detection, domain reasoning,
  transformation, verification, rendering, or step construction than as a new
  calculus rule

When those signals appear, the next cycle should usually choose block 11
(`Architecture, observability, and runtime`) even if a narrow calculus case is
available. The goal is not a large rewrite. The retained move should be one
small boundary improvement that makes future calculus generalization safer.

Preferred architecture moves:

- extract one coherent route family while preserving behavior and call order
- isolate a family-owned domain-condition builder instead of adding another
  display-side condition patch
- isolate post-calculus presentation from internal canonical matching
- isolate derivative/antiderivative residual verification routes before
  broadening integration families
- move repeated didactic step construction into a shared helper only after the
  mathematical policy is identical
- add observability or a support-matrix classification when ownership is still
  unclear

Rejected architecture moves:

- broad rewrites of calculus routing without a measured blocker
- generic registries that hide rule priority
- abstractions that merge families with different domain, branch, sign,
  orientation, or presentation policy
- global simplifier changes whose only purpose is to make one calculus result
  prettier

Selection rule:

- after two nearby retained or rejected cycles expose the same pipeline shape,
  prefer extraction or consolidation before adding another local variant
- if a candidate requires a broad result-shape check after integration or
  differentiation, first look for a narrower source-side route boundary
- if verification is the blocker, fix the bounded verifier/residual path before
  promoting more primitives from the same family

## Block-Based Calculus Maturity Plan

The calculus campaign should now be planned by maturity blocks, not by an
unbounded list of examples. Each auto-improvement cycle should select one
concrete sub-gate (block bullet) inside one block, then validate it with the
normal ROI and guardrail process. The blocks are written as prose gates, not
checkboxes.

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

Use these blocks as the active plan:

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
    - prefer extraction/consolidation when repeated calculus cycles add local
      helpers for the same pipeline shape across sibling families
    - make post-calculus presentation and residual verification explicit
      boundaries before broadening formulas that depend on them
    - keep domain-condition builders owned by the family or policy that proves
      them, not by late display cleanup
    - improve observability before broadening matchers if the next calculus move
      cannot be localized safely
    - keep embedded and pressure lanes green; calculus maturity does not justify
      broad hot-path regressions
    - Progress (2026-06-10): two retained Phase 1 extractions
      (general_integration_backend module split; affine trig power residual
      family) — see ENGINE_COHESION_REFACTORING_STRATEGY.md Retained
      extractions; remaining interleaved zones in
      calculus_residual_support/mod.rs (~12.4k lines) keep this block open.

12. **Hybrid algorithmic integration backend**
    - introduce a separate backend boundary before adding general integration
      algorithms
    - carry method tags, assumptions, verification status, and residual reasons
      as structured data
    - start with behavior-preserving routing of existing verified families,
      then add discovery-only algorithmic probes
    - promote a backend family only when antiderivative verification, real-domain
      policy, constant policy, runtime, and fallback behavior are explicit
    - keep the educational route conservative and didactic even when a backend
      can produce a broader result
    - Status (2026-06-10): boundary, shared verifier, probe runner,
      observability lanes, and the first public family (Hermite positive
      quadratic) are live; see GENERAL_INTEGRATION_BACKEND_ROADMAP.md phase
      status.

13. **Definite integrals and the fundamental theorem**
   - definite integration over the real domain via the fundamental theorem,
     with continuity/pole checks on the integration interval before applying
     it; interval domain checking is the mathematical core, not display sugar
     over indefinite integration
   - didactic trace: antiderivative step, evaluation at the bounds, and an
     interval-condition explanation
   - honest residuals for improper integrals and non-elementary
     antiderivatives
   - support-matrix rows once a minimal command surface exists
   - Status (2026-06-11): first rung live — `integrate(f, x, a, b)` parses
     (arity-4 whitelist), evaluates via FTC with the interval certificate
     before substitution (linear poles inside the closed interval ->
     undefined; uncertifiable conditions -> honest residual), and has five
     matrix rows plus block13 harness axes. The three-step narration
     landed the same day (antiderivative rebuilt on a scratch context,
     evaluation at the bounds, pole detection for undefined results).
     Bound display landed next (\int_{a}^{b} in the trace). Symbolic bounds for
     unconditional antiderivatives landed next (the area function
     integrate(f, x, a, t); conditional integrands with symbolic bounds
     stay honest residuals). The improper policy landed next:
     infinite bounds evaluate via boundary limits with the certificate
     extended to (half-)infinite intervals, divergence reports the honest
     infinite value, and the narrator renders lim notation. The exp(-x) gap closed
     next (a reciprocal-exp recognizer at the educational Div arm;
     integrate(exp(-x), x, 0, infinity) = 1 composes). The x/e^x by-parts gap
     closed next (rebuild-and-delegate from the Div arm; the Gamma-style
     integrate(x*exp(-x), x, 0, infinity) = 1 composes). The ln(|x|) gap closed
     next (abs-wrapped unbounded tails resolve at both infinities;
     divergent log integrals report signed infinity). The exponential-quotient
     narrator landed next (rewrite quoted as the intermediate, table or
     by-parts title by numerator shape). The cyclic family closed next
     (one-line delegation: sin(x)/e^x and the improper damped
     oscillation = 1/2); tan(x)/e^x and Gaussian shapes remain honest
     residuals. The interval certificate then became SELF-CONTAINED
     (risk scan of the integrand: denominators factor by factor, ln/sqrt
     domains, trig poles via a rational pi enclosure) fixing four
     pre-existing wrong-finite-value soundness bugs, and learned
     polynomial positivity and the derivative-cofactor route - certified
     conditions are discharged from the display. Pi-multiple bounds landed
     next (exact r + q*pi endpoints; sec^2 on [pi/4, pi/3] = sqrt(3)-1,
     poles at pi/2 located exactly). One-sided finite limits
     learned composition next (scaling, additive combination with
     infinity awareness, power-log dominance: limit(x*ln(x), x, 0+) = 0)
     - the enabler for boundary-convergent improper integrals, whose
     wiring (one-sided boundary values at touched endpoints in the
     definite path) is the next rung, together with the product-to-sum
     trig rewrites for Fourier-style definites.

Selection rule:

- every calculus cycle should name its `calculus_maturity_block`
- every promoted calculus row should state which block gate it advances
- if a candidate does not advance a block gate, treat it as discovery pressure
  or reject it as a near-duplicate
- broad integration work must name block 12 and either improve the backend
  boundary or promote one verified algorithmic family; otherwise reject it as
  unsafe search
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

## Calculus Runtime Policy

As the calculus engine generalizes, runtime must be interpreted per block, not
only through the global embedded equivalence lane.

`embedded_equivalence_context` is still the broadest guardrail for common
simplification/equivalence traffic. Calculus needs additional runtime visibility
because `diff`, `limit`, and `integrate` have different cost centers:

- `diff`: rule dispatch, chain/product/quotient expansion, domain collection,
  and final presentation cleanup
- `limit`: finite/infinite routing, domain path checks, one-sided policy, and
  safe pre-simplification
- `integrate`: method selection, substitution detection, partial fractions,
  residual policy, and verification by differentiation

For calculus lanes, prefer normalized metrics:

- elapsed seconds
- case count
- `avg_case_ms`
- p95/max case time when available
- top slow case IDs
- family/regime breakdown of slow cases

Runtime regressions are acceptable only when the retained mathematical value is
clear. Use this rule of thumb:

- flat or improved runtime with more supported/domain-safe cases is a clean win
- roughly 10-20% slowdown can be retained if it unlocks a reusable public
  family, stronger domain policy, or verified integration method
- more than roughly 20-30% slowdown should be rejected or deferred unless the
  capability gain is major and the cost is localized
- any embedded collateral slowdown must be justified separately

Do not accept a calculus change that merely adds a near-duplicate matrix row,
presentation preference, or residual wrapper while making a broad calculus lane
materially slower.

When runtime worsens but the capability is valuable, document the tradeoff in
the cycle report and leave the next runtime candidate explicit.

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

## Deferred Horizons

These are explicitly out of the current block plan, named so their boundary is
visible and so present-day decisions do not make them needlessly expensive:

- **Complex domain.** Deferred until the real-domain north star is mature.
  Guardrails for today: keep branch policy explicit and family-owned (never
  implicit in display cleanup), keep `ln(|...|)` versus `ln(...)` decisions
  owned by the family that proves them, and keep condition predicates
  structured and extensible rather than hard-coding real-only assumptions
  into public contracts where a structured representation costs the same.

  **Semantic model decision (2026-06-10, binding for future work).** When the
  complex track starts, it uses the standard single-valued principal-branch
  CAS model — not a multivalued/Riemann-surface model:

  - every multivalued function denotes its principal branch: `Arg z` in
    `(-pi, pi]`, `Log z = ln|z| + i*Arg z`, `sqrt(z) = exp(Log(z)/2)`,
    `z^w = exp(w * Log z)`, with the branch cut on the negative real axis
  - a complex result is a contract, mirroring the backend result model: it
    must carry which branch choices were made, the Arg convention, and any
    cut-crossing assumptions as structured conditions — never silent branch
    hops in display cleanup (`sqrt(z^2) != z` in general; identities that
    need `Arg` restrictions must state them)
  - rules valid on all of `C` (Gaussian arithmetic, polynomial identities)
    are unconditional; rules valid only off a cut carry the cut condition
  - staged growth, hybrid-pattern style (contract first, then families):
    (1) `C`-algebra — Gaussian arithmetic, `(a+b*i)^n` expansion,
    conjugates, modulus, Gauss identities; already partially live behind
    the `--value-domain complex` axis (`i^2 -> -1`,
    `(2+3i)(2-3i) -> 13`); (2) `C`-elementary — Euler, principal
    `Log`/`Arg`, powers with cuts (note: `ln(-1)` currently returns
    `undefined` under the real semantics and is revisited here);
    (3) complex analysis (holomorphic calculus, residues) — explicitly
    questioned for the educational scope; do not plan it without a
    curriculum case.
- **Series and Taylor expansions.** Standard curriculum, not yet planned;
  belongs after definite integrals (block 13) earns a command surface.
- **Improper integrals.** Tracked as the residual policy of block 13, not as
  a separate engine track.
- **Differential equations.** Out of scope for this strategy document.

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
