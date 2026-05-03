# Engine Test Corpus Roadmap

This document is a derived strategy under
[ENGINE_IMPROVEMENT_AUTOMATION.md](/Users/javiergimenezmoya/developer/math/docs/ENGINE_IMPROVEMENT_AUTOMATION.md).

It should drive an iteration when the ROI selector chooses `coverage`, or when
corpus work is the right vehicle for a `robustness` iteration.

For calculus-specific coverage, read this together with
[CALCULUS_ENGINE_STRATEGY.md](/Users/javiergimenezmoya/developer/math/docs/CALCULUS_ENGINE_STRATEGY.md).
Calculus corpora should grow as bounded vertical slices that also pressure the
pre-calculus core.

This document proposes the next high-ROI test corpora to improve the
simplification engine itself, not only the didactic layer.

This document now serves two purposes at once:

- durable corpus policy
- current promotion priority guidance

Read it with this status in mind:

- `embedded_equivalence_context_corpus.csv` already exists and is already part
  of the live scorecard guardrail
- high-temperature generation is still primarily a discovery feeder, not an
  automatic promotion lane
- the next corpus move is therefore usually growth and rebalance of existing
  guardrails, not creation of a brand-new corpus from zero

It is motivated by the `0/1` mixed-corpus work:

- `/Users/javiergimenezmoya/developer/math/docs/simplify_to_zero_step_by_step_review.csv`
- `/Users/javiergimenezmoya/developer/math/docs/simplify_zero_mixed_corpus.csv`
- `/Users/javiergimenezmoya/developer/math/docs/generated/simplify_zero_mixed_corpus_failures.csv`

The main lesson from that corpus remains important:

- the engine already knew many exact identities
- but often failed when those identities appeared inside a larger context
- targeted corpus generation exposed that gap much better than isolated manual
  examples

So the continuing job is to design corpora that force the engine to solve real
structural problems:

- local equivalence inside context
- orientation and sign robustness
- domain-sensitive simplification
- loop resistance
- budget and performance stability

## Goal

Use generated corpora to discover missing engine capabilities, especially when:

- an identity works only in its naked form
- the same identity fails when embedded in a larger expression
- the engine depends too much on sign orientation or canonical order
- semantic correctness changes across domain modes
- the simplifier reaches the right result but via low-quality or unstable paths

Corpus work is therefore not just “more tests”.

It is the main mechanism for:

- growing the `live` workload intentionally
- preserving a small stable `frozen` ruler
- separating useful completeness growth from accidental runtime tax

## Why Corpus-Driven Testing Matters

Manual examples are good for discovering families.

Generated corpora are better for discovering structural weaknesses.

A single generated family can reveal:

- over-specialized rules
- missing passthrough handling
- missing negated orientation
- dependence on syntactic order
- failure to re-enter a useful rule after an intermediate rewrite
- accidental performance cliffs

The recent mixed `0/1` corpus did exactly that. It exposed a broad engine gap
that was invisible from individual examples:

- exact-zero identities were recognized in isolation
- but not when embedded in additive context

One more lesson matters for strategy:

- broad contextual corpora are also runtime guardrails
- a new matcher can improve one narrow family while slowing down the whole
  embedded corpus
- that slowdown is real engine cost, not benchmark noise

So `embedded equivalence` should be read in two dimensions:

- correctness: pass/fail
- cost: elapsed time under the corpus

That makes corpus strategy and orchestrator strategy tightly connected.
For the orchestrator-specific side of that work, see
[ORCHESTRATOR_OBSERVABILITY_STRATEGY.md](/Users/javiergimenezmoya/developer/math/docs/ORCHESTRATOR_OBSERVABILITY_STRATEGY.md).

That is the model to repeat.

## Corpus Roles: Frozen, Live, Stress

These are corpus-role labels, not the current scorecard profile names.

Today the runnable scorecard profiles are `fast`, `guardrail`, `pressure`, and
`full`.
This file uses `frozen / live / stress` to describe how a corpus should behave
strategically over time.

Not every corpus should play the same role.

We need three distinct roles:

- `frozen`
  - a small representative benchmark snapshot
  - intentionally stable across normal engine work
  - used to track baseline overhead tax on already-known traffic
- `live`
  - the current representative guardrail corpus
  - grows as new mathematical families and wrappers become product-relevant
- `stress`
  - larger, deeper, or more combinatorial expressions
  - used to expose scaling cliffs and performance instability

### Why The Split Matters

If the only corpus we keep is the growing current workload, elapsed time becomes
hard to interpret:

- maybe the engine got slower
- maybe the corpus simply got broader
- maybe the engine got more complete and that extra cost is worth paying

Keeping a `frozen` corpus gives us a stable ruler for overhead.

Keeping a `live` corpus preserves relevance.

Keeping a `stress` corpus preserves scale awareness.

### Promotion And Placement Policy

Preferred path:

1. explore in targeted regressions or local pressure slices
2. promote to `live` once the family is stable and representative
3. promote only a small durable subset to `frozen` when the goal is long-term
   overhead tracking
4. add larger or more entangled variants to `stress`, not to `frozen`

Generated high-temperature expressions should enter before step 1 as a discovery
feeder, not as immediate `live` promotions.

Preferred feeder path:

1. generate a composed candidate from trusted equivalence seeds
2. use it to discover a missing abstraction, brittle route, or didactic gap
3. retain the smallest durable guardrail artifact afterward

## Corpus 1: Embedded Equivalence In Context

### Purpose

Check whether the engine can exploit a known equivalence when it is wrapped by a
larger algebraic context.

### Generation Pattern

Given equivalent expressions `expr1 ~ expr2`, generate:

- `a + expr1 - (a + expr2)` -> `0`
- `a*expr1 - a*expr2` -> `0`
- `expr1/k - expr2/k` -> `0`
- `expr1 + passthrough - (expr2 + passthrough)` -> `0`
- `(expr1 + c)/(expr2 + c)` -> `1` when safe

Then deliberately generate hotter composed variants by combining several such
pairs inside a single larger expression, for example:

- additive sums of independent zero-equivalent blocks
- one family embedded inside another family's wrapper
- nested wrapper variants with higher `shell_depth`
- cases with extra additive or multiplicative `noise_budget`
- sign-flipped or reordered orientations of the same core identity

### Why It Matters

This is the most direct way to test whether the engine understands
equivalence locally, not only globally.

It forces the simplifier to:

- recognize a useful identity inside a larger expression
- remove a zero-equivalent subexpression
- preserve surrounding structure correctly
- avoid getting stuck on the wrapper instead of the core identity
- avoid paying broad no-match overhead for narrow shortcuts

### Typical Bugs It Reveals

- identity only works when the whole expression matches
- failure to simplify `1 + zero_identity`
- failure inside numerators or denominators
- failure after a harmless passthrough term is added
- failure to propagate equivalence through multiplication or division

### Example Families

- trig sum-to-product
- trig phase shift
- hyperbolic angle sum/difference
- logarithmic expansion/contraction
- rationalization
- nested fractions

### Why This Has High ROI

This directly improves real user input, because most interactive expressions are
not naked textbook identities. They are identities embedded in context.

It also has high ROI as a performance guardrail:

- it is broad enough to catch hot-path matcher regressions
- it penalizes shortcuts that are too expensive outside their real family
- it helps distinguish a reusable engine improvement from a local benchmark hack

That is one of the main reasons not to refactor the orchestrator blindly.
The orchestrator should become more observable before it becomes more abstract.

It is also one of the best reasons to generate composed expressions instead of
waiting only for manual examples:

- isolated identities often already work
- failures appear when several valid local identities are forced to coexist
- those compositions expose route ordering, fallback tax, and sign fragility
  much faster than isolated curation alone

### Curation Policy

This corpus should be curated, not inflated.

Add new entries when they introduce:

- a new mathematical family
- a new wrapper behavior
- a new cross-family composition
- a regression-prone structural interaction already observed in engine work

For every promoted mathematical family, also evaluate a `derive` shadow case:

- if the family has a natural target-form direction, add the smallest
  representative `source -> target` pair to derive only when it improves
  reachability, multi-step path quality, or removes a magical jump
- if the family is already represented in derive, do not add a duplicate row
  just because the embedded wrapper changed
- if branch/domain semantics are unresolved, record the defer reason rather
  than promoting an unsound or fake derivation

This keeps engine coverage and derive coverage coupled without turning either
corpus into a dumping ground.

Do not add large batches of near-duplicate variants that all exercise the same
matcher route.

Do not automatically promote every successful or failing high-temperature
expression either.

For generated composite cases, promote only when the case adds one of:

- a new family interaction
- a new wrapper axis
- a new shell-depth level
- a new `noise_budget` level with real retained value
- a representative case of a regression-prone failing route

The objective is contextual completeness per case, not corpus size for its own
sake.

### Runtime Policy

`embedded_equivalence_context_corpus.csv` should also be tracked as an elapsed
time metric.

That runtime should appear in the scorecard as a first-class guardrail with
baseline comparison, not just as a manually copied observation.

A meaningful slowdown in this corpus should be treated as a regression unless
there is a strong, explicit justification such as:

- a new family of mathematically important cases now works
- a previous timeout/overflow/nontermination path is now closed
- the engine gained robustness that could not be achieved more cheaply

In other words:

- pass/fail is necessary
- elapsed time still matters
- a slower embedded corpus is acceptable only when the functional or robustness
  gain is clearly worth it

That runtime should also be interpreted by suite role:

- `frozen`
  - baseline tax
- `live`
  - current representative workload
- `stress`
  - scaling behavior

The default workflow should therefore compare each meaningful run against a
recent baseline and make the delta visible in the generated scorecard.

When a corpus slice exposes a strong local runtime win that still loses on the
global embedded guardrail, that case should be recorded as a combination
candidate, not forgotten.

Use
[ENGINE_COMBINATION_LEDGER.md](/Users/javiergimenezmoya/developer/math/docs/ENGINE_COMBINATION_LEDGER.md)
to capture:

- which slice improved
- which global corpus regressed
- which shapes dominated the local hotspot
- what extra scoping or reuse might make the idea safe later

That ledger is especially useful for corpus-guided work because many
performance-sensitive ideas are only wrong in *placement*, not in mathematics.

## Slow CI Tests And Embedded Runtime

Slow tests observed in `make ci` are relevant to corpus strategy only after
they are classified correctly.

There are three different phenomena that can look like the same problem:

- compile / runner noise
- real engine runtime pathology
- test verification pathology

That distinction matters because only the second class should normally drive
engine runtime changes.

The third class is especially dangerous for corpus work:

- the engine may already be fast on the real product path
- but the test verifies equivalence through a much more expensive isolated or
  assertion-only route
- “fixing” that by broadening engine shortcuts can easily slow down the
  embedded corpus for no real product gain

So the policy is:

1. reproduce the slow test narrowly outside `make ci`
2. classify it before editing engine code
3. prefer a test-side fix when the engine already behaves well on the real path
4. require an embedded corpus rerun for any retained engine-side fix

Operationally, keep the current slow-test state in:

- [SLOW_CI_TEST_LEDGER.md](/Users/javiergimenezmoya/developer/math/docs/SLOW_CI_TEST_LEDGER.md)

Use the embedded corpus as the guardrail only for retained engine-side fixes,
not as a reason to avoid fixing a pathological test harness.

## Discovery Feeder: High-Temperature Composition

### Purpose

Generate larger composed expressions from already-valid equivalence seeds in
order to reveal failures that do not appear in isolated rows.

This corpus role is mainly for:

- coverage discovery
- robustness discovery
- performance cliff discovery
- didactic review of nontrivial multi-step transformations

### Why It Should Exist

Recent engine work has repeatedly shown the same pattern:

- each identity term works alone
- several terms together reveal a new weakness
- fixing that weakness often improves a reusable engine abstraction

So composed generation is not just “more random cases”.

It is a way to search the missing abstraction space faster.

### Construction Policy

Build these expressions from trusted seeds, not arbitrary syntax.

Recommended knobs:

- number of identity blocks combined
- family diversity per expression
- wrapper mix
- `noise_budget`
- `shell_depth`
- sign/orientation perturbation
- cross-family nesting

Temperature should mean increasing those knobs intentionally, not unconstrained
randomness.

### Evaluation Loop

For each generated expression:

1. run it through the cheap discovery lane
2. record whether it:
   - simplifies correctly
   - panics, overflows, loops, or times out
   - resolves only through a late expensive fallback
   - produces low-quality steps or broken highlights
3. cluster failures by structural signature
4. fix the reusable family-level weakness
5. validate retained fixes against `embedded` runtime and the other relevant
   guardrails
6. choose the correct promotion target

### Promotion Policy

Promotion targets should be chosen conservatively:

- `unit test`
  - when the case captures a narrow matcher bug or very specific regression
- `live`
  - when a small representative contextual form improves durable coverage
- `stress`
  - when the hotter form is valuable for scale pressure but too large or too
    duplicative for `live`
- `frozen`
  - only when a minimal stable subset becomes a long-term overhead ruler

This means the full hot expression is often *not* the retained artifact.

The retained artifact should be the smallest case that still guards the lesson.

### Didactic Review Policy

Generated composite expressions are also good didactic probes, because they
surface:

- magical steps
- weak sub-steps
- brittle or misleading `before/after` highlighting
- explanations that are too narrow for the actual family

But didactic review should come after semantic retention, not before it.

The right sequence is:

1. make the expression semantically robust
2. verify global runtime retention
3. then improve steps, sub-steps, and highlighting if the explanation can be
   generalized usefully

## Corpus 2: Orientation, Sign, And Canonicalization Robustness

### Purpose

Check whether the same algebraic fact works regardless of sign orientation,
syntactic order, or equivalent canonical form.

### Generation Pattern

For each identity or rewrite family, generate:

- `expr1 - expr2`
- `expr2 - expr1`
- `-(expr1 - expr2)`
- `expr1 + (-expr2)`
- reordered additive terms
- reordered multiplicative factors
- equivalent but differently-associated trees

### Why It Matters

Many engine bugs are not mathematical failures. They are orientation failures.

The engine knows:

- `A - B -> 0`

but misses:

- `B - A -> 0`
- `-A + B -> 0`
- `B + (-A) -> 0`

### Typical Bugs It Reveals

- rule only matches one sign orientation
- rule depends on exact left/right layout
- canonicalization happens too late
- no re-entry after sign normalization
- spurious no-op steps such as `Canonicalize Division`

### Why This Has High ROI

It tends to unlock many cases with one small generalization in the engine.

The recent additive-zero improvements came largely from this type of issue.

## Corpus 3: Domain Frontier And Semantic Safety

### Purpose

Check whether simplification remains correct across `strict`, `generic`, and
`assume` domain modes.

### Generation Pattern

Reuse the same equivalence families under expressions involving:

- `log`
- `sqrt`
- reciprocal terms
- `abs`
- even and odd powers
- hidden denominator constraints

Run each case under multiple domain modes and compare:

- result
- required conditions
- assumptions
- displayed guards

### Why It Matters

A large class of bugs is not algebraic but semantic:

- simplification is algebraically valid
- but changes the domain or silently assumes something false

### Typical Bugs It Reveals

- missing nonzero guard
- redundant positivity/nonnegativity guards
- wrong use of `abs`
- invalid cancellation through zero-risk denominators
- inconsistent behavior between modes

### Why This Has High ROI

This is one of the best ways to harden correctness, not just didactics.

## Corpus 4: Loop And Oscillation Detection

### Purpose

Find families where the engine reaches a long, noisy, or cyclic sequence instead
of taking the short route.

### Generation Pattern

For each test case, record:

- final result
- step count
- repeated rule names
- repeated normalized expressions
- repeated local before/after patterns

Flag cases such as:

- same normalized expression seen twice
- same rule family repeating without progress
- long tails of canonicalization/no-op rewrites

### Why It Matters

A simplifier can be algebraically correct and still be poor as an engine if it:

- wastes budget
- produces unstable traces
- depends on lucky rule ordering

### Typical Bugs It Reveals

- two-rule oscillation
- canonicalization loops
- expansion followed by immediate contraction
- failure to short-circuit after reaching a known normal form

### Why This Has High ROI

It improves both performance and didactic quality at once.

## Corpus 5: Budget And Performance Stability

### Purpose

Detect regressions where a new rule fixes a family but makes unrelated inputs
slower or more expensive.

### Generation Pattern

For curated corpora, record:

- pass/fail
- timing
- step count
- node count
- depth
- long-path rate
- warnings such as `depth_overflow`

Compare before/after snapshots when new rule families are added.

### Why It Matters

A highly targeted rule can be mathematically correct but still be too expensive
if it:

- triggers too often
- allocates large intermediate trees
- runs costly equivalence checks on broad candidates

### Typical Bugs It Reveals

- expensive rules with weak candidate filters
- broad scans over additive terms
- rules that dominate hot paths in debug mode
- accidental budget explosions

### Why This Has High ROI

It protects the engine from “fix one family, slow everything else”.

## Corpus 6: Didactic Fidelity Versus Engine Correctness

### Purpose

Separate “the engine found the answer” from “the path is pedagogically good”.

### Generation Pattern

For each curated expression, record:

- final result
- step count
- visible rule names
- substep count
- presence of redundant or tautological substeps
- before/after coherence
- highlight continuity

This is similar to the existing derive and simplify review CSVs, but should be
used after the engine corpus has already established correctness.

### Why It Matters

A correct engine can still produce bad step-by-step output if:

- the useful local rewrite is hidden
- the step title is too generic
- substeps repeat the parent step
- highlights do not follow the real focus

### Why This Has High ROI

It prevents engine progress from being masked by poor presentation.

## Corpus 7: Mixed-Family Interaction

### Purpose

Test what happens when two correct identities from different families are mixed
in the same expression.

### Generation Pattern

Start from known good simplifications to `0` and combine them in ways such as:

- `expr_zero_A + expr_zero_B`
- `expr_zero_A - expr_zero_B`
- `(expr_zero_A + 1)/(expr_zero_B + 1)`
- `a*expr_zero_A + b*expr_zero_B`

### Why It Matters

This often reveals that the engine understands each family separately but lacks
normalization across family boundaries.

### Typical Bugs It Reveals

- `ln` versus `log` normalization gaps
- `abs`-aware versus non-`abs` variants not meeting
- mixed trig and algebraic normal forms failing to converge
- one family’s output shape not matching the other family’s matcher

### Why This Has High ROI

The mixed `0/1` corpus repeatedly exposes exactly this kind of problem whenever
residual failures remain after a broader cleanup pass.

## Corpus 8: Calculus Vertical Slices As Pre-Calculus Pressure

### Purpose

Build calculus capability without separating it from the simplification,
equivalence, domain, and didactic machinery that makes the engine reliable.

Calculus corpus work should test two things at once:

- the calculus command returns the right result or residual
- the pre-calculus core can simplify, explain, and validate the intermediate
  and final expressions
- the final public calculus form is readable when the mathematical capability
  already exists, without forcing a global canonical simplification preference

### Generation Pattern

Start with small vertical slices, not broad random calculus syntax.

Differentiation examples:

- polynomial and rational derivatives
- product, quotient, and chain rule cases
- `exp`, `ln`, trig, and inverse-trig derivatives where policy is clear
- derivative results that require simplification but should not become magical
- post-diff presentation rows where the internal result is correct but awkward,
  such as `diff(arctan(sqrt(x)), x)` preferring a final reciprocal-root form
  over `x^(-1/2)/(2*x + 2)`

Limit examples:

- polynomial and rational limits at infinity
- simple finite point limits only under explicit policy
- safe pre-simplification noise such as `+0`, `*1`, and structural zero terms
- residual cases where unsupported behavior must stay explicit

Integration examples:

- powers, sums, and constant multiples
- table-supported `exp`, `sin`, `cos`, and `1/x` cases
- simple linear substitution only when the substitution trace is explicit
- antiderivatives that can be checked by differentiating within supported
  families
- post-integral presentation rows where a verified antiderivative should keep a
  compact reciprocal denominator instead of expanding it for display

### Why It Matters

Calculus turns pre-calculus strength into public mathematical capability.

It also exposes gaps that ordinary simplification corpora can miss:

- derivative outputs with awkward but equivalent forms
- cancellation and factoring needed after product or quotient rules
- domain assumptions around `ln`, `sqrt`, inverse trig, and division
- integration constants and unsupported residual behavior
- presentation-only defects where the result is correct but less product-ready
  than a domain-equivalent compact form
- step traces that are correct but too magical for educational use

### Promotion Policy

Promote calculus cases conservatively:

- use unit tests for a narrow calculus rule or family
- use CLI/API contract tests when public command behavior changes
- use presentation contract tests when the retained value is only the final
  calculus display form
- use didactic/highlight tests when the visible trace is the retained value
- use pressure or generated discovery for large composed calculus expressions
- promote to live guardrails only when the family is stable, representative,
  and cheap enough to keep

Do not promote a large calculus expression just because it found a bug.
Promote the smallest representative that preserves the lesson.

Presentation cases should remain especially small. They are valuable when they
establish a reusable preference such as reciprocal-root display, compact
denominator factoring, or avoiding post-integration denominator expansion. They
are not valuable when they only encode one string-shaped anecdote.

### Why This Has High ROI

This track lets the project start building calculus now while continuing to
strengthen the pre-calculus heart of the engine.

The best retained cases improve both:

- user-visible calculus behavior
- simplification/equivalence/domain/didactic quality below it

## Suggested Priority Order

Recommended order for maximum engine ROI:

1. Embedded equivalence in context
2. Orientation, sign, and canonicalization robustness
3. Mixed-family interaction
4. Calculus vertical slices when they reuse and harden pre-calculus
5. Domain frontier and semantic safety
6. Loop and oscillation detection
7. Budget and performance stability
8. Didactic fidelity review

## Current Promotion Priority

The highest-ROI current move is to keep growing and rebalancing the existing
embedded contextual guardrail:

### Embedded Equivalence In Context

Use already validated equivalence pairs and generate:

- additive wrappers
- multiplicative wrappers
- common-denominator wrappers
- shifted quotients
- passthrough terms

Treat this as growth of an existing live guardrail, not as a proposal to create
that corpus from scratch.

It is still the most likely place to produce another real leap in engine
quality, because it directly tests whether equivalence is usable locally, not
only in a top-level naked identity.

## How To Grow `embedded_equivalence_context_corpus.csv`

Use a family-first policy.

For each promoted family:

1. define one root equivalence pair
2. explore hotter composed variants in discovery lanes first
3. embed the retained representative case in a small, fixed wrapper set
4. add at most one or two composed variants if they exercise a genuinely new
   path and still justify their runtime cost

Recommended wrapper set:

- additive passthrough
- scaled difference
- common denominator
- shifted quotient

Only add more wrappers when the family exposes a new failure mode under them.

## Promotion Checklist For Embedded Cases

Promote a family into `embedded_equivalence_context_corpus.csv` when all of
these are true:

- it represents a real mathematical family, not a one-off anecdote
- it covers a wrapper or composition axis not already guarded well
- the engine behavior is stable enough to serve as a long-lived regression guard
- the case is likely to catch future contextual regressions
- the added runtime tax on `embedded` is proportionate to the new retained
  coverage value

Keep a family in metamorphic pressure lanes instead when:

- the family is still moving too fast
- the common normal form is not stable yet
- the case is useful for pressure but not ready to become a guardrail
- the expression is valuable mainly because it is hot, wide, or deeply composed

That separation matters:

- `embedded` is for curated contextual guardrails
- metamorphic slices are for exploration and pressure

## Current Promotion Seeds

Good seeds for this next corpus:

- `sin(x) + sin(y) ~ 2*sin((x+y)/2)*cos((x-y)/2)`
- `cos(x) - cos(y) ~ -2*sin((x+y)/2)*sin((x-y)/2)`
- `3*sin(x) + 4*cos(x) ~ 5*sin(x + arctan(4/3))`
- `sinh(x+y) ~ sinh(x)*cosh(y) + cosh(x)*sinh(y)`
- `ln(x^2-y^2) ~ ln(x-y) + ln(x+y)`
- `1/(sqrt(a)+sqrt(b)) ~ (sqrt(a)-sqrt(b))/(a-b)`
- `(x^3-b^3)/(x-b) ~ x^2 + x*b + b^2`

Then wrap them as:

- `a + expr1 - (a + expr2)`
- `k*expr1 - k*expr2`
- `(expr1 + 1)/(expr2 + 1)`
- `(expr1 / q) - (expr2 / q)`

## Success Criteria

A corpus is worth keeping only if it improves at least one of:

- correctness
- semantic soundness
- convergence stability
- didactic path quality
- post-calculus presentation quality for supported `diff`, `limit`, or
  `integrate` results
- performance predictability
- derive bridgeability when the same identity has a meaningful target form
- calculus bridgeability when a pre-calculus family can become a bounded
  `diff`, `limit`, or `integrate` capability without unsafe assumptions

For `embedded_equivalence_context_corpus.csv`, add one more filter:

- each added family should increase contextual mathematical coverage, not merely
  duplicate an already-covered local shape

If a corpus only rewards ad hoc string-shape patches, it should be rejected or
reframed.

That is the core principle:

- corpus design should expose missing engine abstractions
- not reward one-off special cases
