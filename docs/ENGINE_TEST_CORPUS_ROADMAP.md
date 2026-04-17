# Engine Test Corpus Roadmap

This document proposes the next high-ROI test corpora to improve the
simplification engine itself, not only the didactic layer.

It is motivated by the recent `0/1` mixed-corpus work:

- `/Users/javiergimenezmoya/developer/math/docs/simplify_to_zero_step_by_step_review.csv`
- `/Users/javiergimenezmoya/developer/math/docs/simplify_zero_mixed_corpus.csv`
- `/Users/javiergimenezmoya/developer/math/docs/generated/simplify_zero_mixed_corpus_failures.csv`

The main lesson from that corpus is important:

- the engine already knew many exact identities
- but often failed when those identities appeared inside a larger context
- targeted corpus generation exposed that gap much better than isolated manual
  examples

So the next step is to design corpora that force the engine to solve real
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

### Curation Policy

This corpus should be curated, not inflated.

Add new entries when they introduce:

- a new mathematical family
- a new wrapper behavior
- a new cross-family composition
- a regression-prone structural interaction already observed in engine work

Do not add large batches of near-duplicate variants that all exercise the same
matcher route.

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

The default workflow should therefore compare each meaningful run against a
recent baseline and make the delta visible in the generated scorecard.

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

The remaining six failures in the current mixed `0/1` corpus are exactly this
kind of problem.

## Suggested Priority Order

Recommended order for maximum engine ROI:

1. Embedded equivalence in context
2. Orientation, sign, and canonicalization robustness
3. Mixed-family interaction
4. Domain frontier and semantic safety
5. Loop and oscillation detection
6. Budget and performance stability
7. Didactic fidelity review

## Recommended Immediate Next Step

The most useful next corpus after the current `0/1` mixed corpus is:

### Embedded Equivalence In Context

Use already validated equivalence pairs and generate:

- additive wrappers
- multiplicative wrappers
- common-denominator wrappers
- shifted quotients
- passthrough terms

This is the most likely corpus to produce another real leap in engine quality,
because it directly tests whether equivalence is usable locally, not only in a
top-level naked identity.

## How To Grow `embedded_equivalence_context_corpus.csv`

Use a family-first policy.

For each promoted family:

1. define one root equivalence pair
2. embed it in a small, fixed wrapper set
3. add at most one or two composed variants if they exercise a genuinely new path

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

Keep a family in metamorphic pressure lanes instead when:

- the family is still moving too fast
- the common normal form is not stable yet
- the case is useful for pressure but not ready to become a guardrail

That separation matters:

- `embedded` is for curated contextual guardrails
- metamorphic slices are for exploration and pressure

## Concrete Example Seeds

Good seeds for this next corpus:

- `sin(x) + sin(y) ~ 2*sin((x+y)/2)*cos((x-y)/2)`
- `cos(x) - cos(y) ~ -2*sin((x+y)/2)*sin((x-y)/2)`
- `3*sin(x) + 4*cos(x) ~ 5*sin(x + arctan(4/3))`
- `sinh(x+y) ~ sinh(x)*cosh(y) + cosh(x)*sinh(y)`
- `log(x^2-y^2) ~ log(x-y) + log(x+y)`
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
- performance predictability

For `embedded_equivalence_context_corpus.csv`, add one more filter:

- each added family should increase contextual mathematical coverage, not merely
  duplicate an already-covered local shape

If a corpus only rewards ad hoc string-shape patches, it should be rejected or
reframed.

That is the core principle:

- corpus design should expose missing engine abstractions
- not reward one-off special cases
