# Derive Roadmap

This document defines the technical roadmap for turning `derive` into a first-class educational feature of ExpliCAS.

Current public behavior:

- `derive <expr1>, <expr2>`
- `derive(<expr1>, <expr2>)`

Current implementation entrypoint:

- `/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/derive_command.rs`

Today, `derive` is intentionally conservative. It only succeeds when the target can be reached through a short list of safe strategies:

- `simplify`
- `collect`
- `factor`
- `simplify -> collect`
- `simplify -> factor`

That is a good base, but it is not enough to make `derive` the flagship educational tool.

## Goal

Given two equivalent expressions, if the first is less simplified than the second, ExpliCAS should be able to:

1. Recognize that the second expression is a reachable pedagogical target.
2. Reuse the engine's real algebraic rules instead of inventing ad hoc transformations.
3. Produce a short, correct, human-readable step-by-step path from the first to the second.

## Core Thesis

Improving `NF-convergent` helps, but it is not the main objective for `derive`.

Why:

- `NF-convergent` measures how often equivalent expressions collapse to the same normal form.
- `derive` needs something stronger: reachability of a user-specified target form.

Examples:

- `x^2 - 1 -> (x - 1)(x + 1)` is a factor target, not the canonical simplified form.
- `a*x + b*x + c -> (a+b)*x + c` is a collected target, not a generic NF target.
- `ln(x^3) + ln(y^2) -> ln(x^3*y^2)` is a contracted-log target.

So the real target for `derive` is:

- strong semantic equivalence checking
- plus controlled navigation between families of mathematical forms
- plus didactic paths that do not feel magical

## Non-Goals

This roadmap explicitly rejects the following approaches:

- blind BFS over all engine rewrites
- unconstrained graph search over expression space
- target-specific hacks hardcoded for isolated examples
- a fake derivation layer that does not reuse the engine's real transformations
- a single global normal form pretending to solve all derive targets

## Design Principles

### 1. Strategy-Led, Not Search-Led

`derive` should remain strategy-based.

Good:

- classify the requested target
- attempt a small number of safe transformation families
- validate each stage semantically

Bad:

- apply arbitrary rewrites until something happens to match structurally

### 2. Reuse Real Engine Rules

The derivation path must come from real engine transforms wherever possible.

That keeps:

- correctness aligned with the rest of the CAS
- step quality aligned with normal simplification
- maintenance cost low

### 3. Family-Oriented Reachability

`derive` should reason in terms of target families:

- simplified
- factored
- collected
- expanded
- log-contracted
- log-expanded
- trig-contracted
- trig-expanded
- rationalized
- solve-prep
- integrate-prep

The planner does not need a universal theorem prover. It needs robust movement between these families.

### 4. Educational Output First

A derivation that is correct but opaque is not enough.

Success means:

- few steps
- understandable step names
- didactic substeps where needed
- no internal-engine noise

## Current State

Current strengths:

- safe surface API
- equivalent-vs-not-equivalent distinction
- clear unsupported-target fallback
- existing reuse of `simplify`, `factor`, and `collect`

Current limitations:

- no explicit target-form classification layer
- no strategy registry beyond a hardcoded list
- no derive-specific test corpus by family
- no planner-level scoring of alternative paths
- no dedicated derive metrics

## Success Metrics

`derive` should be measured with its own metrics, not only with generic metamorphic metrics.

### Primary Metrics

- `derive_reachability_rate`
  - percentage of curated equivalent pairs whose exact target is reached
- `derive_supported_equiv_rate`
  - percentage of equivalent pairs that are not reported as `EquivalentButUnsupported`
- `derive_mean_step_count`
  - average visible steps for successful derives
- `derive_long_path_rate`
  - percentage of successes exceeding a pedagogical threshold, for example `> 8` visible steps
- `derive_timeout_rate`
  - percentage of derive runs hitting budget/timeout

### Secondary Metrics

- `NF-convergent`
- `Proved-symbolic`
- `Numeric-only`
- `Inconclusive`

These still matter, but they are supporting indicators, not the main KPI for `derive`.

### Educational Metrics

- visible-step duplication rate
- trivial-substep rate
- unsupported-but-equivalent rate by family
- target family coverage

## Proposed Architecture

The long-term shape should be:

### A. Parse and Resolve

Keep the current front door:

- parse source
- parse target
- resolve references/session bindings
- reject unknown functions early

### B. Target Classification

Add a lightweight classifier that tries to answer:

- what form does the target look like?
- which variable, if any, is being collected?
- does the target look factored?
- does it look expanded?
- does it look log-contracted or log-expanded?
- does it look trig-contracted or trig-expanded?

Suggested module shape:

- `derive/target_form.rs`
- `derive/target_classifier.rs`

### C. Strategy Registry

Replace the current fixed if-chain with a registry of derive strategies.

Each strategy should expose:

- label
- applicability predicate
- attempt function
- expected target families
- budget profile

Suggested module shape:

- `derive/strategy.rs`
- `derive/strategies/*`

### D. Planner

The planner should stay shallow.

Recommended behavior:

1. classify target
2. rank candidate strategies
3. try a small ordered set
4. stop on first exact target hit
5. if none succeed, report equivalent-but-unsupported

This is not a graph search engine. It is a bounded strategy planner.

Suggested module shape:

- `derive/planner.rs`

### E. Didactic Rendering

Once a strategy succeeds, reuse the engine steps, then apply didactic cleanup:

- suppress trivial local noise
- merge preparation steps when pedagogically correct
- add family-specific substeps only when they add value

## Target Families To Support

The expansion order should be intentional.

### Phase 1 Families

- simplify
- factor
- collect
- expand

These are algebraically central and have the highest educational value.

### Phase 2 Families

- log contraction
- log expansion
- trig contraction
- trig expansion
- rationalization

These are common in textbook transformations and often requested by users explicitly.

### Phase 3 Families

- solve-prep
- integrate-prep
- partial telescoping-like transforms
- canonical fraction reshaping

These are useful, but more domain-sensitive.

## Roadmap

## Phase 0. Baseline and Instrumentation

Goal:

- stop treating `derive` as a side feature
- measure it directly

Deliverables:

- create a curated derive-pairs corpus
- add derive family labels
- add derive-specific stats output
- add unsupported-equivalent diagnostics by family

Suggested test file:

- `/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/derive_contract_tests.rs`

Suggested corpus file:

- `/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/derive_pairs.csv`

Minimum curated families:

- combine-like-terms
- factor
- collect
- expand
- logs
- trig
- rationals
- radicals

Acceptance criteria:

- derive corpus exists
- each case is labeled by family
- test output reports reachability and unsupported-equivalent counts

## Phase 1. Extract Target Classification

Goal:

- make `derive` reason explicitly about target shape

Deliverables:

- `TargetForm` enum
- target classifier
- variable-of-interest detection for collect-like targets
- exact-target structural matcher after normalization

Suggested `TargetForm` sketch:

```rust
enum TargetForm {
    Simplified,
    Factored,
    Collected { var: String },
    Expanded,
    LogContracted,
    LogExpanded,
    TrigContracted,
    TrigExpanded,
    Rationalized,
    Unknown,
}
```

Acceptance criteria:

- current `derive` behavior is preserved
- target classifier routes existing `simplify/factor/collect` cases correctly
- no regression in `make ci`

## Phase 2. Strategy Registry and Planner

Goal:

- remove hardcoded strategy sequencing from one file

Deliverables:

- strategy trait
- registry
- planner with ranked attempts
- derive trace/debug output for which strategy succeeded

Acceptance criteria:

- current strategies are migrated without behavior loss
- planner tries a bounded set only
- unsupported-but-equivalent remains explicit

## Phase 3. Add Expand As A First-Class Derive Strategy

Goal:

- support targets that are explicitly expanded, not just simplified or factored

Example:

- `derive (x+1)^2, x^2 + 2*x + 1`

Deliverables:

- `expand`
- `simplify -> expand`
- small-polynomial normalization reuse where appropriate

Acceptance criteria:

- curated expand pairs succeed
- no major step explosion
- derive steps remain human-readable

## Phase 4. Logarithmic Families

Goal:

- make `derive` robust for textbook log transformations

Examples:

- `derive ln(x^3) + ln(y^2), ln(x^3*y^2)`
- `derive ln(x*y), ln(x) + ln(y)`

Deliverables:

- `log-contract`
- `log-expand`
- domain-aware acceptance checks

Acceptance criteria:

- curated log corpus reaches target forms
- no domain loss in displayed requirements

## Phase 5. Trigonometric Families

Goal:

- support derive for common trig rewrites without opening oscillation risks

Examples:

- `derive 2*sin(x)*cos(x), sin(2*x)`
- `derive tan(2*x), (sin(2*x))/(cos(2*x))`

Deliverables:

- narrow target-aware trig contraction/expansion strategies
- planner guardrails to avoid oscillatory families unless the target explicitly asks for them

Acceptance criteria:

- curated trig pairs succeed
- no new metamorphic regressions
- no planner loops

## Phase 6. Rationalization and Radical/Power Families

Goal:

- support didactically important transitions between radical and power forms

Examples:

- `derive 1/(sqrt(x)-1), (sqrt(x)+1)/(x-1)`
- `derive sqrt(x)*x^(2/3), x^(7/6)`

Deliverables:

- rationalize strategy
- radical/power presentation-aware success checks
- notation-style preservation where appropriate

Acceptance criteria:

- web/JSON output stays stylistically coherent
- derive can reach radical targets and power targets intentionally

## Phase 7. Didactic Quality Pass

Goal:

- ensure successful derive paths are not only correct but teachable

Deliverables:

- derive-specific didactic audit
- rules for pruning trivial steps and redundant single substeps
- family-specific human step titles where needed

Suggested new audit:

- `/Users/javiergimenezmoya/developer/math/docs/DERIVE_DIDACTIC_AUDIT.md`

Acceptance criteria:

- no duplicated parent/substep narratives
- no trivial one-line substeps that restate the parent step
- derive examples in web and CLI read naturally

## Phase 8. UX and Surface Parity

Goal:

- make `derive` consistent across CLI, REPL, JSON, and web

Deliverables:

- same parsing surface in REPL and `eval`
- `input_latex` for `derive(...)`
- stable JSON contract for status:
  - success
  - already-at-target
  - equivalent-but-unsupported
  - not-equivalent

Acceptance criteria:

- web and REPL cannot diverge semantically
- examples in `web/examples.csv` stay green

## Testing Plan

## 1. Contract Tests

Curated derive pairs should assert:

- exact target reached
- expected strategy used
- max step count not exceeded
- no unsupported-equivalent when the family is marked supported

## 2. Metamorphic Guardrails

Use metamorphic tests to ensure new derive-friendly strategies do not degrade the core engine:

- no increase in `Numeric-only`
- no unexpected rise in `Inconclusive`
- no timeout growth in the benchmark slices

## 3. Didactic Audits

Curated derive examples should be exported and reviewed for:

- redundant steps
- magical jumps
- opaque internal rule names
- noisy local rewrites

## 4. Performance Guardrails

`derive` should remain bounded.

Each new strategy should define:

- max planner attempts
- max rewrite budget
- max visible steps

No strategy should introduce a general rewrite search.

## Suggested Initial Corpus

These are good early targets for the derive roadmap:

- `derive x + x, 2*x`
- `derive a*x + b*x + c, (a+b)*x + c`
- `derive x^2 - 1, (x-1)*(x+1)`
- `derive (x+1)^2, x^2 + 2*x + 1`
- `derive ln(x^3) + ln(y^2), ln(x^3*y^2)`
- `derive ln(x*y), ln(x) + ln(y)`
- `derive 2*sin(x)*cos(x), sin(2*x)`
- `derive 1/(sqrt(x)-1), (sqrt(x)+1)/(x-1)`
- `derive (x - 1)*(x^5 + x^4 + x^3 + x^2 + x + 1), x^6 - 1`

## Recommended File Layout

If the roadmap is executed, a clean structure would be:

- `/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/derive_command.rs`
  - thin public orchestration layer
- `/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/derive/target_form.rs`
- `/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/derive/target_classifier.rs`
- `/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/derive/strategy.rs`
- `/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/derive/planner.rs`
- `/Users/javiergimenezmoya/developer/math/crates/cas_solver/src/derive/strategies/`

That would keep parsing, planning, and strategy execution separated cleanly.

## Practical Recommendation

Do not start by chasing a universal derive engine.

Start with:

1. derive corpus + metrics
2. target classification
3. strategy registry
4. expand/log families

That path gives the highest ROI while preserving the current conservative architecture.

## Definition of Done

`derive` becomes a flagship feature when all of the following are true:

- it succeeds on a broad curated family corpus
- it reaches user-requested targets, not just canonical NFs
- it stays conservative and bounded
- it does not regress metamorphic quality
- it produces steps a student can actually follow

Until then, the right mindset is:

- not “make derive smarter at any cost”
- but “make derive reliable, teachable, and explicitly target-aware”
