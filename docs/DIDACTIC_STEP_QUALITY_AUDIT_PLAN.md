# Didactic Step Quality Audit

## Goal

Create a reproducible audit loop for simplification `step by step` quality and
didactic substeps.

This track is intentionally different from metamorphic testing:

- metamorphic tests ask whether the engine is mathematically correct
- this audit asks whether the simplification trace is understandable for a
  human reader

## Substep Policy

Didactic substeps exist for one reason:

- explain a step that would otherwise feel like a magic jump

They do **not** exist to pad the trace.

So a good substep must do at least one of these:

- show a concrete intermediate mathematical state that is not already visible in
  the parent step
- isolate a local transformation that makes a non-obvious jump understandable
- name a temporary substitution or contextual identification that is actually
  needed to follow the algebra

And a bad substep is any of these:

- a generic template such as “use the identity ...” when no visible
  intermediate is shown
- a paraphrase of the parent step title
- a description of the maneuver without new math
- a duplicate of the parent step `before/after`

Important consequence:

- if a parent step is already self-explanatory, the correct number of substeps
  is often `0`
- adding a generic substep is worse than emitting no substeps at all

## Scope

The audit covers simplification expressions that are known to be didactically
interesting:

- rationalization
- nested fractions
- local focus / combine-like-terms
- cancellations and exact quotients
- trig identities
- inverse-trig reductions
- radicals / perfect squares

## Deliverables

1. A curated corpus file in
   [didactic_step_quality_cases.csv](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/tests/didactic_step_quality_cases.csv)
2. A reproducible runner in
   [didactic_step_quality_audit.rs](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/tests/didactic_step_quality_audit.rs)
3. A generated Markdown report at
   [DIDACTIC_STEP_QUALITY_AUDIT_REPORT.md](/Users/javiergimenezmoya/developer/math/docs/generated/DIDACTIC_STEP_QUALITY_AUDIT_REPORT.md)

## Runner Contract

For each corpus case the runner must:

1. Simplify the expression with `steps on`
2. Capture the human-oriented CLI simplification lines
3. Capture the wire/web step payload
4. Write both views into a Markdown report

The report is intended for manual review, so readability matters more than
compactness.

## Review Rubric

When reviewing the generated report, look for:

- steps that feel like a magic jump
- missing intermediate algebra that a human would expect
- weak or generic rule descriptions
- substeps that duplicate the main step instead of clarifying it
- substeps that only describe the maneuver without exposing an intermediate
  state
- formula-template substeps that add wording but no visible math
- missing local focus when only a subpart changed
- web payloads whose `before_latex` / `after_latex` are awkward or misleading
- cases with good math but poor narrative ordering

## First-Pass Heuristic Flags

The generated report surfaces lightweight flags to guide review:

- `no steps emitted`
- `no wire substeps emitted`
- `single step with no didactic substeps`
- `wire substeps with missing math sides`

These are review hints only, not hard failures of mathematical correctness.

In particular:

- `single step with no didactic substeps` is only a problem when the step still
  feels magical
- it is not a problem when the step is already direct and self-explanatory

## Execution

Validation:

```bash
cargo test -p cas_didactic --test didactic_step_quality_audit -- --nocapture
```

Report generation:

```bash
cargo test -p cas_didactic --test didactic_step_quality_audit \
  didactic_step_quality_audit_generates_markdown_report \
  -- --ignored --exact --nocapture
```

## Stopping Rule

This track should iterate in a loop:

1. refresh the report
2. review bad traces
3. improve step/substep quality
4. regenerate and compare

It should stop when the remaining complaints are sparse and specific, not when
the trace becomes longer by default.

The target is not “more substeps”.

The target is:

- fewer magical jumps
- no didactic filler
- no generic template noise
- no substep that exists only to make the UI look more detailed
