# Derive Didactic Audit

This audit tracks the educational quality of `derive`, independently from the
generic simplification audit.

Normalization reference:

- [DIDACTIC_SUBSTEP_NORMALIZATION.md](/Users/javiergimenezmoya/developer/math/docs/DIDACTIC_SUBSTEP_NORMALIZATION.md)

It exists for a different reason than correctness tests:

- `derive` is target-driven
- it should reach the requested target form
- and it should do so with steps that read naturally in CLI and web/JSON

Current audit entrypoints:

- contract and metrics:
  - [/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/derive_contract_tests.rs](/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/derive_contract_tests.rs)
- derive-specific didactic audit:
  - [/Users/javiergimenezmoya/developer/math/crates/cas_didactic/tests/derive_didactic_audit.rs](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/tests/derive_didactic_audit.rs)
- curated corpus:
  - [/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/derive_pairs.csv](/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/derive_pairs.csv)

Commands:

```bash
cargo test -p cas_solver --test derive_contract_tests -- --nocapture
cargo test -p cas_didactic --test derive_didactic_audit -- --nocapture
cargo test -p cas_didactic --test derive_didactic_audit derive_didactic_audit_generates_markdown_report -- --ignored --exact --nocapture
```

Generated report:

- [/Users/javiergimenezmoya/developer/math/docs/generated/DERIVE_DIDACTIC_AUDIT.md](/Users/javiergimenezmoya/developer/math/docs/generated/DERIVE_DIDACTIC_AUDIT.md)

Current quality gates:

- every curated `derived` pair must emit web steps
- no curated `derived` pair may emit a single substep that duplicates the parent
  step exactly
- no curated `derived` pair may keep a generic template substep that only says
  “use the identity” or similar without exposing any visible intermediate math
- no curated `derived` pair may keep a substep whose only job is to restate or
  rename the parent step
- no curated `derived` pair may keep generic placeholder math in a substep body
  when the concrete local rewrite is known
- the report must expose per-case CLI lines and web/JSON steps so new
  redundancies can be reviewed before growing more families

## What A Derive Substep Is For

`derive` is especially sensitive to fake didactic detail, because the feature is
already target-driven. If the trace reaches the requested target but the
substeps are only decorative, the result reads as magical anyway.

So in `derive`, a substep is justified only when it helps explain a jump that
the parent step does not already make visible.

A valid `derive` substep should do at least one of these:

- show an intermediate algebraic form hidden inside the parent step
- expose a local identity application that would otherwise feel abrupt
- make a substitution or pattern recognition explicit when that is necessary to
  follow the move

And when it exists, it should still follow the normalized public shape:

```text
[title]
[specific expression]
->
[specific expression]
```

So in `derive` too:

- the title may be generic
- the math must be specific
- prose belongs in the title, not inside the math lines

A `derive` substep is noise if it does any of these:

- paraphrases the main rule title
- describes the maneuver without new math
- repeats the same `before/after` as the parent step
- injects a formula-template line without any visible intermediate state
- uses placeholder math like `a`, `b`, `u`, `n` when the step already knows
  the concrete rewritten block

Important rule:

- if a `derive` step is already self-explanatory, prefer `0` substeps
- a clean direct step is better than a padded step-by-step trace

This audit should keep `derive` honest on that point: the goal is not to make
the trace look denser, but to make non-obvious transformations less magical.

This audit is intentionally conservative. It is meant to stop obvious regressions
in teachability, not to freeze the exact wording of every step forever.
