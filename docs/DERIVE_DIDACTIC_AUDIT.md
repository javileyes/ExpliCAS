# Derive Didactic Audit

This audit tracks the educational quality of `derive`, independently from the
generic simplification audit.

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
- the report must expose per-case CLI lines and web/JSON steps so new
  redundancies can be reviewed before growing more families

This audit is intentionally conservative. It is meant to stop obvious regressions
in teachability, not to freeze the exact wording of every step forever.
