# Post-V2.4 Verification & Guards Roadmap

> Created: 2026-03-15
>
> This roadmap starts after
> [ROADMAP_V2_1_TO_V2_4.md](/Users/javiergimenezmoya/developer/math/docs/ROADMAP_V2_1_TO_V2_4.md)
> was closed for its conservative scope.

---

## Baseline

What is already true today:

- `solve --check` is stable for discrete solutions.
- Non-discrete verification no longer collapses everything into one bucket:
  - `VerifiedUnderGuard`
  - `NeedsSampling`
  - `NotCheckable`
- Failed discrete verification can surface a conservative counterexample hint.
- Branch-sensitive residuals (`log`, `sqrt`, inverse trig) are excluded from
  hint generation by policy.
- The active guard vocabulary is aligned across runtime and docs:
  - `NonZero`
  - `Positive`
  - `NonNegative`
  - `EqZero`
  - `EqOne`

What this roadmap is for:

- improving verification UX without weakening soundness
- expanding native guard recognition carefully
- preventing verification-path performance regressions

What this roadmap is not for:

- reopening the old `NeZero / NeOne` design
- broad symbolic search for counterexample hints
- turning every interval/union into a symbolic proof claim

---

## Active Backlog

## 1. Verification UX Contracts

Status: done.

Goal:
- make `solve --check` outputs easier to read without changing proof policy

Concrete scope:
- add one curated snapshot/contract suite for representative `solve --check`
  outputs:
  - all discrete verified
  - partially verified with non-discrete note
  - needs sampling with guard hint
  - failed discrete with counterexample hint
  - failed discrete with suppressed hint
- keep the display wording stable enough for regression detection

Current coverage already in place:
- session-render contracts exist for:
  - `AllVerified`
  - `NeedsSampling` with interval guard hint (`x > 0`)
  - `NeedsSampling` with union guard hint (`x != 0`)
  - mixed conditional with verified discrete branch + non-discrete note
  - `AllReals -> NotCheckable`
- lower render contracts exist for:
  - failed discrete with counterexample hint
  - failed discrete with suppressed hint

Done when:
- the public render surface is pinned for the five cases above
- wording changes become intentional instead of accidental

## 2. Native Guard Recognition Expansion

Status: done.

Goal:
- cover a few more non-discrete families natively before falling back to
  `NeedsSampling`

Current native families:
- `x > 0`
- `x >= 0`
- `x < 0`
- `x <= 0`
- `x != 0`

Current coverage already in place:
- direct `verify_solution_set(...)` contracts exist for:
  - `x > 0`
  - `x >= 0`
  - `x < 0`
  - `x <= 0`
  - `x != 0`
- public `solve --check` session-render contracts exist for:
  - `x > 0`
  - `x < 0`
  - `x <= 0`
  - `x != 0`
- `x >= 0` remains covered directly in verification contracts; there is no
  dedicated public session case yet because the current public examples already
  pin the same `NeedsSampling` wording through neighboring interval families

Rules:
- only claim `VerifiedUnderGuard` when the guard-shaped justification is explicit
- otherwise stay in `NeedsSampling`

Outcome:
- 2 additional interval families (`x < 0`, `x <= 0`) are now covered
  end-to-end
- solver/render contracts cover the public path without widening proof claims
- native recognition is slightly broader, but still conservative

## 3. Counterexample Hint Policy Tightening

Status: done.

Goal:
- make hint policy more explicit, not more aggressive

Concrete scope:
- document the current suppression taxonomy in one place
- decide whether suppressed hints should remain silent or surface a neutral
  note such as `hint omitted for branch-sensitive residual`
- if surfaced, do it only for debug/explain-oriented output, not by default

Current policy now fixed and tested:
- suppressed hint families are:
  - `log` / `ln`
  - `sqrt`
  - inverse trig (`asin/acos/atan/...` and `arcsin/arccos/arctan/...`)
- suppression remains silent in normal user-facing output
- explain-oriented solve render (`Verbose` / debug) may surface a neutral note
  explaining that the hint was omitted for a branch-sensitive residual
- both surfaced hints and suppression notes remain gated by `hints_enabled`
- the same `hints_enabled` gate is now available through the public verification
  formatter API, not only through the solve command render path
- that public formatter API is now covered by integration contracts
- failed discrete verification still surfaces a concrete hint when the residual
  is finite, probeable, and outside those branch-sensitive families
- render/session contracts already pin both:
  - surfaced hint behavior
  - suppressed-hint behavior

Non-goal:
- expanding beyond the current tiny literal set unless perf and soundness stay
  clearly acceptable

Outcome:
- suppression taxonomy is explicit in runtime code and tests
- user-facing behavior is intentional and stable

## 4. Verification Performance Guardrails

Status: done.

Goal:
- keep verification-path improvements from regressing solver responsiveness

Existing asset:
- dedicated verification bench in
  `/Users/javiergimenezmoya/developer/math/crates/cas_engine/benches/profile_cache.rs`
  under `solver_verification_inherited_steps`
- reproducible make entrypoints:
  - `make bench-engine-verification`
  - `make bench-engine-verification-save BASELINE=...`
  - `make bench-engine-verification-compare BASELINE=...`
- named baseline `verification_guardrails_post_v24` is now the current
  reproducible reference for this bench group
- immediate compare against that named baseline showed:
  - `quadratic_two_roots_steps_on`: no change detected
  - `failed_discrete_with_hint_steps_on`: change stayed within noise threshold
  - `failed_discrete_log_hint_suppressed_steps_on`: no change detected
  - `needs_sampling_positive_interval_steps_on`: no change detected
  - `needs_sampling_nonzero_union_steps_on`: no change detected
- practical read: the earlier local "regressed" marker on the unnamed baseline
  was stale-reference noise, not a reproduced runtime regression

Current coverage already in place:
- `quadratic_two_roots_steps_on`
- `failed_discrete_with_hint_steps_on`
- `failed_discrete_log_hint_suppressed_steps_on`
- `needs_sampling_positive_interval_steps_on`
- `needs_sampling_nonzero_union_steps_on`
- ignored repro tests now mirror the same families in
  `/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/repro_bench.rs`
- cheap structural guards now protect:
  - the `AllVerified` quadratic-two-roots path with inherited `steps = on`
  - the failed-discrete-hint path, both with default steps and with inherited
    `steps = on`
  - the suppressed-log failed-discrete path with inherited `steps = on`
  - the native `x > 0` `NeedsSampling` path with inherited `steps = on`
  - the native `x != 0` `NeedsSampling` path, both with default steps and with
    inherited `steps = on`
  against obvious node-growth regressions in normal test runs

Concrete scope:
- identify 1-2 verification-heavy classroom cases worth protecting
- decide whether they belong in:
  - criterion benches
  - ignored repro tests
  - cheap structural guards in normal CI
- ensure counterexample-hint probing does not become a hidden hotspot

Done when:
- at least one verification-path perf guardrail is explicit
- the chosen guardrail is documented and reproducible

## 5. Guard Model Consolidation

Status: done.

What is already done:
- `Positive(x) -> NonZero(x)`
- `Positive(x) -> NonNegative(x)`
- `EqOne(x) -> NonZero(x)`
- `EqOne(x) -> Positive(x)`
- `EqOne(x) -> NonNegative(x)`

What remains worth deciding:
- whether any additional implication is both mathematically safe and
  pedagogically useful
- whether more guard simplification belongs in `cas_ast::ConditionSet` or
  should stay as solver-level proof knowledge

Default recommendation:
- do not add more implications without a concrete user-facing win

Done when:
- either one new implication is justified and tested, or this area is
  explicitly frozen

---

## Execution Order

1. Verification UX contracts.
2. Verification performance guardrails.
3. Native guard recognition expansion.
4. Counterexample hint policy tightening.
5. Optional extra guard implication, only if justified by a concrete case.

Why this order:
- the first two items improve safety and observability
- only then is it worth extending verification claims

---

## Definition Of Done

This roadmap can be considered complete for its conservative scope when:

- `solve --check` has stable public contracts for the main verification states
- at least one verification perf guardrail is in place
- native non-discrete guard recognition is slightly broader but still
  conservative
- hint suppression policy is explicit instead of implicit
- no new guard implication is added without tests and a user-facing reason
