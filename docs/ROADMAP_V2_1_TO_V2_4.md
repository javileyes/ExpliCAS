# Roadmap V2.1 -> V2.4: Status Refresh

> Status refresh: 2026-03-15
>
> This file is no longer a speculative roadmap. It is the current triage view of
> the original V2.1 -> V2.4 plan: what is done, what is still worth doing, and
> what was superseded by the current design.
>
> Current conclusion: the roadmap is functionally complete for the conservative
> scope retained here.

---

## Summary

### Closed

- Issue #1: output polish for `otherwise:` is done.
- Issue #2: REPL iconic snapshots are done.
- Issue #3: explain mode is done.
- Issue #4: stable solver API is done.
- Issue #5: `solve --check` exists and is stable.
- Issue #6: non-discrete verification states are done for the narrow,
  conservative scope defined here.
- Issue #10: clear denominators with guards is done.

### Remaining work inside this roadmap

- none for the conservative scope retained here

### Superseded by the current design

- Issue #8 and Issue #9, as written, are outdated.
- The current guard model uses `NonZero`, `Positive`, `NonNegative`,
  `EqZero`, `EqOne`, not `NeZero` / `NeOne`.
- Part of the intended simplification work already exists in the current
  `ConditionSet` implementation.

### Separate track, not this roadmap

- Metamorphic benchmark hardening, `numeric-only` reduction, `domain-frontier`
  classification and `safe-window` mirrors now live in:
  - [METAMORPHIC_NEXT_LEVEL_PLAN.md](/Users/javiergimenezmoya/developer/math/docs/METAMORPHIC_NEXT_LEVEL_PLAN.md)
  - [METAMORPHIC_TESTING.md](/Users/javiergimenezmoya/developer/math/docs/METAMORPHIC_TESTING.md)
- Performance guardrails for pathological simplification cases are now tracked
  by repro/bench tests and CI contracts, not by this V2.1 -> V2.4 roadmap.

---

## Closed Foundation

### Issue #1: `otherwise:` output polish

Status: done.

Evidence:
- console and display use `otherwise:` without the extra `if`
- REPL snapshots cover the format

### Issue #2: REPL iconic snapshots

Status: done.

Evidence:
- the iconic solver cases are snapshotted and stable
- snapshot tests run in CI

### Issue #3: explain mode

Status: done.

Evidence:
- `explain on|off` exists
- solve output already surfaces:
  - assumptions used
  - blocked simplifications

### Issue #4: stable solver API

Status: done.

Evidence:
- stable exports are re-exported from the public API
- compile-contract tests exist for the exposed types

### Issue #5: `solve --check`

Status: done for discrete solutions.

What exists today:
- one-shot `solve --check ...`
- semantic toggle for solve checking
- stable verified / unverifiable / not-checkable display

What is intentionally not counted as done here:
- symbolic verification of intervals, unions, and all-reals branches

### Issue #10: clear denominators with guards

Status: done.

Evidence:
- denominator guards are preserved in the solve result
- rational-equation classroom cases are covered by contract tests

---

## Active Backlog

## Issue #6: Verification for non-discrete solution sets

Status: done for the current conservative scope.

Why it still matters:
- today, non-discrete outputs are mapped directly to `NotCheckable`
- that is honest, but educationally incomplete

Current behavior:
- `AllReals` -> `not checkable (infinite set: all reals)`
- `Continuous` -> `verification requires numeric sampling`, with native guard
  hints for simple interval families such as ``x > 0`` / ``x >= 0``
- `Union` -> `verification requires numeric sampling`, with native guard hints
  for simple unions such as ``x != 0``
- simple conditional non-discrete branches with explicit guards
  (`NonZero`, `Positive`, `NonNegative`) can already surface as
  `verified symbolically under guard`
- mixed conditional outputs no longer hide non-discrete branches when one
  discrete branch verifies: `solve --check` now keeps an explicit note such as
  `some non-discrete branches require numeric sampling`
- solver/API contract tests now cover:
  - intervals coming from positivity guards
  - unions of intervals
  - conditional branches with non-discrete `then`
  - the fallback `AllReals -> not checkable`

Done when:
- [x] solve checking can distinguish:
  - `verified symbolically under guard`
  - `requires numeric sampling`
  - `not checkable`
- [x] it never overclaims verification
- [x] coverage exists for:
  - intervals coming from positivity guards
  - unions of intervals
  - conditional branches with non-discrete `then`

Recommended scope:
- start narrow
- only prove guarded non-discrete cases when the solver already has enough
  structural information to justify the branch
- do not attempt general quantified proof machinery

Intentionally out of scope:
- general quantified proof for arbitrary intervals and unions
- turning every native interval into `verified under guard`
- symbolic coverage claims without an explicit guard-shaped justification

## Issue #7: Counterexample hint on failed verification

Status: done for the current conservative scope.

Why it still matters:
- today, failed discrete verification tells the truth, but does not help the
  student understand why

What exists today:
- failed discrete verification can attach a tiny counterexample hint when the
  substituted residual still depends on free parameters
- the probe set is intentionally tiny and deterministic: `0`, `1`, `2`, `-1`
- undefined and non-finite probes are skipped instead of being reported as
  evidence
- branch/domain-sensitive residuals with explicit `log`, `sqrt`, or
  inverse-trig structure are conservatively excluded from counterexample hints
- when a concrete probe is found, the verification display surfaces it inline
  under the failed solution
- when no probe is found, nothing is invented
- solver/render contract coverage exists both at verification-summary level and
  at `solve` render/session level

Done when:
- [x] on a failed discrete verification, the solver tries a tiny counterexample
  search over simple literals such as `0`, `1`, `2`, `-1` when applicable
- [x] if a counterexample is found, the UI surfaces it as a hint
- [x] if none is found, nothing is invented
- [x] tests cover:
  - a real counterexample found
  - no counterexample found
  - branch/domain-sensitive cases where probing must be suppressed

Recommended scope:
- only for discrete failures
- only for small literal probes
- no symbolic search tree

What is still missing:
- a broader search strategy, if we ever want hints beyond tiny deterministic
  literals
- any policy expansion beyond the current conservative suppression set

Priority:
- lower than Issue #6
- best done after the verification result model for non-discrete sets is
  cleaned up

---

## Superseded / Rewrite Required

## Issue #8: `NeZero / NeOne`

Status: obsolete as written.

Why it is outdated:
- the real implementation uses `NonZero`, not `NeZero`
- the guard system already contains:
  - `NonZero`
  - `Positive`
  - `NonNegative`
  - `EqZero`
  - `EqOne`

What already exists:
- redundancy simplification:
  - `Positive(x) -> NonZero(x)`
  - `EqOne(x) -> NonZero(x)`
- contradiction detection:
  - `EqZero(x)` with `NonZero(x)`
  - `EqZero(x)` with `Positive(x)`
  - `EqZero(x)` with `EqOne(x)`

What remains worth deciding:
- [ ] explicitly decide whether `NeOne` is ever needed at all
- [x] if not, update design docs to stop referencing it
- [ ] if yes, justify a concrete user-facing use case before adding it

Pragmatic recommendation:
- do not implement `NeZero` / `NeOne` just to satisfy the old roadmap wording
- refresh the docs around the actual current guard vocabulary instead

## Issue #9: Extra guard implications

Status: partially done, partially still open as a policy choice.

Already done:
- `Positive(x) -> NonZero(x)`
- `EqOne(x) -> NonZero(x)`
- `EqOne(x) -> Positive(x)`
- `EqOne(x) -> NonNegative(x)`

Still open only if we want it:
- none for the currently retained real-only guard model

Pragmatic recommendation:
- treat this as a small polish task on the guard model
- not as a major roadmap item

---

## Post-Roadmap Follow-up

What still makes sense after closing this roadmap:

1. Optional: one small extra implication if it proves pedagogically useful.
2. Optional: broader counterexample-hint search, but only if we accept a less
   conservative policy.
3. Ongoing verification/guards backlog now lives in:
   [ROADMAP_POST_V2_4_VERIFICATION_GUARDS.md](/Users/javiergimenezmoya/developer/math/docs/ROADMAP_POST_V2_4_VERIFICATION_GUARDS.md)

---

## Definition Of Done For This Roadmap

This roadmap can be considered complete when:

- discrete solve checking remains stable
- non-discrete solve checking no longer collapses everything into
  `not checkable`
- failed discrete checks can optionally provide a basic counterexample hint
- the documentation no longer refers to guard predicates that the codebase
  does not actually use

Current status against that definition:

- [x] discrete solve checking remains stable
- [x] non-discrete solve checking no longer collapses everything into
  `not checkable`
- [x] failed discrete checks can provide a conservative counterexample hint
- [x] the active documentation is fully refreshed away from the old guard
  naming (historical references remain only in this roadmap note)

That makes this roadmap complete for the conservative scope retained here.
