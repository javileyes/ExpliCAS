# Engine Combination Ledger

This document is a derived strategy under
[ENGINE_IMPROVEMENT_AUTOMATION.md](/Users/javiergimenezmoya/developer/math/docs/ENGINE_IMPROVEMENT_AUTOMATION.md).

It should drive an iteration only when the ROI selector chooses `combination`
and there is already a causally coherent complementary hypothesis.

This document records engine changes that produced a strong local win but were
not retained because they regressed a global guardrail, usually
`embedded_equivalence_context`.

The purpose is not to preserve rejected patches as a backlog of random ideas.

The purpose is to preserve *combinable hypotheses*:

- local improvements that may become globally good when paired with
  - a cheap gate
  - a narrower call-site
  - cache or signature reuse
  - a follow-up patch that removes the new global cost they introduce

That makes this ledger part of the engine improvement workflow, not an archive
of failed work.

Its role is narrow:

- preserve high-value local wins that failed global retention
- explain why they failed
- make future combination attempts evidence-driven

Its role is not:

- to justify retrying every rejected optimization
- to override the benchmark guardrails
- to compete with `runtime`, `coverage`, or `observability` as a default mode

## When To Add An Entry

Add an entry when all of these are true:

- the local profiler or narrow slice improves materially
- the global guardrail gets worse enough that the change should not be retained
- there is a concrete technical hypothesis for why the local win did not scale
- there is a plausible complementary change that could make the idea safe later

Do not add entries for:

- trivial coding mistakes
- changes with no measurable local win
- changes whose only explanation is “benchmark noise”

## Required Fields

Each entry should capture:

- area:
  - orchestrator / arithmetic / derive planner / corpus generation / etc.
- local lane:
  - exact command or corpus slice
- local win:
  - concrete profiler labels and deltas
- global result:
  - scorecard baseline, elapsed delta, and release rerun when available
- why it regressed globally:
  - the best current causal explanation
- what could make it combinable later:
  - gate, cache, narrower call-site, route split, ordering change, etc.
- status:
  - `rejected`, `observe-only`, `candidate for combination`, `superseded`

## Combination Rules

Combining rejected local wins is only worth doing when the combination is
causally coherent.

Good combination patterns:

- `expensive fast path + cheap gate`
- `repeated extraction + cache/reuse`
- `local win in shared helper + move to exact call-site`
- `hotspot shift + second patch that attacks the shifted hotspot`

Bad combination patterns:

- two broad helper-level fast paths that both add traffic
- more canonicalization in already shared hot paths
- combinations whose only rationale is “both looked good locally”

The burden of proof stays the same:

- rerun the narrow local slice
- rerun the scorecard with baseline
- rerun the direct release corpus if the scorecard is good enough

## Current Entries

### 2026-04-18: Binary-Add Surface Plain-Algebraic Gate

- area:
  - [arithmetic.rs](/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/arithmetic.rs)
  - `try_build_exact_trig_phase_shift_zero_scope_rewrite(...)`
  - `Expr::Add(lhs, rhs)` branch feeding `rule.phase_shift.binary_add_match`
- status:
  - `candidate for combination`
- local lane:
  - `CAS_PROFILE_ORCHESTRATOR_SHORTCUTS=1 cargo run -q -p cas_solver --example run_embedded_equivalence_context_corpus`
- attempted change:
  - reject `binary_add_match` when either side is a surface plain algebraic term
    (`a`, `-a`, `a*k`, `a/b`, etc.) before entering the full phase-shift matcher
- profiler signal:
  - `rule.phase_shift.binary_add_match`: `2336.640ms -> 1569.232ms`
  - `rule.phase_shift.binary_add_match.forward_try`: `822.854ms -> 789.083ms`
  - `rule.phase_shift.binary_add_match.reverse_try`: `1512.833ms -> 779.327ms`
  - filtered pair count stayed aligned with the previous broad trig-presence gate:
    - `516 -> 444`
- useful retained signal:
  - the surviving impossible traffic is concentrated in superficially plain
    algebraic terms paired against exact shifted trigs
  - the new surface classifier leaves a clearer split:
    - `surface_pair_shape.lhs_plain_algebraic`
    - `surface_pair_shape.rhs_plain_algebraic`
    - `surface_pair_shape.neither_plain_algebraic`
- global runtime result:
  - clean release rerun:
    - `cargo run --release -q -p cas_solver --example run_embedded_equivalence_context_corpus`
    - `1116/1116`
    - `0` fallos
    - `Elapsed: 7.58s`
  - retained baseline before the attempt:
    - `6.98s`
  - decision:
    - rejected as runtime change
- best current explanation:
  - even the cheap surface classifier still adds enough hot-path work at the
    binary-add call-site to lose globally
  - the fact that it filters almost exactly the same `72` pairs as the broader
    trig-presence gate suggests the real issue is not the screen cost alone, but
    that this caller is not large enough to justify any extra runtime filtering
- plausible combination later:
  - reuse term-shape metadata already produced by the surrounding additive view
    instead of recomputing per pair
  - or push the screen into an upstream phase where pair construction is cheaper
    and the same metadata is already materialized

### 2026-04-18: Binary-Add Trig-Pair Gate

- area:
  - [arithmetic.rs](/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/arithmetic.rs)
  - `try_build_exact_trig_phase_shift_zero_scope_rewrite(...)`
  - `Expr::Add(lhs, rhs)` branch feeding `rule.phase_shift.binary_add_match`
- status:
  - `candidate for combination`
- local lane:
  - `CAS_PROFILE_ORCHESTRATOR_SHORTCUTS=1 cargo run -q -p cas_solver --example run_embedded_equivalence_context_corpus`
- attempted change:
  - require both binary-add terms to contain `sin/cos` before entering
    `try_find_trig_phase_shift_cancellation_match(...)`
- profiler signal:
  - `rule.phase_shift.binary_add_match.trig_pair_gate`: `516` attempts, `444` hits, `72` misses
  - `rule.phase_shift.binary_add_match`: `2336.640ms -> 1573.269ms`
  - `rule.phase_shift.binary_add_match.forward_try`: `822.854ms -> 788.003ms`
  - `rule.phase_shift.binary_add_match.reverse_try`: `1512.833ms -> 784.381ms`
- useful retained signal:
  - the filtered traffic is real and impossible:
    - `a  ||  sin(x + 1/4*pi) * sqrt(2)`
    - `-a  ||  -(sin(x + 1/4*pi) * sqrt(2))`
    - `a*k  ||  k * sin(x + 1/4*pi) * sqrt(2)`
  - so `binary_add_match` really is receiving non-trig terms that cannot
    participate in phase-shift cancellation
- global runtime result:
  - clean release rerun:
    - `cargo run --release -q -p cas_solver --example run_embedded_equivalence_context_corpus`
    - `1116/1116`
    - `0` fallos
    - `Elapsed: 7.09s`
  - retained baseline before the attempt:
    - `6.98s`
  - decision:
    - rejected as runtime change
- best current explanation:
  - the gate cuts real dead traffic, but the recursive builtin scan at this
    binary-add call-site still costs more than the saved matcher work in release
- plausible combination later:
  - reuse trig-shape metadata already known by the caller instead of rescanning
    both terms
  - or add a stronger O(1) syntactic screen for the specific non-trig families
    that dominate the samples

### 2026-04-17: Direct-Core Trig Presence Gate

- area:
  - [arithmetic.rs](/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/arithmetic.rs)
  - `try_build_direct_core_equivalence_rewrite(...)`
- status:
  - `candidate for combination`
- local lane:
  - `CAS_PROFILE_ORCHESTRATOR_SHORTCUTS=1 cargo run -q -p cas_solver --example run_embedded_equivalence_context_corpus`
- attempted change:
  - gate the direct-core phase-shift matcher call so it only runs when both
    cores contain `sin/cos`
- local profiler signal:
  - `rule.phase_shift.direct_core_pair_candidate_gate`: `35529` attempts,
    `34879` hits, `650` misses
  - `rule.phase_shift.route.exact_try`: `73588` attempts vs previous `74380`
  - `rule.phase_shift.route.final_linear_compare_try`: `72680` attempts vs
    previous `74364`
- interpretation:
  - the gate was correct but weak
  - only a small fraction of direct-core traffic was actually algebraic enough
    to be filtered by “both sides must contain trig”
  - the true hotspots remained overwhelmingly large after the gate
- global runtime result:
  - clean release rerun:
    - `cargo run --release -q -p cas_solver --example run_embedded_equivalence_context_corpus`
    - `1116/1116`
    - `0` fallos
    - `Elapsed: 28.63s`
  - decision:
    - rejected as runtime change
- best current explanation:
  - this call-site does contribute some impossible traffic, but not enough to
    justify even a cheap builtin scan
  - the dominant residual cost still comes from genuinely trig-shaped but
    ultimately non-matching traffic deeper in `exact_try` and
    `final_linear_compare_try`
- plausible combination later:
  - reuse caller metadata stronger than bare trig presence
  - or split the late `exact_try` traffic by exact-shift plausibility without
    adding another generic recursive scan at the direct-core entry

### 2026-04-17: Exact-Scope Pair Candidate Gate

- area:
  - [arithmetic.rs](/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/arithmetic.rs)
  - `try_build_exact_trig_phase_shift_zero_scope_rewrite(...)`
- status:
  - `candidate for combination`
- local lane:
  - `CAS_PROFILE_ORCHESTRATOR_SHORTCUTS=1 cargo run -q -p cas_solver --example run_embedded_equivalence_context_corpus -- --family trig_contract --limit 24`
- attempted change:
  - add a call-site-only plausibility gate before `rule.phase_shift.exact_scope_pair_match`
  - require:
    - both sides contain `sin/cos`
    - and at least one side contain `pi` or inverse-trig shift markers
- local result:
  - no meaningful gain on the narrow slice
  - `Elapsed`: stayed at `1.28s`
  - `rule.phase_shift.exact_scope_pair_candidate_gate`: `24/24` hits
  - so the local `trig_contract` lane was already mostly inside the real family
- global profiler result:
  - `rule.phase_shift.exact_scope_pair_match`: `984 -> 186`
  - `rule.phase_shift.exact_scope_pair_candidate_gate`: `984 attempts`, `186` hits
  - but the true global hotspots barely moved:
    - `rule.phase_shift.route.exact_try`: `74380 -> 73588`
    - `rule.phase_shift.route.final_linear_compare_try`: `74364 -> 73572`
- global runtime result:
  - clean release rerun:
    - `cargo run --release -q -p cas_solver --example run_embedded_equivalence_context_corpus`
    - `1116/1116`
    - `0` fallos
    - `Elapsed: 29.70s`
  - decision:
    - rejected as runtime change
- best current explanation:
  - the gate successfully cuts dead traffic inside `exact_scope_pair_match`
  - but that traffic is not the dominant contributor to the retained global cost
  - the additional recursive scans for `sin/cos` and `pi/atan` at this call-site
    cost more than the saved work in the true hot path
- useful retained signal:
  - `exact_scope_pair_match` does contain plenty of impossible algebraic pairs
  - but reducing that subroute alone is not enough; the real cost still lives in
    later broad routes such as `exact_try` and `final_linear_compare_try`
- plausible combination later:
  - reuse metadata already computed while forming `focus_expr` / `remaining_expr`
    instead of rescanning both expressions
  - or move the route split even later, closer to the actual `exact_try` caller,
    where the semantic information is stronger and the gate can be cheaper

### 2026-04-17: Pair-Candidate Gate At Phase-Shift Matcher Entry

- area:
  - [arithmetic.rs](/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/arithmetic.rs)
  - `try_find_trig_phase_shift_cancellation_match(...)`
- status:
  - `candidate for combination`
- local lane:
  - `CAS_PROFILE_ORCHESTRATOR_SHORTCUTS=1 cargo run -q -p cas_solver --example run_embedded_equivalence_context_corpus -- --family trig_contract --limit 24`
- attempted change:
  - add a cheap gate at matcher entry requiring:
    - both sides to contain `sin/cos`
    - and at least one side to contain a phase-shift marker (`pi` or `atan`)
- local win:
  - `Elapsed`: `1.34s -> 1.28s`
  - profiler total: `934.146ms -> 901.985ms`
  - `rule.phase_shift.route.linear_focus_try`: `160 attempts -> 24`
  - `rule.phase_shift.route.general_try`: `140 attempts -> 4`
- global profiler result:
  - `CAS_PROFILE_ORCHESTRATOR_SHORTCUTS=1 cargo run -q -p cas_solver --example run_embedded_equivalence_context_corpus`
  - `rule.phase_shift.route.exact_try`: `74380 attempts -> 1978`
  - `rule.phase_shift.route.final_linear_compare_try`: `74364 attempts -> 1962`
  - new gate cut:
    - `rule.phase_shift.route.pair_candidate_gate`: `72058` attempts, `2050` hits
- global runtime result:
  - clean release rerun:
    - `cargo run --release -q -p cas_solver --example run_embedded_equivalence_context_corpus`
    - `1116/1116`
    - `0` fallos
    - `Elapsed: 27.99s`
  - baseline retained before the attempt:
    - `26.79s`
  - decision:
    - rejected as runtime change
- best current explanation:
  - the gate removes a huge amount of profiler-visible impossible traffic,
    especially before `exact_try` and `final_linear_compare_try`
  - but the recursive builtin and `pi/atan` scans at matcher entry are still too
    expensive on the normal hot path when the profiler is off
  - in other words, this was the right *semantic filter* at the wrong *cost
    location*
- useful retained signal:
  - the bad traffic is real
  - the profitable direction is still to cut phase-shift matching before
    `exact_try`
  - but the cut must reuse information already known at the caller instead of
    rescanning both expressions at matcher entry
- plausible combination later:
  - call-site-specific gating in `binary_add_match` or `direct_core_equivalence`
    using already available structure
  - cached term metadata for:
    - contains `sin/cos`
    - contains `pi`
    - contains inverse-trig shift markers
  - route splitting before `try_find_trig_phase_shift_cancellation_match(...)`
    so the matcher does not have to rediscover phase-shift plausibility itself

### 2026-04-17: Exact Target Raw Supported-Arg Fast Path

- area:
  - [arithmetic.rs](/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/arithmetic.rs)
  - `extract_exact_phase_shift_term_data_for_cancellation(...)`
- status:
  - `candidate for combination`
- local lane:
  - `CAS_PROFILE_ORCHESTRATOR_SHORTCUTS=1 cargo run -q -p cas_solver --example run_embedded_equivalence_context_corpus -- --family trig_contract --limit 24`
- attempted change:
  - after the existing `pi` gate, resolve these exact raw shapes directly before
    falling back to the general supported-arg extractor:
    - `pi / 4 + x`
    - `x - pi / 4`
    - `pi / 3 + x`
    - `pi / 6 + x`
- local win:
  - `rule.phase_shift.exact_scope_rewrite`: `44.411ms -> 9.546ms`
  - `rule.phase_shift.exact_scope_pair_match`: `43.971ms -> 9.130ms`
  - `root.div.03.shifted_quotient_nested_zero_core`: `16.221ms -> 7.156ms`
- global result:
  - baseline scorecard:
    - `/tmp/engine_improvement_scorecard_embedded_after_linear_focus_exact_target_pi_gated_fastpath.json`
  - scorecard result:
    - `26.88s -> 29.41s`
  - decision:
    - rejected as runtime change
- best current explanation:
  - the direct raw fast path is excellent for the narrow `trig_contract` slice,
    but too expensive when installed inside a shared extraction path that is hit
    by much broader traffic in `embedded`
- useful retained signal:
  - raw exact shapes are common in the hot slice
  - the real retained hotspot still sits in the general path:
    - `rule.phase_shift.exact_target_extract.shifted_arg_match`
    - `rule.phase_shift.supported_arg.normalized_simplify_fallback`
- plausible combination later:
  - do **not** revive this as a helper-global fast path
  - instead, combine it with one of:
    - call-site-only reuse in `exact_scope_pair_match`
    - cached exact target signature
    - a narrower route that proves the caller is already in the exact phase-shift
      family before paying the raw matcher

### 2026-04-17: Exact Target Raw Third/Sixth Fast Path

- area:
  - [arithmetic.rs](/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/arithmetic.rs)
  - `extract_exact_phase_shift_term_data_for_cancellation(...)`
- status:
  - `candidate for combination`
- local lane:
  - `CAS_PROFILE_ORCHESTRATOR_SHORTCUTS=1 cargo run -q -p cas_solver --example run_embedded_equivalence_context_corpus -- --family trig_contract --limit 24`
- attempted change:
  - after the existing `pi` gate, and only when `has_sqrt_two == false`,
    resolve raw exact `third/sixth` targets directly before calling the general
    supported-arg extractor
  - intended covered shapes:
    - `pi / 3 + x`
    - `pi / 6 + x`
    - `x - pi / 6`
- local win:
  - `rule.phase_shift.exact_scope_rewrite`: `43.741ms -> 9.855ms`
  - `rule.phase_shift.exact_scope_pair_match`: `43.272ms -> 9.433ms`
  - `root.div.03.shifted_quotient_nested_zero_core`: `15.749ms -> 7.155ms`
  - `rule.phase_shift.exact_target_extract.shifted_arg_match`: `34.237ms -> 0.118ms`
- global result:
  - baseline scorecard:
    - `/tmp/engine_improvement_scorecard_embedded_after_linear_focus_exact_target_pi_gated_fastpath.json`
  - scorecard result:
    - `26.88s -> 28.83s`
  - decision:
    - rejected as runtime change
- best current explanation:
  - even scoped to `third/sixth`, the raw fast path still adds enough shared
    extraction traffic to regress `embedded`, despite being excellent in the
    narrow `trig_contract` slice
- useful retained signal:
  - the expensive misses in the retained path are genuinely concentrated in
    `quarter -> third/sixth` fallback churn
  - the profitable part is the *routing information*, not the raw fast path
    itself
- plausible combination later:
  - pair this idea with exact-target signature reuse that is already computed by
    the caller, instead of re-extracting inside `extract_exact_phase_shift_term_data_for_cancellation(...)`
  - or narrow it even further to a route that has already proven it is in the
    `exact_scope_pair_match -> linear_to_shifted` family before attempting raw
    extraction

### 2026-04-17: Exact Target Kind-Hinted Extractor

- area:
  - [arithmetic.rs](/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/arithmetic.rs)
  - `find_linear_focus_phase_shift_cancellation_match(...)`
  - `extract_exact_phase_shift_term_data_for_cancellation(...)`
- status:
  - `candidate for combination`
- local lane:
  - `CAS_PROFILE_ORCHESTRATOR_SHORTCUTS=1 cargo run -q -p cas_solver --example run_embedded_equivalence_context_corpus -- --family trig_contract --limit 24`
- attempted change:
  - reuse the already known `exact_data.kind` from `linear_focus`
  - force exact-target extraction to try only the single denominator compatible
    with that expected `Quarter`/`Third`/`Sixth` family
- local win:
  - `rule.phase_shift.exact_scope_rewrite`: `43.741ms -> 10.903ms`
  - `rule.phase_shift.exact_scope_pair_match`: `43.272ms -> 10.452ms`
  - `root.div.03.shifted_quotient_nested_zero_core`: `15.749ms -> 8.024ms`
- global result:
  - baseline scorecard:
    - `/tmp/engine_improvement_scorecard_embedded_after_linear_focus_exact_target_pi_gated_fastpath.json`
  - scorecard result:
    - `26.88s -> 29.64s`
  - decision:
    - rejected as runtime change
- best current explanation:
  - even though the hint is semantically correct for the local
    `linear_to_shifted` family, specializing the exact-target extractor in place
    still regresses broader `embedded` traffic
  - that implies the global cost is not only “too many denominators tried”; it
    is also *where* the extraction happens and how often the route is reached
- useful retained signal:
  - denominator pruning alone is not enough
  - the next viable combination still points to one-shot target-signature reuse,
    not to further specialization inside the extractor call itself
- plausible combination later:
  - compute exact target signature once in `exact_scope_pair_match` and reuse it
    across `linear_focus` and later compare stages
  - or add a cache keyed by the target term before it enters the extractor

### 2026-04-18: Target-Exact Cache Inside Shared Matcher

- area:
  - [arithmetic.rs](/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/arithmetic.rs)
  - `try_find_trig_phase_shift_cancellation_match(...)`
  - `find_linear_focus_phase_shift_cancellation_match(...)`
  - `final_linear_compare`
- status:
  - `candidate for combination`
- attempted change:
  - cache `extract_exact_phase_shift_term_data_for_cancellation(target_expr)` once
    per matcher call and reuse it across:
    - `linear_focus.exact_target_match`
    - `exact_try.target_exact_extract`
    - `final_linear_compare.target_exact_linear`
- local lane:
  - `CAS_PROFILE_ORCHESTRATOR_SHORTCUTS=1 cargo run -q -p cas_solver --example run_embedded_equivalence_context_corpus -- --family trig_contract --limit 24`
  - result:
    - `Elapsed: 1.21s -> 1.23s`
    - no meaningful local win
- global result:
  - release rerun:
    - `6.49s -> 6.70s`
    - `1116/1116`, `0` fallos
  - decision:
    - rejected as runtime change
- best current explanation:
  - even scoped to the target side only, the extra cache plumbing inside the
    shared matcher adds more overhead than the repeated target extraction it
    removes
  - this suggests the profitable reuse point is not “inside the matcher”, but
    one layer above, where caller metadata can be passed in without reopening
    more mutable state and branching in the hot path
- plausible combination later:
  - reuse an already materialized exact-target signature from the caller of
    `exact_scope_pair_match`, instead of introducing cache logic inside
    `try_find_trig_phase_shift_cancellation_match(...)`
