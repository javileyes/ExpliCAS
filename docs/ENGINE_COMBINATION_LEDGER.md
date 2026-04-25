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

### 2026-04-25: Nested-Fraction Three-Core Live Corpus Promotion

- area:
  - corpus generation / embedded equivalence live promotion
  - [embedded_equivalence_context_corpus.csv](/Users/javiergimenezmoya/developer/math/docs/embedded_equivalence_context_corpus.csv)
- status:
  - `rejected`
- local lane:
  - `python3 scripts/engine_embedded_candidate_smoke.py --json --row '((u*v)/(u+v) - 1/(1/u + 1/v)) + (sec(y) - 1/cos(y)) + (ln(a)+ln(b)-ln(a*b)),0,combined_additive_zero,nested_fraction_trigreciprocal_log_contract_three_core_combined_zero,nested_fraction,(u*v)/(u+v),1/(1/u + 1/v),collapse exact zero additive subexpressions'`
  - `cargo run --release -q -p cas_solver --example run_embedded_equivalence_context_corpus -- --wrapper combined_additive_zero --family nested_fraction`
- local value:
  - smoke passed `1/1`
  - local `nested_fraction` slice passed `4/4`
  - projected coverage gain was `combined_additive_zero.multi_core_families: 2/23 -> 3/23`
- global result:
  - retained baseline before the attempt:
    - `embedded_equivalence_context`: `1332/1332`, `Elapsed: 3.87s`
  - attempted promotion:
    - first guardrail: `1333/1333`, `Elapsed: 5.06s`
    - rerun guardrail: `1333/1333`, `Elapsed: 5.10s`
  - pressure suite still passed:
    - `simplify_zero_mixed`: `450/450`
  - decision:
    - rejected as live corpus promotion
- best current explanation:
  - the three-core row combines nested-fraction cancellation, trig reciprocal,
    and symbolic log contraction in a way that is semantically valid but too
    expensive for a single live row
  - the smoke lane itself took `1.35s`, so the cost appears attached to this
    candidate shape rather than to scorecard noise
- plausible combination later:
  - keep this shape as a discovery/stress candidate, not live
  - find a cheaper `nested_fraction` multi-core representative without symbolic
    log contraction, or first add profiling that isolates which sub-block makes
    this exact composition hot

### 2026-04-21: Fast Solve-Prep Neg-Rewritten Default-Simplify Route

- area:
  - [arithmetic.rs](/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/arithmetic.rs)
  - `try_build_fast_solve_prep_exact_zero_scope_rewrite(...)`
- status:
  - `candidate for combination`
- local lane:
  - `CAS_PROFILE_ORCHESTRATOR_SHORTCUTS=1 CAS_PROFILE_ORCHESTRATOR_SHORTCUT_FILTER=rule.direct_identity.try.zero_scope_fast_solve_prep,rule.fast_solve_prep.,rule.solve_prep. cargo run --release -q -p cas_solver --example run_embedded_equivalence_context_corpus -- --limit 480`
- attempted change:
  - promote an extra `neg_rewritten` + `default_simplify` comparison before
    falling back to `candidate_total_zero`
- local profiler signal:
  - `rule.fast_solve_prep.route.neg_rewritten_default_simplify_match`: `4` hits
  - `rule.fast_solve_prep.route.candidate_total_zero`: `8 -> 4`
  - the moved traffic stayed concentrated in the same shape bucket:
    - `add(add, div) || neg(mul)`
- global runtime result:
  - retained baseline on the same profiled embedded slice:
    - `Elapsed: 762.60ms`
    - `rule.direct_identity.try.zero_scope_fast_solve_prep`: `47.855ms`
    - `TOTAL`: `72.885ms`
  - attempted route enabled:
    - `Elapsed: 805.35ms`
    - `rule.direct_identity.try.zero_scope_fast_solve_prep`: `102.883ms`
    - `TOTAL`: `127.485ms`
  - decision:
    - rejected as runtime change
- best current explanation:
  - the extra whole-expression `default_simplify` comparison is broad enough
    that it costs more than the saved `candidate_total_zero` traffic on the
    shared `fast_solve_prep` path
  - the local win is real, but the call-site is still too broad for this check
    to pay for itself as a default branch
- follow-up combination check:
  - paired later with a retained `remaining_shifted_square` gate in
    `fast_solve_prep`
  - local movement improved:
    - `rule.fast_solve_prep.route.candidate_total_zero`: `8 -> 4`
    - `rule.fast_solve_prep.route.neg_rewritten_default_simplify_match`: `4`
  - but the global guarded slice still regressed:
    - retained gated baseline:
      - `Elapsed: 753.61ms`
      - `TOTAL: 71.558ms`
    - gated + neg-rewritten route:
      - `Elapsed: 824.95ms`
      - `TOTAL: 128.302ms`
  - conclusion:
    - `remaining_shifted_square` is not a narrow enough partner for this route
- plausible combination later:
  - pair it with a still narrower syntactic gate specific to the surviving
    `add(add, div) || neg(mul)` pocket, not merely “remaining has shifted
    square”
  - or reuse caller metadata so the extra negated-compare only runs on a
    subset already known to have exact cancellation plausibility

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

### 2026-04-21: One-Pass Quadratic Term Analyzer

- area:
  - [quadratic_coeffs.rs](/Users/javiergimenezmoya/developer/math/crates/cas_solver_core/src/quadratic_coeffs.rs)
  - `extract_quadratic_coefficients(...)`
  - `analyze_term(...)`
- status:
  - `rejected as runtime change`
- attempted change:
  - replace the current `contains_var`-driven branching in `analyze_term(...)`
    with a one-pass recursive analyzer for `Mul`/`Div`/`Neg`
  - keep constant subtrees opaque and only rebuild coefficients after recursive
    degree analysis
- local lane:
  - `CAS_PROFILE_ORCHESTRATOR_SHORTCUTS=1 CAS_PROFILE_ORCHESTRATOR_SHORTCUT_FILTER=rule.direct_identity.try.zero_scope_fast_solve_prep,rule.fast_solve_prep.,rule.solve_prep. cargo run --release -q -p cas_solver --example run_embedded_equivalence_context_corpus -- --family solve_prep`
  - result:
    - retained baseline before attempt:
      - `Elapsed: 37.85ms`
      - `TOTAL: 42.830ms`
      - `rule.solve_prep.build_candidate.extract_coeffs: 10.264ms`
    - attempted change, hot rerun:
      - `Elapsed: 41.36ms`
      - `TOTAL: 46.217ms`
      - `rule.solve_prep.build_candidate.extract_coeffs: 10.755ms`
- global result:
  - guardrail lane:
    - `CAS_PROFILE_ORCHESTRATOR_SHORTCUTS=1 CAS_PROFILE_ORCHESTRATOR_SHORTCUT_FILTER=rule.direct_identity.try.zero_scope_fast_solve_prep,rule.fast_solve_prep.,rule.solve_prep. cargo run --release -q -p cas_solver --example run_embedded_equivalence_context_corpus -- --limit 480`
  - retained baseline before attempt:
    - `Elapsed: 706.67ms`
    - `TOTAL: 38.352ms`
    - `rule.solve_prep.build_candidate.extract_coeffs: 9.063ms`
  - attempted change, hot rerun:
    - `Elapsed: 716.12ms`
    - `TOTAL: 41.026ms`
    - `rule.solve_prep.build_candidate.extract_coeffs: 9.715ms`
  - decision:
    - reverted
- best current explanation:
  - the repeated `contains_var(...)` scans are not the dominant cost in the
    retained `solve_prep` pocket
  - replacing them with a more generic recursive analyzer adds its own control
    flow and coefficient rebuilding overhead before any real candidate is
    formed
- useful retained signal:
  - `extract_coeffs` should not be attacked with a broad analyzer rewrite in
    shared core
  - the next viable move should be narrower:
    - route-specific prefilters before quadratic extraction
    - or observability inside coefficient extraction to separate
      coefficient-shape cost from target-variable scans

### 2026-04-21: Deferred `solve_prep` `c` Simplification

- area:
  - [arithmetic.rs](/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/arithmetic.rs)
  - `extract_profiled_solve_prep_nonzero_quadratic_coefficients(...)`
- status:
  - `rejected as runtime change`
- attempted change:
  - stop simplifying the extracted constant term `c` inside `solve_prep`
  - rely on later `tail` / final-candidate simplification to normalize it
- local lane:
  - `CAS_PROFILE_ORCHESTRATOR_SHORTCUTS=1 CAS_PROFILE_ORCHESTRATOR_SHORTCUT_FILTER=rule.direct_identity.try.zero_scope_fast_solve_prep,rule.fast_solve_prep.,rule.solve_prep. cargo run --release -q -p cas_solver --example run_embedded_equivalence_context_corpus -- --family solve_prep`
  - retained baseline before attempt:
    - `Elapsed: 47.44ms`
    - `TOTAL: 56.502ms`
    - `rule.solve_prep.extract.simplify_c: 4.958ms`
  - attempted change:
    - `Elapsed: 57.03ms`
    - `TOTAL: 113.983ms`
    - `rule.solve_prep.extract.defer_simplify_c: 0.001ms`
    - but `rule.solve_prep.build_candidate.extract_coeffs: 10.973ms -> 16.229ms`
    - and `rule.fast_solve_prep.gate.focus_remaining_var_mismatch: 16 -> 56`
- global result:
  - guardrail lane:
    - `CAS_PROFILE_ORCHESTRATOR_SHORTCUTS=1 CAS_PROFILE_ORCHESTRATOR_SHORTCUT_FILTER=rule.direct_identity.try.zero_scope_fast_solve_prep,rule.fast_solve_prep.,rule.solve_prep. cargo run --release -q -p cas_solver --example run_embedded_equivalence_context_corpus -- --limit 480`
  - retained baseline before attempt:
    - `Elapsed: 723.97ms`
    - `rule.solve_prep.build_candidate.extract_coeffs: 9.735ms`
  - attempted change:
    - `Elapsed: 761.06ms`
    - `rule.solve_prep.build_candidate.extract_coeffs: 15.329ms`
    - `rule.fast_solve_prep.gate.focus_remaining_var_mismatch: 16 -> 56`
  - decision:
    - reverted
- best current explanation:
  - `simplify_c` is expensive, but removing it broadens the upstream shape
    space enough to reopen a larger dead-traffic pocket than the cost it saves
  - the saved work is real; the placement is wrong
- plausible combination later:
  - defer `c` normalization only after a narrower route-specific prefilter
  - or only for one build-route once the `focus_remaining_var_mismatch` pocket
    is already closed

### 2026-04-21: Narrow `pos_generic` Negative-Linear `c` Deferral

- area:
  - [arithmetic.rs](/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/arithmetic.rs)
  - `extract_profiled_solve_prep_nonzero_quadratic_coefficients(...)`
- status:
  - `rejected as runtime change`
- attempted change:
  - defer `c` only for the `pos_generic` pocket measured as:
    - `a.shape = atom`
    - `b.shape = neg_atom`
    - `c.shape = addsub_with_div`
    - `focus_no_shifted_square`
  - this was narrow enough to make the focused regression
    `a*x^2 - b*x + (c + d - d)` pass in fast `solve_prep`
- local lane:
  - `CAS_PROFILE_ORCHESTRATOR_SHORTCUTS=1 CAS_PROFILE_ORCHESTRATOR_SHORTCUT_FILTER=rule.solve_prep.extract.simplify_a,rule.solve_prep.extract.simplify_a.shape.,rule.solve_prep.extract.simplify_b,rule.solve_prep.extract.simplify_b.shape.,rule.solve_prep.extract.simplify_c,rule.solve_prep.extract.simplify_c.build.,rule.solve_prep.extract.simplify_c.shape.,rule.solve_prep.extract.pos_generic.,rule.solve_prep.extract.defer_simplify_c.,rule.direct_identity.try.zero_scope_fast_solve_prep,rule.fast_solve_prep.,rule.solve_prep. cargo run --release -q -p cas_solver --example run_embedded_equivalence_context_corpus -- --family solve_prep`
  - retained baseline before attempt:
    - `Elapsed: 43.27ms`
    - `TOTAL: 52.250ms`
    - `rule.fast_solve_prep.try.collect_rewrites: 16`
    - `rule.solve_prep.build_candidate.extract_coeffs: 16`
  - attempted change:
    - `Elapsed: 47.28ms`
    - `TOTAL: 57.535ms`
    - `rule.fast_solve_prep.try.collect_rewrites: 16 -> 72`
    - `rule.solve_prep.build_candidate.extract_coeffs: 16 -> 104`
    - `rule.solve_prep.extract.defer_simplify_c.pos_generic_scale: 4`
    - sample dead traffic reopened as `:: b`
- decision:
  - reverted
- best current explanation:
  - the pocket is real, but skipping `c` normalization there is still early enough
    to broaden the rewrite search around the raw focus
  - the next viable move is not another `defer_c` variant
  - it should be either:
    - a dedicated negative-linear symbolic route that keeps `c` normalized
    - or a prefilter earlier than coefficient extraction

### 2026-04-25: `nested_fraction` Three-Core Coverage Candidate

- area:
  - [embedded_equivalence_context_corpus.csv](/Users/javiergimenezmoya/developer/math/docs/embedded_equivalence_context_corpus.csv)
  - `combined_additive_zero`
- status:
  - `rejected as live coverage row`
- attempted row:
  - `nested_fraction_trigreciprocal_factor_three_core_combined_zero`
  - `(1/(1/u + 1/v) - (u*v)/(u+v)) + (sec(y) - 1/cos(y)) + (a^2-b^2 - (a-b)*(a+b))`
- local result:
  - candidate smoke passed:
    - `1/1` cases
    - `runner_elapsed=1.34s`
    - `wall=1.664s`
  - focused slice passed:
    - `combined_additive_zero` x `nested_fraction`
    - `4/4` cases
    - `Elapsed: 1.30s`
  - `make engine-fast` passed
- global result:
  - `make engine-scorecard` passed correctness:
    - `embedded_equivalence_context: 1344/1344`
    - `combined_additive_zero: 102`
    - `nested_fraction: 4`
    - `multi_core_family_count: 14`
  - but the embedded lane cost rose from the retained baseline:
    - `Elapsed: 4.89s -> 6.13s`
- decision:
  - reverted the live corpus row
  - keep the expression as a discovered stress candidate, not as retained
    coverage
- best current explanation:
  - the candidate proves the missing structural axis, but the nested fraction
    core is expensive enough that composing it with two additional exact-zero
    cores is a poor live guardrail tradeoff
- plausible combination later:
  - find a cheaper `nested_fraction` representative for the same three-core
    axis
  - or improve nested fraction runtime before promoting this shape

### 2026-04-25: `fraction_decompose` Three-Core Coverage Candidate

- area:
  - [embedded_equivalence_context_corpus.csv](/Users/javiergimenezmoya/developer/math/docs/embedded_equivalence_context_corpus.csv)
  - `combined_additive_zero`
- status:
  - `rejected before live promotion`
- attempted row:
  - `fraction_decompose_trigreciprocal_factor_three_core_combined_zero`
  - `((a*x+b)/(x+c) - (a + (b-a*c)/(x+c))) + (sec(y) - 1/cos(y)) + (p^2-q^2 - (p-q)*(p+q))`
- local lane:
  - `python3 scripts/engine_embedded_candidate_smoke.py --timeout-seconds 6 --expect pass --row ...`
- local result:
  - timed out before promotion:
    - `status=timeout`
    - `wall=6.005s`
- decision:
  - did not add the row to the live corpus
  - keep this exact shape as stress/discovery only
- best current explanation:
  - the `fraction_decompose` core is stable in existing wrappers, but this
    three-core composition is too hot for live guardrail promotion
  - the low family count should not be closed by adding a broad
    `trig reciprocal + factor` companion around this core
- plausible combination later:
  - look for a cheaper `fraction_decompose` multi-core representative
  - or improve the fraction-decompose residual route before attempting this
    coverage axis again
