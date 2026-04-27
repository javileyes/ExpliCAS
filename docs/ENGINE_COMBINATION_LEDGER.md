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

### 2026-04-27: `log_exp_inverse` Log10 Factor Coverage Runtime Rejection

- area:
  - corpus generation / embedded equivalence candidate promotion
  - [embedded_equivalence_context_corpus.csv](/Users/javiergimenezmoya/developer/math/docs/embedded_equivalence_context_corpus.csv)
  - `combined_additive_zero` x `log_exp_inverse`
- status:
  - `rejected as live coverage row`
- attempted row:
  - `log10_power_alias_factor_combined_zero`
  - `(10^(y*log10(x)) - x^y) + (p^2-q^2 - (p-q)*(p+q))`
- local lane:
  - exact candidate smoke with `scripts/engine_embedded_candidate_smoke.py`
  - focused slice: `combined_additive_zero` x `log_exp_inverse`
- local result:
  - candidate smoke passed:
    - `1/1` cases
    - `runner_elapsed=6.55ms`
  - focused slice passed:
    - `6/6` cases
    - `Elapsed: 111.44ms`
  - family balance would improve from 12 low families to 11 low families
- global result:
  - `make engine-scorecard` passed correctness:
    - `embedded_equivalence_context: 1387/1387`
    - `combined_additive_zero: 136`
    - `log_exp_inverse: 18`
  - but embedded runtime rose from the recent retained baseline:
    - `Elapsed: 5.59s -> 6.26s`
    - `+0.67s`, about `+12.0%`
- decision:
  - reverted the live corpus row
  - keep the expression as a discovered stress candidate, not as retained
    coverage
- best current explanation:
  - the base-10 logarithmic power alias is semantically correct, but this
    forward orientation under a factor companion appears to activate heavier
    contextual log/power matching than its coverage value justifies as one live
    row
- plausible combination later:
  - retry only after a cheaper base-10 alias gate or narrower contextual route
  - or promote a lower-cost representative for the same `log_exp_inverse`
    coverage gap

### 2026-04-27: `integrate_prep` Dirichlet Additive-Composition Timeout Discovery

- area:
  - generated discovery / embedded equivalence candidate smoke
  - [embedded_equivalence_context_corpus.csv](/Users/javiergimenezmoya/developer/math/docs/embedded_equivalence_context_corpus.csv)
  - `combined_additive_zero` x `integrate_prep`
- status:
  - `resolved; minimal live row promoted`
- attempted row:
  - `integrate_prep_dirichlet_factor_collect_three_core_combined_zero`
  - `(sin(5*x/2)/sin(x/2) - (1 + 2*cos(x) + 2*cos(2*x))) + (p^2-q^2 - (p-q)*(p+q)) + (u*v + u*w - u*(v+w))`
- local lane:
  - `python3 scripts/engine_embedded_candidate_smoke.py --json --timeout-seconds 6 --row ...`
- local result:
  - attempted three-core candidate timed out before live promotion:
    - `status=timeout`
    - `wall_elapsed_seconds=6.004`
    - no runner summary was emitted before timeout
  - isolating probes:
    - root Dirichlet `combined_additive_zero` shape: pass,
      `runner_elapsed=3.59ms`
    - Dirichlet plus factor two-core shape: timeout at `10.004s`
    - Dirichlet plus collect two-core shape: timeout at `10.004s`
- resolution:
  - added a direct seven-term `combined_additive_zero` route for Dirichlet
    plus one independent exact-zero core
  - promoted smallest live row:
    - `integrate_prep_dirichlet_factor_combined_zero`
    - `(sin(5*x/2)/sin(x/2) - (1 + 2*cos(x) + 2*cos(2*x))) + (p^2-q^2 - (p-q)*(p+q))`
  - post-fix candidate smokes:
    - Dirichlet plus factor: `pass`, `runner_elapsed=9.44ms`
    - Dirichlet plus collect: `pass`, `runner_elapsed=10.08ms`
- decision:
  - promote the two-core row to live coverage
  - keep the original three-core row as a stress/discovery shape until a later
    cycle validates that exact three-core composition under full guardrails
- best current explanation:
  - the weakness was the additive-composition route, not Dirichlet itself
  - the arithmetic core could prove each piece, but the orchestrator rejected
    the seven-term composition and fell through to expensive generic paths
- plausible follow-up:
  - isolate whether the timeout is caused by additive-zero decomposition,
    trigonometric Dirichlet matching after flattening, or equivalence fallback
    search when the Dirichlet core is not the only additive term
  - retain a focused regression only after the failing route is understood;
    the generated three-core row is too expensive for live promotion today

### 2026-04-27: `radical_power` Passthrough Three-Core Discovery

- area:
  - generated discovery / embedded equivalence candidate smoke
  - [embedded_equivalence_context_corpus.csv](/Users/javiergimenezmoya/developer/math/docs/embedded_equivalence_context_corpus.csv)
  - `combined_additive_zero` x `radical_power`
- status:
  - `resolved; live row promoted`
- attempted row:
  - `radical_power_passthrough_factor_collect_three_core_combined_zero`
  - `((sqrt(x^3)+a) - (abs(x)*sqrt(x)+a)) + (p^2-q^2 - (p-q)*(p+q)) + (u*v + u*w - u*(v+w))`
- local lane:
  - `python3 scripts/engine_embedded_candidate_smoke.py --json --timeout-seconds 6 --row ...`
- local result:
  - attempted three-core candidate failed before live promotion:
    - `status=fail`
    - `passed=0`
    - `failed=1`
    - `runner_elapsed=23.78ms`
  - isolating probes passed:
    - root radical passthrough shape: pass
    - radical passthrough plus factor two-core shape: pass
    - radical three-core shape without the `+ a` passthrough: pass
- decision:
  - do not promote the passthrough row to the live corpus
  - keep this as discovery pressure for a future `radical_power` isolation or
    runtime cycle
- best current explanation:
  - this is not evidence that `radical_power` is generally broken
  - the current signature points to a narrower composition gap: the radical
    passthrough core becomes fragile only when it is part of a three-core
    additive-zero composition
- plausible follow-up:
  - isolate whether the miss is caused by additive passthrough cancellation
    inside the radical core or by exact-zero decomposition after the first
    radical rewrite

### 2026-04-26: `collect` Common-Tail Three-Core Discovery

- area:
  - generated discovery / embedded equivalence candidate smoke
  - [embedded_equivalence_context_corpus.csv](/Users/javiergimenezmoya/developer/math/docs/embedded_equivalence_context_corpus.csv)
  - `combined_additive_zero` x `collect`
- status:
  - `resolved`
- attempted row:
  - `collect_common_symbolic_factor_trigexpand_three_core_combined_zero`
  - `(x*y + x*z + w - (x*(y+z) + w)) + (p^2-q^2 - (p-q)*(p+q)) + (sin(2*t) - 2*sin(t)*cos(t))`
- local lane:
  - `python3 scripts/engine_embedded_candidate_smoke.py --json --timeout-seconds 6 --row ...`
- local result:
  - attempted three-core candidate failed before live promotion:
    - `status=fail`
    - `passed=0`
    - `failed=1`
    - `runner_elapsed=11.15ms`
  - isolating probes passed:
    - root collect shape with the `+ w` passthrough: pass
    - collect plus factor two-core shape: pass
    - collect plus trig two-core shape: pass
    - three-core shape without the `+ w` passthrough: pass
  - related probes failed:
    - reordered three-core shape with the `+ w` passthrough: fail
    - same collect/factor shape with trig reciprocal instead of trig expand: fail
- 2026-04-27 rejected runtime attempt:
  - tried a bounded three-core additive-zero partition route for generated
    compositions with one shared-passthrough collect core plus two small exact
    zero cores
  - local candidate became correct:
    - `status=pass`
    - `passed=1`
    - `failed=0`
    - `runner_elapsed=443.23ms`
  - global embedded guardrail rejected the promotion:
    - `embedded_equivalence_context`: `1377/1377`, `failed=0`
    - elapsed increased to `9.24s` for the guardrail run, far above the prior
      `5.27s` order of magnitude for only one extra row
  - runtime changes and live corpus promotion were not retained
- 2026-04-27 adjacent coverage retained:
  - promoted the minimal stable two-core representative instead of the failed
    three-core discovery:
    - `collect_common_symbolic_factor_with_tail_factor_combined_zero`
    - `(x*y + x*z + w - (x*(y + z) + w)) + (p^2-q^2 - (p-q)*(p+q))`
  - local smoke before promotion:
    - `status=pass`
    - `runner_elapsed=12.76ms`
  - this does not resolve the observe-only three-core row; it preserves the
    already-stable shared-tail collect shape in live coverage while keeping the
    hotter composition as discovery pressure
- decision:
  - do not promote the row to the live corpus
  - keep the expression as generated-discovery pressure for a future `collect`
    isolation or runtime cycle
- best current explanation:
  - the weakness was the additive-zero partition route, not `collect` itself
  - the engine could prove the collect passthrough core and the two other cores
    separately, but the live combined route did not allow a five-term
    passthrough core inside a three-core composition
  - the rejected fix widened the route enough to trigger expensive direct
    identity probing in embedded traffic
- plausible follow-up:
  - optimize a narrower partition recognizer before attempting promotion again;
    a local pass is not sufficient while the one-row smoke remains hundreds of
    milliseconds
  - isolate the reordered and trig-reciprocal variants before broadening the
    route again

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

### 2026-04-27: Exact Trig Two-Term Zero-Scope Runtime Candidate

- area:
  - [arithmetic.rs](/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/arithmetic.rs)
  - `simplify_zero_mixed` pressure `sum`
- status:
  - `retained`
- change:
  - added a normalized two-term path for
    `try_build_exact_trig_equivalence_zero_scope_rewrite`
  - two-term exact trig scopes compare `lhs`/`rhs` directly and avoid the
    generic subset loop
  - three-or-more-term exact trig scopes still use the existing generic path
- local pressure result:
  - focused filtered baseline:
    - `rule.direct_identity.try.zero_scope_exact_trig_equivalence`:
      `11489` attempts, `11040.624ms`, `960.97us` avg
  - after change:
    - `rule.direct_identity.try.zero_scope_exact_trig_equivalence`:
      `5221` attempts, `2083.946ms`, `399.15us` avg
  - `sum@0+100` simplify avg:
    - `68.36ms -> 54.84ms` in the filtered profile
    - `55.48ms` in the full local profile
- guardrails:
  - targeted arithmetic tests:
    - `exact_trig_equivalence_zero_scope`: `8/8`
    - `maybe_exact_trig_equivalence_zero_scope_candidate`: `5/5`
    - `direct_small_zero_additive_combination_shortcut`: `9/9`
  - `make engine-fast`: passed
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1398/1398`
    - `simplify_strict`: `16475/16475`
  - `make engine-scorecard-pressure`: passed
    - `simplify_zero_mixed`: `450/450`
    - elapsed `13.37s -> 10.56s`
    - `sum` simplify avg `65.77ms -> 51.67ms`
- decision:
  - retained as runtime improvement
- next:
  - the new pressure head is phase-shift expansion/comparison on trig-heavy
    two-term scopes; investigate a narrower gate or a cheaper exact path there

### 2026-04-27: Phase-Shift Single-Fragment Runtime Gate

- area:
  - [arithmetic.rs](/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/arithmetic.rs)
  - `simplify_zero_mixed` pressure `sum`
- status:
  - `retained`
- change:
  - reject binary phase-shift bridge pairs where one side is a single plain
    trig term and the other side is an exact/general shifted trig term
  - keep full additive linear-combination vs shifted pairs active
- local pressure result:
  - focused filtered baseline:
    - `rule.direct_identity.try.expand_trig_phase_shift`:
      `306` attempts, `2099.771ms`, `6862us` avg
    - `rule.phase_shift.route.shifted_try`:
      `1326` attempts, `1748.453ms`
    - `rule.phase_shift.binary_add_match`:
      `333` attempts, `1193.440ms`
  - after change:
    - `rule.direct_identity.try.expand_trig_phase_shift`:
      `102` attempts, `330.292ms`, `3238us` avg
    - `rule.phase_shift.route.shifted_try`:
      `252` attempts, `258.625ms`
    - `rule.phase_shift.binary_add_match.productive_term_family_gate`:
      `111` attempts, `237.065ms`
  - `sum@0+100` simplify avg:
    - `52.89ms -> 48.82ms` in the filtered profile
    - `50.37ms` in the full local profile
- guardrails:
  - targeted phase-shift tests:
    - `expand_trig_phase_shift_rule`: `5/5`
    - `trig_phase_shift_cancellation_match`: `6/6`
    - `collapse_exact_zero_three_term_subset_rule_matches_phase_shift`: `2/2`
  - `make engine-fast`: passed
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1398/1398`
    - `simplify_strict`: `16475/16475`
  - `make engine-scorecard-pressure`: passed
    - `simplify_zero_mixed`: `450/450`
    - elapsed `10.56s -> 9.16s`
    - `sum` simplify avg `51.67ms -> 44.75ms`
- decision:
  - retained as runtime improvement
- next:
  - remaining pressure head is the double-angle embedded factor family around
    `2*cos(2*x)*sin(x) - (4*cos(x)^2*sin(x) - 2*sin(x))`; investigate a
    narrower direct route or cached target simplification for that family

### 2026-04-27: Embedded Double-Angle Factor Runtime Narrowing

- area:
  - [arithmetic.rs](/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/arithmetic.rs)
  - `simplify_zero_mixed` pressure `sum`
- status:
  - `retained`
- change:
  - restricted `zero_scope_embedded_double_angle_factor` to embedded
    `sin(2*x)` factors, because embedded `cos(2*x)` mixed-polynomial cases
    are already covered by the dedicated `cos_variant` /
    `mixed_double_angle_poly` routes
  - added an arity/default-fallback guard for the exact
    `cos(2*x)` polynomial zero-scope path, while preserving factored
    remaining forms with a top-level additive-factor check
- local pressure result:
  - baseline `sum@0+50`:
    - `50/50`, simplify avg `49.85ms`
    - `rule.direct_identity.try.zero_scope_embedded_double_angle_factor`:
      `314` attempts, `2` hits, `650.356ms`
  - after change `sum@0+50`:
    - `50/50`, simplify avg `47.12ms`
    - slow double-angle mixed cases around `#33/#189`: steady-state
      `~529-562ms` instead of the previous `~672ms` local baseline
    - `rule.direct_identity.try.zero_scope_embedded_double_angle_factor`:
      `2` attempts, `2` hits, `5.740ms`
  - pressure scorecard:
    - `simplify_zero_mixed`: `450/450`
    - elapsed `8.11s`
    - `sum` simplify avg `39.56ms`
- guardrails:
  - targeted arithmetic tests:
    - `embedded_double_angle`: `8/8`
    - `mixed_double_angle`: `7/7`
    - `exact_zero_cos_double_angle_polynomial_matches_factored_remaining_regression`: `1/1`
  - `cargo fmt -- --check`: passed
  - `make engine-fast`: passed
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1398/1398`
    - `simplify_strict`: `16475/16475`
  - `make engine-scorecard-pressure`: passed
    - `simplify_zero_mixed`: `450/450`
- rejected local candidates:
  - early small-core mixed double-angle polynomial route improved targeted
    coverage but regressed the pressure profile, so it was not promoted
  - arithmetic small log/double-angle partition probes and additional
    orchestrator root/base partition probes did not move the hot eval route
    and were reverted
- decision:
  - retained as runtime improvement
- next:
  - remaining pressure heads are exact trig equivalence and phase-shift routes
    for product-to-sum families such as `2*cos(x)*cos(y) - cos(x+y) - cos(x-y)`;
    investigate similarly narrow family gates there

### 2026-04-27: Exact-Trig Nested Default Fallback Runtime Gate

- area:
  - [arithmetic.rs](/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/arithmetic.rs)
  - `simplify_zero_mixed` pressure `sum`
- status:
  - `retained`
- change:
  - disabled default-simplify fallback comparisons inside
    `try_build_exact_trig_equivalence_zero_scope_rewrite` when already running
    under `run_default_simplify`
  - kept exact/structural cancellation checks active, and kept outermost
    default-simplify fallback behavior unchanged
  - applied the same nested fallback guard to the two-term exact trig zero-scope
    helper
- local pressure result:
  - `case#89` product-to-sum/log composition:
    - steady-state simplify `~267.75ms -> ~181.62ms`
    - `rule.direct_identity.try.zero_scope_exact_trig_equivalence`:
      `3000` attempts, `0` hits, `~1436.6ms -> ~722.6ms`
  - `sum@0+100` profiled slice:
    - `100/100`
    - simplify avg `41.50ms -> 38.17ms`
    - exact-trig label `9505` attempts / `3992.559ms` -> `5221` attempts /
      `908.348ms`
  - pressure scorecard:
    - `simplify_zero_mixed`: `450/450`
    - elapsed `8.11s -> 7.25s`
    - `sum` simplify avg `39.56ms -> 35.24ms`
- guardrails:
  - targeted arithmetic tests:
    - `exact_trig_equivalence_zero_scope`: `8/8`
    - `maybe_exact_trig_equivalence_zero_scope_candidate`: `5/5`
  - `cargo fmt -- --check`: passed
  - `make engine-fast`: passed
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1398/1398`
    - `simplify_strict`: `16475/16475`
  - `make engine-scorecard-pressure`: passed
    - `simplify_zero_mixed`: `450/450`
- rejected local candidates:
  - generic direct product-to-sum zero-scope route passed unit tests but did not
    change the hot eval route and worsened `case#89`
  - structural product-to-sum candidate rejection also left the exact-trig
    no-match traffic unchanged
- decision:
  - retained as runtime improvement
- next:
  - after exact-trig fallback cost drops, the next pressure head shifts back to
    phase-shift expansion/comparison; investigate a similarly nested-safe
    fallback gate for phase-shift generated candidate comparisons

### 2026-04-27: Phase-Shift Binary Fragment Pre-Reject

- area:
  - [arithmetic.rs](/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/arithmetic.rs)
  - `simplify_zero_mixed` pressure `sum`
- status:
  - `retained`
- change:
  - added a cheap surface pre-reject for binary phase-shift fragments where one
    side is a plain trig term and the other side carries a phase-shift signal
  - used that pre-reject before direct two-term phase-shift identity and
    expansion probes, avoiding expensive generated-candidate routes for
    fragments that cannot cancel alone
  - added a direct fast zero-scope path for the full general arctan phase-shift
    triple so the complete `a*sin(x)+b*cos(x)-r*sin(x+atan(b/a))` shape keeps
    its didactic two-step rewrite when reached as a whole
- local pressure result:
  - `case#49` general phase-shift/log composition:
    - steady-state simplify `~218.67ms -> ~17.30ms`
    - initial single run simplify `237.44ms -> 28.13ms`
    - hot `two_term_phase_shift_identity`, `expand_trig_phase_shift`, and
      `shifted_try` probes dropped out of the case top profile after the
      pre-reject
  - profiled `sum@0+100` slice:
    - `100/100`
    - simplify avg `36.24ms -> 32.45ms`
  - pressure scorecard:
    - `simplify_zero_mixed`: `450/450`
    - elapsed `7.25s -> 6.45s`
    - `sum` simplify avg `35.24ms -> 31.23ms`
    - `sum@0+100` simplify avg `34.47ms -> 30.45ms`
    - `sum@700+100` simplify avg `36.00ms -> 32.00ms`
- guardrails:
  - targeted arithmetic tests:
    - `phase_shift`: `54/54`
  - `cargo fmt -- --check`: passed
  - `make engine-fast`: passed
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1398/1398`
    - `simplify_strict`: `16475/16475`
  - `make engine-scorecard-pressure`: passed
    - `simplify_zero_mixed`: `450/450`
  - `git diff --check`: passed
- rejected local candidates:
  - a nested `default_simplify` fallback gate inside supported phase-shift
    argument normalization passed focused tests but worsened `sum@0+100`
    (`36.24ms -> 38.34ms`) and was reverted
  - the general arctan triple fast path alone was correct but did not move the
    mixed pressure profile until paired with the binary fragment pre-reject
- decision:
  - retained as runtime improvement
- next:
  - remaining pressure heads are double-angle mixed fragments such as
    `2*cos(2*x)*sin(x) - (4*cos(x)^2*sin(x) - 2*sin(x))`; investigate a narrow
    early gate or fast zero-scope route for those fragments next

### 2026-04-27: Surface Trig Power Numeric Fallback Reject

- area:
  - [arithmetic.rs](/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/arithmetic.rs)
  - `simplify_zero_mixed` pressure `sum`
- status:
  - `retained`
- change:
  - added a direct-core pre-reject before expensive `default_simplify` for
    scaled symbolic `sin`/`cos` power monomials against numeric atoms, e.g.
    `4*cos(x)^2` vs `1`
  - extended the existing surface trig power-gap reject to scaled plain-vs-power
    pairs, e.g. `2*sin(x)` vs `3*sin(x)^2`
  - preserved special-angle traffic by requiring symbolic trig arguments without
    `pi` constants or nested function calls
- local pressure result:
  - `case#33` double-angle/log composition:
    - steady-state simplify `~578.33ms -> ~513.19ms`
    - profiled steady-state simplify `~577.06ms -> ~523.45ms`
  - `case#189` double-angle/nested-fraction composition:
    - steady-state simplify `~580.10ms -> ~517.39ms`
  - profiled `case#33` route movement:
    - `rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.other`:
      `1600` attempts / `~332.304ms -> 32` attempts / `~13.275ms`
    - new reject
      `rule.direct_core_equivalence.route.default_simplify_surface_trig_power_numeric_atom_reject`:
      `1568` hits
  - profiled `sum@0+100` slice:
    - `100/100`
    - simplify avg `30.45ms -> 29.80ms`
  - pressure scorecard:
    - `simplify_zero_mixed`: `450/450`
    - elapsed `6.45s -> 6.08s`
    - `sum` simplify avg `31.23ms -> 29.37ms`
    - `sum@0+100` simplify avg `30.45ms -> 28.68ms`
    - `sum@700+100` simplify avg `32.00ms -> 30.07ms`
- guardrails:
  - targeted arithmetic tests:
    - `reject_scaled_surface_trig_power_vs_numeric_atom_before_default_simplify`: `2/2`
    - `reject_plain_surface_trig_power_gap_before_default_simplify_matches_scaled_gap`: `1/1`
  - `cargo fmt -- --check`: passed
  - `make engine-fast`: passed
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1398/1398`
    - `simplify_strict`: `16475/16475`
  - `make engine-scorecard-pressure`: passed
    - `simplify_zero_mixed`: `450/450`
- rejected local candidates:
  - adding pure log cores to `small_direct_zero_core` and the two-core
    composition gate passed focused tests for
    `ln(x^3)+ln(y^2)-ln(x^3*y^2)`, but did not move the live route and worsened
    `case#33` steady-state (`~578ms -> ~604-618ms`); reverted
  - extending the same composition gate to the concrete nested-fraction core in
    `case#189` failed the focused rewrite test before promotion; reverted
- decision:
  - retained as runtime improvement
- next:
  - the remaining hot heads are still full double-angle mixed compositions; a
    future cycle should investigate a direct exact-zero route for the complete
    `2*cos(2*x)*sin(x) - (4*cos(x)^2*sin(x) - 2*sin(x))` composition rather than
    broadening small-core partition gates

### 2026-04-27: Compact Direct Small-Zero Pair Before Didactic Chunking

- area:
  - [orchestrator.rs](/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/orchestrator.rs)
  - `simplify_zero_mixed` pressure `sum`
- status:
  - `retained`
- change:
  - added a `StepsMode::Compact`-only early route for
    `try_standard_direct_small_zero_pair_shortcut` before the targeted
    additive-combination didactic path
  - this preserves the richer chunked explanation path for full step collection,
    but avoids building expensive chunk-pair substeps on canonical eval calls
    where user-visible steps are off and `Compact` is only used internally
  - added a cached-profile compact pipeline regression for the
    `log_product + mixed double-angle` sum to lock the root-shortcut route
- local pressure result:
  - `case#33` double-angle/log composition:
    - steady-state simplify `~445.53ms -> ~0.09ms`
    - single profiled simplify `471.59ms -> 0.85ms`
    - profile now shows
      `root.addsub.00.direct_small_zero_pair.compact_first`: `4/4` hits
  - `case#3001` ln-abs/double-angle composition:
    - steady-state simplify `~477.53ms -> ~0.09ms`
    - single profiled simplify `~497.92ms -> 0.99ms`
  - pressure scorecard:
    - `simplify_zero_mixed`: `450/450`
    - elapsed `6.08s -> 0.24s`
    - `sum` simplify avg `29.37ms -> 0.26ms`
    - `sum@0+100` simplify avg `28.68ms -> 0.12ms`
    - `sum@700+100` simplify avg `30.07ms -> 0.40ms`
- guardrails:
  - targeted orchestrator tests:
    - `trig_mixed_double_angle_sum_regression`: `2/2`
  - `cargo fmt -- --check`: passed
  - `make engine-fast`: passed
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1398/1398`
    - `simplify_strict`: `16475/16475`
  - `make engine-scorecard-pressure`: passed
    - `simplify_zero_mixed`: `450/450`
- decision:
  - retained as runtime improvement
- next:
  - the new pressure head is `case#3145`:
    `(tan(x)*cot(x) - 1) + (sin(x)^2 - (1 - cos(2*x))/2)`, around
    `30.7ms` steady-state; investigate a compact/direct pair route for
    tan-cot plus half-angle square without weakening the full didactic path

### 2026-04-27: Compact Tan-Cot Half-Angle Pair Shortcut

- area:
  - [orchestrator.rs](/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/orchestrator.rs)
  - `simplify_zero_mixed` pressure `sum@700+100`
- status:
  - `retained`
- investment:
  - `investment_class`: runtime
  - `primary_dimension`: runtime
  - `secondary_dimension`: observability/robustness
  - `cohesion_scope`: compact-only root add/sub small-zero pair routing
  - `behavior_change_expected`: yes, runtime route and compact-step granularity
    only; full didactic step collection remains on the existing path
- change:
  - added a narrow compact matcher for the exact pair
    `tan(x)*cot(x)-1` with `sin(x)^2 - (1 - cos(2*x))/2`
  - installed it before the generic
    `root.addsub.00.direct_small_zero_pair.compact_first` route, only when
    `collect_steps && StepsMode::Compact`
  - added a regression test for the compact shortcut so the optimized path is
    covered independently from the broader generic matcher
- local pressure result:
  - `case#3145`:
    - before: steady-state simplify `~32.25ms`
    - after: initial simplify `1.10ms`, steady-state simplify `~0.07ms`
    - profile now shows
      `root.addsub.00.tan_cot_half_angle_pair.compact_first`: `4/4` hits,
      `0.312ms` total
  - local `sum@700+100` slice:
    - `100/100`
    - elapsed `29.29ms`
    - simplify avg `0.12ms`
  - pressure scorecard:
    - `simplify_zero_mixed`: `450/450`
    - elapsed `0.24s -> 0.21s`
    - `sum` simplify avg `0.26ms -> 0.11ms`
    - `sum@0+100` simplify avg `0.12ms -> 0.12ms`
    - `sum@700+100` simplify avg `0.40ms -> 0.09ms`
- guardrails:
  - targeted orchestrator test:
    - `compact_tan_cot_half_angle_pair_shortcut_handles_sum_regression`: `1/1`
  - `cargo fmt -- --check`: passed
  - `make engine-fast`: passed
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1398/1398`
    - `derive_contract`: reachability `1.000`, supported equivalence `1.000`
    - `simplify_strict`: `16475/16475`
  - `make engine-scorecard-pressure`: passed
    - `simplify_zero_mixed`: `450/450`
- decision:
  - retained as runtime improvement
- next:
  - current pressure heads are now low-ms/sub-ms hyperbolic/log combinations;
    inspect `case#3049`
    `(2*ln(abs(x*y)) - 2*ln(abs(x)) - 2*ln(abs(y))) + (cosh(x)^2 - sinh(x)^2 - 1)`
    or the `sum@0+100` head `case#25` before widening any generic matcher

### 2026-04-27: Direct Hyperbolic Pythagorean Zero-Scope Orientation

- area:
  - [arithmetic.rs](/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/arithmetic.rs)
  - `simplify_zero_mixed` pressure `sum@700+100`
- status:
  - `retained`
- investment:
  - `investment_class`: runtime
  - `primary_dimension`: runtime
  - `secondary_dimension`: robustness/coverage-reuse
  - `cohesion_scope`: direct hyperbolic equivalence helper in `arithmetic.rs`
  - `behavior_change_expected`: yes, exact-zero route selection only
- change:
  - reused the existing hyperbolic Pythagorean rewrite inside the exact
    hyperbolic cancellation helper
  - added signed-additive recognition for the `AddView` subset form of
    `cosh(x)^2 - sinh(x)^2`, so
    `cosh(x)^2 - sinh(x)^2 - 1` can prove directly through
    `zero_scope_exact_hyperbolic_equivalence`
  - added focused tests for both the public exact-zero additive rule and the
    private exact hyperbolic zero-scope helper
- local pressure result:
  - `case#3049`:
    `(2*ln(abs(x*y)) - 2*ln(abs(x)) - 2*ln(abs(y))) + (cosh(x)^2 - sinh(x)^2 - 1)`
    - before: profiled steady-state simplify `~0.23ms`
    - after: profiled steady-state simplify `~0.19ms`
    - route movement:
      - `rule.direct_identity.try.zero_scope_exact_hyperbolic_equivalence`:
        `8` attempts / `0` hits -> `4` attempts / `4` hits
      - direct-identity probe attempts: `56 -> 28`
  - local `sum@700+100` slice:
    - `100/100`
    - `case#3049` steady-state simplify `~0.17ms` in the local window run
  - pressure scorecard:
    - `simplify_zero_mixed`: `450/450`
    - elapsed `0.21s -> 0.206s`
    - `sum` simplify avg held at `0.11ms`
    - `sum@700+100` simplify avg held at `0.09ms`
- guardrails:
  - targeted arithmetic tests:
    - `collapse_exact_zero_additive_subexpression_matches_direct_hyperbolic_pythagorean_zero_scope`: `1/1`
    - `direct_exact_hyperbolic_zero_scope_matches_additive_pythagorean_orientation`: `1/1`
  - `make engine-fast`: passed
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1398/1398`
    - `derive_contract`: reachability `1.000`, supported equivalence `1.000`
    - `simplify_strict`: `16475/16475`
  - `make engine-scorecard-pressure`: passed
    - `simplify_zero_mixed`: `450/450`
- rejected local candidates:
  - simply calling the existing `try_rewrite_hyperbolic_pythagorean_sub_expr`
    did not move the live profile because the zero-scope subset builder presents
    the focus as a signed additive expression rather than the original `Sub`
    node; retained only after adding the signed-additive recognizer
- decision:
  - retained as a small runtime/robustness improvement
- next:
  - the next pressure head is again `case#25`
    `(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (cosh(x) + sinh(x) - e^x)`;
    current profiling suggests no broad orchestrator cliff, so prefer a
    narrow log/hyperbolic helper only if a new profile shows repeated misses
    rather than adding another generic additive shortcut

### 2026-04-27: Pressure Steady-Rerun Millisecond Rendering

- area:
  - [engine_improvement_scorecard.py](/Users/javiergimenezmoya/developer/math/scripts/engine_improvement_scorecard.py)
  - [test_engine_improvement_scorecard.py](/Users/javiergimenezmoya/developer/math/scripts/test_engine_improvement_scorecard.py)
  - `simplify_zero_mixed` pressure observability
- status:
  - `retained`
- investment:
  - `investment_class`: observability
  - `primary_dimension`: observability
  - `secondary_dimension`: runtime-selection
  - `cohesion_scope`: scorecard markdown rendering only
  - `behavior_change_expected`: no engine behavior change; report precision
    only
- trigger:
  - profiled the current pressure heads before touching files:
    - `case#25` showed steady-state simplify around `0.13ms`; profile hit
      `rule.direct_identity.try.two_term_safe_hyperbolic` `4/4`
    - `case#53` showed steady-state simplify around `0.15ms`; same
      hyperbolic direct-identity route hit `4/4`
  - this did not expose a broad runtime cliff, but the generated Markdown was
    still printing steady rerun medians as `0.00s`, hiding the sub-ms ordering
- change:
  - added `format_runtime_duration` so sub-second runtime values render in ms
    rather than second-rounded `0.00s`
  - applied it to the pressure scorecard's steady-state engine rerun summary
  - added unit coverage for both the formatter and the rendered Markdown
    pressure section
- local pressure result:
  - generated pressure Markdown now reports the top steady reruns as:
    - `case#25`: `median_simplify=0.11ms`, `median_wire=0.13ms`,
      `median_wall=0.37ms`
    - `case#53`: `median_simplify=0.13ms`, `median_wire=0.16ms`,
      `median_wall=0.39ms`
    - `case#57`: `median_simplify=0.12ms`, `median_wire=0.14ms`,
      `median_wall=0.38ms`
  - pressure scorecard:
    - `simplify_zero_mixed`: `450/450`
    - elapsed `207.56ms`
    - `sum` avg simplify `0.11ms`
- guardrails:
  - `python3 -m unittest scripts/test_engine_improvement_scorecard.py`: passed
    - `12` tests
  - `make engine-fast`: passed
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1398/1398`
    - `derive_contract`: reachability `1.000`, supported equivalence `1.000`
    - `simplify_strict`: `16475/16475`
  - `make engine-scorecard-pressure`: passed
    - `simplify_zero_mixed`: `450/450`
- decision:
  - retained as an observability improvement; this does not make the engine
    more powerful directly, but it prevents future cycles from losing useful
    runtime signal once pressure cases are sub-ms
- next:
  - keep the current `case#25` / `case#53` profiles as low-priority runtime
    candidates unless a later profile shows repeated misses or a broader
    non-hyperbolic route; otherwise continue corpus/coverage growth

### 2026-04-27: Pressure Aggregate Millisecond Rendering

- area:
  - [engine_improvement_scorecard.py](/Users/javiergimenezmoya/developer/math/scripts/engine_improvement_scorecard.py)
  - [test_engine_improvement_scorecard.py](/Users/javiergimenezmoya/developer/math/scripts/test_engine_improvement_scorecard.py)
  - `simplify_zero_mixed` pressure observability
- status:
  - `retained`
- investment:
  - `investment_class`: observability
  - `primary_dimension`: observability
  - `secondary_dimension`: runtime-selection
  - `cohesion_scope`: scorecard markdown rendering only
  - `behavior_change_expected`: no engine behavior change; report precision
    only
- trigger:
  - the live `combined_additive_zero` corpus is now balanced at the current
    target (`23/23` families, each non-`simplify` family at `6` cases), so a
    coverage promotion from the old `collect` discovery was not the highest ROI
  - after the previous steady-rerun formatter improvement, the same pressure
    Markdown still rounded aggregate rows such as `difference simplify` to
    `0.00s`, hiding real `4ms`-range costs
- change:
  - reused `format_runtime_duration` for `Mixed Zero Pressure` aggregate
    elapsed/simplify/wall fields in composition hotspots, engine hotspots, and
    window slices
  - added Markdown assertions that block regressions back to second-rounded
    `simplify=0.00s`
- pressure result:
  - generated pressure Markdown now reports:
    - composition hotspots:
      - `sum`: `elapsed=77.29ms`, `simplify=21.38ms`
      - `shifted_quotient`: `elapsed=59.35ms`, `simplify=9.15ms`
      - `difference`: `elapsed=26.35ms`, `simplify=4.42ms`
    - engine hotspots:
      - `sum simplify=21.38ms wall=77.29ms`
      - `difference simplify=4.42ms wall=26.35ms`
    - window slices now render as ms, including
      `sum@700+100 elapsed=23.14ms`
  - pressure scorecard:
    - `simplify_zero_mixed`: `450/450`
    - elapsed `205.04ms`
- guardrails:
  - `python3 -m unittest scripts/test_engine_improvement_scorecard.py`: passed
    - `12` tests
  - `make engine-fast`: passed
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1398/1398`
    - `derive_contract`: reachability `1.000`, supported equivalence `1.000`
    - `simplify_strict`: `16475/16475`
  - `make engine-scorecard-pressure`: passed
    - `simplify_zero_mixed`: `450/450`
- decision:
  - retained as a harness/observability improvement; it does not increase
    mathematical power directly, but it preserves the timing signal now that
    promoted pressure lanes are mostly sub-second and often sub-100ms
- next:
  - use the clearer pressure aggregates to decide between sparse-wrapper
    coverage growth and a narrow runtime pass only if a future profile shows
    repeated misses rather than already-hot successful direct routes

### 2026-04-27: Squared Passthrough Collect Common-Factor Coverage

- area:
  - [embedded_equivalence_context_corpus.csv](/Users/javiergimenezmoya/developer/math/docs/embedded_equivalence_context_corpus.csv)
  - `squared_passthrough_zero x collect`
- status:
  - `retained`
- investment:
  - `investment_class`: coverage
  - `primary_dimension`: wrapper spread
  - `secondary_dimension`: shell_depth / source-shape breadth
  - `cohesion_scope`: one embedded corpus row
  - `behavior_change_expected`: no engine behavior change; corpus promotion
    only
- trigger:
  - `squared_passthrough_zero` remained sparse after the previous observability
    pass
  - `collect` had only linear representatives under this wrapper, while
    common symbolic coefficient collection was already a stable low-cost shape
    in other contexts
- change:
  - promoted one live row:
    `(((x*y + x*z + w)^2) + m) - (((x*(y + z) + w)^2) + m) -> 0`
  - this keeps the promotion minimal and avoids turning a single stable smoke
    into a batch expansion
- candidate smoke:
  - `1/1` passed
  - runner elapsed `4.28ms`
  - complexity `l3_nested_or_composed`
  - shell depth `3`
- filtered result:
  - `squared_passthrough_zero --family collect`: `3/3`
  - elapsed `2.94ms`
  - bucket moved from `2 -> 3`
- guardrails:
  - `make engine-fast`: passed
  - `make engine-scorecard-pressure`: passed
    - `simplify_zero_mixed`: `450/450`
    - elapsed `206.39ms`
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1399/1399`
    - average case runtime `2.852ms`
    - sparse bucket: `squared_passthrough_zero x collect total=3 failed=0`
    - `derive_contract`: reachability `1.000`, supported equivalence `1.000`
    - `simplify_strict`: `16475/16475`
- decision:
  - retained as a coverage promotion; it does not increase runtime power by
    itself, but it adds a guarded representative proving this wrapper/family
    combination remains correct for a broader collect source shape
- next:
  - continue sparse-wrapper coverage with another one-row smoke from the
    remaining `squared_passthrough_zero` families at count `2`, or spend a
    cycle on observability if sparse-wrapper selection becomes hard to audit

### 2026-04-27: Squared Passthrough Finite Product Coverage

- area:
  - [embedded_equivalence_context_corpus.csv](/Users/javiergimenezmoya/developer/math/docs/embedded_equivalence_context_corpus.csv)
  - `squared_passthrough_zero x finite_telescoping`
- status:
  - `retained`
- investment:
  - `investment_class`: coverage
  - `primary_dimension`: wrapper spread
  - `secondary_dimension`: finite telescoping source-shape breadth
  - `cohesion_scope`: one embedded corpus row
  - `behavior_change_expected`: no engine behavior change; corpus promotion
    only
- trigger:
  - `squared_passthrough_zero x finite_telescoping` only covered the sum
    telescoping shape
  - the product telescoping shape was already stable in simple wrappers and in
    combined additive coverage, but not under the squared passthrough wrapper
- change:
  - promoted one live row:
    `(((product((k+1)/k, k, 1, n))^2) + m) - (((n+1)^2) + m) -> 0`
  - this is the minimal product representative; no double-squared variant and
    no extra additive/composition core were added
- candidate smoke:
  - `1/1` passed
  - runner elapsed `16.41ms`
  - complexity `l3_nested_or_composed`
  - shell depth `3`
- filtered result:
  - `squared_passthrough_zero --family finite_telescoping`: `3/3`
  - elapsed `37.96ms`
  - bucket moved from `2 -> 3`
- guardrails:
  - `make engine-fast`: passed
  - `make engine-scorecard-pressure`: passed
    - `simplify_zero_mixed`: `450/450`
    - elapsed `221.86ms`
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1400/1400`
    - average case runtime `3.164ms`
    - sparse bucket: `squared_passthrough_zero x finite_telescoping total=3 failed=0`
    - `derive_contract`: reachability `1.000`, supported equivalence `1.000`
    - `simplify_strict`: `16475/16475`
  - repeated embedded-only check:
    - `1400/1400`
    - elapsed `4.39s`
- decision:
  - retained as coverage, not as runtime work
  - the global embedded elapsed was higher than the previous guardrail run, but
    the candidate is locally bounded (`16.41ms` smoke, `37.96ms` filtered
    family), has zero failures, and adds a distinct finite-product shape rather
    than another sum/noise variant
- next:
  - continue sparse-wrapper growth only with low-cost one-row smokes; prefer a
    cheaper non-finite family next if embedded elapsed stays around `4.4s`

### 2026-04-27: Squared Passthrough Expand Difference Coverage

- area:
  - [embedded_equivalence_context_corpus.csv](/Users/javiergimenezmoya/developer/math/docs/embedded_equivalence_context_corpus.csv)
  - `squared_passthrough_zero x expand`
- status:
  - `retained`
- investment:
  - `investment_class`: coverage
  - `primary_dimension`: wrapper spread
  - `secondary_dimension`: sign/orientation robustness
  - `cohesion_scope`: one embedded corpus row
  - `behavior_change_expected`: no engine behavior change; corpus promotion
    only
- trigger:
  - `squared_passthrough_zero x expand` only covered the common-factor sum
    representative and its double-squared shell
  - after the finite-product promotion showed higher embedded timing, the next
    coverage row needed to be a cheap non-finite representative
- change:
  - promoted one live row:
    `(((a*(b-c))^2) + m) - (((a*b - a*c)^2) + m) -> 0`
  - this adds subtraction distribution under the squared passthrough wrapper
    without adding the double-squared variant or extra composition
- candidate smoke:
  - `1/1` passed
  - runner elapsed `4.41ms`
  - complexity `l3_nested_or_composed`
  - shell depth `3`
- filtered result:
  - `squared_passthrough_zero --family expand`: `3/3`
  - elapsed `2.99ms`
  - bucket moved from `2 -> 3`
- guardrails:
  - `make engine-fast`: passed
  - `make engine-scorecard-pressure`: passed
    - `simplify_zero_mixed`: `450/450`
    - elapsed `209.54ms`
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1401/1401`
    - average case runtime `2.948ms`
    - sparse bucket: `squared_passthrough_zero x expand total=3 failed=0`
    - `derive_contract`: reachability `1.000`, supported equivalence `1.000`
    - `simplify_strict`: `16475/16475`
- decision:
  - retained as a low-cost coverage promotion
  - it broadens the squared wrapper's expand coverage from positive
    distribution to subtraction/sign handling while keeping local and global
    runtime within guardrail expectations
- next:
  - continue one-row sparse-wrapper coverage in a remaining count-`2` family,
    favoring similarly cheap algebraic representatives before heavier log,
    radical, or telescoping shapes

### 2026-04-27: Squared Passthrough Fraction Difference Combine Coverage

- area:
  - [embedded_equivalence_context_corpus.csv](/Users/javiergimenezmoya/developer/math/docs/embedded_equivalence_context_corpus.csv)
  - `squared_passthrough_zero x fraction_combine`
- status:
  - `retained`
- investment:
  - `investment_class`: coverage
  - `primary_dimension`: wrapper spread
  - `secondary_dimension`: sign/orientation robustness
  - `cohesion_scope`: one embedded corpus row
  - `behavior_change_expected`: no engine behavior change; corpus promotion
    only
- trigger:
  - `squared_passthrough_zero x fraction_combine` only covered the same
    denominator sum representative and its double-squared shell
  - the same denominator difference representative was already stable in
    simple wrappers but not under the squared passthrough wrapper
- change:
  - promoted one live row:
    `(((a/d - b/d)^2) + m) - ((((a-b)/d)^2) + m) -> 0`
  - this adds fraction subtraction/combine coverage without adding a
    double-squared variant, general-denominator variant, or composition core
- candidate smoke:
  - `1/1` passed
  - runner elapsed `21.08ms`
  - complexity `l3_nested_or_composed`
  - shell depth `3`
- filtered result:
  - `squared_passthrough_zero --family fraction_combine`: `3/3`
  - elapsed `16.44ms`
  - bucket moved from `2 -> 3`
- guardrails:
  - `make engine-fast`: passed
  - `make engine-scorecard-pressure`: passed
    - `simplify_zero_mixed`: `450/450`
    - elapsed `205.74ms`
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1402/1402`
    - average case runtime `2.889ms`
    - sparse bucket: `squared_passthrough_zero x fraction_combine total=3 failed=0`
    - `derive_contract`: reachability `1.000`, supported equivalence `1.000`
    - `simplify_strict`: `16475/16475`
- decision:
  - retained as coverage
  - local filtered cost is bounded and the final embedded guardrail improved
    relative to the previous run, so this does not create the kind of runtime
    tax seen in heavier finite/log candidates
- next:
  - continue sparse-wrapper coverage with another count-`2` family; the next
    cheap candidates are likely `power_merge`, `polynomial_product`, or
    `fraction_expand` before heavier log/radical/telescoping shapes

### 2026-04-27: Squared Passthrough Fraction Difference Expand Coverage

- area:
  - [embedded_equivalence_context_corpus.csv](/Users/javiergimenezmoya/developer/math/docs/embedded_equivalence_context_corpus.csv)
  - `squared_passthrough_zero x fraction_expand`
- status:
  - `retained`
- investment:
  - `investment_class`: coverage
  - `primary_dimension`: wrapper spread
  - `secondary_dimension`: sign/orientation robustness and fraction
    expansion subtraction
  - `cohesion_scope`: one embedded corpus row
  - `behavior_change_expected`: no engine behavior change; corpus promotion
    only
- trigger:
  - `squared_passthrough_zero x fraction_expand` covered the same denominator
    sum representative and its double-squared shell
  - the previous retained `fraction_combine` difference row confirmed the
    inverse orientation is cheap, so this completes the paired expansion
    shape under the squared wrapper
- change:
  - promoted one live row:
    `((((a-b)/d)^2) + m) - (((a/d - b/d)^2) + m) -> 0`
  - this covers `(a-b)/d -> a/d - b/d` without adding a double-squared
    variant, a general-denominator variant, or a composition core
- candidate smoke:
  - `1/1` passed
  - runner elapsed `16.48ms`
  - complexity `l3_nested_or_composed`
  - shell depth `3`
- filtered result:
  - `squared_passthrough_zero --family fraction_expand`: `3/3`
  - elapsed `18.27ms`
  - bucket moved from `2 -> 3`
- guardrails:
  - `make engine-fast`: passed
  - `make engine-scorecard-pressure`: passed
    - `simplify_zero_mixed`: `450/450`
    - elapsed `201.81ms`
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1403/1403`
    - average case runtime `2.865ms`
    - sparse bucket: `squared_passthrough_zero x fraction_expand total=3 failed=0`
    - `derive_contract`: reachability `1.000`, supported equivalence `1.000`
    - `simplify_strict`: `16475/16475`
- decision:
  - retained as coverage
  - local and global runtime stayed bounded while the sparse squared-wrapper
    bucket moved out of the count-`2` set
- next:
  - continue low `squared_passthrough_zero` coverage with cheap
    `power_merge` or `polynomial_product` before heavier
    log/radical/telescoping shapes

### 2026-04-27: Squared Passthrough Symbolic Power Merge Coverage

- area:
  - [embedded_equivalence_context_corpus.csv](/Users/javiergimenezmoya/developer/math/docs/embedded_equivalence_context_corpus.csv)
  - `squared_passthrough_zero x power_merge`
- status:
  - `retained`
- investment:
  - `investment_class`: coverage
  - `primary_dimension`: wrapper spread
  - `secondary_dimension`: symbolic exponent regime under squared passthrough
  - `cohesion_scope`: one embedded corpus row
  - `behavior_change_expected`: no engine behavior change; corpus promotion
    only
- trigger:
  - `squared_passthrough_zero x power_merge` was still at count `2`
  - the squared rows only covered fixed fractional powers and the
    double-squared variant, while simple wrappers already covered symbolic
    `x^a*x^b -> x^(a+b)`
  - a competing `polynomial_product` cube-difference squared candidate also
    passed smoke, but was less cheap locally (`13.56ms` vs `3.54ms`)
- change:
  - promoted one live row:
    `(((x^a*x^b)^2) + m) - (((x^(a+b))^2) + m) -> 0`
  - this adds symbolic exponent coverage without adding a double-squared
    variant, quotient-power variant, or cross-family composition
- candidate smoke:
  - `1/1` passed
  - runner elapsed `3.54ms`
  - complexity `l3_nested_or_composed`
  - shell depth `3`
- filtered result:
  - `squared_passthrough_zero --family power_merge`: `3/3`
  - elapsed `4.39ms`
  - bucket moved from `2 -> 3`
- guardrails:
  - `make engine-fast`: passed
  - `make engine-scorecard-pressure`: passed
    - `simplify_zero_mixed`: `450/450`
    - elapsed `201.34ms`
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1404/1404`
    - average case runtime `2.942ms`
    - sparse bucket: `squared_passthrough_zero x power_merge total=3 failed=0`
    - `derive_contract`: reachability `1.000`, supported equivalence `1.000`
    - `simplify_strict`: `16475/16475`
- decision:
  - retained as coverage
  - the candidate broadens the squared wrapper from fixed fractional exponents
    to symbolic exponent merge with low filtered cost and no global failures
- next:
  - continue low `squared_passthrough_zero` coverage with
    `polynomial_product` as the next cheap candidate, then reassess heavier
    log/radical/telescoping families

### 2026-04-27: Squared Passthrough Polynomial Product Difference Coverage

- area:
  - [embedded_equivalence_context_corpus.csv](/Users/javiergimenezmoya/developer/math/docs/embedded_equivalence_context_corpus.csv)
  - `squared_passthrough_zero x polynomial_product`
- status:
  - `retained`
- investment:
  - `investment_class`: coverage
  - `primary_dimension`: wrapper spread
  - `secondary_dimension`: sign/orientation robustness for polynomial product
  - `cohesion_scope`: one embedded corpus row
  - `behavior_change_expected`: no engine behavior change; corpus promotion
    only
- trigger:
  - `squared_passthrough_zero x polynomial_product` was still at count `2`
  - the squared rows covered the cube-sum product and its double-squared shell,
    while the cube-difference product was already stable in simple wrappers
- change:
  - promoted one live row:
    `((((x-a)*(x^2+a*x+a^2))^2) + m) - (((x^3-a^3)^2) + m) -> 0`
  - this covers `(x-a)*(x^2+a*x+a^2) -> x^3-a^3` without adding a
    double-squared variant, higher-degree product, or cross-family composition
- candidate smoke:
  - `1/1` passed
  - runner elapsed `3.93ms`
  - complexity `l3_nested_or_composed`
  - shell depth `3`
- filtered result:
  - `squared_passthrough_zero --family polynomial_product`: `3/3`
  - elapsed `2.97ms`
  - bucket moved from `2 -> 3`
- guardrails:
  - `make engine-fast`: passed
  - `make engine-scorecard-pressure`: passed
    - `simplify_zero_mixed`: `450/450`
    - elapsed `203.59ms`
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1405/1405`
    - average case runtime `2.868ms`
    - sparse bucket: `squared_passthrough_zero x polynomial_product total=3 failed=0`
    - `derive_contract`: reachability `1.000`, supported equivalence `1.000`
    - `simplify_strict`: `16475/16475`
- decision:
  - retained as coverage
  - it adds the missing sign variant for the squared polynomial-product
    representative while keeping local and global runtime bounded
- next:
  - continue low `squared_passthrough_zero` coverage with the cheapest
    remaining simple algebraic families, likely `radical_power` or
    `rationalize`, before heavier log/telescoping cases

### 2026-04-27: Squared Passthrough Radical Power Sqrt-Cube Coverage

- area:
  - [embedded_equivalence_context_corpus.csv](/Users/javiergimenezmoya/developer/math/docs/embedded_equivalence_context_corpus.csv)
  - `squared_passthrough_zero x radical_power`
- status:
  - `retained`
- investment:
  - `investment_class`: coverage
  - `primary_dimension`: wrapper spread
  - `secondary_dimension`: radical/power regime after prior simplification
  - `cohesion_scope`: one embedded corpus row
  - `behavior_change_expected`: no engine behavior change; corpus promotion
    only
- trigger:
  - `squared_passthrough_zero x radical_power` was still at count `2`
  - existing squared rows covered `x^(3/2) -> abs(x)*sqrt(x)` and its
    double-squared shell, while simple wrappers already covered the explicit
    radical source `sqrt(x^3) -> abs(x)*sqrt(x)`
  - a competing `rationalize_linear_root_squared` candidate passed
    symbolically but exceeded the local smoke slow threshold (`70.10ms`), so it
    was not promoted in this cycle
- change:
  - promoted one live row:
    `(((sqrt(x^3))^2) + m) - (((abs(x)*sqrt(x))^2) + m) -> 0`
  - this covers explicit radical source shape without adding a double-squared
    variant, passthrough variant, or cross-family composition
- candidate smoke:
  - `1/1` passed
  - runner elapsed `13.94ms`
  - complexity `l3_nested_or_composed`
  - shell depth `3`
- filtered result:
  - `squared_passthrough_zero --family radical_power`: `3/3`
  - elapsed `5.05ms`
  - bucket moved from `2 -> 3`
- guardrails:
  - `make engine-fast`: passed
  - `make engine-scorecard-pressure`: passed
    - `simplify_zero_mixed`: `450/450`
    - elapsed `201.32ms`
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1406/1406`
    - average case runtime `2.859ms`
    - sparse bucket: `squared_passthrough_zero x radical_power total=3 failed=0`
    - `derive_contract`: reachability `1.000`, supported equivalence `1.000`
    - `simplify_strict`: `16475/16475`
- decision:
  - retained as coverage
  - it broadens the squared wrapper from odd half-power syntax to explicit
    radical syntax with bounded local and global cost
- next:
  - revisit `rationalize` only with a cheaper representative or stress-only
    treatment; otherwise continue low `squared_passthrough_zero` coverage with
    simple non-log families such as `fraction_decompose` or `solve_prep`

### 2026-04-27: Squared Passthrough Fraction Decompose Scaled Linear Coverage

- area:
  - [embedded_equivalence_context_corpus.csv](/Users/javiergimenezmoya/developer/math/docs/embedded_equivalence_context_corpus.csv)
  - `squared_passthrough_zero x fraction_decompose`
- status:
  - `retained`
- investment:
  - `investment_class`: coverage
  - `primary_dimension`: wrapper spread
  - `secondary_dimension`: scaled linear fraction decomposition
  - `cohesion_scope`: one embedded corpus row
  - `behavior_change_expected`: no engine behavior change; corpus promotion
    only
- trigger:
  - `squared_passthrough_zero x fraction_decompose` was still at count `2`
  - existing squared rows covered the unscaled denominator shape
    `(a*x+b)/(x+c)`, including a double-squared shell
  - simple wrappers already covered the scaled linear denominator
    `(a*x+b)/(c*x+d) -> a/c + (b-a*d/c)/(c*x+d)`
  - `solve_prep` remained a possible low-family target, but prior ledgered
    runtime sensitivity made this narrower fraction-decomposition promotion
    the lower-risk candidate for this cycle
- change:
  - promoted one live row:
    `((((a*x+b)/(c*x+d))^2) + m) - (((a/c + (b-a*d/c)/(c*x+d))^2) + m) -> 0`
  - this covers the scaled linear decomposition shape under a shared squared
    passthrough without adding a double-squared variant or a cross-family
    composition
- candidate smoke:
  - `1/1` passed
  - runner elapsed `6.51ms`
  - complexity `l3_nested_or_composed`
  - shell depth `3`
  - wrapper overhead nodes `20`
- filtered result:
  - `squared_passthrough_zero --family fraction_decompose`: `3/3`
  - elapsed `8.75ms`
  - bucket moved from `2 -> 3`
- guardrails:
  - `make engine-fast`: passed
  - `make engine-scorecard-pressure`: passed
    - `simplify_zero_mixed`: `450/450`
    - elapsed `200.24ms`
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1407/1407`
    - average case runtime `2.843ms`
    - sparse bucket: `squared_passthrough_zero x fraction_decompose total=3 failed=0`
    - `derive_contract`: reachability `1.000`, supported equivalence `1.000`
    - `simplify_strict`: `16475/16475`
- decision:
  - retained as coverage
  - it closes another low squared-wrapper family with a scaled denominator
    representative already known to be stable in simpler wrappers
- next:
  - continue low `squared_passthrough_zero` coverage with remaining count-2
    families, preferring cheap non-log candidates before revisiting
    `rationalize` or the historically sensitive `solve_prep` lane

### 2026-04-27: Squared Passthrough Log Exp Inverse Log10 Coverage

- area:
  - [embedded_equivalence_context_corpus.csv](/Users/javiergimenezmoya/developer/math/docs/embedded_equivalence_context_corpus.csv)
  - `squared_passthrough_zero x log_exp_inverse`
- status:
  - `retained`
- investment:
  - `investment_class`: coverage
  - `primary_dimension`: wrapper spread
  - `secondary_dimension`: base-10 log/power alias
  - `cohesion_scope`: one embedded corpus row plus discovery notes
  - `behavior_change_expected`: no engine behavior change; corpus promotion
    only
- trigger:
  - `squared_passthrough_zero x log_exp_inverse` was still at count `2`
  - existing squared rows covered the natural alias
    `exp(y*log(x)) -> x^y`, including a double-squared shell
  - simple wrappers already covered the base-10 alias
    `10^(y*log10(x)) -> x^y`
  - a prior base-10 combined-additive candidate was rejected for global
    embedded runtime cost, so this cycle required a minimal non-composed smoke
    before promotion
- rejected discovery probes before promotion:
  - `conditional_factor` higher-power extraction under squared passthrough
    failed before live promotion:
    - quartic `x^2` extraction squared: `0/1`, elapsed `15.67ms`
    - matching additive passthrough quartic probe: `1/1`, elapsed `23.12ms`
    - septic `x^3` extraction squared: `0/1`, elapsed `11.43ms`
  - this is a reusable structural gap: conditional factorization by higher
    powers is stable in simpler wrappers but not when both sides are wrapped as
    squared expressions
  - additional heavier squared candidates were not promoted:
    - `integrate_prep` Dirichlet reverse squared: `0/1`, elapsed `14.80ms`
    - `telescoping_fraction` difference-squares split squared: `0/1`,
      elapsed `11.81ms`
    - `trig_expand` phase-shift pair squared: `0/1`, elapsed `73.31ms`
    - `solve_prep` negative linear coefficient squared: `0/1`, elapsed
      `73.60ms`
- change:
  - promoted one live row:
    `(((10^(y*log10(x)))^2) + m) - (((x^y)^2) + m) -> 0`
  - this covers base-10 log-exp inverse under a shared squared passthrough
    without adding double-squared depth or additive cross-family composition
- candidate smoke:
  - `1/1` passed
  - runner elapsed `3.35ms`
  - complexity `l3_nested_or_composed`
  - shell depth `3`
  - wrapper overhead nodes `12`
- filtered result:
  - `squared_passthrough_zero --family log_exp_inverse`: `3/3`
  - elapsed `4.23ms`
  - bucket moved from `2 -> 3`
- guardrails:
  - `make engine-fast`: passed
  - `make engine-scorecard-pressure`: passed
    - `simplify_zero_mixed`: `450/450`
    - elapsed `200.75ms`
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1408/1408`
    - average case runtime `2.834ms`
    - sparse bucket: `squared_passthrough_zero x log_exp_inverse total=3 failed=0`
    - `derive_contract`: reachability `1.000`, supported equivalence `1.000`
    - `simplify_strict`: `16475/16475`
- decision:
  - retained as coverage
  - the minimal log10 squared row closed a low sparse-wrapper family without
    repeating the earlier global runtime rejection from the composed log10
    candidate
- next:
  - stop trying arbitrary heavy squared wrappers as live rows; either improve
    the squared-passthrough route for a documented structural gap such as
    higher-power `conditional_factor`, or continue coverage only with
    smoke-cheap minimal representatives such as simple `log_contract`

### 2026-04-27: Squared Passthrough Log Contract Basic Sum Coverage

- area:
  - [embedded_equivalence_context_corpus.csv](/Users/javiergimenezmoya/developer/math/docs/embedded_equivalence_context_corpus.csv)
  - `squared_passthrough_zero x log_contract`
- status:
  - `retained`
- investment:
  - `investment_class`: coverage
  - `primary_dimension`: wrapper spread
  - `secondary_dimension`: basic log-sum contraction
  - `cohesion_scope`: one embedded corpus row
  - `behavior_change_expected`: no engine behavior change; corpus promotion
    only
- trigger:
  - `squared_passthrough_zero x log_contract` was still at count `2`
  - existing squared rows covered grouped-power contraction
    `ln(x^2)+ln(y^2) -> ln((x*y)^2)`, including a double-squared shell
  - simple wrappers already covered the basic contraction
    `ln(x)+ln(y) -> ln(x*y)`
  - the prior cycle showed that heavy squared candidates should not be promoted
    blindly, so this cycle used a minimal smoke-cheap representative
- change:
  - promoted one live row:
    `(((ln(x)+ln(y))^2) + m) - (((ln(x*y))^2) + m) -> 0`
  - this covers basic log-sum contraction under a shared squared passthrough
    without adding double-squared depth, reciprocal noise, or cross-family
    composition
- candidate smoke:
  - `1/1` passed
  - runner elapsed `26.07ms`
  - complexity `l3_nested_or_composed`
  - shell depth `3`
  - wrapper overhead nodes `13`
- filtered result:
  - `squared_passthrough_zero --family log_contract`: `3/3`
  - elapsed `17.47ms`
  - bucket moved from `2 -> 3`
- guardrails:
  - `make engine-fast`: passed
  - `make engine-scorecard-pressure`: passed
    - `simplify_zero_mixed`: `450/450`
    - elapsed `213.69ms`
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1409/1409`
    - average case runtime `3.102ms`
    - sparse bucket: `squared_passthrough_zero x log_contract total=3 failed=0`
    - `derive_contract`: reachability `1.000`, supported equivalence `1.000`
    - `simplify_strict`: `16475/16475`
- decision:
  - retained as coverage
  - embedded elapsed moved from the prior `3.99s` to `4.37s`, which is below
    the scorecard's material runtime delta threshold while preserving
    correctness and closing the target sparse-wrapper bucket
- next:
  - remaining count-2 squared-wrapper families are mostly heavier or already
    documented as fragile; prefer either a runtime/robustness iteration for the
    `conditional_factor` squared gap or another smoke-cheap minimal family such
    as `log_inverse_power`

### 2026-04-27: Squared Passthrough Log Inverse Power Reversed Coverage

- area:
  - [embedded_equivalence_context_corpus.csv](/Users/javiergimenezmoya/developer/math/docs/embedded_equivalence_context_corpus.csv)
  - `squared_passthrough_zero x log_inverse_power`
- status:
  - `retained`
- investment:
  - `investment_class`: coverage
  - `primary_dimension`: wrapper spread
  - `secondary_dimension`: sign/orientation robustness for log inverse power
  - `behavior_change_expected`: no engine behavior change; corpus promotion
    only
- trigger:
  - `squared_passthrough_zero x log_inverse_power` was still at count `2`
  - existing squared rows covered the forward orientation and a double-squared
    shell
  - this cycle used the reversed orientation as a minimal representative
    instead of adding a deeper or cross-family composition
- change:
  - promoted one live row:
    `(((log(x))^2) + m) - (((x^(log(log(x))/log(x)))^2) + m) -> 0`
  - this tests that the same log inverse-power identity is stable when the
    enclosing subtraction is reversed under the shared squared passthrough
- candidate smoke:
  - `1/1` passed
  - runner elapsed `13.89ms`
  - complexity `l3_nested_or_composed`
  - shell depth `3`
  - wrapper overhead nodes `11`
- filtered result:
  - `squared_passthrough_zero --family log_inverse_power`: `3/3`
  - elapsed `4.64ms`
  - average wrapper overhead nodes `12.33`
  - bucket moved from `2 -> 3`
- guardrails:
  - `make engine-fast`: passed
    - unit smoke/scorecard tests: `20/20`
    - `simplify_add_small`: `435/435`
    - `contextual_strict_fast`: `64/64`
  - `make engine-scorecard-pressure`: passed
    - `simplify_zero_mixed`: `450/450`
    - elapsed `215.82ms`
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1410/1410`
    - average case runtime `3.078ms`
    - sparse bucket: `squared_passthrough_zero x log_inverse_power total=3 failed=0`
    - `derive_contract`: reachability `1.000`, supported equivalence `1.000`
    - `simplify_strict`: `16475/16475`
- decision:
  - retained as coverage
  - the row closed the target sparse-wrapper family without material embedded
    runtime degradation; embedded moved from `1409/1409` at `3.102ms/case` to
    `1410/1410` at `3.078ms/case`
- next:
  - continue only with smoke-cheap minimal squared-wrapper representatives for
    remaining count-2 families, or switch to a runtime/robustness iteration for
    the documented `conditional_factor` higher-power squared gap

### 2026-04-27: Squared Passthrough Telescoping Affine Shift-Gap Coverage

- area:
  - [embedded_equivalence_context_corpus.csv](/Users/javiergimenezmoya/developer/math/docs/embedded_equivalence_context_corpus.csv)
  - `squared_passthrough_zero x telescoping_fraction`
- status:
  - `retained`
- investment:
  - `investment_class`: coverage
  - `primary_dimension`: wrapper spread
  - `secondary_dimension`: affine symbolic shift-gap telescoping core
  - `behavior_change_expected`: no engine behavior change; corpus promotion
    only
- trigger:
  - `squared_passthrough_zero x telescoping_fraction` was still at count `2`
  - existing squared rows covered the shifted quadratic form
    `1/(x+b)-1/(x+c) -> (c-b)/(x^2+(b+c)*x+b*c)`
  - simple wrappers already covered the affine symbolic gap form
    `1/(c-b)*(1/(a*n+b)-1/(a*n+c)) -> 1/((a*n+b)*(a*n+c))`
  - alternative smoke candidates for reversed `trig_expand` and reversed
    `rationalize` passed, but were mostly orientation repeats; this row adds a
    distinct telescoping parameterization under the squared wrapper
- change:
  - promoted one live row:
    `(((1/(c-b)*(1/(a*n+b) - 1/(a*n+c)))^2) + m) - (((1/((a*n+b)*(a*n+c)))^2) + m) -> 0`
  - this covers affine symbolic denominator shifts without adding
    double-squared depth, reciprocal noise, or cross-family composition
- candidate smoke:
  - `1/1` passed
  - runner elapsed `17.54ms`
  - complexity `l3_nested_or_composed`
  - shell depth `3`
  - wrapper overhead nodes `22`
- filtered result:
  - `squared_passthrough_zero --family telescoping_fraction`: `3/3`
  - elapsed `16.81ms`
  - average wrapper overhead nodes `22.00`
  - bucket moved from `2 -> 3`
- guardrails:
  - `make engine-fast`: passed
    - unit smoke/scorecard tests: `20/20`
    - `simplify_add_small`: `435/435`
    - `contextual_strict_fast`: `64/64`
  - `make engine-scorecard-pressure`: passed
    - `simplify_zero_mixed`: `450/450`
    - elapsed `211.38ms`
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1411/1411`
    - average case runtime `3.040ms`
    - sparse bucket: `squared_passthrough_zero x telescoping_fraction total=3 failed=0`
    - `derive_contract`: reachability `1.000`, supported equivalence `1.000`
    - `simplify_strict`: `16475/16475`
- decision:
  - retained as coverage
  - the row closed the target sparse-wrapper family without material embedded
    runtime degradation; embedded moved from `1410/1410` at `3.078ms/case` to
    `1411/1411` at `3.040ms/case`
- next:
  - remaining count-2 squared-wrapper families are
    `conditional_factor`, `integrate_prep`, `rationalize`, `solve_prep`, and
    `trig_expand`; prefer a smoke-cheap minimal representative unless choosing
    to address the documented `conditional_factor` higher-power gap directly

### 2026-04-27: Squared Passthrough Trig Product-To-Sum Coverage

- area:
  - [embedded_equivalence_context_corpus.csv](/Users/javiergimenezmoya/developer/math/docs/embedded_equivalence_context_corpus.csv)
  - `squared_passthrough_zero x trig_expand`
- status:
  - `retained`
- investment:
  - `investment_class`: coverage
  - `primary_dimension`: wrapper spread
  - `secondary_dimension`: product-to-sum trig expansion under squared shell
  - `behavior_change_expected`: no engine behavior change; corpus promotion
    only
- trigger:
  - `squared_passthrough_zero x trig_expand` was still at count `2`
  - existing squared rows covered only the `sin(2*x) -> 2*sin(x)*cos(x)`
    double-angle shape
  - simple wrappers already covered product-to-sum forms with independent
    angles, so the smallest useful promotion was one squared row for
    `2*sin(x)*cos(y) -> sin(x+y)+sin(x-y)`
  - alternative smoke candidates for reversed `rationalize` and reversed
    `integrate_prep` passed, but they mainly tested orientation rather than a
    distinct core shape
- rejected probe before promotion:
  - `solve_prep_complete_square_symbolic_leading_coeff_squared` failed
    smoke as `0/1`, runner elapsed `45.70ms`
  - the same leading-coefficient complete-square form exists in simpler
    wrappers, so this is a useful future robustness/runtime candidate rather
    than a live promotion
- change:
  - promoted one live row:
    `(((2*sin(x)*cos(y))^2) + m) - (((sin(x+y) + sin(x-y))^2) + m) -> 0`
  - this covers product-to-sum trig expansion without adding double-squared
    depth, reciprocal noise, or cross-family composition
- candidate smoke:
  - `1/1` passed
  - runner elapsed `13.47ms`
  - complexity `l3_nested_or_composed`
  - shell depth `3`
  - wrapper overhead nodes `16`
- filtered result:
  - `squared_passthrough_zero --family trig_expand`: `3/3`
  - elapsed `3.73ms`
  - average wrapper overhead nodes `15.33`
  - bucket moved from `2 -> 3`
- guardrails:
  - `make engine-fast`: passed
    - unit smoke/scorecard tests: `20/20`
    - `simplify_add_small`: `435/435`
    - `contextual_strict_fast`: `64/64`
  - `make engine-scorecard-pressure`: passed
    - `simplify_zero_mixed`: `450/450`
    - elapsed `216.25ms`
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1412/1412`
    - average case runtime `3.123ms`
    - sparse bucket: `squared_passthrough_zero x trig_expand total=3 failed=0`
    - `derive_contract`: reachability `1.000`, supported equivalence `1.000`
    - `simplify_strict`: `16475/16475`
- decision:
  - retained as coverage
  - the row closed the target sparse-wrapper family; embedded moved from
    `1411/1411` at `3.040ms/case` to `1412/1412` at `3.123ms/case`, below the
    material runtime threshold
- next:
  - remaining count-2 squared-wrapper families are `conditional_factor`,
    `integrate_prep`, `rationalize`, and `solve_prep`; prefer either
    smoke-cheap minimal representatives or a focused robustness fix for the
    repeated `solve_prep`/`conditional_factor` squared gaps

### 2026-04-27: Solve Prep Leading-Coefficient Squared Discovery

- area:
  - generated discovery / embedded equivalence candidate smoke
  - [embedded_equivalence_context_corpus.csv](/Users/javiergimenezmoya/developer/math/docs/embedded_equivalence_context_corpus.csv)
  - `squared_passthrough_zero` x `solve_prep`
- status:
  - `resolved`
- attempted row:
  - `solve_prep_complete_square_symbolic_leading_coeff_squared`
  - `(((a*x^2 + b*x + c)^2) + m) - (((a*(x + b/(2*a))^2 + c - b^2/(4*a))^2) + m)`
- local lane:
  - `python3 scripts/engine_embedded_candidate_smoke.py --json --row ...`
- local result:
  - attempted squared candidate failed before live promotion:
    - `status=fail`
    - `passed=0`
    - `failed=1`
    - `runner_elapsed=45.70ms`
- structural read:
  - the same leading-coefficient complete-square identity is already live in
    additive, scaled, common-denominator, and shifted-quotient wrappers
  - the monic complete-square variant already has squared passthrough coverage
  - the miss therefore points to a reusable gap in non-monic `solve_prep`
    complete-square handling under squared wrappers, not to a typo or malformed
    candidate
- decision:
  - resolved by the later square-base fallback robustness iteration
  - the same representative is now promoted live as
    `solve_prep_complete_square_symbolic_leading_coeff_squared`

### 2026-04-27: Squared Passthrough Rationalize Reverse Coverage

- area:
  - [embedded_equivalence_context_corpus.csv](/Users/javiergimenezmoya/developer/math/docs/embedded_equivalence_context_corpus.csv)
  - `squared_passthrough_zero x rationalize`
- status:
  - `retained`
- investment:
  - `investment_class`: coverage
  - `primary_dimension`: wrapper spread
  - `secondary_dimension`: sign/orientation robustness for radical
    rationalization
  - `behavior_change_expected`: no engine behavior change; corpus promotion
    only
- trigger:
  - `squared_passthrough_zero x rationalize` was still at count `2`
  - existing squared rows covered direct orientation and double-squared depth
    for `1/(sqrt(a)+sqrt(b)) -> (sqrt(a)-sqrt(b))/(a-b)`
  - combined-additive rows already covered reversed rationalization in noisier
    contexts, so the smallest missing live representative was the reversed
    squared wrapper with no extra family composition
- rejected probe before promotion:
  - `factor_out_square_with_division_quartic_squared` for
    `squared_passthrough_zero x conditional_factor` failed smoke as `0/1`,
    runner elapsed `20.21ms`
  - this was not re-entered as a new discovery because the same higher-power
    conditional-factor squared gap is already documented in this ledger
- change:
  - promoted one live row:
    `((((sqrt(a)-sqrt(b))/(a-b))^2) + m) - (((1/(sqrt(a)+sqrt(b)))^2) + m) -> 0`
  - this covers reversed radical rationalization without adding double-squared
    depth, reciprocal noise, or cross-family composition
- candidate smoke:
  - `1/1` passed
  - runner elapsed `2.70ms`
  - complexity `l3_nested_or_composed`
  - shell depth `3`
  - wrapper overhead nodes `16`
- filtered result:
  - `squared_passthrough_zero --family rationalize`: `3/3`
  - elapsed `5.03ms`
  - average wrapper overhead nodes `17.33`
  - bucket moved from `2 -> 3`
- guardrails:
  - `make engine-fast`: passed
    - unit smoke/scorecard tests: `20/20`
    - `simplify_add_small`: `435/435`
    - `contextual_strict_fast`: `64/64`
  - `make engine-scorecard-pressure`: passed
    - `simplify_zero_mixed`: `450/450`
    - elapsed `212.24ms`
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1413/1413`
    - average case runtime `3.057ms`
    - sparse bucket: `squared_passthrough_zero x rationalize total=3 failed=0`
    - `derive_contract`: reachability `1.000`, supported equivalence `1.000`
    - `simplify_strict`: `16475/16475`
- decision:
  - retained as coverage
  - the row closed the target sparse-wrapper family; embedded moved from
    `1412/1412` at `3.123ms/case` to `1413/1413` at `3.057ms/case`, below the
    material runtime threshold
- next:
  - remaining count-2 squared-wrapper families are `conditional_factor`,
    `integrate_prep`, and `solve_prep`; prefer `integrate_prep` if continuing
    low-risk corpus closure, or switch to robustness for the documented
    `conditional_factor` and `solve_prep` squared gaps

### 2026-04-27: Integrate Prep Dirichlet Squared Discovery

- area:
  - generated discovery / embedded equivalence candidate smoke
  - [embedded_equivalence_context_corpus.csv](/Users/javiergimenezmoya/developer/math/docs/embedded_equivalence_context_corpus.csv)
  - `squared_passthrough_zero` x `integrate_prep`
- status:
  - `resolved`
- attempted rows:
  - `integrate_prep_dirichlet_basic_squared`
  - `(((1 + 2*cos(x) + 2*cos(2*x))^2) + m) - (((sin(5*x/2)/sin(x/2))^2) + m)`
  - `integrate_prep_dirichlet_reverse_squared`
  - `(((sin(5*x/2)/sin(x/2))^2) + m) - (((1 + 2*cos(x) + 2*cos(2*x))^2) + m)`
- local lane:
  - `python3 scripts/engine_embedded_candidate_smoke.py --json --row ...`
- local result:
  - direct squared candidate failed before live promotion:
    - `status=fail`
    - `passed=0`
    - `failed=1`
    - `runner_elapsed=15.26ms`
  - reversed squared candidate failed before live promotion:
    - `status=fail`
    - `passed=0`
    - `failed=1`
    - `runner_elapsed=4.56ms`
- resolution probe:
  - after the square-base fallback robustness work, both attempted orientations
    pass candidate smoke:
    - direct: `1/1`, runner elapsed `15.18ms`
    - reverse: `1/1`, runner elapsed `3.34ms`
  - promoted only the direct row as the minimal live representative:
    `integrate_prep_dirichlet_basic_squared`
- structural read:
  - the same Dirichlet kernel identity is already live in additive, scaled,
    common-denominator, shifted-quotient, and combined-additive wrappers
  - both squared orientations fail, while the Morrie product identity remains
    stable under squared wrappers
  - this points to an `integrate_prep` subfamily gap for Dirichlet forms under
    squared passthrough, not to a malformed candidate
- decision:
  - resolved by the square-base fallback robustness route
  - close the observe-only discovery and retain one minimal live corpus row for
    Dirichlet squared passthrough coverage

### 2026-04-27: Squared Passthrough Integrate Prep Morrie Reverse Coverage

- area:
  - [embedded_equivalence_context_corpus.csv](/Users/javiergimenezmoya/developer/math/docs/embedded_equivalence_context_corpus.csv)
  - `squared_passthrough_zero x integrate_prep`
- status:
  - `retained`
- investment:
  - `investment_class`: coverage
  - `primary_dimension`: wrapper spread
  - `secondary_dimension`: sign/orientation robustness for Morrie product
  - `behavior_change_expected`: no engine behavior change; corpus promotion
    only
- trigger:
  - `squared_passthrough_zero x integrate_prep` was still at count `2`
  - existing squared rows covered direct orientation and double-squared depth
    for `cos(x)*cos(2*x)*cos(4*x) -> sin(8*x)/(8*sin(x))`
  - the Dirichlet squared candidates exposed a subfamily gap and were kept as
    observe-only discovery, so the smallest stable live promotion was the
    reversed Morrie product under the same squared wrapper
- rejected probe before promotion:
  - `integrate_prep_dirichlet_basic_squared` failed smoke as `0/1`, runner
    elapsed `15.26ms`
  - `integrate_prep_dirichlet_reverse_squared` failed smoke as `0/1`, runner
    elapsed `4.56ms`
  - both are documented in the adjacent observe-only discovery section
- change:
  - promoted one live row:
    `(((sin(8*x)/(8*sin(x)))^2) + m) - (((cos(x)*cos(2*x)*cos(4*x))^2) + m) -> 0`
  - this covers reversed Morrie product preparation without adding
    double-squared depth, reciprocal noise, or cross-family composition
- candidate smoke:
  - `1/1` passed
  - runner elapsed `13.50ms`
  - complexity `l3_nested_or_composed`
  - shell depth `3`
  - wrapper overhead nodes `18`
- filtered result:
  - `squared_passthrough_zero --family integrate_prep`: `3/3`
  - elapsed `3.47ms`
  - average wrapper overhead nodes `19.33`
  - bucket moved from `2 -> 3`
- guardrails:
  - `make engine-fast`: passed
    - unit smoke/scorecard tests: `20/20`
    - `simplify_add_small`: `435/435`
    - `contextual_strict_fast`: `64/64`
  - `make engine-scorecard-pressure`: passed
    - `simplify_zero_mixed`: `450/450`
    - elapsed `220.27ms`
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1414/1414`
    - average case runtime `3.098ms`
    - sparse bucket: `squared_passthrough_zero x integrate_prep total=3 failed=0`
    - observe-only discoveries: `3`
    - `derive_contract`: reachability `1.000`, supported equivalence `1.000`
    - `simplify_strict`: `16475/16475`
- decision:
  - retained as coverage
  - the row closed the target sparse-wrapper family; embedded moved from
    `1413/1413` at `3.057ms/case` to `1414/1414` at `3.098ms/case`, below the
    material runtime threshold
- next:
  - only `conditional_factor` and `solve_prep` remain at count `2` for
    `squared_passthrough_zero`; both have documented squared-wrapper gaps, so
    prefer a robustness iteration over more corpus closure

### 2026-04-27: Squared Passthrough Conditional Factor Square-Base Robustness

- area:
  - [arithmetic.rs](/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/arithmetic.rs)
  - [embedded_equivalence_context_corpus.csv](/Users/javiergimenezmoya/developer/math/docs/embedded_equivalence_context_corpus.csv)
  - `squared_passthrough_zero x conditional_factor`
- status:
  - `retained`
- investment:
  - `investment_class`: robustness
  - `primary_dimension`: squared wrapper robustness for conditional factor
  - `secondary_dimension`: square-base equivalence after passthrough
    cancellation
  - `cohesion_scope`: narrow exact-zero helpers in `arithmetic.rs`
  - `behavior_change_expected`: yes
- trigger:
  - `conditional_factor` was one of the last count-2
    `squared_passthrough_zero` families
  - the quartic factor-out base already passed additive/scaled/common/
    shifted wrappers, but the squared passthrough candidate failed before
    promotion
- baseline probe:
  - candidate row:
    `(((a*x^4 + b*x^3 + c*x^2 + d)^2) + m) - (((x^2*(a*x^2 + b*x + c + d/x^2))^2) + m) -> 0`
  - smoke before change: `0/1`, `status=fail`, runner elapsed `8.58ms`
  - failure shape after the normal flow canceled `+m`: residual stayed as
    `(a*x^4 + b*x^3 + c*x^2 + d)^2 - (x^2*(...))^2`
- change:
  - added square-base equivalence handling for shared passthrough cores
  - added exact-zero handling for a bare difference of equivalent squares
    after the passthrough terms have already been canceled
  - added focused unit coverage for both the shared-passthrough form and the
    exposed `A^2 - B^2` residual
  - promoted one live corpus row for the quartic squared conditional-factor
    representative
- candidate smoke:
  - after change: `1/1` passed
  - runner elapsed `6.61ms`
  - complexity `l3_nested_or_composed`
  - shell depth `3`
  - wrapper overhead nodes `28`
- filtered result:
  - `squared_passthrough_zero --family conditional_factor`: `3/3`
  - elapsed `4.51ms`
  - bucket moved from `2 -> 3`
- guardrails:
  - unit tests:
    - `shared_passthrough_square_base_equivalence_keeps_quartic_conditional_factor_regression`
    - `collapse_exact_zero_additive_subexpression_matches_quartic_conditional_factor_square_difference`
  - `make engine-fast`: passed
    - unit smoke/scorecard tests: `20/20`
    - `simplify_add_small`: `435/435`
    - `contextual_strict_fast`: `64/64`
  - `make engine-scorecard-pressure`: passed
    - `simplify_zero_mixed`: `450/450`
    - elapsed `223.47ms`
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1415/1415`
    - average case runtime `3.286ms`
    - `squared_passthrough_zero`: `70/70`
    - sparse bucket: `squared_passthrough_zero x conditional_factor total=3 failed=0`
    - observe-only discoveries: `3`
    - `derive_contract`: reachability `1.000`, supported equivalence `1.000`
    - `simplify_strict`: `16475/16475`, `0` timeouts
  - isolated embedded rerun after guardrail:
    - `1415/1415`, elapsed `4.64s`
- decision:
  - retained as robustness plus one minimal live promotion
  - the iteration converts a previously documented structural weakness into a
    bounded engine route; no failures or timeouts were introduced
- next:
  - `solve_prep` remains the only count-2 `squared_passthrough_zero` family;
    prefer a robustness iteration for its non-monic leading-coefficient
    squared gap before adding more simple corpus rows

### 2026-04-27: Squared Passthrough Solve Prep Square-Base Robustness

- area:
  - [arithmetic.rs](/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/arithmetic.rs)
  - [embedded_equivalence_context_corpus.csv](/Users/javiergimenezmoya/developer/math/docs/embedded_equivalence_context_corpus.csv)
  - `squared_passthrough_zero x solve_prep`
- status:
  - `retained`
- investment:
  - `investment_class`: robustness
  - `primary_dimension`: squared wrapper robustness for non-monic
    `solve_prep`
  - `secondary_dimension`: bare `A^2 - B^2` residual after passthrough
    cancellation
  - `cohesion_scope`: narrow exact-zero fallback in `arithmetic.rs`
  - `behavior_change_expected`: yes
- trigger:
  - `solve_prep` was the last count-2 `squared_passthrough_zero` family
  - the leading-coefficient complete-square candidate had already been
    documented as a structural miss under squared passthrough
- baseline probe:
  - candidate row:
    `(((a*x^2 + b*x + c)^2) + m) - (((a*(x + b/(2*a))^2 + c - b^2/(4*a))^2) + m) -> 0`
  - smoke before change: `0/1`, `status=fail`, runner elapsed `290.95ms`
  - negative-linear sibling probe also failed before this change:
    `0/1`, runner elapsed `239.15ms`
- change:
  - extended the difference-of-equivalent-square-bases helper with a fallback
    that proves the base residual using `try_build_exact_zero_identity_rewrite_direct`
  - added focused unit coverage for the exposed `A^2 - B^2` residual and the
    full squared-passthrough wrapper
  - promoted one live corpus row for the non-monic leading-coefficient
    complete-square representative
  - closed the prior pending `solve_prep` discovery section as resolved
- candidate smoke:
  - after change: `1/1` passed
  - runner elapsed `19.00ms`
  - complexity `l3_nested_or_composed`
  - shell depth `3`
  - wrapper overhead nodes `20`
- filtered result:
  - `squared_passthrough_zero --family solve_prep`: `3/3`
  - elapsed `12.51ms`
  - bucket moved from `2 -> 3`
- guardrails:
  - unit tests:
    - `collapse_exact_zero_additive_subexpression_matches_solve_prep_square_difference`
    - `collapse_exact_zero_additive_subexpression_matches_solve_prep_squared_passthrough`
  - `make engine-fast`: passed
    - unit smoke/scorecard tests: `20/20`
    - `simplify_add_small`: `435/435`
    - `contextual_strict_fast`: `64/64`
  - `make engine-scorecard-pressure`: passed
    - `simplify_zero_mixed`: `450/450`
    - elapsed `223.12ms`
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1416/1416`
    - average case runtime `3.305ms`
    - `squared_passthrough_zero`: `71/71`
    - sparse bucket: `squared_passthrough_zero x solve_prep total=3 failed=0`
    - `derive_contract`: reachability `1.000`, supported equivalence `1.000`
    - `simplify_strict`: `16475/16475`, `0` timeouts
- decision:
  - retained as robustness plus one minimal live promotion
  - all `squared_passthrough_zero` families now have at least three live cases
    with zero embedded failures
- next:
  - with sparse squared-wrapper family counts closed, prefer an observability
    or runtime iteration over further corpus growth unless a new structural
    miss appears in the generated discovery ledger

### 2026-04-27: Squared Passthrough Dirichlet Discovery Closure

- area:
  - [embedded_equivalence_context_corpus.csv](/Users/javiergimenezmoya/developer/math/docs/embedded_equivalence_context_corpus.csv)
  - `squared_passthrough_zero x integrate_prep`
- status:
  - `retained`
- investment:
  - `investment_class`: coverage
  - `primary_dimension`: Dirichlet kernel semantic subfamily under squared
    passthrough
  - `secondary_dimension`: stale observe-only discovery closure
  - `behavior_change_expected`: no engine code change; prior square-base
    fallback work already unlocked the candidate
- trigger:
  - `Integrate Prep Dirichlet Squared Discovery` was the only remaining
    observe-only discovery in the generated scorecard
  - rerunning both originally failed squared orientations showed that the
    current engine now proves them
- probes:
  - direct candidate smoke:
    - `integrate_prep_dirichlet_basic_squared`: `1/1`
    - runner elapsed `15.18ms`
  - reverse candidate smoke:
    - `integrate_prep_dirichlet_reverse_squared`: `1/1`
    - runner elapsed `3.34ms`
- change:
  - promoted one minimal live row:
    `integrate_prep_dirichlet_basic_squared`
  - marked the prior Dirichlet squared discovery as `resolved`
  - did not promote the reverse row because it is now confirmed by smoke and
    would mainly duplicate the same squared Dirichlet subfamily in live corpus
- filtered result:
  - `squared_passthrough_zero --family integrate_prep`: `4/4`
  - elapsed `3.82ms`
- guardrails:
  - `make engine-fast`: passed
    - unit smoke/scorecard tests: `20/20`
    - `simplify_add_small`: `435/435`
    - `contextual_strict_fast`: `64/64`
  - `make engine-scorecard-pressure`: passed
    - `simplify_zero_mixed`: `450/450`
    - elapsed `219.47ms`
  - `make engine-scorecard`: passed
    - `embedded_equivalence_context`: `1417/1417`
    - average case runtime `3.169ms`
    - `squared_passthrough_zero`: `72/72`
    - sparse bucket: `squared_passthrough_zero x integrate_prep total=4 failed=0`
    - observe-only discoveries: `0`
    - `derive_contract`: reachability `1.000`, supported equivalence `1.000`
    - `simplify_strict`: `16475/16475`, `0` timeouts
- decision:
  - retained as coverage
  - this closes the last generated discovery without changing runtime code
- next:
  - with generated discoveries at zero, prefer an observability/runtime cycle
    over more live corpus growth unless a fresh structural miss appears

### 2026-04-27: Direct Small-Zero Pair Flag-Scan Cache Rejection

- area:
  - [orchestrator.rs](/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/orchestrator.rs)
  - `root.addsub.00.direct_small_zero_pair`
  - `root.direct_small_zero_composition.candidate.*`
- status:
  - `rejected`
- investment:
  - `investment_class`: runtime
  - `primary_dimension`: direct-small-zero no-match cost
  - `secondary_dimension`: scanner reuse / cheap-gate discipline
  - `behavior_change_expected`: no semantic behavior change
- trigger:
  - the profiled embedded slice showed persistent no-match cost:
    - `root.addsub.00.direct_small_zero_pair`: `905` attempts, `895` misses
    - `root.addsub.00.direct_small_zero_pair.compact_first`: `321` attempts,
      `296` misses
    - `root.direct_small_zero_composition.candidate.three_core_groups`: `75`
      attempts, `56` misses
- rejected candidate:
  - computing `HotDirectSmallZeroFamilyFlags` for both sides up front in
    `matches_direct_small_zero_pair_root`
  - a narrower variant that replaced only the log/division partner-side scans
    with one flag scan
- focused result:
  - baseline filtered profile:
    - total profiled time `175.419ms`
    - add/sub direct route `46.597ms`
    - compact direct route `40.722ms`
  - up-front flag-cache attempt:
    - total profiled time `191.111ms`
    - add/sub direct route `50.616ms`
    - compact direct route `44.505ms`
  - narrow log/division scan-cache attempt:
    - repeat totals `190.931ms` and `184.403ms`
    - still no retained local win over baseline
- guardrails:
  - targeted direct-small-zero tests stayed green:
    - `cargo test -q -p cas_engine direct_small_zero_pair_shortcut -- --nocapture`
      `28/28`
  - `cargo fmt -- --check`: passed
  - `git diff --check`: passed
- decision:
  - rejected and reverted; no runtime code retained
  - the evidence says not to scan/cache broad flags before a narrower candidate
    gate proves they are needed
- next:
  - prefer a more specific gate for the actual `three_core_groups` miss shapes
    or richer observability of the full expression/context before trying more
    broad scan reuse in direct-small-zero pair matching
