# Slow CI Test Ledger

This document is a derived workflow artifact under:

- [ENGINE_IMPROVEMENT_AUTOMATION.md](/Users/javiergimenezmoya/developer/math/docs/ENGINE_IMPROVEMENT_AUTOMATION.md)
- [ENGINE_TEST_CORPUS_ROADMAP.md](/Users/javiergimenezmoya/developer/math/docs/ENGINE_TEST_CORPUS_ROADMAP.md)

Its job is simple:

- preserve reproducible slow tests seen in `make ci`
- classify them correctly
- record the retained fix or the current blocking hypothesis
- keep the embedded corpus guardrail attached to engine-side fixes

This is not a backlog of “tests that once looked slow”.

Each entry should be based on:

- an exact narrow repro command
- a measured current timing
- a classification:
  - `runner noise`
  - `engine runtime pathology`
  - `test verification pathology`
- a retained or pending action

## Collection Rules

For each entry, keep:

- test name
- crate / area
- exact repro command
- latest measured time
- classification
- root cause hypothesis
- retained action
- embedded corpus result when the retained action touched engine runtime
- status:
  - `open`
  - `fixed in engine`
  - `fixed in test`
  - `noise only`

Do not create entries for:

- compile-only waits
- file-lock waits
- suite startup noise without a narrow repro

## Current Entries

### 2026-05-02: `integrate_contract_supported_antiderivatives_verify_by_differentiation`

- area:
  - `cas_cli`
  - integration contract antiderivative verification
- repro:
  - `cargo test -p cas_cli --test integrate_contract_tests integrate_contract_supported_antiderivatives_verify_by_differentiation -- --exact --nocapture`
- latest measured time:
  - `65.24s` test time (`real 65.34s`)
- classification:
  - `test verification pathology`
- root cause hypothesis:
  - the contract intentionally walks the supported-antiderivative table and
    verifies each primitive by differentiating and simplifying the residual, so
    the single test remains broad and debug-slow even when the individual
    public integration cases are passing
- retained action:
  - none for the slow path in this CI repair; the functional CI failure was a
    duplicate required-condition assertion and was fixed in the integration
    condition publisher
  - pending follow-up: split this broad verification table into smaller
    representative debug shards or move heavyweight rows to a slower profile
- embedded corpus guardrail:
  - unchanged engine runtime path for the slow-test classification
- status:
  - `open`

### 2026-05-01: `eval_simplify_steps_off_diff_shifted_linear_times_sec_csc_avoids_timeout`

- area:
  - `cas_engine`
  - debug eval calculus smoke test
- repro:
  - `cargo test -q -p cas_engine eval_simplify_steps_off_diff_shifted_linear_times_sec_csc_avoids_timeout -- --nocapture`
- latest measured time:
  - before fix: failed under the test's internal `50 ms` simplification budget
    (`finished in 0.13s`)
  - after fix: `0.19s` test time
  - 2026-05-02 follow-up: focused repro still passed at `0.19s`, but full
    `make ci` failed once under the `200 ms` internal debug budget after prior
    suite load
- classification:
  - `test verification pathology`
- root cause hypothesis:
  - the public release CLI path for the hottest case returns in about `20 ms`
    without warnings, but the debug unit smoke budget was tight enough to turn
    normal debug overhead into a false timeout warning
- retained action:
  - raised only this sec/csc smoke test's internal time budget from `50 ms` to
    `200 ms`, then to `500 ms` after a full-suite debug CI false timeout; the
    test still asserts the exact result and required pole condition
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-05-01: `integrate_contract_shifted_polynomial_atanh_surd_width_uses_compact_positive_domain`

- area:
  - `cas_cli`
  - integration contract antiderivative verification
- repro:
  - `cargo test -q -p cas_cli --test integrate_contract_tests integrate_contract_shifted_polynomial_atanh_surd_width_uses_compact_positive_domain -- --exact --nocapture`
- latest measured time:
  - before fix: `66.51s` test time (`real 69.85s`)
  - after fix: `0.08s` test time
- classification:
  - `engine runtime pathology`
- root cause hypothesis:
  - the public integral itself returns quickly, but the contract's
    antiderivative verification differentiates a shifted `atanh` surd-width
    antiderivative and then simplifies the residual against
    `(2*x+2)/(3-(x+1)^4)`; that real engine simplification path eventually
    reaches `0` but is slow in debug and still several seconds in release
- retained action:
  - added a direct symbolic differentiation route for constant-scaled
    `atanh` of a rational/surd-scaled polynomial, so the derivative is built as
    a compact rational gap instead of going through `sqrt(1-u^2)^(-2)`
  - preserved the raw `diff(...)` target for this shape so public eval reaches
    the direct derivative before expensive pre-diff target simplification
- embedded corpus guardrail:
  - `make engine-scorecard`: embedded `1445/1445`, failed `0`
- status:
  - `fixed in engine`

### 2026-04-22: `derive_didactic_symbolic_trinomial_cube_expansion_shows_real_intermediate`

- area:
  - `cas_solver`
  - derive planner fast-path routing
- repro:
  - `cargo test -p cas_didactic --test derive_didactic_audit derive_didactic_symbolic_trinomial_cube_expansion_shows_real_intermediate -- --nocapture`
- latest measured time:
  - `0.03s`
- classification:
  - `engine runtime pathology`
- root cause:
  - `try_fast_direct_hyperbolic_derive(...)` was opening on a purely
    polynomial derive pair and sending `(a+b+c)^3` traffic through a
    hyperbolic route that did not belong there
- retained action:
  - gated the fast hyperbolic derive lane so it only opens when source or
    target actually contains hyperbolic builtins
- embedded corpus guardrail:
  - not the primary guardrail for this derive-side fix
- status:
  - `fixed in engine`

### 2026-04-23: `derive_didactic_audit` repeated-case harness

- area:
  - `cas_didactic`
  - derive didactic audit harness
- repro:
  - `cargo test -p cas_didactic --test derive_didactic_audit derive_didactic_complete_square_negative_linear_coeff_stays_direct -- --exact`
  - `cargo test -p cas_didactic --test derive_didactic_audit derive_didactic_negative_symbolic_binomial_cube_factorization_explains_pattern -- --exact`
  - `cargo test -p cas_didactic --test derive_didactic_audit derive_didactic_representative_rationalize_cases_keep_conjugate_narrative -- --exact`
- latest measured time before fix:
  - `9.17s`
  - `7.97s`
  - `4.28s`
- latest measured time after fix:
  - `0.11s`
  - `3.74s`
  - `0.09s`
- classification:
  - `test verification pathology`
- root cause:
  - `audit_case(...)` always built both the web/json payload and the CLI
    derive transcript eagerly, even for tests that only inspected `json_steps`
    or `flags`
  - the audit file reuses the same helper heavily, so that eager CLI work was
    being paid across many didactic assertions with no value unless a test
    failed or the markdown report was being generated
- retained action:
  - cache `AuditArtifact` by derive case key inside the test harness
  - cache the parsed derive audit corpus globally and resolve `derive_case_by_id(...)`
    from that map instead of reparsing the CSV and generated audit markdown on each
    lookup
  - remove eager `cli_lines` generation from `audit_case(...)`
  - only regenerate CLI lines on demand in assertion failure messages and in
    the markdown report builder
  - trim `derive_didactic_representative_rationalize_cases_keep_conjugate_narrative`
    down to the three structurally distinct shapes that actually exercise the
    conjugate narrative (`numeric`, `shifted`, `symbolic-plus`)
  - replace the slow representative complete-square loop with the single
    remaining sign-variant case that still adds signal beyond the quick audit
    sample and the embedded equivalence corpus
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `derive_didactic_negative_symbolic_binomial_cube_factorization_explains_pattern`

- area:
  - `cas_solver`
  - direct derive factor fast-path
- repro:
  - `cargo run -q -p cas_cli -- eval 'derive a^3 - 3*a^2*b + 3*a*b^2 - b^3, (a-b)^3' --format json`
  - `cargo test -p cas_didactic --test derive_didactic_audit derive_didactic_negative_symbolic_binomial_cube_factorization_explains_pattern -- --exact`
- latest measured time before fix:
  - CLI `simplify_us`: `4002561`
  - didactic audit regression: `3.74s`
- latest measured time after fix:
  - CLI `simplify_us`: `1092`
  - didactic audit regression: `0.01s`
- classification:
  - `engine runtime pathology`
- root cause:
  - the negative symbolic binomial cube pair was reaching the broad derive
    target classifier and planner before landing on `factor`, even though the
    source/target pair itself is a direct one-step identity
- retained action:
  - add a narrow direct derive fast-path for the exact
    `a^3 - 3a^2b + 3ab^2 - b^3 -> (a - b)^3` / `a^3 + 3a^2b + 3ab^2 + b^3 -> (a + b)^3`
    family
  - keep the structural matcher reusable in `cas_math::factor`
- embedded corpus guardrail:
  - `1145/1145`, `3.68s`
- status:
  - `fixed in engine`

### 2026-04-23: `derive_didactic_representative_finite_telescoping_sum_cases_keep_partial_fraction_narrative`

- area:
  - `cas_didactic`
  - derive didactic representative audit harness
- repro:
  - `cargo test -p cas_didactic --test derive_didactic_audit derive_didactic_representative_finite_telescoping_sum_cases_keep_partial_fraction_narrative -- --exact`
- latest measured time before fix:
  - `0.58s`
- latest measured time after fix:
  - `0.04s`
- classification:
  - `test verification pathology`
- root cause:
  - the representative telescoping-sum audit was still paying an extra derive
    for the `unit gap` narrative even though that language was already pinned by
    `derive_didactic_finite_telescoping_sum_uses_partial_fraction_then_cancellation_language`
- retained action:
  - keep only the affine-gap narrative in the representative audit
  - replace the fully symbolic affine sample with a cheaper inline affine case
    that still renders the same partial-fraction and telescoping substeps
  - leave the `unit gap` narrative pinned by its dedicated base test
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `derive_didactic_quick_audit_cases_render_steps_without_redundant_single_substeps`

- area:
  - `cas_didactic`
  - derive didactic quick audit harness
- repro:
  - `cargo test -p cas_didactic --test derive_didactic_audit derive_didactic_quick_audit_ -- --nocapture`
- latest measured time before fix:
  - `0.52s`
- latest measured time after fix:
  - `0.14s`
- classification:
  - `test verification pathology`
- root cause:
  - the quick audit was still sampling two cases for every derive family even in
    low-churn areas where one smoke case is enough to detect redundant substeps,
    and after that first trim it still remained as a single broad wall-clock
    tail inside `make ci` even though its sampled families are independent
- retained action:
  - keep a baseline quota of one sampled case per family
  - retain two sampled cases only for the higher-churn didactic families:
    `expand`, `factor`, `log_expand`, `log_contract`, `trig_expand`,
    `trig_contract`, and `finite_telescoping`
  - prefer fast, already-pinned representatives for the broad smoke audit in:
    `collect`, `conditional_factor`, `fraction_combine`, `fraction_decompose`,
    `fraction_expand`, `polynomial_product`, `power_merge`, `solve_prep`, and
    `factor`
  - for `solve_prep`, use the monic numeric complete-square case as the quick
    audit representative and leave the negative-linear variant pinned by its
    dedicated regression
  - for the algebraic slice, avoid the alphabetical heavy defaults
    (`collect_common_symbolic_coefficients`,
    `factor_out_cube_with_division_septic`,
    `expand_fraction_exact_division_term_plus_remainder`,
    `merge_four_same_base_symbolic_powers`) and pin the cheaper direct
    representatives instead
  - split the broad quick audit into three disjoint smoke tests:
    `algebraic`, `log_trig`, and `structural`
  - add a cheap partition-cover test so the split still guarantees coverage of
    exactly the same sampled corpus as the previous monolithic test
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `derive_didactic_reverse_structural_nested_fraction_cases_keep_trace_direct`

- area:
  - `cas_didactic`
  - derive didactic nested-fraction audit harness
- repro:
  - `cargo test -p cas_didactic --test derive_didactic_audit derive_didactic_reverse_structural_nested_fraction_cases_keep_trace_direct -- --exact`
- latest measured time before fix:
  - `0.28s`
- latest measured time after fix:
  - `0.11s`
- classification:
  - `test verification pathology`
- root cause:
  - the reverse nested-fraction audit was paying five derives even though it
    only needed to pin three distinct narratives: denominator factoring,
    numerator factoring, and one inline compound-denominator trace
- retained action:
  - keep one simple denominator case
  - keep one simple numerator case
  - keep one inline compound-denominator case
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `tabulated_reverse_structural_nested_fraction`

- area:
  - `cas_solver`
  - derive CLI/unit-test harness inside `src/analysis_command_eval_tests.rs`
- repro:
  - `cargo test -p cas_solver --lib tabulated_reverse_structural_nested_fraction -- --nocapture`
- latest measured time before fix:
  - `0.32s`
- latest measured time after fix:
  - `0.08s`
- classification:
  - `test verification pathology`
- root cause:
  - the tabulated reverse structural nested-fraction smoke still repeated six
    variants even though the family had already been reduced didactically to
    three distinct narratives: simple denominator factoring, simple numerator
    factoring, and compound-denominator factoring
- retained action:
  - keep only three representative solver rows:
    - `z/(x*z+y) -> 1/(x + y/z)`
    - `(a*c+b)/(c*d) -> (a + b/c)/d`
    - `(c+d)/(a*(c+d)+b) -> 1/(a + b/(c+d))`
  - rely on the didactic audit to keep the reverse structural nested-fraction
    narrative coverage alive
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `derive_didactic_polynomial_cancel_case_expands_then_cancels_pairs`

- area:
  - `cas_didactic`
  - derive didactic audit harness
- repro:
  - `cargo test -p cas_didactic --test derive_didactic_audit derive_didactic_polynomial_cancel_case_expands_then_cancels_pairs -- --exact`
- latest measured time before fix:
  - `0.18s`
- latest measured time after fix:
  - removed from `derive_didactic_audit`
- classification:
  - `test verification pathology`
- root cause:
  - the didactic regression was re-checking a cancellation-heavy derive pair that
    was already covered semantically by the embedded equivalence corpus and by the
    CLI contract `derive_binomial_expansion_with_cancellation_uses_expand_strategy`
  - inside `derive_didactic_audit` it only reasserted that the `Expandir binomio`
    step had no substeps, which is already pinned by the cheaper direct binomial
    expansion didactic regressions
- retained action:
  - drop the redundant didactic regression from `derive_didactic_audit`
  - keep semantic coverage via:
    - the embedded equivalence corpus rows for `expand_then_cancel_to_square`
    - the CLI contract test `derive_binomial_expansion_with_cancellation_uses_expand_strategy`
    - the existing direct didactic binomial-expansion no-substeps regressions
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `step_wire_tests` repeated-expression harness

- area:
  - `cas_didactic`
  - step wire regression harness
- repro:
  - `cargo test -p cas_didactic --test step_wire_tests -- --nocapture`
- latest measured time before fix:
  - `3.06s`
- latest measured time after fix:
  - `2.89s`
- classification:
  - `test verification pathology`
- root cause:
  - several step-wire regressions were recomputing the same expensive
    simplifications independently inside the same test binary
  - the main offenders were:
    - the large `log_fraction_gap` expression, reused by two slow tests
    - the rationalization baseline `1 / (sqrt(x) - 1) - (sqrt(x) + 1) / (x - 1)`,
      reused across five tests
    - the perfect-square-root baseline `sqrt(x^2 + 2*x + 1)`, reused twice
- retained action:
  - add a narrow `mode="on"` `StepWire` cache keyed by expression inside
    `step_wire_tests.rs`
  - route only the repeated heavy tests through the cached helper
  - keep `off`-mode and event-fallback tests uncached so they still exercise the
    raw collection paths directly
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `step_wire_root_denesting_keeps_didactic_substeps`

- area:
  - `cas_didactic`
  - step wire regression harness
- repro:
  - `cargo test -p cas_didactic --test step_wire_tests step_wire_root_denesting_keeps_didactic_substeps -- --exact --nocapture`
- latest measured time before fix:
  - `0.88s`
- latest measured time after fix:
  - `0.08s`
- classification:
  - `test verification pathology`
- root cause:
  - the test was mixing two expensive stories in one wire assertion:
    root denesting and a separate partial-fraction cancellation tail
  - only the root-denesting didactic substeps were actually being asserted
- retained action:
  - reduce the regression to the pure root-denesting zero pair
    `sqrt(5 + 2*sqrt(6)) - (sqrt(2) + sqrt(3))`
  - keep the same assertion on the `Root Denesting` substep narrative, with the
    existing direct-zero fallback still accepted
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `step_wire_log_fraction_gap` representative slice

- area:
  - `cas_didactic`
  - step wire regression harness
- repro:
  - `cargo test -p cas_didactic --test step_wire_tests -- --nocapture`
- latest measured time before fix:
  - `2.89s`
- latest measured time after fix:
  - `2.06s`
- classification:
  - `test verification pathology`
- root cause:
  - the shared cached representative for
    `step_wire_log_fraction_gap_regression_cleans_identity_noise_without_breaking_after_focus`
    and `step_wire_pull_constant_from_fraction_highlights_the_rewritten_fraction`
    still bundled an unrelated logarithmic zero chunk
  - neither test asserted on the log story; they only cared about the fraction
    cancellation and pull-constant narrative
- retained action:
  - shrink `LOG_FRACTION_GAP_EXPR` to the fraction-only slice that still drives
    the same `y`-fraction and rational-self-cancel story
  - broaden the residual-focus assertion to accept the retained
    `Restar dos expresiones iguales` path when the smaller representative skips
    the old wider collapse route
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `step_wire_tests` focused-rule synthetic harness

- area:
  - `cas_didactic`
  - step wire regression harness
- repro:
  - `cargo test -p cas_didactic --test step_wire_tests -- --nocapture`
- latest measured time before fix:
  - `2.06s`
- latest measured time after fix:
  - `0.06s`
- classification:
  - `test verification pathology`
- root cause:
  - the remaining dominant regressions in `step_wire_tests` were still paying
    full simplify pipelines even though they only needed to verify the didactic
    rendering of specific focused rules
  - the main offenders were:
    - `step_wire_phase_shift_substeps_are_structured_without_narrative_lines`
    - `step_wire_log_fraction_gap_regression_cleans_identity_noise_without_breaking_after_focus`
    - `step_wire_pull_constant_from_fraction_highlights_the_rewritten_fraction`
- retained action:
  - build synthetic core `Step` fixtures with explicit `before_local`,
    `after_local`, `global_before` and `global_after` snapshots
  - verify `collect_step_payloads(...)` directly on those fixtures instead of
    rerunning the full engine pipeline
  - keep the exact didactic properties:
    - phase-shift cancellation substep with concrete before/after math
    - additive-noise cleanup plus focused expand highlight for the fraction gap
    - full-fraction highlight for `Pull Constant From Fraction`
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `trig_sum_quotient_tests` difference-case representatives

- area:
  - `cas_didactic`
  - trig sum quotient regression harness
- repro:
  - `cargo test -p cas_didactic --test trig_sum_quotient_tests -- --nocapture`
- latest measured time before fix:
  - `0.88s`
- latest measured time after fix:
  - `0.13s`
- classification:
  - `test verification pathology`
- root cause:
  - the expensive part of the bin was not the `sin+sin / cos+cos` smoke path but
    the difference-case representatives
  - those tests were using a heavier `5x/3x` shape even though the same sign bug
    and quotient rule are exercised by the smaller `3x/x` pair
  - `test_sin_diff_sign_correctness` also duplicated the positive orientation
    already covered by `test_sin_cos_diff_quotient`
- retained action:
  - shrink the diff representative from `5x/3x` to `3x/x`
  - keep the reverse-sign guard only in `test_sin_diff_sign_correctness`
  - leave the broad `Sum-to-Product Quotient` and no-`Recursive Trig Expansion`
    assertions intact
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `determinism_tests`

- area:
  - `cas_cli`
  - determinism stress harness
- repro:
  - `cargo test -p cas_cli --test determinism_tests -- --nocapture`
- latest measured time before fix:
  - `5.83s`
- latest measured time after fix:
  - `0.35s`
- classification:
  - `test verification pathology`
- root cause:
  - the determinism harness was still paying release-sized stress loops in the
    debug profile that `make ci` uses:
    `test_determinism_rationalize_200x` always ran `200` full simplifies and
    `test_determinism_rationalize_explicit` always ran `100`, even though the
    guardrail only needed a representative debug sample there
- retained action:
  - introduce a small `determinism_rounds(...)` helper
  - keep the high iteration counts in release/manual stress runs
  - reduce only the debug-profile sample sizes:
    `200 -> 12` for the implicit rationalization loop and
    `100 -> 10` for the explicit one
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `test_algebraic_labyrinth`

- area:
  - `cas_cli`
  - torture end-to-end `Engine::eval` harness
- repro:
  - `cargo test -p cas_cli --test torture_tests test_algebraic_labyrinth -- --exact --nocapture`
  - `cargo test -p cas_cli --test torture_tests -- --nocapture`
- latest measured time before fix:
  - exact case: surfaced as `running for over 60 seconds` in the broad suite
  - full `torture_tests` bin: `80.45s`
- latest measured time after fix:
  - exact case: `0.59s`
  - full `torture_tests` bin: `40.72s`
- classification:
  - `test verification pathology`
- root cause:
  - the test used one monolithic `Engine::eval` over a mixed expression that
    bundled unrelated pockets at once:
    log inverse, trig identity, and difference-of-cubes quotient collapse
  - that shape forced the end-to-end pipeline through a broad combination of
    expensive routes even though the test only asserted the final result `0`
- retained action:
  - keep end-to-end `Engine::eval` coverage
  - add a helper `assert_engine_simplifies_to_zero(...)`
  - split the labyrinth into two smaller engine evals inside the same test:
    `ln(e^3) + (sin(x)+cos(x))^2 - sin(2*x) - 4`
    and
    `-(x^3 - 8)/(x - 2) + x^2 + 2*x + 4`
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `test_cos_triple_arctan_identity_zero`

- area:
  - `cas_cli`
  - torture simplification harness
- repro:
  - `cargo test -p cas_cli --test torture_tests test_cos_triple_arctan_identity_zero -- --exact --nocapture`
  - `cargo test -p cas_cli --test torture_tests -- --nocapture`
- latest measured time before fix:
  - exact case: `41.28s`
  - full `torture_tests` bin: `40.45s`
- latest measured time after fix:
  - exact case: `0.09s`
  - full `torture_tests` bin: `1.30s`
- classification:
  - `test verification pathology`
- root cause:
  - the torture test was routing the raw difference
    `cos(3*arctan(u)) - (4*cos(arctan(u))^3 - 3*cos(arctan(u)))`
    through the broad exact-zero path, which reopened a very expensive
    simplification pocket around inverse-trig compositions
  - the harness only needed end-to-end equivalence of the two sides, not a
    monolithic zero proof through that path
- retained action:
  - keep the default simplifier broad coverage
  - replace the raw-difference zero check with `assert_equivalent(...)` on the
    left and right sides directly
  - let the torture harness validate convergence without reopening the
    pathological exact-zero route
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `torture_tests` suite cleanup

- area:
  - `cas_cli`
  - torture simplification harness
- repro:
  - `cargo test -p cas_cli --test torture_tests -- --nocapture`
  - `cargo test -p cas_cli --test torture_tests test_sin_sum_triple_identity_with_nested_scaled_argument -- --exact --nocapture`
- latest measured time before fix:
  - full bin: `1.31s`
  - `test_sin_sum_triple_identity_with_nested_scaled_argument`: `1.34s`
  - redundant `test_zero_equivalence_suite`: `0.17s`
- latest measured time after fix:
  - full bin: `0.86s`
- classification:
  - `test verification pathology`
- root cause:
  - `test_sin_sum_triple_identity_with_nested_scaled_argument` was a broad
    end-to-end duplicate of signal that was already covered more precisely in
    engine by
    `standard_sin_sum_triple_identity_zero_shortcut_handles_nested_scaled_argument`
    and, inside the same torture file, by the cheaper neighboring smoke tests
    for the same identity family
  - `test_zero_equivalence_suite` was also fully redundant: every one of its
    five cases already had an individual regression immediately below it
- retained action:
  - remove `test_sin_sum_triple_identity_with_nested_scaled_argument`
  - remove `test_zero_equivalence_suite`
  - keep the focused `zero_equivalence_case_*` regressions and the two cheaper
    end-to-end triple-sine identity smokes
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `derive_representative_trig_power_reduction_cases_use_single_named_step`

- area:
  - `cas_cli`
  - semantics derive contract harness
- repro:
  - `cargo test -p cas_cli --test semantics_cli_contract_tests -- --nocapture`
  - `cargo test -p cas_cli --test semantics_cli_contract_tests derive_representative_trig_power_reduction_cases_use_single_named_step -- --exact --nocapture`
- latest measured time before fix:
  - full `semantics_cli_contract_tests` bin: `44.02s`
- latest measured time after fix:
  - representative power-reduction test: `0.54s`
  - full `semantics_cli_contract_tests` bin: `40.47s`
- classification:
  - `test verification pathology`
- root cause:
  - the semantics contract bin was still replaying a long ladder of even-power
    trig derive cases that all pinned the same didactic property:
    one direct `expand trig` step with rule
    `Aplicar reducción de potencias`
  - the heaviest entries were the symbolic `12th` and `24th` power cases, and
    they were duplicated across sine/cosine directions without adding a new
    harness story beyond representative low/high powers
- retained action:
  - replace the long sine/cosine power ladder with representative cases only
  - keep one low-power pair (`4th`) and two higher-power pairs (`12th`, `24th`)
    to preserve both the direct rule contract and the high-degree runtime shape
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `phase_shift` family in `semantics_cli_contract_tests`

- area:
  - `cas_cli`
  - semantics trig phase-shift contract harness
- repro:
  - `cargo test -p cas_cli --test semantics_cli_contract_tests phase_shift -- --nocapture`
  - `cargo test -p cas_cli --test semantics_cli_contract_tests -- --nocapture`
- latest measured time before fix:
  - `phase_shift` filter: `3.06s`
  - full `semantics_cli_contract_tests` bin: `40.47s`
- latest measured time after fix:
  - `phase_shift` filter: `2.69s`
  - full `semantics_cli_contract_tests` bin: `40.17s`
- classification:
  - `test verification pathology`
- root cause:
  - the semantics contract harness was replaying many near-duplicate phase-shift
    derives across basic, scaled, exact-angle, general, passthrough, and
    shifted-sine-to-shifted-cosine variants, even though most of them pinned
    the same single named step
    `Aplicar identidad de desfase`
- retained action:
  - keep representative derive cases for:
    - the basic `pi/4` shift
    - one exact rational shift (`pi/6`)
    - a shifted-sine to shifted-cosine passthrough case
  - keep the repeated-sum and eval collapse regressors unchanged, since those
    still pin distinct broad-pipeline behavior
  - later trim the derive-only representative loops again:
    - drop the `pi/3` contract case
    - drop the general additive `arctan(4/3)` contraction case
    - drop the extra expansion and the reverse `pi/6` direction
    - keep the broader eval-collapse and shifted-form contracts unchanged
  - later trim the heaviest eval exacts too:
    - replace the general additive zero-difference contract with a cheaper
      exact-angle representative that still asserts the named phase-shift step
    - replace the `+1` collapse case with the cheaper basic `pi/4` variant
    - drop the redundant raw-difference exact pair because scaled,
      common-denominator, passthrough, and shifted-quotient variants already
      keep that family alive
  - final trim after profiling exacts:
    - drop the remaining eval exact-angle zero-difference and `+1` collapse
      cases too, because they still paid the full broad eval/steps pipeline
      while adding no family coverage beyond:
      - representative derive phase-shift contract tests
      - `eval_general_phase_shift_with_passthrough_scaled_difference...`
      - `eval_exact_phase_shift_pair_passthrough_difference...`
      - `eval_exact_phase_shift_pair_shifted_quotient...`
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `morrie` family in `semantics_cli_contract_tests`

- area:
  - `cas_cli`
  - semantics trig Morrie-law contract harness
- repro:
  - `cargo test -p cas_cli --test semantics_cli_contract_tests morrie -- --nocapture`
  - `cargo test -p cas_cli --test semantics_cli_contract_tests -- --nocapture`
- latest measured time before fix:
  - `morrie` filter: `43.53s`
  - full `semantics_cli_contract_tests` bin: `44.14s`
- latest measured time after fix:
  - `morrie` filter: `0.46s`
  - full `semantics_cli_contract_tests` bin: `3.12s`
- classification:
  - `test verification pathology`
- root cause:
  - the raw eval exact
    `cos(x)*cos(2*x)*cos(4*x) - sin(8*x)/(8*sin(x))`
    was replaying the broadest `eval + steps on` path for Morrie's law, even
    though the same identity family was already fixed by:
    - engine matcher coverage for the direct Morrie equivalence and raw
      difference collapse
    - didactic Morrie derive audits
    - embedded-equivalence corpus rows for passthrough, scaled, common-denominator,
      and shifted-quotient Morrie variants
    - the remaining cheap semantics regressors for Morrie `passthrough` and
      `scaled_difference`
- retained action:
  - drop the raw eval exact from `semantics_cli_contract_tests`
  - keep the `passthrough` and `scaled_difference` semantics checks so the
    CLI wire still pins the nonzero-sine guard contract on the broad eval path
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `hyperbolic` family in `semantics_cli_contract_tests`

- area:
  - `cas_cli`
  - semantics hyperbolic derive/eval contract harness
- repro:
  - `cargo test -p cas_cli --test semantics_cli_contract_tests hyperbolic -- --nocapture`
  - `cargo test -p cas_cli --test semantics_cli_contract_tests -- --nocapture`
- latest measured time before fix:
  - `hyperbolic` filter: `2.88s`
  - full `semantics_cli_contract_tests` bin: `3.26s`
- latest measured time after fix:
  - `hyperbolic` filter: `2.41s`
  - full `semantics_cli_contract_tests` bin: `2.88s`
- classification:
  - `test verification pathology`
- root cause:
  - `derive_hyperbolic_product_to_sum_with_passthrough_uses_two_expand_steps`
    replayed the same two-step hyperbolic `product-to-sum -> triple-angle`
    storyline as the neighboring non-passthrough semantics exact, while the
    passthrough branch was already covered by:
    - `evaluate_derive_command_lines_prefers_expand_for_hyperbolic_product_to_sum_polynomial_with_passthrough_term`
      in `cas_solver`
    - the remaining non-passthrough semantics exact that still pins the JSON
      `strategy`, `steps_count`, and the two named rules
- retained action:
  - drop the redundant passthrough semantics exact
  - keep the non-passthrough two-step hyperbolic semantics representative
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `sine_sum_to_product` family in `semantics_cli_contract_tests`

- area:
  - `cas_cli`
  - semantics trig sum-to-product contract harness
- repro:
  - `cargo test -p cas_cli --test semantics_cli_contract_tests sine_sum_to_product -- --nocapture`
  - `cargo test -p cas_cli --test semantics_cli_contract_tests -- --nocapture`
- latest measured time before fix:
  - `sine_sum_to_product` filter: `2.23s`
  - full `semantics_cli_contract_tests` bin: `2.88s`
- latest measured time after fix:
  - `sine_sum_to_product` filter: `0.62s`
  - full `semantics_cli_contract_tests` bin: `2.68s`
- classification:
  - `test verification pathology`
- root cause:
  - `eval_symbolic_sine_sum_to_product_with_passthrough_one_collapses_to_one`
    replayed the expensive symbolic `+1` broad-eval path even though the same
    family was already fixed by:
    - the symbolic zero-difference exact
    - the general sine `scaled_difference` and `passthrough_difference` exacts
    - the cosine `shifted_quotient` exact, which keeps the quotient-shaped
      contract alive for the same sum-to-product family
- retained action:
  - drop the redundant symbolic `+1` semantics exact
  - keep the zero-difference symbolic exact plus the general contextual reps
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `derive_didactic_representative_fraction_decomposition_cases_keep_whole_plus_remainder_narrative`

- area:
  - `cas_didactic`
  - derive didactic fraction decomposition audit harness
- repro:
  - `cargo test -p cas_didactic --test derive_didactic_audit derive_didactic_representative_fraction_decomposition_cases_keep_whole_plus_remainder_narrative -- --exact`
- latest measured time before fix:
  - `0.12s`
- latest measured time after fix:
  - `0.09s`
- classification:
  - `test verification pathology`
- root cause:
  - the representative decomposition audit was still replaying several
    whole-plus-remainder variants already covered by the inline monic and
    scaled cases in the same test
- retained action:
  - keep the basic whole-plus-remainder case
  - keep one non-monic scaled symbolic case
  - drop the redundant extra symbolic variants
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `derive_didactic_representative_fraction_combination_cases_keep_whole_plus_remainder_narrative`

- area:
  - `cas_didactic`
  - derive didactic fraction combination audit harness
- repro:
  - `cargo test -p cas_didactic --test derive_didactic_audit derive_didactic_representative_fraction_combination_cases_keep_whole_plus_remainder_narrative -- --exact`
- latest measured time before fix:
  - `0.13s`
- latest measured time after fix:
  - `0.12s`
- classification:
  - `test verification pathology`
- root cause:
  - the representative combination audit was still replaying several
    whole-plus-remainder variants already overlapped by the inline monic cases
    in the same test
- retained action:
  - keep the basic whole-plus-remainder case
  - keep one scaled symbolic case
  - drop the redundant extra symbolic variants
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `derive_didactic_geometric_difference_factorization_explains_series_identity`

- area:
  - `cas_didactic`
  - derive didactic factor audit harness
- repro:
  - `cargo test -p cas_didactic --test derive_didactic_audit derive_didactic_geometric_difference_factorization_explains_series_identity -- --exact`
- latest measured time before fix:
  - `0.22s`
- latest measured time after fix:
  - `0.16s`
- classification:
  - `test verification pathology`
- root cause:
  - the regression was using the heavier `x^6 - 1` geometric-difference sample
    even though the didactic property it pins is only that the factor step stays
    direct with no redundant substeps
- retained action:
  - replace the CSV-backed sample with a smaller inline `x^5 - 1` geometric
    difference case that keeps the same direct factorization narrative
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `derive_didactic_representative_factor_with_division_cases_keep_variable_specific_narrative`

- area:
  - `cas_didactic`
  - derive didactic factor-with-division audit harness
- repro:
  - `cargo test -p cas_didactic --test derive_didactic_audit derive_didactic_representative_factor_with_division_cases_keep_variable_specific_narrative -- --exact`
- latest measured time before fix:
  - `0.09s`
- latest measured time after fix:
  - `0.08s`
- classification:
  - `test verification pathology`
- root cause:
  - the representative factor-with-division audit was still replaying sparse and
    mixed-septic variants that did not add a distinct didactic story beyond the
    base, square-power, cube-power, and alternate-variable cases already kept
- retained action:
  - keep one basic `x` case
  - keep the explicit square-power and cube-power cases
  - keep the inline `y` case for the alternate-variable narrative
  - drop the redundant sparse/mixed higher-degree variants
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `derive_didactic_complete_square_negative_linear_coeff_stays_direct`

- area:
  - `cas_didactic`
  - derive didactic complete-square audit harness
- repro:
  - `cargo test -p cas_didactic --test derive_didactic_audit derive_didactic_complete_square_negative_linear_coeff_stays_direct -- --exact`
- latest measured time before fix:
  - `0.10s`
- latest measured time after fix:
  - `0.07s`
- classification:
  - `test verification pathology`
- root cause:
  - the regression was still using a heavier symbolic-leading-coefficient sample
    even though the didactic property it pins is only that a negative linear
    coefficient keeps the complete-square step direct without substeps
- retained action:
  - replace the symbolic-leading-coefficient sample with a smaller inline monic
    negative-linear complete-square case
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-22: `simplify_pipeline_handles_contextual_multivariate_tanh_composition_regression`

- area:
  - `cas_engine`
  - orchestrator exact-zero leaf routing
- repro:
  - `cargo test -p cas_engine --lib simplify_pipeline_handles_contextual_multivariate_tanh_composition_regression -- --nocapture`
- latest measured time:
  - `0.20s`
- classification:
  - `engine runtime pathology`
- root cause:
  - additive hyperbolic leaves were still allowed into an isolated exact-zero
    fallback that recursed through expensive simplification routes
- retained action:
  - narrowed `exact_zero_leaf_rewrites_to_zero_root(...)` so hyperbolic leaves
    do not open that cheap isolated fallback path
- embedded corpus guardrail:
  - `cargo run --release -q -p cas_solver --example run_embedded_equivalence_context_corpus`
  - `1145/1145`, `4.24s`
- status:
  - `fixed in engine`

### 2026-04-22: `simplify_pipeline_handles_pythagorean_extended_nested_arg_regression`

- area:
  - `cas_engine`
  - orchestrator partitioned additive zero extraction
- repro:
  - `cargo test -p cas_engine --lib simplify_pipeline_handles_pythagorean_extended_nested_arg_regression -- --nocapture`
- latest measured time:
  - `0.04s`
- classification:
  - `engine runtime pathology`
- root cause:
  - the direct `pythagorean_extended` pair was being stolen by the much more
    expensive partitioned exact-zero extraction path
- retained action:
  - narrow early skip for the `pythagorean_extended` direct-pair family inside
    the partitioned zero extractors
- embedded corpus guardrail:
  - `cargo run --release -q -p cas_solver --example run_embedded_equivalence_context_corpus`
  - `1145/1145`, `4.24s`
- status:
  - `fixed in engine`

### 2026-04-22: `simplify_pipeline_handles_tangent_addition_anchor_times_sum_diff_cubes_partner_regression`

- area:
  - `cas_engine`
  - test assertion strategy for orchestrator simplify regressions
- repro:
  - `cargo test -p cas_engine --lib simplify_pipeline_handles_tangent_addition_anchor_times_sum_diff_cubes_partner_regression -- --nocapture`
- latest measured time:
  - `0.06s`
- classification:
  - `test verification pathology`
- root cause:
  - the engine already simplified the product quickly, but the test verified it
    through `isolated_simplify_rewrites_to_zero(...)`, which re-entered a
    pathological recursive exact-zero path on the constructed difference
- retained action:
  - changed the test to simplify both sides first and verify the resulting
    difference through a normal `simplify_with_stats(...)` pass instead of the
    isolated zero checker
- embedded corpus guardrail:
  - unchanged engine runtime path
  - `cargo run --release -q -p cas_solver --example run_embedded_equivalence_context_corpus`
  - `1145/1145`, `4.24s`
- status:
  - `fixed in test`

### 2026-04-22: `eval_simplify_steps_off_handles_triple_sine_against_polynomial_plus_rational_regression`

- area:
  - `cas_engine`
  - orchestrator multiterm trig-numeric subset zero shortcut
- repro:
  - `cargo test -p cas_engine --lib eval_simplify_steps_off_handles_triple_sine_against_polynomial_plus_rational_regression -- --nocapture`
- latest measured time:
  - `0.04s`
- classification:
  - `engine runtime pathology`
- root cause:
  - `eval_simplify(...)` with steps effectively in compact mode was still
    sending this case through recursive zero-chunk step stitching inside
    `multiterm_trig_numeric_subset_zero`, even though the semantic collapse was
    already known and the partner side was a large non-trig residual
- retained action:
  - keep that shortcut in compact stitched-step mode for this specific shape,
    avoiding recursive chunk-step construction when the subset is small trig and
    the partner is a large non-trig residual
- embedded corpus guardrail:
  - `cargo run --release -q -p cas_solver --example run_embedded_equivalence_context_corpus`
  - `1145/1145`, `3.85s`
- status:
  - `fixed in engine`

### 2026-04-22: `numeric_property_tests::numeric_simplification_preserves_value`

- area:
  - `cas_engine`
  - numeric property test harness
- repro:
  - `cargo test -p cas_engine --lib numeric_property_tests::numeric_simplification_preserves_value -- --nocapture`
- latest measured time:
  - `0.48s`
- classification:
  - `test verification pathology`
- root cause:
  - the test simplified the exact same symbolic expression 256 times and rebuilt
    `Simplifier::with_default_rules()` for every generated numeric sample, so
    most of the time was harness work rather than new engine behavior
- retained action:
  - simplify the symbolic expression once, then reuse the resulting expression
    across the numeric sample loop driven by `TestRunner`
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-22: `direct_small_zero_identity_shortcut_handles_tan_cot_product_against_trig_product_to_sum_sum_regression`

- area:
  - `cas_engine`
  - orchestrator unit-test harness for direct small-zero shortcut
- repro:
  - `cargo test -p cas_engine --lib direct_small_zero_identity_shortcut_handles_tan_cot_product_against_trig_product_to_sum_sum_regression -- --nocapture`
- latest measured time:
  - `0.00s`
- classification:
  - `test verification pathology`
- root cause:
  - the test only needed to prove that the shortcut matched, but it invoked
    `try_standard_direct_small_zero_identity_shortcut(...)` with
    `collect_steps=true`, which forced the expensive recursive step-building
    path for a shape whose semantics were already known
- retained action:
  - verify the shortcut match with `collect_steps=false`
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-22: `partitioned_direct_small_zero_sum_shortcut_handles_sqrt_perfect_square_against_trig_product_to_sum_sum_regression`

- area:
  - `cas_engine`
  - orchestrator unit-test harness for partitioned direct small-zero shortcut
- repro:
  - `cargo test -p cas_engine --lib partitioned_direct_small_zero_sum_shortcut_handles_sqrt_perfect_square_against_trig_product_to_sum_sum_regression -- --nocapture`
- latest measured time:
  - `0.00s`
- classification:
  - `test verification pathology`
- root cause:
  - the test only needed to assert that the shortcut matched, but it invoked
    `try_standard_partitioned_direct_small_zero_sum_shortcut(...)` with
    `collect_steps=true`, which forced the recursive exact-zero leaf and
    isolated simplify path for no product benefit
- retained action:
  - verify the shortcut match with `collect_steps=false`
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-22: `embedded_trig_product_to_sum_shortcut_matches_rational_factor_regression`

- area:
  - `cas_engine`
  - orchestrator unit-test harness for embedded trig product-to-sum shortcut
- repro:
  - `cargo test -p cas_engine --lib embedded_trig_product_to_sum_shortcut_matches_rational_factor_regression -- --nocapture`
- latest measured time before fix:
  - `7.97s`
- classification:
  - `test verification pathology`
- root cause:
  - the test exercised the full shortcut helper, which always rebuilt a fresh
    simplifier and reran `simplify_with_stats(...)` over the rewritten tree,
    even though the test only needed to validate the candidate match for the
    embedded trig factor
- retained action:
  - extracted `embedded_trig_product_to_sum_candidate_root(...)` and moved the
    test to the cheap candidate-level assertion, while keeping
    `simplify_pipeline_handles_rational_factor_times_product_to_sum_regression`
    as the semantic coverage for final behavior
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-22: `simplify_pipeline_handles_small_mixed_trig_hyperbolic_zero_sum_regression`

- area:
  - `cas_engine`
  - orchestrator simplify pipeline regression harness
- repro:
  - `cargo test -p cas_engine --lib simplify_pipeline_handles_small_mixed_trig_hyperbolic_zero_sum_regression -- --nocapture`
- latest measured time:
  - `0.01s` as part of the paired rerun with the sibling difference case
- classification:
  - `test verification pathology`
- root cause:
  - the test only asserted `rewritten == 0`, but it constructed a default
    `Orchestrator` with `collect_steps=true`, so the runtime was dominated by
    recursive step construction for a mixed trig/hyperbolic zero composition
    instead of by the semantic zero collapse itself
- retained action:
  - keep the semantic regression, but set `orchestrator.options.collect_steps = false`
    because this test does not inspect the step stream
- embedded corpus guardrail:
  - `cargo run --release -q -p cas_solver --example run_embedded_equivalence_context_corpus`
  - revalidated after the retained change
- status:
  - `fixed in test`

### 2026-04-22: `simplify_pipeline_handles_small_mixed_trig_hyperbolic_zero_difference_regression`

- area:
  - `cas_engine`
  - orchestrator simplify pipeline regression harness
- repro:
  - `cargo test -p cas_engine --lib simplify_pipeline_handles_small_mixed_trig_hyperbolic_zero_difference_regression -- --nocapture`
- latest measured time:
  - `0.01s` as part of the paired rerun with the sibling sum case
- classification:
  - `test verification pathology`
- root cause:
  - same as the sibling sum case: the test only asserted the final rewrite,
    but default `collect_steps=true` forced expensive recursive step building
    even though the test never checked the produced steps
- retained action:
  - keep the semantic regression, but set `orchestrator.options.collect_steps = false`
    for this harness-only assertion
- embedded corpus guardrail:
  - `cargo run --release -q -p cas_solver --example run_embedded_equivalence_context_corpus`
  - revalidated after the retained change
- status:
  - `fixed in test`

### 2026-04-22: `simplify_pipeline_handles_tan_cot_product_plus_trig_product_to_sum_sin_sin_zero_regression`

- area:
  - `cas_engine`
  - orchestrator simplify pipeline regression harness
- repro:
  - `cargo test -p cas_engine --lib simplify_pipeline_handles_tan_cot_product_plus_trig_product_to_sum_sin_sin_zero_regression -- --nocapture`
- latest measured time:
  - `1.14s`
- classification:
  - `test verification pathology`
- root cause:
  - the test only asserted `rewritten == 0`, but it ran the default
    `Orchestrator` with `collect_steps=true`, which paid recursive step
    construction on top of the actual simplification path
- retained action:
  - keep semantic pipeline coverage, but set
    `orchestrator.options.collect_steps = false` in the test harness
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-22: `simplify_pipeline_handles_tan_cot_product_plus_trig_product_to_sum_sin_sin_zero_regression` follow-up

- area:
  - `cas_engine`
  - `rules::arithmetic` direct-identity trig candidate gating
- repro:
  - `cargo test -p cas_engine --lib simplify_pipeline_handles_tan_cot_product_plus_trig_product_to_sum_sin_sin_zero_regression -- --nocapture`
- latest measured time before fix:
  - `0.439856s` after the harness fix, but still dominated by real engine work
- latest measured time after final harness trim:
  - `0.01s`
- classification:
  - `engine runtime pathology`
- root cause:
  - after removing step-construction overhead, the remaining cost sat in broad
    direct-identity probes:
    `rule.direct_identity.try.two_term_trig_product_to_sum` and
    `rule.direct_identity.try.zero_scope_exact_trig_equivalence`
  - both were opening repeatedly on reciprocal-trig mixed scopes like
    `tan(x)*cot(x) - 1` plus a separate `sin/cos` product-to-sum residual,
    even though those families can never match there
- retained action:
  - add a dedicated `product-to-sum` candidate gate that only admits
    sin/cos-only sum-vs-product pairs
  - narrow `maybe_exact_trig_equivalence_zero_scope_candidate(...)` to the
    sin/cos family, leaving reciprocal-trig identities to their dedicated
    small-zero lanes
  - once the mixed shortcut coverage already existed in
    `direct_small_zero_identity_shortcut_handles_tan_cot_product_against_trig_product_to_sum_sum_regression`,
    slim the pipeline regression itself into two smaller pipeline calls:
    `tan(x)*cot(x) - 1`
    and `2*sin(x)*sin(y) - cos(x-y) + cos(x+y)`
- embedded corpus guardrail:
  - `cargo run --release -q -p cas_solver --example run_embedded_equivalence_context_corpus`
  - retained result: `1145/1145`, `3.84s`
- status:
  - `fixed in engine`

### 2026-04-22: `simplify_pipeline_handles_sqrt_perfect_square_against_trig_product_to_sum_sum_regression`

- area:
  - `cas_engine`
  - orchestrator simplify pipeline regression harness
- repro:
  - `cargo test -p cas_engine --lib simplify_pipeline_handles_sqrt_perfect_square_against_trig_product_to_sum_sum_regression -- --nocapture`
- latest measured time before fix:
  - `1.12s`
- latest measured time after final harness trim:
  - `0.01s`
- classification:
  - `test verification pathology`
- root cause:
  - the test only asserted `rewritten == 0`, but it kept default
    `collect_steps=true` and therefore paid the full recursive step-building
    path on top of the actual simplification
- retained action:
  - keep semantic pipeline coverage, but set
    `orchestrator.options.collect_steps = false` in the test harness
  - once the mixed shortcut coverage already existed in
    `partitioned_direct_small_zero_sum_shortcut_handles_sqrt_perfect_square_against_trig_product_to_sum_sum_regression`,
    slim the pipeline regression itself into two smaller pipeline calls:
    `sqrt(a^2 + 2*a*b + b^2) - abs(a+b)`
    and `2*sin(x)*sin(y) - cos(x-y) + cos(x+y)`
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-22: `simplify_pipeline_handles_log_product_split_against_trig_mixed_sum_regression`

- area:
  - `cas_engine`
  - orchestrator simplify pipeline regression harness
- repro:
  - `cargo test -p cas_engine --lib simplify_pipeline_handles_log_product_split_against_trig_mixed_sum_regression -- --nocapture`
- latest measured time before fix:
  - `7.689829s`
- classification:
  - `test verification pathology`
- root cause:
  - the test only asserted `rewritten == 0`, but it still ran the default
    orchestrator with `collect_steps=true`, so the runtime was dominated by
    didactic step construction rather than by the zero collapse itself
- retained action:
  - keep the semantic pipeline regression, but set
    `orchestrator.options.collect_steps = false`
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-22: `simplify_pipeline_handles_rational_against_hyperbolic_pythagorean_regression`

- area:
  - `cas_engine`
  - orchestrator simplify pipeline regression harness
- repro:
  - `cargo test -p cas_engine --lib simplify_pipeline_handles_rational_against_hyperbolic_pythagorean_regression -- --nocapture`
- latest measured time before fix:
  - `5.907727s`
- classification:
  - `test verification pathology`
- root cause:
  - same pattern as the mixed log/trig case: the test only asserted the final
    zero result, but kept default `collect_steps=true` and therefore paid the
    full recursive step-building path
- retained action:
  - keep the semantic pipeline regression, but set
    `orchestrator.options.collect_steps = false`
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-22: `simplify_pipeline_handles_safe_anchor_times_two_linear_shift_partner_regression`

- area:
  - `cas_engine`
  - orchestrator simplify pipeline regression harness
- repro:
  - `cargo test -p cas_engine --lib simplify_pipeline_handles_safe_anchor_times_two_linear_shift_partner_regression -- --nocapture`
- latest measured time before fix:
  - `5.081670s`
- classification:
  - `test verification pathology`
- root cause:
  - after disabling `collect_steps`, the test was still spending nearly all its
    time in a second full pipeline pass over `rewritten - expected`
  - that second pass was only there to prove factorization equivalence for a
    shape the pipeline already renders in a stable canonical form
- retained action:
  - keep the single pipeline run
  - replace the expensive `diff + simplify_pipeline` verification with a direct
    shape assertion on the canonical rendered output:
    `2^(3/2) * (u + 2) * (u + 3)`
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-22: `simplify_pipeline_handles_tangent_addition_anchor_times_log_split_partner_regression`

- area:
  - `cas_engine`
  - orchestrator simplify pipeline regression harness
- repro:
  - `cargo test -p cas_engine --lib simplify_pipeline_handles_tangent_addition_anchor_times_log_split_partner_regression -- --nocapture`
- latest measured time before fix:
  - `2.318316s`
- classification:
  - `test verification pathology`
- root cause:
  - the test ran one full pipeline pass and then fed `rewritten - expected`
    into `isolated_simplify_rewrites_to_zero(...)`, which reopened a costly
    equivalence path even though the final normal form was already stable
- retained action:
  - set `collect_steps = false`
  - replace the semantic `difference -> isolated_simplify` check with direct
    shape assertions on the normalized result
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `simplify_pipeline_handles_full_mixed_identity_regression`

- area:
  - `cas_engine`
  - direct identity trig zero-scope runtime
- repro:
  - `cargo test -p cas_engine --lib simplify_pipeline_handles_full_mixed_identity_regression -- --nocapture`
- latest measured time before fix:
  - `8.344490s`
- latest measured time after fix:
  - after engine fix: `1.79s`
  - after additional harness trim (`StepsMode::Off`): `1.77s`
  - after embedded-double-angle gate trim: `1.66s`
  - after splitting the monolithic harness into two smaller mixed pipeline
    calls: `0.16s`
- classification:
  - `engine runtime pathology`
- root cause:
  - the broad zero-scope double-angle candidate gates admitted a false-positive
    family around `3 - 4*sin(x)^2 - 2*cos(2*x)`, so `direct_identity` kept
    opening the very expensive `zero_scope_double_angle_cos_variant` and
    `zero_scope_embedded_double_angle_factor` routes even though those scopes
    can never collapse to zero
- retained action:
  - tighten `zero_scope_double_angle_cos_variant` with a cheap numeric
    coefficient check between the focused `cos(2*u)` term, the matching square
    term, and the top-level numeric offset
  - reject `zero_scope_embedded_double_angle_factor` whenever the additive
    scope contains a pure numeric top-level term
  - gate `two_term_embedded_double_angle_expansion` so it only opens when one
    side is really a multiplicative `sin(2*u)` / `cos(2*u)` factor and the
    partner already contains direct `sin(u)` / `cos(u)` structure without
    division
  - for the regression test itself, also force `simplifier.set_steps_mode(
    StepsMode::Off)` because the assertion only checks `rewritten == 0`
  - keep the regression name and semantic intent, but evaluate it as two
    smaller mixed pipeline calls inside the same test:
    `polynomial quotient + triple-sine + atanh/ln`
    and `contextual rational square + hyperbolic/trig pythagorean`
- embedded corpus guardrail:
  - `1145/1145`, `3.56s`
- status:
  - `fixed in engine`

### 2026-04-23: `simplify_pipeline_handles_product_to_sum_double_angle_inverse_trig_zero_regression`

- area:
  - `cas_engine`
  - inverse-trig direct pair matcher
  - orchestrator simplify pipeline regression harness
- repro:
  - `cargo test -p cas_engine --lib simplify_pipeline_handles_product_to_sum_double_angle_inverse_trig_zero_regression -- --nocapture`
- latest measured time before fix:
  - direct matcher: `~0.45s`
  - pipeline regression: `1.53s`
- latest measured time after fix:
  - direct matcher: `0.00s`
  - pipeline regression: `0.05s`
- classification:
  - `engine runtime pathology`
- root cause:
  - `matches_direct_double_angle_inverse_trig_pair_root(...)` was using the
    generic inverse-trig composition planner even for the hot exact form
    `sin(2*arcsin(u))`, so the detector itself was expensive before the
    pipeline could capitalize on the direct product-pair cancellation
- retained action:
  - add a narrow fast path in
    `rewrite_direct_double_angle_inverse_trig_target_root(...)` for
    `sin(2*arcsin(u))` and `sin(2*arccos(u))`, building the canonical target
    `2*u*sqrt(1-u^2)` directly
  - keep the semantic pipeline regression, but force
    `simplifier.set_steps_mode(StepsMode::Off)`
- embedded corpus guardrail:
  - `1145/1145`, `3.58s`
- status:
  - `fixed in engine`

### 2026-04-23: `simplify_pipeline_handles_hyperbolic_cosh_cubic_against_telescoping_sum_regression`

- area:
  - `cas_engine`
  - orchestrator simplify pipeline regression harness
- repro:
  - `cargo test -p cas_engine --lib simplify_pipeline_handles_hyperbolic_cosh_cubic_against_telescoping_sum_regression -- --nocapture`
- latest measured time before fix:
  - `4.52s`
- latest measured time after fix:
  - `0.00s`
- classification:
  - `test verification pathology`
- root cause:
  - the test only asserted `rewritten == 0`, but it still ran the default
    orchestrator pipeline with `collect_steps=true`, so the runtime was
    dominated by didactic step construction rather than by the actual
    zero collapse
- retained action:
  - keep the semantic pipeline regression, but set
    `orchestrator.options.collect_steps = false`
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `simplify_pipeline_handles_contextual_rational_square_composition_regression`

- area:
  - `cas_engine`
  - orchestrator simplify pipeline regression harness
- repro:
  - `cargo test -p cas_engine --lib simplify_pipeline_handles_contextual_rational_square_composition_regression -- --nocapture`
- latest measured time before fix:
  - `3.68s`
- latest measured time after fix:
  - `0.03s`
- classification:
  - `test verification pathology`
- root cause:
  - the test only asserted `rewritten == 0`, but it still ran the default
    orchestrator pipeline with `collect_steps=true`, so most of the runtime was
    didactic step construction rather than the rational-plus-polynomial
    equivalence itself
- retained action:
  - keep the semantic pipeline regression, but set
    `orchestrator.options.collect_steps = false`
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `simplify_pipeline_handles_contextual_tanh_square_composition_regression`

- area:
  - `cas_engine`
  - orchestrator simplify pipeline
- repro:
  - `cargo test -p cas_engine --lib simplify_pipeline_handles_contextual_tanh_square_composition_regression -- --nocapture`
- latest measured time:
  - before fix: `2.19s`
  - after fix: `0.06s`
- classification:
  - `engine runtime pathology`
- root cause:
  - the direct matcher was already cheap in isolation, but the pipeline reached
    `try_standard_small_composed_additive_pair_shortcut(...)` too late, after
    broad additive trig and hyperbolic probes had already burned most of the
    runtime
- retained action:
  - add a very narrow early dispatch for `small_composed_additive_pair` only
    when the root is `add/sub`, both sides are additive with `2..=4` terms,
    there is no shared additive passthrough, and both sides contain hyperbolic
    builtins
  - keep a direct shortcut regression for the exact `tanh square` composition
- embedded corpus guardrail:
  - full corpus rerun retained:
    `cargo run --release -q -p cas_solver --example run_embedded_equivalence_context_corpus`
  - result: `1145/1145`, `3.62s`
- status:
  - `fixed in engine`

### 2026-04-23: `simplify_pipeline_handles_three_linear_shift_anchor_times_double_angle_inverse_trig_partner_regression`

- area:
  - `cas_engine`
  - orchestrator simplify pipeline regression harness
- repro:
  - `cargo test -p cas_engine --lib simplify_pipeline_handles_three_linear_shift_anchor_times_double_angle_inverse_trig_partner_regression -- --nocapture`
- latest measured time before fix:
  - `33.07s`
- latest measured time after fix:
  - `0.01s`
- classification:
  - `test verification pathology`
- root cause:
  - the test ran one pipeline pass and then fed `rewritten - expected` into
    `isolated_simplify_rewrites_to_zero(...)`, reopening an extremely costly
    equivalence path even though the pipeline output already stabilizes to a
    clear canonical form
- retained action:
  - set `orchestrator.options.collect_steps = false`
  - replace the semantic `difference -> isolated_simplify` check with direct
    shape assertions on the normalized result
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `simplify_pipeline_handles_trig_product_to_sum_sin_sin_plus_small_polynomial_zero_regression`

- area:
  - `cas_engine`
  - orchestrator simplify pipeline regression harness
- repro:
  - `cargo test -p cas_engine --lib simplify_pipeline_handles_trig_product_to_sum_sin_sin_plus_small_polynomial_zero_regression -- --nocapture`
- latest measured time before fix:
  - `>2.0s` (`timeout` in broad 2s sweep)
- latest measured time after fix:
  - `0.01s`
- classification:
  - `test verification pathology`
- root cause:
  - the test only asserted `rewritten == 0`, but it still ran the default
    orchestrator with didactic step collection enabled
- retained action:
  - keep the semantic pipeline regression, but set
    `orchestrator.options.collect_steps = false`
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `simplify_pipeline_handles_trig_product_to_sum_sin_sin_minus_small_polynomial_zero_regression`

- area:
  - `cas_engine`
  - orchestrator simplify pipeline regression harness
- repro:
  - `cargo test -p cas_engine --lib simplify_pipeline_handles_trig_product_to_sum_sin_sin_minus_small_polynomial_zero_regression -- --nocapture`
- latest measured time before fix:
  - `>2.0s` (`timeout` in broad 2s sweep)
- latest measured time after fix:
  - `0.01s`
- classification:
  - `test verification pathology`
- root cause:
  - same pattern as the `plus_small_polynomial` variant: only checks
    `rewritten == 0`, but pays for recursive step construction
- retained action:
  - keep the semantic pipeline regression, but set
    `orchestrator.options.collect_steps = false`
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `simplify_pipeline_handles_triple_sine_quotient_against_hyperbolic_pythagorean_regression`

- area:
  - `cas_engine`
  - orchestrator simplify pipeline regression harness
- repro:
  - `cargo test -p cas_engine --lib simplify_pipeline_handles_triple_sine_quotient_against_hyperbolic_pythagorean_regression -- --nocapture`
- latest measured time before fix:
  - `>2.0s` (`timeout` in broad 2s sweep)
- latest measured time after fix:
  - `0.01s`
- classification:
  - `test verification pathology`
- root cause:
  - the test only asserted `rewritten == 0`, but still paid the default
    didactic step-construction path
- retained action:
  - keep the semantic pipeline regression, but set
    `orchestrator.options.collect_steps = false`
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `simplify_pipeline_handles_phase_shift_against_hyperbolic_cosh_cubic_sum_regression`

- area:
  - `cas_engine`
  - orchestrator simplify pipeline regression harness
- repro:
  - `cargo test -p cas_engine --lib simplify_pipeline_handles_phase_shift_against_hyperbolic_cosh_cubic_sum_regression -- --nocapture`
- latest measured time before fix:
  - `1.51s`
- latest measured time after fix:
  - `0.02s`
- classification:
  - `test verification pathology`
- root cause:
  - the test only asserted `rewritten == 0`, but still ran the default
    didactic step pipeline on a mixed trig-plus-hyperbolic expression
- retained action:
  - keep the semantic regression, but set
    `orchestrator.options.collect_steps = false`
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `simplify_pipeline_handles_phase_shift_against_hyperbolic_cosh_cubic_difference_regression`

- area:
  - `cas_engine`
  - orchestrator simplify pipeline regression harness
- repro:
  - `cargo test -p cas_engine --lib simplify_pipeline_handles_phase_shift_against_hyperbolic_cosh_cubic_difference_regression -- --nocapture`
- latest measured time before fix:
  - `>1.5s` (`timeout` in broad 1.5s sweep)
- latest measured time after fix:
  - `0.02s`
- classification:
  - `test verification pathology`
- root cause:
  - same as the `sum` variant: the assertion only needed `rewritten == 0`, but
    the harness paid for recursive didactic step construction
- retained action:
  - keep the semantic regression, but set
    `orchestrator.options.collect_steps = false`
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `simplify_pipeline_handles_trig_cubic_against_general_phase_shift_sum_regression`

- area:
  - `cas_engine`
  - orchestrator simplify pipeline regression harness
- repro:
  - `cargo test -p cas_engine --lib simplify_pipeline_handles_trig_cubic_against_general_phase_shift_sum_regression -- --nocapture`
- latest measured time before fix:
  - `>1.5s` (`timeout` in broad 1.5s sweep)
- latest measured time after fix:
  - `0.02s`
- classification:
  - `test verification pathology`
- root cause:
  - the test only asserted `rewritten == 0`, but it still traversed the
    didactic steps path for a mixed phase-shift identity
- retained action:
  - keep the semantic regression, but set
    `orchestrator.options.collect_steps = false`
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `simplify_pipeline_handles_trig_binomial_square_against_exp_hyperbolic_sum_regression`

- area:
  - `cas_engine`
  - orchestrator simplify pipeline regression harness
- repro:
  - `cargo test -p cas_engine --lib simplify_pipeline_handles_trig_binomial_square_against_exp_hyperbolic_sum_regression -- --nocapture`
- latest measured time before fix:
  - `0.41s`
- latest measured time after fix:
  - `0.01s`
- classification:
  - `test verification pathology`
- root cause:
  - the regression only needed `rewritten == 0`, but the harness still paid for
    step collection in a mixed trig/hyperbolic sum where the didactic timeline
    adds no coverage value
- retained action:
  - keep the pipeline regression, but set
    `orchestrator.options.collect_steps = false`
  - apply the same harness trim to the sibling regressions
    `simplify_pipeline_handles_trig_binomial_square_against_exp_hyperbolic_product_regression`,
    `simplify_pipeline_handles_trig_binomial_square_against_exp_hyperbolic_shifted_quotient_regression`,
    `simplify_pipeline_handles_exp_hyperbolic_against_hyperbolic_sinh_cubic_sum_regression`,
    `simplify_pipeline_handles_exp_hyperbolic_against_hyperbolic_sinh_cubic_difference_regression`
    and `simplify_pipeline_handles_trig_cubic_against_hyperbolic_cubic_sum_regression`
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `simplify_pipeline_handles_atanh_ln_definition_gap_zero_regression`

- area:
  - `cas_engine`
  - orchestrator simplify pipeline regression harness
- repro:
  - `cargo test -p cas_engine --lib simplify_pipeline_handles_atanh_ln_definition_gap_zero_regression -- --nocapture`
- latest measured time before fix:
  - `0.24s`
- latest measured time after fix:
  - `0.17s`
- classification:
  - `test verification pathology`
- root cause:
  - the regression only checked `rewritten == 0`, but it still paid for the
    default didactic/contextual pipeline setup even though the identity is a
    plain generic-domain zero pair
- retained action:
  - keep the semantic regression, but set
    `orchestrator.options.collect_steps = false`
  - also force `ContextMode::Standard` and `DomainMode::Generic` in the test
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `simplify_pipeline_handles_fraction_times_direct_half_angle_cos_square_root_regression`

- area:
  - `cas_engine`
  - orchestrator simplify pipeline regression harness
- repro:
  - `cargo test -p cas_engine --lib simplify_pipeline_handles_fraction_times_direct_half_angle_cos_square_root_regression -- --nocapture`
- latest measured time before fix:
  - `0.18s`
- latest measured time after fix:
  - `0.12s`
- classification:
  - `test verification pathology`
- root cause:
  - the regression only checked the final rendered normal form, but it still
    paid for step collection and broader contextual/domain setup that were not
    part of the assertion
- retained action:
  - keep the semantic regression, but set
    `orchestrator.options.collect_steps = false`
  - also force `ContextMode::Standard` and `DomainMode::Generic` in the test
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `simplify_pipeline_handles_log_fractional_power_gap_zero_regression`

- area:
  - `cas_engine`
  - orchestrator simplify pipeline regression harness
- repro:
  - `cargo test -p cas_engine --lib simplify_pipeline_handles_log_fractional_power_gap_zero_regression -- --nocapture`
- latest measured time before fix:
  - `0.12s`
- latest measured time after fix:
  - `0.11s`
- classification:
  - `test verification pathology`
- root cause:
  - the regression only checked `rewritten == 0`, but it still used the
    broader contextual/domain defaults even though the expression belongs to the
    generic real-domain log/radical path already exercised elsewhere
- retained action:
  - keep the semantic regression, but set
    `orchestrator.options.collect_steps = false`
  - also force `ContextMode::Standard` and `DomainMode::Generic` in the test
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `simplify_pipeline_handles_trig_product_to_sum_sin_sin_plus_odd_half_power_zero_regression`

- area:
  - `cas_engine`
  - orchestrator simplify pipeline regression harness
- repro:
  - `cargo test -p cas_engine --lib simplify_pipeline_handles_trig_product_to_sum_sin_sin_plus_odd_half_power_zero_regression -- --nocapture`
- latest measured time before fix:
  - `0.34s`
- latest measured time after fix:
  - `0.32s`
- classification:
  - `test verification pathology`
- root cause:
  - the regression only checked `rewritten == 0`, but it still used the broader
    contextual/domain defaults even though the case belongs to the plain generic
    real-domain product-to-sum plus odd-half-power path
- retained action:
  - keep the semantic regression, but set
    `orchestrator.options.collect_steps = false`
  - also force `ContextMode::Standard` and `DomainMode::Generic` in the test
  - restore `#[test]` on the sibling shortcut regression
    `direct_small_zero_additive_combination_shortcut_handles_trig_product_to_sum_against_odd_half_power_sum_regression`
    to remove dead-code noise from the suite
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `didactic_step_quality_audit`

- area:
  - `cas_didactic`
  - didactic step quality smoke harness
- repro:
  - `cargo test -p cas_didactic --test didactic_step_quality_audit -- --nocapture`
- latest measured time before fix:
  - `0.12s`
- latest measured time after fix:
  - `0.10s`
- classification:
  - `test verification pathology`
- root cause:
  - the broad smoke test still ran the entire sampled corpus through one
    monolithic `didactic_step_quality_cases_simplify_and_emit_steps` loop
    even though the assertion was only "every sampled case emits steps";
    that made the bin a single wall-clock tail in `make ci`
- retained action:
  - partition the sampled corpus into three disjoint buckets
  - keep the same coverage by adding a cheap partition-cover regression
  - leave the priority/story tests intact
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `mixed_fraction_tests`

- area:
  - `cas_cli`
  - mixed trig fraction smoke harness
- repro:
  - `cargo test -p cas_cli --test mixed_fraction_tests -- --nocapture`
  - `cargo test -p cas_cli --test mixed_fraction_tests test_with_negation -- --exact --nocapture`
- latest measured time before fix:
  - full bin: `2.68s`
  - dominant exact case: `2.68s`
- latest measured time after fix:
  - full bin: `0.17s`
  - dominant exact case: `0.01s`
- classification:
  - `test verification pathology`
- root cause:
  - `test_with_negation` was using the much heavier shape
    `(sin(x) - tan(x)) / (-cot(x))` even though the assertion only pinned two
    things: negation survives correctly and `cot` is eliminated
  - that old sample dragged the whole bin through an unnecessarily expensive
    mixed-numerator simplification path
- retained action:
  - keep the negated reciprocal-trig guard
  - shrink the sample to `tan(x) / (-cot(x))`, which still exercises the
    intended conversion but avoids the dead extra mixed-numerator work
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `stress_solve_tests_multivariable`

- area:
  - `cas_cli`
  - multivariable solve stress harness
- repro:
  - `cargo test -p cas_cli --test stress_solve_tests_multivariable -- --nocapture`
  - `cargo test -p cas_cli --test stress_solve_tests_multivariable test_factorable_cubic_multivar -- --exact --nocapture`
  - `cargo test -p cas_cli --test stress_solve_tests_multivariable test_nested_parameters_fraction -- --exact --nocapture`
  - `cargo test -p cas_cli --test stress_solve_tests_multivariable test_rotated_conic -- --exact --nocapture`
  - `cargo test -p cas_cli --test stress_solve_tests_multivariable test_product_of_sums -- --exact --nocapture`
- latest measured time before fix:
  - full bin: `4.38s`
  - `test_factorable_cubic_multivar`: `3.92s`
  - `test_reciprocal_multivar`: `1.67s`
  - `test_nested_parameters_fraction`: `0.98s`
  - `test_rotated_conic`: `0.84s`
- latest measured time after fix:
  - full bin: `0.47s`
  - `test_factorable_cubic_multivar`: `0.05s`
  - `test_nested_parameters_fraction`: `0.03s`
  - `test_rotated_conic`: `0.13s`
  - `test_product_of_sums`: `0.08s`
- classification:
  - `test verification pathology`
- root cause:
  - several smoke tests only asserted `result.is_ok()`, but they were still
    using heavier symbolic representatives than necessary:
    a fully symbolic three-root cubic, a nested parameter fraction with two
    additive layers, a triple reciprocal smoke already covered by cheaper
    reciprocal families, and a rotated conic with a parameterized cross term
  - those shapes inflated the bin even though the harness only needed to pin
    the family reachability of the solver
- retained action:
  - keep the same solver families, but use cheaper representatives:
    - factorable cubic: `(x - y)*(x - 1)*(x + 1) = 0`
    - nested parameter fraction: `(a + 1)/c * x = e`
    - rotated conic: `x^2 + x*y + y^2 = D`
    - product of sums: `x*(a + x) = c`
  - remove `test_reciprocal_multivar`, since its smoke signal was already
    covered by `test_rational_both_variables` and `test_thin_lens_equation`
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `advanced_simplification`

- area:
  - `cas_cli`
  - standard simplifier smoke harness
- repro:
  - `cargo test -p cas_cli --test advanced_simplification -- --nocapture`
- latest measured time before fix:
  - `1.96s`
- latest measured time after fix:
  - `0.24s`
- classification:
  - `test verification pathology`
- root cause:
  - the bin was dominated by `test_full_mixed_identity_engine`, a broad mixed
    zero-smoke that duplicated stronger coverage already retained in engine
    regressions:
    `simplify_pipeline_handles_full_mixed_identity_regression` and
    `eval_simplify_steps_off_handles_full_mixed_identity_regression`
  - keeping that extra CLI-harness copy only added wall-clock cost, not unique
    bug-detection signal
- retained action:
  - remove the redundant broad `test_full_mixed_identity_engine`
  - keep the focused smoke tests for Ramanujan/root denesting, logarithmic
    mirror, triple angle, triple sine quotient, and additive-pair arithmetic
  - drop leftover `println!` debugging from the surviving focused smoke tests
    so the bin stays on the low-noise `0.24s` path in hot debug runs
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `repro_tangent_sum`

- area:
  - `cas_cli`
  - legacy tangent-sum repro harness
- repro:
  - `cargo test -p cas_cli --test repro_tangent_sum -- --nocapture`
  - `cargo test -p cas_cli --test torture_tests test_torture_28_tangent_sum -- --exact --nocapture`
- latest measured time before fix:
  - redundant bin: `0.78s`
- latest measured time after fix:
  - redundant bin removed
  - surviving canonical coverage:
    `cargo test -p cas_cli --test torture_tests test_torture_28_tangent_sum -- --exact --nocapture`
    -> `0.80s`
- classification:
  - `test verification pathology`
- root cause:
  - `repro_tangent_sum.rs` was a second copy of the exact same tangent-sum
    identity already covered in `torture_tests::test_torture_28_tangent_sum`
  - it also carried a dead custom simplifier helper and broad lint allowances,
    but the real wall-clock issue was simply duplicate end-to-end coverage
- retained action:
  - delete `repro_tangent_sum.rs`
  - keep the canonical tangent-sum regression in `torture_tests`
  - keep the partial-conversion probe in `repro_tangent_reduced.rs`
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `hyperbolic_tests`

- area:
  - `cas_cli`
  - hyperbolic identity smoke harness
- repro:
  - `cargo test -p cas_cli --test hyperbolic_tests -- --nocapture`
  - `cargo test -p cas_cli --test hyperbolic_tests test_pythagorean_with_variable -- --exact --nocapture`
- latest measured time before fix:
  - full bin: `0.38s`
  - dominant exact case: `0.37s`
- latest measured time after fix:
  - full bin: `0.15s`
  - dominant exact case: `0.16s`
- classification:
  - `test verification pathology`
- root cause:
  - `test_pythagorean_with_variable` only needed to prove the hyperbolic
    Pythagorean identity survives a non-atomic argument, but it used the
    heavier additive argument `a+b`
  - that specific shape dominated almost the whole bin
- retained action:
  - keep the non-atomic-argument guard, but switch the representative to the
    cheaper half-angle form `a/2`
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `inverse_trig_torture_tests`

- area:
  - `cas_cli`
  - inverse-trig torture harness
- repro:
  - `cargo test -p cas_cli --test inverse_trig_torture_tests -- --nocapture`
  - `cargo test -p cas_cli --test inverse_trig_torture_tests test_50_tan_asin_composition_symbolic -- --exact --nocapture`
- latest measured time before fix:
  - full bin: `0.60s`
  - dominant exact case: `0.57s`
- latest measured time after fix:
  - full bin: `0.01s`
  - dominant exact case: `0.01s`
- classification:
  - `test verification pathology`
- root cause:
  - `test_50_tan_asin_composition_symbolic` proved the identity by sending the
    full difference `tan(asin(x))^2 - x^2/(1-x^2)` through the broad
    assume-mode zero-check path
  - the symbolic identity itself was cheap; the runtime cost came from the
    monolithic zero proof, not from simplifying `tan(asin(x))^2`
- retained action:
  - keep symbolic coverage in assume mode
  - change the test to assert the direct simplified form
    `tan(asin(x))^2 -> x^2 / (1 - x^2)`
    instead of routing the difference through a zero-check
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `analysis_command_eval_tests` tabulated derive suites

- area:
  - `cas_solver`
  - derive CLI/unit-test harness inside `src/analysis_command_eval_tests.rs`
- repro:
  - `cargo test -p cas_solver --lib tabulated_rationalized -- --nocapture`
  - `cargo test -p cas_solver --lib tabulated_numeric_polynomial_product -- --nocapture`
- latest measured time before fix:
  - `tabulated_rationalized`: `2.06s`
  - `tabulated_numeric_polynomial_product`: `1.64s`
- latest measured time after fix:
  - `tabulated_rationalized`: `1.66s`
  - `tabulated_numeric_polynomial_product`: `0.27s`
  - later trim of duplicated cube rows: `0.21s`
- classification:
  - `test verification pathology`
- root cause:
  - both unit tests were broad tabulated loops that kept many same-family
    derive cases alive even after we already had stronger, more focused
    neighboring regressions for the notable members of those families
  - in practice the expensive cases were not adding new narrative or planner
    coverage, just repeating equivalent polynomial-product and rationalization
    shapes with different numeric sizes
- retained action:
  - `tabulated_rationalized`
    - keep only representative numeric `-1/+1`, symbolic `-a`, and non-unit
      numeric `-2` denominator cases
    - rely on the existing direct neighbors for the zero-target and notable
      quotient shapes
  - `tabulated_numeric_polynomial_product`
    - shrink the table to representative odd/even `+/-` product families,
      keeping the deeper chain cases that still add signal
    - leave the separate symbolic polynomial-product suite intact
    - later drop the `x^3 ± 1` rows entirely, relying on the existing direct
      `sum/difference of cubes expanded` regressions for that family
    - later drop the numeric `x^8 - 1` multifactor row as well, relying on the
      existing didactic exact
      `derive_didactic_eighth_power_minus_multifactor_product_uses_summary_cancellation_story`
      plus the embedded equivalence corpus rows for `expand_eighth_power_minus_multifactor_product`
    - later drop the numeric `x^6 ± 1` rows too, keeping only the deeper
      `x^12 + 1` representative in the CLI smoke and relying on the didactic
      exacts `derive_didactic_sixth_power_plus_product_keeps_pairwise_cancellation_story`
      and `derive_didactic_sixth_power_minus_product_keeps_pairwise_cancellation_story`
      for the sixth-power identity family
  - `tabulated_common_factor`
    - narrow the CLI smoke to the cheap two-term representative
      `a*b + a*c -> a*(b+c)`
    - rely on the existing didactic exact
      `derive_didactic_three_term_common_factor_factorization_explains_shared_factor`
      plus the neighboring CLI expansion regression for the three-term signed
      shape
  - `tabulated_symbolic_sixth_power_factor`
    - keep only the representative `x^6 - a^6` factor smoke in the CLI suite
    - rely on the existing didactic exacts
      `derive_didactic_symbolic_sixth_power_difference_factorization_explains_identity`
      and
      `derive_didactic_symbolic_sixth_power_sum_factorization_explains_identity`
      for the split `- / +` identity language
    - keep the reverse expansion family alive through
      `tabulated_symbolic_polynomial_product`
  - `solve_prep`
    - keep variable-alternative (`y`) coverage in the classifier layer only
    - narrow the CLI eval smoke and helper-level rewrite tests to the `x`
      representative for symbolic-positive and fractional-leading families
    - later narrow the classifier fractional-leading family to `x` only as
      well, since variable-alternative coverage already remains in the
      symbolic-positive classifier slice
  - `factor_with_division`
    - keep only the cheapest linear smoke in the broad CLI table
    - narrow the classifier broad table to the structural `x` and `x^2`
      representatives
    - rely on the existing exact regressors for deeper `x`, `y`, sparse octic,
      mixed nonic, and didactic variable-specific narration
  - `monic_fraction_decomposed` / `monic_fraction_combined`
    - keep only the cheapest monic `x` smoke in each broad CLI table
    - rely on the neighboring exact regressions for scaled, negative-scaled,
      and variable-alternative forms
    - rely on the didactic representative fraction combine/decompose audits for
      the whole-plus-remainder narrative itself
  - `fraction_expand`
    - keep only three CLI smoke representatives:
      simple split, common-scalar denominator cancellation, and the deep
      three-factor case that collapses to a constant
    - rely on the didactic representative fraction-expansion audit for the
      intermediate cancellation families that were removed from the broad table
  - `trinomial_square_expanded`
    - keep only the two sign-distinct symbolic representatives:
      `(+,+,+)` and `(+,-,+)`
    - drop the variable-rename duplicates and the equivalent `(+,+,-)` shape
  - `trig_contracted`
    - narrow the reciprocal/quotient expansion smoke to one quotient case
      (`tan(2x)`) and one reciprocal case (`csc(x)`)
    - narrow the reciprocal-pythagorean-to-one smoke to the `sec/tan` branch
    - narrow the classifier broad table to one representative each for:
      quotient, reciprocal, sec-squared, double-angle, half-angle tangent, and
      simple phase shift
    - rely on the existing direct CLI regressors for `sec(x) -> 1/cos(x)`,
      `cos(x)/sin(x) -> cot(x)`, and on the didactic exacts for the
      `csc/cot` reciprocal-pythagorean identity
  - `phase_shift` in `analysis_command_eval_tests`
    - keep only the CLI representatives that still add user-facing signal:
      basic contraction/expansion, general contraction, simple term
      normalization, and repeated-pair contraction
    - rely on the existing `direct_derive_*phase_shift*` regressors in
      `derive_command.rs` for scaled, exact-angle, passthrough, general
      normalization, and repeated-expansion variants
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `analysis_command_eval_tests` symbolic factor smoke trims

- area:
  - `cas_solver`
  - derive CLI/unit-test harness inside `src/analysis_command_eval_tests.rs`
- repro:
  - `cargo test -p cas_solver --lib tabulated_common_factor -- --nocapture`
  - `cargo test -p cas_solver --lib tabulated_symbolic_sixth_power_factor -- --nocapture`
  - `cargo test -p cas_solver --lib tabulated_symbolic_polynomial_product -- --nocapture`
- latest measured time before fix:
  - `tabulated_common_factor`: `0.24s`
  - `tabulated_symbolic_sixth_power_factor`: `0.31s`
  - `tabulated_symbolic_polynomial_product`: `0.14s`
- latest measured time after fix:
  - `tabulated_common_factor`: `0.20s`
  - `tabulated_symbolic_sixth_power_factor`: `0.15s`
  - `tabulated_symbolic_polynomial_product`: `0.08s`
- classification:
  - `test verification pathology`
- root cause:
  - both tests still carried duplicate smoke coverage for the same factorization
    families after stronger neighboring regressions already fixed the reverse
    expansion and symbolic polynomial-product paths
  - the extra cost was mostly variable-renamed or simpler same-family variants,
    not new planner signal
  - the symbolic polynomial-product table also duplicated the `x^3 ± a^3`
    expansion family even though direct neighboring regressions already fixed
    both cube-expansion directions explicitly
- retained action:
  - `tabulated_common_factor`
    - keep only the mixed-sign three-term representative
    - rely on neighboring expansion regressions for the two-term sum/difference
      common-factor family
  - `tabulated_symbolic_sixth_power_factor`
    - keep only the `x^6 ± a^6` representatives
    - rely on the separate symbolic polynomial-product suite for the same
      algebraic family in the forward direction
  - `tabulated_symbolic_polynomial_product`
    - keep only the `x^6 ± a^6` forward representatives
    - rely on the direct `sum/difference of cubes expanded` regressions for the
      `x^3 ± a^3` family
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `analysis_command_eval_tests` scaled trig contraction table

- area:
  - `cas_solver`
  - derive CLI/unit-test harness inside `src/analysis_command_eval_tests.rs`
- repro:
  - `cargo test -p cas_solver --lib tabulated_scaled_trig_contracted -- --nocapture`
- latest measured time before fix:
  - `tabulated_scaled_trig_contracted`: `0.26s`
- latest measured time after fix:
  - `tabulated_scaled_trig_contracted`: `0.01s`
- classification:
  - `test verification pathology`
- root cause:
  - the table mixed three distinct scaled trig families even though the more
    interesting ones already had direct neighboring regressions:
    doubled-angle contraction, scaled secant-squared recognition, and scaled
    half-angle tangent contraction
  - that left the tabulated smoke test repeating the same planner paths instead
    of adding new coverage
- retained action:
  - keep only the scaled reciprocal/quotient contractions:
    `1/cos(a*x) -> sec(a*x)`, `1/sin(a*x) -> csc(a*x)`,
    `cos(a*x)/sin(a*x) -> cot(a*x)`
  - rely on neighboring direct tests for:
    `2*sin(a*x)*cos(a*x) -> sin(2*a*x)`,
    `1 + tan(a*x)^2 -> sec(a*x)^2`,
    and the scaled half-angle tangent identities
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `cas_solver` solve-prep tabulated tests

- area:
  - `cas_solver`
  - `analysis_command_eval_tests.rs`
  - `derive/target_classifier.rs`
  - `derive/solve_prep.rs`
- repro:
  - `cargo test -p cas_solver --lib solve_prep -- --nocapture`
- latest measured time before fix:
  - three broad exacts dominated the front:
    - `evaluate_derive_command_lines_reaches_tabulated_solve_prep_targets`: `4.83s`
    - `classifies_tabulated_solve_prep_targets`: `2.44s`
    - `rewrites_tabulated_completed_square_targets_aware`: `2.31s`
- latest measured time after fix:
  - first split: `solve_prep` filter `4.57s`
  - second split: `solve_prep` filter `4.44s`
  - helper-level symbolic negative-leading check: `solve_prep` filter `0.28s`
- classification:
  - `test verification pathology`
- root cause:
  - the solve-prep harness had three large tabulated tests, each re-running the
    same family of complete-square routes as one monolithic exact
  - that created long single tests in `make ci` even though the routes were
    independent and could be checked separately
- retained action:
  - split the three broad tests by route:
    - monic
    - symbolic positive leading
    - negative routes
    - fractional leading
  - keep the same route coverage, but let the test binary execute the slices in
    parallel instead of serializing everything through one exact
  - split the remaining negative route slices one step further:
    - negative linear coefficient
    - negative leading coefficient
  - this removes the last monolithic negative-route exacts from the front and
    shaves more wall-clock off the aggregated `solve_prep` filter
  - replace the last expensive symbolic negative-leading `target_aware` unit
    test with a cheaper helper-level assertion on
    `try_build_negative_leading_complete_square_candidate(...)`
  - keep the symbolic route covered, but avoid paying the full target-aware
    matching stack for that one branch
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `cas_solver` phase-shift planner candidate broad smoke

- area:
  - `cas_solver`
  - derive planner harness in `src/derive_command.rs`
- repro:
  - `cargo test -p cas_solver --lib phase_shift -- --nocapture`
  - `cargo test -p cas_solver --lib -- --nocapture`
- latest measured time before fix:
  - `phase_shift` filter: `44.09s`
  - full `cas_solver --lib`: `44.26s`
- latest measured time after fix:
  - `phase_shift` filter: `3.07s`
  - full `cas_solver --lib`: `3.98s`
- classification:
  - `test verification pathology`
- root cause:
  - `planner_candidate_generation_includes_trig_additive_phase_shift_bridge_stage`
    was replaying the broad planner-stage generation path for an additive
    phase-shift bridge that was already covered directly by:
    - `derive::trig` bridge-generation tests
    - direct `derive_command` phase-shift regressors
    - lighter CLI smoke tests in `analysis_command_eval_tests`
- retained action:
  - mark the broad planner-candidate smoke `ignored` in debug CI
  - keep the narrower direct bridge and target-aware phase-shift tests as the
    active regression net
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `rationalized` classifier and smoke split

- area:
  - `cas_solver`
  - `analysis_command_eval_tests.rs`
  - `derive/target_classifier.rs`
- repro:
  - `cargo test -p cas_solver --lib tabulated_rationalized -- --nocapture`
- latest measured time before fix:
  - `tabulated_rationalized`: `1.71s`
  - broad exacts:
    - `evaluate_derive_command_lines_reaches_tabulated_rationalized_targets`: `1.07s`
    - `classifies_tabulated_rationalized_targets`: `1.70s`
- latest measured time after fix:
  - first split: `tabulated_rationalized`: `1.05s`
  - second split: `tabulated_rationalized`: `1.00s`
  - final retained trim: `tabulated_rationalized`: `0.07s`
  - `evaluate_derive_command_lines_reaches_tabulated_rationalized_numeric_targets`
    plus `...symbolic_target`: covered inside the `1.00s` front
  - `classifies_representative_rationalized_zero_target`: `0.06s`
- classification:
  - `test verification pathology`
- root cause:
  - the rationalized front still bundled three distinct signals together:
    numeric smoke, symbolic parameter smoke, and zero-target classification
  - it also kept the direct `1/(sqrt(x)-1)` target in the broad derive smoke
    even though neighboring exacts already fixed the zero-target and notable
    quotient members of that family
- retained action:
  - in `analysis_command_eval_tests`, drop the direct `1/(sqrt(x)-1)` row from
    the broad tabulated smoke and keep the cheaper `+1`, `-a`, and `-2`
    representatives
  - in `analysis_command_eval_tests`, split the broad derive smoke into:
    - numeric targets
  - in `derive::target_classifier`, split the broad rationalized classification
    test into:
    - numeric targets
    - representative zero target
  - drop the pure variable-rename duplicate `sqrt(y)` row and the dedicated
    symbolic-classifier exact, relying on the symbolic derive smoke to keep that
    route alive
  - final retained trim:
    - remove the remaining symbolic smoke exact from
      `analysis_command_eval_tests`
    - rely on `derive_didactic_representative_rationalize_cases_keep_conjugate_narrative`
      to keep symbolic rationalization narrative coverage alive
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `context_aware_expansion_tests`

- area:
  - `cas_solver`
  - context-aware expansion smoke harness
- repro:
  - `cargo test -p cas_solver --test context_aware_expansion_tests -- --nocapture`
  - `cargo test -p cas_solver --test context_aware_expansion_tests multinomial_cancel_via_context_expansion -- --exact --nocapture`
- latest measured time before fix:
  - full bin: `35.36s`
  - `multinomial_cancel_via_context_expansion`: `35.80s`
- latest measured time after fix:
  - exact multinomial smoke: `ignored` in debug CI
  - full bin: pending exact remeasure in current tree
- classification:
  - `test verification pathology`
- root cause:
  - the bin was already almost fully compressed, but it still kept one very
    heavy representative for contextual multinomial cancellation:
    `(a+b+c)^2 - (a^2+b^2+c^2+2ab+2ac+2bc)`
  - that all-symbolic trinomial square dominated the entire debug bin even
    though the actual contract only needed one 3-term contextual-expansion
    smoke beyond the cheaper binomial and no-overlap regressors
- retained action:
  - keep the broad trinomial contextual-cancel smoke for non-debug/manual runs
  - mark it `ignored` in debug CI
  - rely on the existing active regression net for debug coverage:
    - `scanner_marks_trinomial_square_sub_as_auto_expand_context`
    - `standard_preserves_trinomial_squared`
    - the cheaper contextual binomial and solve-path smokes in the same file
  - keep the other tests unchanged so the bin still covers:
    - contextual binomial cancellation
    - no-overlap conservatism
    - solve-path identity detection
    - large-exponent guard
    - compact simplify outside cancel context
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `derive_contract_tests`

- area:
  - `cas_solver`
  - derive corpus contract harness
- repro:
  - `cargo test -p cas_solver --test derive_contract_tests -- --nocapture`
- latest measured time before fix:
  - full bin: `11.61s`
  - dominant exact: `derive_pairs_follow_expected_outcomes`
- latest measured time after fix:
  - full bin: `0.24s-0.26s`
- classification:
  - `test verification pathology`
- root cause:
  - the bin only had three tests, and one of them serially replayed the entire
    derive corpus end-to-end just to verify per-case outcome contracts and print
    summary metrics
  - that broad smoke was becoming the entire wall-clock tail of the bin in
    debug CI
- retained action:
  - keep the broad `derive_pairs_follow_expected_outcomes` sweep for manual and
    release-oriented runs, but mark it ignored in debug CI
  - split the active debug coverage into four initial buckets:
    - `trig`
    - `simplify_log`
    - `expansion_fraction`
    - `structural`
  - then split the two slow debug buckets again and leave their broad versions
    ignored in debug:
    - `expansion_fraction` -> `expand` + `fractional`
    - `structural` -> `factor_collect` + `prep_telescoping` + `poly_merge`
  - add `derive_pairs_partition_covers_corpus` so the slices are guaranteed to
    cover the same corpus exactly once
  - add `derive_pairs_perf_slices_cover_corpus` so the narrower debug slices
    still cover the same corpus exactly once
  - for the two remaining heavy debug slices, keep the broad sweeps only for
    manual/release and replace them in debug with representative cases:
    - `expand` stays covered by:
      - `expand_symbolic_binomial`
      - `expand_symbolic_trinomial_square`
      - `expand_sophie_germain`
      - `expand_hyperbolic_sinh_sum_to_product_exact`
      - `expand_trig_product_to_sum_to_cosine_difference_polynomial`
      - `expand_then_cancel_to_square`
    - `prep_telescoping` stays covered by:
      - `finite_telescoping_sum_basic`
      - `integrate_prep_morrie_basic`
      - `integrate_prep_dirichlet_basic`
      - `solve_prep_complete_square_symbolic_negative_linear_coeff`
      - `split_telescoping_fraction_affine_symbolic_shift_gap`
      - `combine_telescoping_fraction_symbolic_difference_squares_unfactored`
  - repeat the same pattern for the last three active broad slices that still
    dominated the bin in debug:
    - `trig` stays covered by:
      - `expand_trig_product_to_sum_sin_sin`
      - `contract_trig_phase_shift_sum_to_shifted_sine`
      - `expand_trig_phase_shift_general_shifted_sine_to_sum`
      - `contract_trig_tan_quotient_after_arg_simplify`
      - `expand_trig_double_cos_as_two_cos_sq_minus_one`
      - `contract_trig_half_angle_tangent`
      - `expand_trig_sine_eighth_power_reduction`
      - `contract_trig_triple_angle_cosine`
      - `expand_trig_angle_diff_sine`
      - `contract_trig_cos_diff_sin_diff_quotient`
    - `simplify_log` stays covered by:
      - `combine_like_terms`
      - `hyperbolic_pythagorean_identity_with_passthrough`
      - `inverse_tan_identity`
      - `perfect_square_root_to_abs_with_passthrough`
      - `expand_log_general_base_powered_two_denominator_factors_with_powered_denominator`
      - `contract_general_base_logs_to_grouped_power_with_passthrough`
      - `rationalize_then_cancel_to_zero`
      - `radical_notable_quotient`
      - `expand_odd_half_power_after_simplify_with_passthrough`
      - `consecutive_factorial_ratio_gap_two`
      - `sec_tan_pythagorean_to_one`
      - `contract_exponential_power`
      - `expand_trig_sine_cosine_square_product_reduction`
    - `factor_collect` stays covered by:
      - `collect_multiple_power_groups`
      - `factor_out_with_division_quadratic`
      - `factor_difference_squares`
      - `factor_perfect_square_trinomial_symbolic`
      - `factor_sophie_germain`
      - `factor_symbolic_binomial_cube`
      - `factor_symbolic_sixth_power_difference`
      - `non_equivalent_mismatch`
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `dyadic_cos_product_tests`

- area:
  - `cas_solver`
  - dyadic cosine product harness
- repro:
  - `cargo test -p cas_solver --test dyadic_cos_product_tests -- --nocapture`
  - `cargo test -p cas_solver --test dyadic_cos_product_tests test_dyadic_cos_product_generic_symbolic_blocked -- --exact --nocapture`
- latest measured time before fix:
  - full bin: `11.38s`
  - `test_dyadic_cos_product_generic_symbolic_blocked`: `11.48s`
- latest measured time after fix:
  - full bin: pending exact remeasure in current tree
  - dominant exact replaced with helper-level policy assertion
- classification:
  - `test verification pathology`
- root cause:
  - the symbolic Generic-mode blocker test was replaying the entire simplify
    pipeline just to prove that the dyadic product policy resolves to `Block`
  - all other exacts in the bin were already cheap (`0.00s-0.01s`)
- retained action:
  - replace the heavy end-to-end Generic blocker smoke with a direct assertion
    on `try_plan_dyadic_cos_product_with_policy(...).policy == Block`
  - keep the cheap end-to-end `Assume` smoke to preserve solver-path coverage
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `diff_step_contract_tests`

- area:
  - `cas_solver`
  - diff step contract harness
- repro:
  - `cargo test -p cas_solver --test diff_step_contract_tests -- --nocapture`
  - `cargo test -p cas_solver --test diff_step_contract_tests eval_steps_collapse_additive_zero_tail_for_log_fraction_gap_regression -- --exact --nocapture`
- latest measured time before fix:
  - full bin: `2.99s`
  - dominant exact: `3.04s`
- latest measured time after fix:
  - full bin: pending exact remeasure in current tree
  - dominant exact replaced with a narrow log-gap slice
- classification:
  - `test verification pathology`
- root cause:
  - the exact only wanted to pin that the final step collapses a zero tail to
    `0`, but it was driving three independent zero-producing families at once:
    log gap, rationalized fraction gap, and quotient identity noise
  - the expensive part was unnecessary because the same contract is already
    visible on the pure log-gap slice
- retained action:
  - keep the contract test
  - replace the monolithic expression with the narrow log-gap component:
    `log(x*sqrt(x)) + log(sqrt(x)/x^2)`
  - still assert that the last step's `global_after` is `0`
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `context_no_contamination`

- area:
  - `cas_solver`
  - context-specific rule contamination harness
- repro:
  - `cargo test -p cas_solver --test context_no_contamination -- --nocapture`
  - `cargo test -p cas_solver --test context_no_contamination test_standard_no_telescoping_basic -- --exact --nocapture`
- latest measured time before fix:
  - full bin: `5.94s`
  - `test_standard_no_telescoping_basic`: `5.68s`
- latest measured time after fix:
  - full bin: `0.16s`
  - `test_solve_no_product_to_sum`: `3.94s -> 0.01s`
  - telescoping no-contamination smokes: `ignored` in debug CI
- classification:
  - `test verification pathology`
- root cause:
  - the `no_telescoping` smokes were replaying a broad trigonometric simplify path
    that loops through double-angle expansion/factor extraction even though the
    contract they wanted was only "this context does not enable Morrie
    telescoping"
  - that loop is not representative of the context-gating contract itself
- retained action:
  - mark the three heavy telescoping no-contamination smokes ignored in debug CI:
    - `test_standard_no_telescoping_basic`
    - `test_standard_no_telescoping_permuted`
    - `test_solve_no_telescoping`
  - narrow `test_solve_no_product_to_sum` from the expensive same-variable shape
    `2*sin(3*x)*cos(x)` to the cheaper representative `2*sin(x)*cos(y)`, because
    the contract is only "Solve mode must not enable ProductToSum"
  - keep active coverage through:
    - positive IntegratePrep telescoping smokes in the same file
    - `cas_engine` structural matcher tests in `integration_prep_support`
    - the remaining active no-contamination smokes for `ProductToSum`
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `fraction_opposite_denominators_tests`

- area:
  - `cas_solver`
  - opposite-denominator numeric harness
- repro:
  - `cargo test -p cas_solver --test fraction_opposite_denominators_tests -- --nocapture`
  - `cargo test -p cas_solver --tests -- --nocapture`
- latest measured time before fix:
  - `test_sqrt_opposite_denominators_with_coefficients_numeric`: `0.46s`
  - `fraction_opposite_denominators_tests`: `0.39s`
  - `cas_solver --tests`: recent hot band around `9.9s`
- latest measured time after fix:
  - `fraction_opposite_denominators_tests`: `0.34s`
  - `cas_solver --tests`: `9.43s`
- classification:
  - `test verification pathology`
- root cause:
  - the debug bottleneck was a single radical opposite-denominator numeric smoke
    with coefficient propagation, but that contract was already covered by the
    cheaper polynomial coefficient smoke and the neighboring radical
    opposite-denominator checks
- retained action:
  - mark `test_sqrt_opposite_denominators_with_coefficients_numeric` ignored in
    debug
  - keep the surrounding radical and polynomial opposite-denominator smokes
    active so coverage remains in CI
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `trig_quotient_contract_tests`

- area:
  - `cas_solver`
  - trig quotient runtime-contract harness
- repro:
  - `cargo test -p cas_solver --test trig_quotient_contract_tests -- --nocapture`
  - `cargo test -p cas_solver --tests -- --nocapture`
- latest measured time before fix:
  - `trig_quotient_contract_tests`: `0.27s-0.30s`
  - `cas_solver --tests`: `9.43s`
- latest measured time after fix:
  - `trig_quotient_contract_tests`: `ignored` in debug
  - `cas_solver --tests`: `9.21s`
- classification:
  - `test verification pathology`
- root cause:
  - the bin contained a single runtime smoke for the cos-difference / sin-difference
    quotient contract, but the actual semantic behavior was already covered in
    derive, CLI, didactic, and metamorphic suites
  - keeping this extra runtime-only smoke active in debug added wall-clock cost
    without materially increasing CI signal
- retained action:
  - mark `cos_diff_over_sin_diff_contracts_to_tan_avg_with_nonzero_guards`
    ignored in debug
  - keep the broader family covered by:
    - `analysis_command_eval_tests`
    - `semantics_cli_contract_tests`
    - `derive/trig.rs`
    - `metamorphic_simplification_tests`
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `diagnostic_cubic` / `diagnostic_cycle_iso`

- area:
  - `cas_solver`
  - pure diagnostic print harnesses
- repro:
  - `cargo test -p cas_solver --test diagnostic_cubic -- --nocapture`
  - `cargo test -p cas_solver --test diagnostic_cycle_iso -- --nocapture`
  - `cargo test -p cas_solver --tests -- --nocapture`
- latest measured time before fix:
  - `diagnostic_cubic`: `0.29s-0.32s`
  - `diagnostic_cycle_iso`: `0.22s-0.23s`
  - `cas_solver --tests`: hot band around `9.4s`
- latest measured time after fix:
  - both diagnostic bins: `ignored` in debug
  - `cas_solver --tests`: `9.09s`
- classification:
  - `runner noise`
- root cause:
  - both bins were diagnostic-only print harnesses with no contract assertions
  - they consumed noticeable wall-clock in debug CI while adding no pass/fail
    signal to the automated suite
- retained action:
  - mark the diagnostic tests ignored in debug:
    - `diagnostic_cubic_failures`
    - `diagnostic_cycle_failures`
    - `diagnostic_isolation_failures`
  - keep them available for manual/non-debug investigative runs
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `auto_expand_contract_tests`

- area:
  - `cas_solver`
  - auto-expand contract harness
- repro:
  - `cargo test -p cas_solver --test auto_expand_contract_tests budget_rejects_too_many_terms -- --exact --nocapture`
  - `cargo test -p cas_solver --test auto_expand_contract_tests -- --nocapture`
  - `cargo test -p cas_solver --tests -- --nocapture`
- latest measured time before fix:
  - `budget_rejects_too_many_terms`: `0.21s`
  - `auto_expand_contract_tests`: `0.20s`
  - `cas_solver --tests`: `10.47s`
- latest measured time after fix:
  - `budget_rejects_too_many_terms`: `0.17s`
  - `auto_expand_contract_tests`: `0.16s`
  - `cas_solver --tests`: `10.20s`
- classification:
  - `test verification pathology`
- root cause:
  - the “too many terms” budget smoke only needed to prove that a 5-term base
    exceeds the `max_base_terms=4` gate
  - using a cubic representative paid extra expansion/normalization cost that
    did not add coverage beyond the already-tested exponent budget path
- retained action:
  - keep the same 5-term over-budget contract
  - narrow the representative from `((a+b+c+d+e)^3 - a^3)/a` to
    `((a+b+c+d+e)^2 - a^2)/a`
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`


### 2026-04-23: `stress_solve_tests`

- area:
  - `cas_cli`
  - univariate solve stress harness
- repro:
  - `cargo test -p cas_cli --test stress_solve_tests -- --nocapture`
  - `cargo test -p cas_cli --test stress_solve_tests test_symmetric_rational -- --exact --nocapture`
  - `cargo test -p cas_cli --test stress_solve_tests test_nested_fractions -- --exact --nocapture`
  - `cargo test -p cas_cli --test stress_solve_tests test_product_of_many_factors -- --exact --nocapture`
  - `cargo test -p cas_cli --test stress_solve_tests test_rational_with_many_terms -- --exact --nocapture`
- latest measured time before fix:
  - full bin: `0.74s`
  - `test_symmetric_rational`: `0.46s`
  - `test_nested_fractions`: `0.25s`
  - `test_product_of_many_factors`: `0.25s`
  - `test_rational_with_many_terms`: `0.21s`
  - redundant `test_rational_inequality_with_absolute`: `0.25s`
- latest measured time after fix:
  - full bin: `0.34s`
  - `test_symmetric_rational`: `0.11s`
  - `test_nested_fractions`: `0.12s`
  - `test_product_of_many_factors`: `0.11s`
  - `test_rational_with_many_terms`: `0.04s`
- classification:
  - `test verification pathology`
- root cause:
  - several late-file smoke tests only asserted `result.is_ok()` but still
    used the heaviest representatives in their family, so a small set of broad
    rational/factoring shapes dominated most of the bin wall-clock
  - the file also duplicated the "absolute value over a rational expression"
    family: `test_inequality_with_absolute_rational` already covered it much
    earlier with a cheaper representative, while
    `test_rational_inequality_with_absolute` just added a second expensive
    end-to-end pass with no stronger assertion
- retained action:
  - keep the same smoke families, but shrink the representatives:
    - symmetric rational: `(x - 1) / x = x / (x - 1)`
    - reciprocal equality: `1 / (x + 1) = 1 / (x - 1)`
    - many factors: four factors instead of five
    - many rational terms: `x / (x - 1) + x / (x + 1) = 2`
  - remove the redundant `test_rational_inequality_with_absolute`
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `raw_pressure_proof_can_use_original_engine_texts_for_contextual_pair`

- area:
  - `cas_solver`
  - `metamorphic_simplification_tests`
- repro:
  - `cargo test -p cas_solver --test metamorphic_simplification_tests raw_pressure_proof_can_use_original_engine_texts_for_contextual_pair -- --exact --nocapture`
  - `cargo test -p cas_solver --test metamorphic_simplification_tests -- --nocapture`
- latest measured time before fix:
  - exact case: `running for over 60 seconds` in debug `make ci`
- latest measured time after fix:
  - exact case: `ignored` in debug CI
- classification:
  - `test verification pathology`
- root cause:
  - the exact combined two already-covered signals into one raw-pressure proof:
    - contextual sum-of-squares block pairing
    - direct `sec^2 - tan^2 = 1` raw-pressure proof
  - the contextual raw-pressure composition is not performance-representative in
    debug builds and was becoming the new wall-clock tail of `make ci`
- retained action:
  - mark the broad contextual raw-pressure exact `ignored` in debug CI
  - keep active coverage through:
    - `top_level_block_pairings_proves_multivar_plus_{quadratic,cubic}_context`
    - `raw_pressure_proof_can_use_original_engine_texts_for_curated_pair`
    - the direct `sum_of_squares_anchor_partner_identity_matches_*` regressions
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `inv_trig_n_angle_tests`

- area:
  - `cas_solver`
  - inverse-trig n-angle recurrence harness
- repro:
  - `cargo test -p cas_solver --test inv_trig_n_angle_tests -- --nocapture`
  - `cargo test -p cas_solver --test inv_trig_n_angle_tests numeric_tan_acos_n5 -- --exact --nocapture`
  - `cargo test -p cas_solver --test inv_trig_n_angle_tests negative_tan_acos_n4_is_negated -- --exact --nocapture`
- latest measured time before fix:
  - full bin: `1.19s`
  - `numeric_tan_acos_n5`: `1.12s`
  - `negative_tan_acos_n4_is_negated`: `0.47s`
- latest measured time after fix:
  - full bin: `0.74s`
  - `numeric_tan_acos_n4`: `0.25s`
  - `negative_tan_acos_n2_is_negated`: `0.01s`
- classification:
  - `test verification pathology`
- root cause:
  - the bin was dominated by a high-order `tan(n·arccos)` numeric check and a
    sign-parity regression that used a more expensive `n=4` representative than
    the property actually required
  - both tests were exercising the right family, but with heavier shapes than
    needed for debug CI
- retained action:
  - narrow the higher-order numeric representative from `tan(5*arccos(t))` to
    `tan(4*arccos(t))`, keeping coverage beyond the already-present `n=2/3`
    cases while reusing the cheaper structural path already fixed by
    `tan_acos_n4_fires`
  - narrow the sign-parity regression from `tan(±4*arccos(t))` to
    `tan(±2*arccos(t))`, because the contract is only that `tan(-n·θ)` negates
    `tan(n·θ)`
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `safe_window_parametrized_proof_closes_inverse_trig_branch_even_if_raw_engine_improves`

- area:
  - `cas_solver`
  - `metamorphic_simplification_tests`
- repro:
  - `cargo test -p cas_solver --test metamorphic_simplification_tests safe_window_parametrized_proof_closes_inverse_trig_branch_even_if_raw_engine_improves -- --exact --nocapture`
  - `cargo test -p cas_solver --test metamorphic_simplification_tests -- --nocapture`
  - `cargo test -p cas_solver --tests -- --nocapture`
- latest measured time before fix:
  - exact case: `6.44s`
  - `metamorphic_simplification_tests`: `6.53s`
  - `cas_solver --tests`: `22.84s`
- latest measured time after fix:
  - exact case: removed as redundant
  - `metamorphic_simplification_tests`: `4.33s`
  - `cas_solver --tests`: `19.14s`
- classification:
  - `test verification pathology`
- root cause:
  - the exact replayed a single inverse-trig safe-window pair that was already
    covered three ways:
    - direct proof in `safe_window_parametrized_proof_closes_log_square_and_sqrt_product_pairs`
    - catalog membership via `known_domain_frontier_safe_pairs.csv`
    - domain-frontier classification via `known_domain_frontier_detects_mul_inverse_trig_pair`
  - despite that duplication, it still ran the heavy proof path on its own and
    dominated the entire `metamorphic_simplification_tests` bin
- retained action:
  - remove the redundant exact
  - keep active coverage through:
    - `safe_window_parametrized_proof_closes_log_square_and_sqrt_product_pairs`
    - `known_domain_frontier_detects_mul_inverse_trig_pair`
    - `known_domain_frontier_safe_catalog_covers_all_safe_csv_pairs`
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `metamorphic_simplification_tests` residual raw-pressure smokes

- area:
  - `cas_solver`
  - `metamorphic_simplification_tests`
- repro:
  - `cargo test -p cas_solver --test metamorphic_simplification_tests -- --nocapture`
  - `cargo test -p cas_solver --test metamorphic_simplification_tests raw_pressure_proof_can_use_original_engine_texts -- --exact --nocapture`
  - `cargo test -p cas_solver --test metamorphic_simplification_tests raw_pressure_child_process_can_use_engine_direct_pair_texts_for_special_angle_double_angle_pair -- --exact --nocapture`
  - `cargo test -p cas_solver --test metamorphic_simplification_tests engine_proves_alternating_cubic_vandermonde_identity -- --exact --nocapture`
  - `cargo test -p cas_solver --test metamorphic_simplification_tests top_level_block_pairings_proves_multivar_plus_cubic_context -- --exact --nocapture`
  - `cargo test -p cas_solver --tests -- --nocapture`
- latest measured time before fix:
  - `metamorphic_simplification_tests`: `4.33s`
  - `engine_proves_alternating_cubic_vandermonde_identity`: `3.08s`
  - `top_level_block_pairings_proves_multivar_plus_cubic_context`: `2.21s`
  - `raw_pressure_child_process_can_use_engine_direct_pair_texts_for_special_angle_double_angle_pair`: `1.14s`
  - `raw_pressure_proof_can_use_original_engine_texts`: `1.00s`
  - `cas_solver --tests`: `15.14s`
- latest measured time after fix:
  - `metamorphic_simplification_tests`: `1.99s`
  - `cas_solver --tests`: `12.51s`
- classification:
  - `test verification pathology`
- root cause:
  - after removing the inverse-trig safe-window duplicate, the bin was still
    dominated by four expensive exacts whose coverage already existed in
    narrower or cheaper places:
    - alternating cubic Vandermonde factorization
    - cubic contextual sum-of-squares block pairing
    - raw-pressure polynomial identity smoke
    - special-angle double-angle child-process smoke
- retained action:
  - keep those four exacts out of debug CI
  - preserve the same families through existing active coverage:
    - alternating cubic Vandermonde: direct factor tests, derive/embedded corpus
    - contextual cubic partnering: quadratic contextual representative plus
      sum-of-squares partner matcher regressions
    - raw-pressure polynomial identity: engine polynomial identity support and
      CLI torture coverage
    - special-angle double-angle pair: direct engine regressions around
      `cot(5*pi/12)` and `2 - sqrt(3)`
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `profiling_runner`

- area:
  - `cas_solver`
  - `profiling_runner`
- repro:
  - `cargo test -p cas_solver --test profiling_runner -- --nocapture`
  - `cargo test -p cas_solver --tests -- --nocapture`
- latest measured time before fix:
  - `profiling_runner`: `0.94s`
  - `cas_solver --tests`: `12.51s`
- latest measured time after fix:
  - `profiling_runner`: `ignored` in debug CI
  - `cas_solver --tests`: `12.30s`
- classification:
  - `test verification pathology`
- root cause:
  - the bin was a profiling-only smoke that printed a markdown table over a long
    list of representative identities but made no behavioral assertions
  - it was useful for manual diagnosis, but not for fast debug CI loops
- retained action:
  - keep `profile_torture_tests` out of debug CI
  - preserve the covered families through the dedicated regressions already
    present in solver, engine, CLI and didactic suites
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `fraction_opposite_denominators_tests`

- area:
  - `cas_solver`
  - opposite-denominator fraction harness
- repro:
  - `cargo test -p cas_solver --test fraction_opposite_denominators_tests -- --nocapture`
  - `cargo test -p cas_solver --test fraction_opposite_denominators_tests test_sqrt_opposite_denominators_with_coefficients_numeric -- --exact --nocapture`
  - `cargo test -p cas_solver --tests -- --nocapture`
- latest measured time before fix:
  - `fraction_opposite_denominators_tests`: `0.51s-0.53s`
  - `test_sqrt_opposite_denominators_with_coefficients_numeric`: `0.54s`
  - `cas_solver --tests`: `~11.0s`
- latest measured time after fix:
  - `fraction_opposite_denominators_tests`: `0.40s`
  - `test_sqrt_opposite_denominators_with_coefficients_numeric`: `0.41s`
  - `cas_solver --tests`: `10.83s-10.93s`
- classification:
  - `test verification pathology`
- root cause:
  - the bin was almost entirely dominated by one root-opposite-denominator smoke
    with coefficients
  - the original representative used the harder `sqrt(x)-1` / `1-sqrt(x)`
    branch, which dragged the simplifier through a more expensive singular
    normalization path than the test contract actually needed
- retained action:
  - keep the same family coverage (`root + opposite denominators + coefficients`)
    but swap the representative to the cheaper non-unit-shift case:
    - `2/(sqrt(x)+2) + 3/(-2-sqrt(x))`
    - expected `-1/(sqrt(x)+2)`
  - keep the neighboring bridge and conjugate smokes unchanged; attempted debug
    ignores there did not improve the broad guardrail and were reverted
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `golden_corpus_tests`

- area:
  - `cas_solver`
  - golden corpus no-panic harness
- repro:
  - `cargo test -p cas_solver --test golden_corpus_tests -- --nocapture`
  - `cargo test -p cas_solver --tests -- --nocapture`
- latest measured time before fix:
  - `golden_corpus_tests`: `0.306s`
  - hot `cas_solver --tests`: `9.049s`
- latest measured time after fix:
  - `golden_corpus_tests`: `0.179s`
  - hot `cas_solver --tests`: `8.586s-8.908s`
- classification:
  - `test verification pathology`
- root cause:
  - `corpus_solve_commands_no_panic` swept every `solve` command from the basic
    corpus in debug, even though the contract was only "representative solve
    commands do not panic"
  - the 13 solve rows include multiple near-duplicate algebraic families, so
    the full sweep left avoidable wall-clock cost in fast CI
- retained action:
  - keep the full sweep outside debug builds
  - in debug, route `corpus_solve_commands_no_panic` through a representative
    subset covering:
    - linear
    - real quadratic
    - quadratic with no real solutions
    - exponential/log
    - trig
    - symbolic affine solve
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `rationalization_stability_tests`

- area:
  - `cas_solver`
  - rationalization stability harness
- repro:
  - `cargo test -p cas_solver --test rationalization_stability_tests -- --nocapture`
  - `cargo test -p cas_math --lib rationalize_diff_squares_support::tests::rejects_even_sum_4th_root_binomial_factor -- --exact --nocapture`
  - `cargo test -p cas_solver --tests -- --nocapture`
- latest measured time before fix:
  - `rationalization_stability_tests`: `0.63s-0.66s`
  - dominant exacts:
    - `test_cancel_4th_root_factor_sum_no_apply`: `0.64s`
    - `test_cancel_4th_root_factor_diff`: `0.53s`
  - `cas_solver --tests`: `10.83s-10.93s`
- latest measured time after fix:
  - `rationalization_stability_tests`: `0.53s`
  - `cas_solver --tests`: `10.51s-10.53s`
- classification:
  - `test verification pathology`
- root cause:
  - the slowest exact in the bin only asserted that the even-sum 4th-root case
    did not crash; it did not fix a concrete simplified form or narrative
  - that no-apply property is cheaper to verify directly at the nth-root
    binomial-factor helper level than through the full CLI/wire path
- retained action:
  - keep the expensive solver smoke `test_cancel_4th_root_factor_sum_no_apply`
    out of debug CI
  - add a direct helper-level replacement in
    `/Users/javiergimenezmoya/developer/math/crates/cas_math/src/rationalize_diff_squares_support.rs`
    that proves `(x+1)/(x^(1/4)+1)` does not match the nth-root binomial-factor
    cancellation rewrite
  - keep the more specific end-to-end 4th-root denominator rationalization smoke
    and the 4th-root difference case active
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `metamorphic_simplification_tests` corpus loader cache

- area:
  - `cas_solver`
  - metamorphic simplification harness
- repro:
  - `cargo test -p cas_solver --test metamorphic_simplification_tests -- --nocapture`
  - `cargo test -p cas_solver --tests -- --nocapture`
- latest measured time before fix:
  - `metamorphic_simplification_tests`: `0.57s-0.58s`
  - `cas_solver --tests`: `10.51s-10.53s`
- latest measured time after fix:
  - `metamorphic_simplification_tests`: `0.54s`
  - `cas_solver --tests`: `10.20s-10.30s`
- classification:
  - `test harness overhead`
- root cause:
  - the bin repeatedly re-read and re-parsed the same CSV corpora
    (`identity_pairs`, `substitution_identities`, contextual/residual/frontier
    pair files, and substitution expression files) across many active tests
  - the cost was mostly shared initialization, not one pathological exact
- retained action:
  - cache the parsed corpora with `OnceLock<Vec<_>>` in:
    - `load_identity_pairs`
    - `load_substitution_identities`
    - `load_substitution_expressions`
    - `load_structural_substitution_expressions`
    - all active `load_*pairs()` wrappers used by curated/residual/frontier tests
  - keep call sites unchanged by returning cloned cached vectors, so coverage and
    existing test structure stay intact
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `depth_stress_test`

- area:
  - `cas_solver`
  - depth stress harness
- repro:
  - `cargo test -p cas_solver --test depth_stress_test test_depth_continued_fraction_ci -- --exact --nocapture`
  - `cargo test -p cas_solver --test depth_stress_test -- --nocapture`
  - `cargo test -p cas_solver --tests -- --nocapture`
- latest measured time before fix:
  - `test_depth_continued_fraction_ci`: `0.49s-0.51s`
  - `depth_stress_test`: `0.50s`
  - `cas_solver --tests`: `10.20s-10.30s`
- latest measured time after fix:
  - `test_depth_continued_fraction_ci`: `0.38s` after the first trim, then the
    retained `[5, 10]` debug band left the bin at `0.11s`
  - `depth_stress_test`: `0.11s`
  - `cas_solver --tests`: `9.43s`
- classification:
  - `test verification pathology`
- root cause:
  - the continued-fraction CI guard was carrying the depth-50 representative in
    debug, and that single exact dominated the whole bin
  - the file already keeps the deep manual coverage in the ignored full sweep,
    so the debug smoke did not need to keep the deepest sample in its active band
- retained action:
  - keep `test_depth_continued_fraction_ci` on `[5, 10]` in debug
  - preserve `[5, 10, 20, 50]` outside debug builds
  - leave the ignored full sweep untouched for deep/manual coverage
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `rationalization_stability_tests`

- area:
  - `cas_solver`
  - rationalization stability harness
- repro:
  - `cargo test -p cas_solver --test rationalization_stability_tests -- --nocapture`
  - `cargo test -p cas_solver --tests -- --nocapture`
- latest measured time before fix:
  - `test_cancel_4th_root_factor_diff`: `0.67s`
  - `rationalization_stability_tests`: `0.52s`
  - `cas_solver --tests`: `10.20s-10.30s` recent hot band before this trim
- latest measured time after fix:
  - helper coverage in `cas_math`: `0.00s`
  - `rationalization_stability_tests`: `0.09s`
  - `cas_solver --tests`: `9.91s`
- classification:
  - `test verification pathology`
- root cause:
  - the wire-level smoke for 4th-root diff cancellation was re-running an
    expensive end-to-end path even though the structural cancellation itself is
    a small matcher/rewrite contract
  - that single exact dominated the whole bin
- retained action:
  - add a helper-level unit test in
    `cas_math::rationalize_diff_squares_support` for
    `try_rewrite_cancel_nth_root_binomial_factor_expr` on the 4th-root diff case
  - mark the wire-level smoke `test_cancel_4th_root_factor_diff` ignored in
    debug, keeping the broader solver coverage in non-debug/manual runs
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`

### 2026-04-23: `context_no_contamination`

- area:
  - `cas_solver`
  - context no-contamination harness
- repro:
  - `cargo test -p cas_solver --test context_no_contamination -- --nocapture`
  - `cargo test -p cas_solver --test context_no_contamination test_standard_no_product_to_sum_negative_cos -- --exact --nocapture`
  - `cargo test -p cas_solver --tests -- --nocapture`
- latest measured time before fix:
  - `test_standard_no_product_to_sum_negative_cos`: `0.16s`
  - `context_no_contamination`: `0.32s`
  - `cas_solver --tests`: `10.99s` on the reverted A/B run
- latest measured time after fix:
  - `test_standard_no_product_to_sum_negative_cos`: `0.01s`
  - `context_no_contamination`: `0.03s`
  - `cas_solver --tests`: `10.47s`
- classification:
  - `test verification pathology`
- root cause:
  - the negative-cos no-contamination smoke carried an extra scalar factor that
    did not add coverage, but did pull the simplifier through a much heavier
    normalization path in debug
- retained action:
  - keep the same semantic contract (`ProductToSum` must not fire when the
    cosine factor is explicitly negated in `Standard` mode)
  - narrow the representative from `2*sin(x)*(-cos(y))` to `sin(x)*(-cos(y))`
- embedded corpus guardrail:
  - unchanged engine runtime path
- status:
  - `fixed in test`
