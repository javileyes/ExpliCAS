# Engine Improvement Scorecard

- Generated: 2026-04-29T06:29:24.747197+00:00
- Git branch: main
- Git commit: `fdd525f4e0f06a01bc5a14090d368c6e6d0a654e`
- Profile: `guardrail`

## Embedded Runtime Guardrail

- Dimension: contextual simplify/equivalence under real wrappers.
- Interpretation: strong for simplify/orchestration quality; not a derive-path metric.
- Elapsed: 3.97s
- Per-case runtime: 2.802ms/case
- Coverage axes: 7 wrappers across 23 families
- Context axes: 4 complexity levels across 6 shell depths
- Largest wrapper share: 20.3%
- Largest wrapper x complexity share: 17.4%
- Combined additive composition: total=147 families=23 collapse_rows=130 collapse_families=23/23 depth4_rows=35 depth4_families=23/23 orientation_rows=41 orientation_families=23/23 multi_core_rows=40 multi_core_families=23/23 min_family_case_count=6 target_family_case_count=6 families_at_min=22/23 families_under_target=0/23 low_family_counts=collect:6, conditional_factor:6, expand:6, factor:6, finite_telescoping:6, fraction_combine:6, fraction_decompose:6, fraction_expand:6, integrate_prep:6, log_contract:6, log_exp_inverse:6, log_expand:6, log_inverse_power:6, nested_fraction:6, polynomial_product:6, power_merge:6, radical_power:6, rationalize:6, solve_prep:6, telescoping_fraction:6, trig_contract:6, trig_expand:6 above_min_family_counts=simplify:15
- Wrappers: additive_passthrough_zero, combined_additive_zero, common_denominator_zero, reciprocal_shifted_difference_zero, scaled_difference_zero, shifted_quotient_one, squared_passthrough_zero
- Sparse wrappers: reciprocal_shifted_difference_zero total=46 failed=0, squared_passthrough_zero total=72 failed=0
- Structural depth: max_shell_depth=5 max_expression_depth=18 avg_wrapper_overhead_nodes=13.21
- Complexity mix: l2_wrapper_plus_noise total=995 failed=0 avg_shell_depth=1.90, l3_nested_or_composed total=407 failed=0 avg_shell_depth=2.61, l1_single_wrapper total=10 failed=0 avg_shell_depth=1.00
- Shell-depth mix: depth 0 total=5 failed=0, depth 1 total=140 failed=0, depth 2 total=1082 failed=0, depth 3 total=105 failed=0, depth 5 total=3 failed=0
- Sparse wrapper x shell-depth buckets: reciprocal_shifted_difference_zero x depth 2 total=2 failed=0, reciprocal_shifted_difference_zero x depth 3 total=21 failed=0, reciprocal_shifted_difference_zero x depth 4 total=23 failed=0, squared_passthrough_zero x depth 3 total=48 failed=0, squared_passthrough_zero x depth 4 total=24 failed=0
- Sparse wrapper x shell-depth family buckets: reciprocal_shifted_difference_zero x depth 2 x collect total=1 failed=0, reciprocal_shifted_difference_zero x depth 2 x fraction_decompose total=1 failed=0, reciprocal_shifted_difference_zero x depth 3 x conditional_factor total=1 failed=0, reciprocal_shifted_difference_zero x depth 3 x expand total=1 failed=0, reciprocal_shifted_difference_zero x depth 3 x factor total=1 failed=0, reciprocal_shifted_difference_zero x depth 3 x finite_telescoping total=1 failed=0, reciprocal_shifted_difference_zero x depth 3 x fraction_combine total=1 failed=0, reciprocal_shifted_difference_zero x depth 3 x fraction_expand total=1 failed=0, reciprocal_shifted_difference_zero x depth 3 x integrate_prep total=1 failed=0, reciprocal_shifted_difference_zero x depth 3 x log_contract total=1 failed=0, reciprocal_shifted_difference_zero x depth 3 x log_exp_inverse total=1 failed=0, reciprocal_shifted_difference_zero x depth 3 x log_expand total=1 failed=0, reciprocal_shifted_difference_zero x depth 3 x log_inverse_power total=1 failed=0, reciprocal_shifted_difference_zero x depth 3 x nested_fraction total=1 failed=0, reciprocal_shifted_difference_zero x depth 3 x polynomial_product total=1 failed=0, reciprocal_shifted_difference_zero x depth 3 x power_merge total=1 failed=0, reciprocal_shifted_difference_zero x depth 3 x radical_power total=1 failed=0, reciprocal_shifted_difference_zero x depth 3 x rationalize total=1 failed=0, reciprocal_shifted_difference_zero x depth 3 x simplify total=1 failed=0, reciprocal_shifted_difference_zero x depth 3 x solve_prep total=1 failed=0, reciprocal_shifted_difference_zero x depth 3 x telescoping_fraction total=1 failed=0, reciprocal_shifted_difference_zero x depth 3 x trig_contract total=1 failed=0, reciprocal_shifted_difference_zero x depth 3 x trig_expand total=1 failed=0, reciprocal_shifted_difference_zero x depth 4 x collect total=1 failed=0, reciprocal_shifted_difference_zero x depth 4 x conditional_factor total=1 failed=0, reciprocal_shifted_difference_zero x depth 4 x expand total=1 failed=0, reciprocal_shifted_difference_zero x depth 4 x factor total=1 failed=0, reciprocal_shifted_difference_zero x depth 4 x finite_telescoping total=1 failed=0, reciprocal_shifted_difference_zero x depth 4 x fraction_combine total=1 failed=0, reciprocal_shifted_difference_zero x depth 4 x fraction_decompose total=1 failed=0, reciprocal_shifted_difference_zero x depth 4 x fraction_expand total=1 failed=0, reciprocal_shifted_difference_zero x depth 4 x integrate_prep total=1 failed=0, reciprocal_shifted_difference_zero x depth 4 x log_contract total=1 failed=0, reciprocal_shifted_difference_zero x depth 4 x log_exp_inverse total=1 failed=0, reciprocal_shifted_difference_zero x depth 4 x log_expand total=1 failed=0, reciprocal_shifted_difference_zero x depth 4 x log_inverse_power total=1 failed=0, reciprocal_shifted_difference_zero x depth 4 x nested_fraction total=1 failed=0, reciprocal_shifted_difference_zero x depth 4 x polynomial_product total=1 failed=0, reciprocal_shifted_difference_zero x depth 4 x power_merge total=1 failed=0, reciprocal_shifted_difference_zero x depth 4 x radical_power total=1 failed=0, reciprocal_shifted_difference_zero x depth 4 x rationalize total=1 failed=0, reciprocal_shifted_difference_zero x depth 4 x simplify total=1 failed=0, reciprocal_shifted_difference_zero x depth 4 x solve_prep total=1 failed=0, reciprocal_shifted_difference_zero x depth 4 x telescoping_fraction total=1 failed=0, reciprocal_shifted_difference_zero x depth 4 x trig_contract total=1 failed=0, reciprocal_shifted_difference_zero x depth 4 x trig_expand total=1 failed=0, squared_passthrough_zero x depth 3 x simplify total=4 failed=0, squared_passthrough_zero x depth 3 x integrate_prep total=3 failed=0, squared_passthrough_zero x depth 3 x collect total=2 failed=0, squared_passthrough_zero x depth 3 x conditional_factor total=2 failed=0, squared_passthrough_zero x depth 3 x expand total=2 failed=0, squared_passthrough_zero x depth 3 x factor total=2 failed=0, squared_passthrough_zero x depth 3 x finite_telescoping total=2 failed=0, squared_passthrough_zero x depth 3 x fraction_combine total=2 failed=0, squared_passthrough_zero x depth 3 x fraction_decompose total=2 failed=0, squared_passthrough_zero x depth 3 x fraction_expand total=2 failed=0, squared_passthrough_zero x depth 3 x log_contract total=2 failed=0, squared_passthrough_zero x depth 3 x log_exp_inverse total=2 failed=0, squared_passthrough_zero x depth 3 x log_expand total=2 failed=0, squared_passthrough_zero x depth 3 x log_inverse_power total=2 failed=0, squared_passthrough_zero x depth 3 x polynomial_product total=2 failed=0, squared_passthrough_zero x depth 3 x power_merge total=2 failed=0, squared_passthrough_zero x depth 3 x radical_power total=2 failed=0, squared_passthrough_zero x depth 3 x rationalize total=2 failed=0, squared_passthrough_zero x depth 3 x solve_prep total=2 failed=0, squared_passthrough_zero x depth 3 x telescoping_fraction total=2 failed=0, squared_passthrough_zero x depth 3 x trig_contract total=2 failed=0, squared_passthrough_zero x depth 3 x trig_expand total=2 failed=0, squared_passthrough_zero x depth 3 x nested_fraction total=1 failed=0, squared_passthrough_zero x depth 4 x nested_fraction total=2 failed=0, squared_passthrough_zero x depth 4 x collect total=1 failed=0, squared_passthrough_zero x depth 4 x conditional_factor total=1 failed=0, squared_passthrough_zero x depth 4 x expand total=1 failed=0, squared_passthrough_zero x depth 4 x factor total=1 failed=0, squared_passthrough_zero x depth 4 x finite_telescoping total=1 failed=0, squared_passthrough_zero x depth 4 x fraction_combine total=1 failed=0, squared_passthrough_zero x depth 4 x fraction_decompose total=1 failed=0, squared_passthrough_zero x depth 4 x fraction_expand total=1 failed=0, squared_passthrough_zero x depth 4 x integrate_prep total=1 failed=0, squared_passthrough_zero x depth 4 x log_contract total=1 failed=0, squared_passthrough_zero x depth 4 x log_exp_inverse total=1 failed=0, squared_passthrough_zero x depth 4 x log_expand total=1 failed=0, squared_passthrough_zero x depth 4 x log_inverse_power total=1 failed=0, squared_passthrough_zero x depth 4 x polynomial_product total=1 failed=0, squared_passthrough_zero x depth 4 x power_merge total=1 failed=0, squared_passthrough_zero x depth 4 x radical_power total=1 failed=0, squared_passthrough_zero x depth 4 x rationalize total=1 failed=0, squared_passthrough_zero x depth 4 x simplify total=1 failed=0, squared_passthrough_zero x depth 4 x solve_prep total=1 failed=0, squared_passthrough_zero x depth 4 x telescoping_fraction total=1 failed=0, squared_passthrough_zero x depth 4 x trig_contract total=1 failed=0, squared_passthrough_zero x depth 4 x trig_expand total=1 failed=0
- Sparse wrapper depth 4 family breadth: reciprocal_shifted_difference_zero depth4_families=23/23 missing=0 cases=23, squared_passthrough_zero depth4_families=23/23 missing=0 cases=24
- Sparse wrapper noise budgets: reciprocal_shifted_difference_zero total=46 failed=0 avg_overhead=19.83 max_overhead=36, squared_passthrough_zero total=72 failed=0 avg_overhead=16.78 max_overhead=28
- Sparse wrapper x complexity buckets: reciprocal_shifted_difference_zero x l3_nested_or_composed total=46 failed=0 avg_shell_depth=3.46, squared_passthrough_zero x l3_nested_or_composed total=72 failed=0 avg_shell_depth=3.33
- Dominant wrapper x complexity buckets: common_denominator_zero x l2_wrapper_plus_noise total=247 failed=0 avg_shell_depth=2.00, scaled_difference_zero x l2_wrapper_plus_noise total=247 failed=0 avg_shell_depth=1.87, shifted_quotient_one x l2_wrapper_plus_noise total=247 failed=0 avg_shell_depth=1.88
- Sparse wrapper l3 family breadth: reciprocal_shifted_difference_zero l3_families=23/23 missing=0 cases=46, squared_passthrough_zero l3_families=23/23 missing=0 cases=72
- Sparse wrapper family breadth: reciprocal_shifted_difference_zero families=23/23 cases=46, squared_passthrough_zero families=23/23 cases=72
- Sparse wrapper family gaps: reciprocal_shifted_difference_zero missing_families=0/23 covered=23 cases=46, squared_passthrough_zero missing_families=0/23 covered=23 cases=72
- Sparse wrapper x family buckets: reciprocal_shifted_difference_zero x collect total=2 failed=0, reciprocal_shifted_difference_zero x conditional_factor total=2 failed=0, reciprocal_shifted_difference_zero x expand total=2 failed=0, reciprocal_shifted_difference_zero x factor total=2 failed=0, reciprocal_shifted_difference_zero x finite_telescoping total=2 failed=0, reciprocal_shifted_difference_zero x fraction_combine total=2 failed=0, reciprocal_shifted_difference_zero x fraction_decompose total=2 failed=0, reciprocal_shifted_difference_zero x fraction_expand total=2 failed=0, reciprocal_shifted_difference_zero x integrate_prep total=2 failed=0, reciprocal_shifted_difference_zero x log_contract total=2 failed=0, reciprocal_shifted_difference_zero x log_exp_inverse total=2 failed=0, reciprocal_shifted_difference_zero x log_expand total=2 failed=0, reciprocal_shifted_difference_zero x log_inverse_power total=2 failed=0, reciprocal_shifted_difference_zero x nested_fraction total=2 failed=0, reciprocal_shifted_difference_zero x polynomial_product total=2 failed=0, reciprocal_shifted_difference_zero x power_merge total=2 failed=0, reciprocal_shifted_difference_zero x radical_power total=2 failed=0, reciprocal_shifted_difference_zero x rationalize total=2 failed=0, reciprocal_shifted_difference_zero x simplify total=2 failed=0, reciprocal_shifted_difference_zero x solve_prep total=2 failed=0, reciprocal_shifted_difference_zero x telescoping_fraction total=2 failed=0, reciprocal_shifted_difference_zero x trig_contract total=2 failed=0, reciprocal_shifted_difference_zero x trig_expand total=2 failed=0, squared_passthrough_zero x simplify total=5 failed=0, squared_passthrough_zero x integrate_prep total=4 failed=0, squared_passthrough_zero x collect total=3 failed=0, squared_passthrough_zero x conditional_factor total=3 failed=0, squared_passthrough_zero x expand total=3 failed=0, squared_passthrough_zero x factor total=3 failed=0, squared_passthrough_zero x finite_telescoping total=3 failed=0, squared_passthrough_zero x fraction_combine total=3 failed=0, squared_passthrough_zero x fraction_decompose total=3 failed=0, squared_passthrough_zero x fraction_expand total=3 failed=0, squared_passthrough_zero x log_contract total=3 failed=0, squared_passthrough_zero x log_exp_inverse total=3 failed=0, squared_passthrough_zero x log_expand total=3 failed=0, squared_passthrough_zero x log_inverse_power total=3 failed=0, squared_passthrough_zero x nested_fraction total=3 failed=0, squared_passthrough_zero x polynomial_product total=3 failed=0, squared_passthrough_zero x power_merge total=3 failed=0, squared_passthrough_zero x radical_power total=3 failed=0, squared_passthrough_zero x rationalize total=3 failed=0, squared_passthrough_zero x solve_prep total=3 failed=0, squared_passthrough_zero x telescoping_fraction total=3 failed=0, squared_passthrough_zero x trig_contract total=3 failed=0, squared_passthrough_zero x trig_expand total=3 failed=0

## Generated Discovery Ledger

- Purpose: keep failed generated candidates visible without promoting them to live corpus.
- Observe-only discoveries: total=0
- Status: no open observe-only generated discoveries.

## Derive Reachability Guardrail

- Dimension: source-to-target bridgeability and path quality.
- Interpretation: measures planner/strategy reachability; not contextual wrapper strength.
- Outcomes: derived=373 unsupported=0 not_equivalent=1
- Path quality: mean_step_count=1.05 long_path_rate=0.00
- Strategy specificity: generic_simplify_expected=0 distinct_expected_strategies=28

## Derive Shadow Pressure

- Dimension: diagnostic engine-to-derive bridgeability over representative engine equivalence rows.
- Interpretation: exposes where known engine/metamorphic identities are not yet reachable or provable as derive targets; diagnostic, not a support gate.
- Outcomes: sampled=50 derived=50 unsupported=0 not_equivalent=0
- Path signal: mean_step_count=1.08 single_step_successes=46 multi_step_successes=4
- Strategy specificity: generic_simplify_strategy_successes=0 distinct_actual_strategies=28
- Generic simplify shadow IDs: none
- Actual strategy counts: finite sums/products:9, expand trig:3, number theory:3, rewrite exponentials:3, rewrite hyperbolics:3, combine powers:2, contract logs:2, log inverse power:2, nested fraction:2, rationalize:2, rewrite radicals:2, cancel fraction:1, +16 more
- Derived family counts: Fundamental exp decomposition:3, Constant sum:2, Pythagorean Identities:2, log_exp_inverse:2, Addition of fractions:1, Binomial Expansion:1, Binomial coefficients (choose function):1, Change of base:1, Completing the square patterns:1, Difference of Squares:1, Exponential of logs:1, Factor-cancel patterns:1, +33 more
- Problem family counts: unsupported=none not_equivalent=none

## Derive Didactic Trace Audit

- Dimension: educational quality of derive target-driven traces and web substeps.
- Interpretation: diagnostic trace-quality lane; not a semantic correctness or runtime metric.
- Outcomes: cases=455 flagged=0 flagged_rate=0.0%
- Substeps: total_web_substeps=470 mean_step_count=1.05
- Flags: no_web_substeps=0 no_web_steps=0

## Simplify Didactic Trace Audit

- Dimension: educational quality of simplify step traces and web substeps.
- Interpretation: diagnostic trace-quality lane; not a semantic correctness or runtime metric.
- Outcomes: cases=14 flagged=0 flagged_rate=0.0%
- Substeps: total_wire_substeps=26 mean_step_count=2.29
- Flags: no_wire_substeps=0 single_step_no_substeps=0 missing_math_sides=0

## Simplify Benchmark Interpretation

- Dimension: broad semantic closure under metamorphic composition.
- Proof-shape mix: quotient=3497 diff=0 composed=13021 (non-composed 21.2%, composed 78.8%).
- Caveat: high `proved-composed` is a strong semantic/robustness signal, but a weaker direct runtime proxy because part of the closure is shaped by the benchmark harness rather than by one raw engine path.
- Runtime interpretation: use `embedded_equivalence_context`, `simplify_zero_mixed`, and orchestrator profiling to localize real engine cost.

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `embedded_equivalence_context` | `pass` | 3.97s | passed=1417 failed=0 total=1417 wrappers=7 families=23 avg_case=2.802ms |
| `derive_contract` | `pass` | 3.93s | derived=373 unsupported=0 not_equivalent=1 mean_step_count=1.05 generic_simplify_expected=0 |
| `derive_shadow_pressure` | `pass` | 0.16s | sampled=50 derived=50 unsupported=0 not_equivalent=0 mean_step_count=1.08 generic_simplify_strategy_successes=0 single_step=46 |
| `derive_didactic_audit` | `pass` | 15.77s | cases=455 flagged=0 no_web_substeps=0 no_web_steps=0 |
| `simplify_didactic_audit` | `pass` | 0.93s | cases=14 flagged=0 no_wire_substeps=0 missing_math_sides=0 |
| `simplify_strict` | `pass` | 17.61s | nf=0 proved=16518 numeric=0 inconclusive=0 timeouts=0 |
