# Engine Improvement Scorecard

- Generated: 2026-04-24T11:34:17.702838+00:00
- Git branch: main
- Git commit: `07f73d6b6b0639394bc828874149ca84f31d9571`
- Profile: `guardrail`

## Embedded Runtime Guardrail

- Dimension: contextual simplify/equivalence under real wrappers.
- Interpretation: strong for simplify/orchestration quality; not a derive-path metric.
- Elapsed: 3.19s
- Coverage axes: 6 wrappers across 23 families
- Context axes: 4 complexity levels across 5 shell depths
- Largest wrapper share: 23.9%
- Largest wrapper x complexity share: 20.4%
- Wrappers: additive_passthrough_zero, combined_additive_zero, common_denominator_zero, scaled_difference_zero, shifted_quotient_one, squared_passthrough_zero
- Sparse wrappers: squared_passthrough_zero total=30 failed=0, combined_additive_zero total=39 failed=0
- Structural depth: max_shell_depth=4 max_expression_depth=18 avg_wrapper_overhead_nodes=11.48
- Complexity mix: l2_wrapper_plus_noise total=980 failed=0 avg_shell_depth=1.90, l3_nested_or_composed total=211 failed=0 avg_shell_depth=2.04, l1_single_wrapper total=10 failed=0 avg_shell_depth=1.00
- Shell-depth mix: depth 0 total=5 failed=0, depth 1 total=140 failed=0, depth 2 total=1026 failed=0, depth 3 total=32 failed=0, depth 4 total=3 failed=0
- Sparse wrapper x shell-depth buckets: combined_additive_zero x depth 0 total=5 failed=0, combined_additive_zero x depth 1 total=17 failed=0, combined_additive_zero x depth 2 total=12 failed=0, combined_additive_zero x depth 3 total=4 failed=0, combined_additive_zero x depth 4 total=1 failed=0, squared_passthrough_zero x depth 3 total=28 failed=0, squared_passthrough_zero x depth 4 total=2 failed=0
- Sparse wrapper noise budgets: combined_additive_zero total=39 failed=0 avg_overhead=13.67 max_overhead=38, squared_passthrough_zero total=30 failed=0 avg_overhead=15.27 max_overhead=24
- Sparse wrapper x complexity buckets: combined_additive_zero x l0_root_pair total=5 failed=0 avg_shell_depth=0.00, combined_additive_zero x l1_single_wrapper total=9 failed=0 avg_shell_depth=1.00, combined_additive_zero x l2_wrapper_plus_noise total=8 failed=0 avg_shell_depth=1.00, combined_additive_zero x l3_nested_or_composed total=17 failed=0 avg_shell_depth=2.35
- Dominant wrapper x complexity buckets: additive_passthrough_zero x l2_wrapper_plus_noise total=246 failed=0 avg_shell_depth=1.88, common_denominator_zero x l2_wrapper_plus_noise total=242 failed=0 avg_shell_depth=2.00, scaled_difference_zero x l2_wrapper_plus_noise total=242 failed=0 avg_shell_depth=1.87
- Sparse wrapper l3 family breadth: combined_additive_zero l3_families=15/23 missing=8 cases=17, squared_passthrough_zero l3_families=23/23 missing=0 cases=30
- Sparse wrapper family breadth: combined_additive_zero families=23/23 cases=39, squared_passthrough_zero families=23/23 cases=30
- Sparse wrapper family gaps: combined_additive_zero missing_families=0/23 covered=23 cases=39, squared_passthrough_zero missing_families=0/23 covered=23 cases=30
- Sparse wrapper x family buckets: combined_additive_zero x simplify total=7 failed=0, combined_additive_zero x collect total=2 failed=0, combined_additive_zero x conditional_factor total=2 failed=0, combined_additive_zero x factor total=2 failed=0, combined_additive_zero x fraction_combine total=2 failed=0, combined_additive_zero x fraction_decompose total=2 failed=0, combined_additive_zero x log_expand total=2 failed=0, combined_additive_zero x polynomial_product total=2 failed=0, combined_additive_zero x solve_prep total=2 failed=0, combined_additive_zero x telescoping_fraction total=2 failed=0, combined_additive_zero x trig_expand total=2 failed=0, combined_additive_zero x expand total=1 failed=0, combined_additive_zero x finite_telescoping total=1 failed=0, combined_additive_zero x fraction_expand total=1 failed=0, combined_additive_zero x integrate_prep total=1 failed=0, combined_additive_zero x log_contract total=1 failed=0, combined_additive_zero x log_exp_inverse total=1 failed=0, combined_additive_zero x log_inverse_power total=1 failed=0, combined_additive_zero x nested_fraction total=1 failed=0, combined_additive_zero x power_merge total=1 failed=0, combined_additive_zero x radical_power total=1 failed=0, combined_additive_zero x rationalize total=1 failed=0, combined_additive_zero x trig_contract total=1 failed=0, squared_passthrough_zero x simplify total=4 failed=0, squared_passthrough_zero x nested_fraction total=3 failed=0, squared_passthrough_zero x factor total=2 failed=0, squared_passthrough_zero x trig_contract total=2 failed=0, squared_passthrough_zero x collect total=1 failed=0, squared_passthrough_zero x conditional_factor total=1 failed=0, squared_passthrough_zero x expand total=1 failed=0, squared_passthrough_zero x finite_telescoping total=1 failed=0, squared_passthrough_zero x fraction_combine total=1 failed=0, squared_passthrough_zero x fraction_decompose total=1 failed=0, squared_passthrough_zero x fraction_expand total=1 failed=0, squared_passthrough_zero x integrate_prep total=1 failed=0, squared_passthrough_zero x log_contract total=1 failed=0, squared_passthrough_zero x log_exp_inverse total=1 failed=0, squared_passthrough_zero x log_expand total=1 failed=0, squared_passthrough_zero x log_inverse_power total=1 failed=0, squared_passthrough_zero x polynomial_product total=1 failed=0, squared_passthrough_zero x power_merge total=1 failed=0, squared_passthrough_zero x radical_power total=1 failed=0, squared_passthrough_zero x rationalize total=1 failed=0, squared_passthrough_zero x solve_prep total=1 failed=0, squared_passthrough_zero x telescoping_fraction total=1 failed=0, squared_passthrough_zero x trig_expand total=1 failed=0

## Derive Reachability Guardrail

- Dimension: source-to-target bridgeability and path quality.
- Interpretation: measures planner/strategy reachability; not contextual wrapper strength.
- Outcomes: derived=279 unsupported=0 not_equivalent=1
- Path quality: mean_step_count=1.03 long_path_rate=0.00

## Simplify Benchmark Interpretation

- Dimension: broad semantic closure under metamorphic composition.
- Proof-shape mix: quotient=3497 diff=0 composed=12978 (non-composed 21.2%, composed 78.8%).
- Caveat: high `proved-composed` is a strong semantic/robustness signal, but a weaker direct runtime proxy because part of the closure is shaped by the benchmark harness rather than by one raw engine path.
- Runtime interpretation: use `embedded_equivalence_context`, `simplify_zero_mixed`, and orchestrator profiling to localize real engine cost.

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `embedded_equivalence_context` | `pass` | 3.19s | passed=1206 failed=0 total=1206 wrappers=6 families=23 |
| `derive_contract` | `pass` | 2.45s | derived=279 unsupported=0 not_equivalent=1 mean_step_count=1.03 |
| `simplify_strict` | `pass` | 18.20s | nf=0 proved=16475 numeric=0 inconclusive=0 timeouts=0 |
