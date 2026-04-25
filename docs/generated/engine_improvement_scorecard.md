# Engine Improvement Scorecard

- Generated: 2026-04-25T07:05:47.461725+00:00
- Git branch: main
- Git commit: `28ccaf6a176705bb322edfbc9ec47da471586e14`
- Profile: `guardrail`

## Embedded Runtime Guardrail

- Dimension: contextual simplify/equivalence under real wrappers.
- Interpretation: strong for simplify/orchestration quality; not a derive-path metric.
- Elapsed: 3.97s
- Coverage axes: 6 wrappers across 23 families
- Context axes: 4 complexity levels across 6 shell depths
- Largest wrapper share: 22.7%
- Largest wrapper x complexity share: 19.4%
- Wrappers: additive_passthrough_zero, combined_additive_zero, common_denominator_zero, scaled_difference_zero, shifted_quotient_one, squared_passthrough_zero
- Sparse wrappers: squared_passthrough_zero total=53 failed=0
- Structural depth: max_shell_depth=5 max_expression_depth=18 avg_wrapper_overhead_nodes=12.12
- Complexity mix: l2_wrapper_plus_noise total=980 failed=0 avg_shell_depth=1.90, l3_nested_or_composed total=272 failed=0 avg_shell_depth=2.42, l1_single_wrapper total=10 failed=0 avg_shell_depth=1.00
- Shell-depth mix: depth 0 total=5 failed=0, depth 1 total=140 failed=0, depth 2 total=1034 failed=0, depth 3 total=34 failed=0, depth 5 total=3 failed=0
- Sparse wrapper x shell-depth buckets: squared_passthrough_zero x depth 3 total=29 failed=0, squared_passthrough_zero x depth 4 total=24 failed=0
- Sparse wrapper x shell-depth family buckets: squared_passthrough_zero x depth 3 x simplify total=4 failed=0, squared_passthrough_zero x depth 3 x factor total=2 failed=0, squared_passthrough_zero x depth 3 x log_expand total=2 failed=0, squared_passthrough_zero x depth 3 x trig_contract total=2 failed=0, squared_passthrough_zero x depth 3 x collect total=1 failed=0, squared_passthrough_zero x depth 3 x conditional_factor total=1 failed=0, squared_passthrough_zero x depth 3 x expand total=1 failed=0, squared_passthrough_zero x depth 3 x finite_telescoping total=1 failed=0, squared_passthrough_zero x depth 3 x fraction_combine total=1 failed=0, squared_passthrough_zero x depth 3 x fraction_decompose total=1 failed=0, squared_passthrough_zero x depth 3 x fraction_expand total=1 failed=0, squared_passthrough_zero x depth 3 x integrate_prep total=1 failed=0, squared_passthrough_zero x depth 3 x log_contract total=1 failed=0, squared_passthrough_zero x depth 3 x log_exp_inverse total=1 failed=0, squared_passthrough_zero x depth 3 x log_inverse_power total=1 failed=0, squared_passthrough_zero x depth 3 x nested_fraction total=1 failed=0, squared_passthrough_zero x depth 3 x polynomial_product total=1 failed=0, squared_passthrough_zero x depth 3 x power_merge total=1 failed=0, squared_passthrough_zero x depth 3 x radical_power total=1 failed=0, squared_passthrough_zero x depth 3 x rationalize total=1 failed=0, squared_passthrough_zero x depth 3 x solve_prep total=1 failed=0, squared_passthrough_zero x depth 3 x telescoping_fraction total=1 failed=0, squared_passthrough_zero x depth 3 x trig_expand total=1 failed=0, squared_passthrough_zero x depth 4 x nested_fraction total=2 failed=0, squared_passthrough_zero x depth 4 x collect total=1 failed=0, squared_passthrough_zero x depth 4 x conditional_factor total=1 failed=0, squared_passthrough_zero x depth 4 x expand total=1 failed=0, squared_passthrough_zero x depth 4 x factor total=1 failed=0, squared_passthrough_zero x depth 4 x finite_telescoping total=1 failed=0, squared_passthrough_zero x depth 4 x fraction_combine total=1 failed=0, squared_passthrough_zero x depth 4 x fraction_decompose total=1 failed=0, squared_passthrough_zero x depth 4 x fraction_expand total=1 failed=0, squared_passthrough_zero x depth 4 x integrate_prep total=1 failed=0, squared_passthrough_zero x depth 4 x log_contract total=1 failed=0, squared_passthrough_zero x depth 4 x log_exp_inverse total=1 failed=0, squared_passthrough_zero x depth 4 x log_expand total=1 failed=0, squared_passthrough_zero x depth 4 x log_inverse_power total=1 failed=0, squared_passthrough_zero x depth 4 x polynomial_product total=1 failed=0, squared_passthrough_zero x depth 4 x power_merge total=1 failed=0, squared_passthrough_zero x depth 4 x radical_power total=1 failed=0, squared_passthrough_zero x depth 4 x rationalize total=1 failed=0, squared_passthrough_zero x depth 4 x simplify total=1 failed=0, squared_passthrough_zero x depth 4 x solve_prep total=1 failed=0, squared_passthrough_zero x depth 4 x telescoping_fraction total=1 failed=0, squared_passthrough_zero x depth 4 x trig_contract total=1 failed=0, squared_passthrough_zero x depth 4 x trig_expand total=1 failed=0
- Sparse wrapper depth 4 family breadth: squared_passthrough_zero depth4_families=23/23 missing=0 cases=24
- Sparse wrapper noise budgets: squared_passthrough_zero total=53 failed=0 avg_overhead=16.96 max_overhead=24
- Sparse wrapper x complexity buckets: squared_passthrough_zero x l3_nested_or_composed total=53 failed=0 avg_shell_depth=3.45
- Dominant wrapper x complexity buckets: additive_passthrough_zero x l2_wrapper_plus_noise total=246 failed=0 avg_shell_depth=1.88, common_denominator_zero x l2_wrapper_plus_noise total=242 failed=0 avg_shell_depth=2.00, scaled_difference_zero x l2_wrapper_plus_noise total=242 failed=0 avg_shell_depth=1.87
- Sparse wrapper l3 family breadth: squared_passthrough_zero l3_families=23/23 missing=0 cases=53
- Sparse wrapper family breadth: squared_passthrough_zero families=23/23 cases=53
- Sparse wrapper family gaps: squared_passthrough_zero missing_families=0/23 covered=23 cases=53
- Sparse wrapper x family buckets: squared_passthrough_zero x simplify total=5 failed=0, squared_passthrough_zero x factor total=3 failed=0, squared_passthrough_zero x log_expand total=3 failed=0, squared_passthrough_zero x nested_fraction total=3 failed=0, squared_passthrough_zero x trig_contract total=3 failed=0, squared_passthrough_zero x collect total=2 failed=0, squared_passthrough_zero x conditional_factor total=2 failed=0, squared_passthrough_zero x expand total=2 failed=0, squared_passthrough_zero x finite_telescoping total=2 failed=0, squared_passthrough_zero x fraction_combine total=2 failed=0, squared_passthrough_zero x fraction_decompose total=2 failed=0, squared_passthrough_zero x fraction_expand total=2 failed=0, squared_passthrough_zero x integrate_prep total=2 failed=0, squared_passthrough_zero x log_contract total=2 failed=0, squared_passthrough_zero x log_exp_inverse total=2 failed=0, squared_passthrough_zero x log_inverse_power total=2 failed=0, squared_passthrough_zero x polynomial_product total=2 failed=0, squared_passthrough_zero x power_merge total=2 failed=0, squared_passthrough_zero x radical_power total=2 failed=0, squared_passthrough_zero x rationalize total=2 failed=0, squared_passthrough_zero x solve_prep total=2 failed=0, squared_passthrough_zero x telescoping_fraction total=2 failed=0, squared_passthrough_zero x trig_expand total=2 failed=0

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
| `embedded_equivalence_context` | `pass` | 3.97s | passed=1267 failed=0 total=1267 wrappers=6 families=23 |
| `derive_contract` | `pass` | 2.82s | derived=279 unsupported=0 not_equivalent=1 mean_step_count=1.03 |
| `simplify_strict` | `pass` | 17.80s | nf=0 proved=16475 numeric=0 inconclusive=0 timeouts=0 |
