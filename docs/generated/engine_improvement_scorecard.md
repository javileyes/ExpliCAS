# Engine Improvement Scorecard

- Generated: 2026-04-23T18:39:44.738443+00:00
- Git branch: main
- Git commit: `c4444c4d0ae05cbf37d9b3b1fe6c3abd3746b6b0`
- Profile: `fast_embedded`

## Embedded Runtime Guardrail

- Dimension: contextual simplify/equivalence under real wrappers.
- Interpretation: strong for simplify/orchestration quality; not a derive-path metric.
- Elapsed: 3.02s
- Coverage axes: 6 wrappers across 23 families
- Context axes: 4 complexity levels across 4 shell depths
- Largest wrapper share: 25.1%
- Largest wrapper x complexity share: 21.4%
- Wrappers: additive_passthrough_zero, combined_additive_zero, common_denominator_zero, scaled_difference_zero, shifted_quotient_one, squared_passthrough_zero
- Structural depth: max_shell_depth=3 max_expression_depth=18 avg_wrapper_overhead_nodes=11.29
- Complexity mix: l2_wrapper_plus_noise total=969 failed=0 avg_shell_depth=1.90, l3_nested_or_composed total=173 failed=0 avg_shell_depth=1.88, l0_root_pair total=2 failed=0 avg_shell_depth=0.00
- Shell-depth mix: depth 0 total=2 failed=0, depth 1 total=124 failed=0, depth 2 total=1010 failed=0, depth 3 total=9 failed=0
- Dominant wrapper x complexity buckets: additive_passthrough_zero x l2_wrapper_plus_noise total=245 failed=0 avg_shell_depth=1.88, common_denominator_zero x l2_wrapper_plus_noise total=241 failed=0 avg_shell_depth=2.00, scaled_difference_zero x l2_wrapper_plus_noise total=241 failed=0 avg_shell_depth=1.87

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `embedded_equivalence_context` | `pass` | 3.02s | passed=1145 failed=0 total=1145 wrappers=6 families=23 |
| `simplify_add_small` | `pass` | 0.61s | passed=1 failed=0 nf=0 proved=435 timeouts=0 |
| `contextual_strict_fast` | `pass` | 4.31s | passed=1 failed=0 nf=0 proved=64 timeouts=0 |
