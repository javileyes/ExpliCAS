# Engine Improvement Scorecard

- Generated: 2026-04-24T08:15:49.743033+00:00
- Git branch: main
- Git commit: `fc167e82511c3e19c664d3eec69917d2fb94a6c5`
- Profile: `guardrail`

## Embedded Runtime Guardrail

- Dimension: contextual simplify/equivalence under real wrappers.
- Interpretation: strong for simplify/orchestration quality; not a derive-path metric.
- Elapsed: 3.16s
- Coverage axes: 6 wrappers across 23 families
- Context axes: 4 complexity levels across 4 shell depths
- Largest wrapper share: 24.9%
- Largest wrapper x complexity share: 21.3%
- Wrappers: additive_passthrough_zero, combined_additive_zero, common_denominator_zero, scaled_difference_zero, shifted_quotient_one, squared_passthrough_zero
- Sparse wrappers: combined_additive_zero total=7 failed=0, squared_passthrough_zero total=11 failed=0
- Structural depth: max_shell_depth=3 max_expression_depth=18 avg_wrapper_overhead_nodes=11.28
- Complexity mix: l2_wrapper_plus_noise total=973 failed=0 avg_shell_depth=1.90, l3_nested_or_composed total=176 failed=0 avg_shell_depth=1.90, l0_root_pair total=5 failed=0 avg_shell_depth=0.00
- Shell-depth mix: depth 0 total=5 failed=0, depth 1 total=124 failed=0, depth 2 total=1014 failed=0, depth 3 total=12 failed=0
- Sparse wrapper x complexity buckets: combined_additive_zero x l0_root_pair total=5 failed=0 avg_shell_depth=0.00, combined_additive_zero x l2_wrapper_plus_noise total=1 failed=0 avg_shell_depth=1.00, combined_additive_zero x l3_nested_or_composed total=1 failed=0 avg_shell_depth=3.00, squared_passthrough_zero x l3_nested_or_composed total=11 failed=0 avg_shell_depth=3.00
- Dominant wrapper x complexity buckets: additive_passthrough_zero x l2_wrapper_plus_noise total=246 failed=0 avg_shell_depth=1.88, common_denominator_zero x l2_wrapper_plus_noise total=242 failed=0 avg_shell_depth=2.00, scaled_difference_zero x l2_wrapper_plus_noise total=242 failed=0 avg_shell_depth=1.87

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
| `embedded_equivalence_context` | `pass` | 3.16s | passed=1155 failed=0 total=1155 wrappers=6 families=23 |
| `derive_contract` | `pass` | 2.56s | derived=279 unsupported=0 not_equivalent=1 mean_step_count=1.03 |
| `simplify_strict` | `pass` | 16.58s | nf=0 proved=16475 numeric=0 inconclusive=0 timeouts=0 |
