# Engine Improvement Scorecard

- Generated: 2026-06-11T15:07:34.042236+00:00
- Git branch: main
- Git commit: `41b08fa6fc3a359a4b0c032d3c79a18ec98955dd`
- Profile: `pressure`

## Generated Discovery Ledger

- Purpose: keep failed generated candidates visible without promoting them to live corpus.
- Observe-only discoveries: total=10
- By area: calculus / integration:4, calculus / runtime:3, calculus / differentiation:2, calculus / robustness:1
- Recent 1: `calculus / integration` - 2026-06-08 - Discovery observe-only: polynomial cosecant/cotangent source-return still emits depth pressure
- Recent 2: `calculus / differentiation` - 2026-06-06 - Observe-only discovery: exact-square atanh scaled-root runtime is not caused by the global empty-domain check
- Recent 3: `calculus / differentiation` - 2026-06-06 - Observe-only discovery: exact-square inverse-root diff runtime is not fixed by raw target preservation

## Calculus Support Matrix Signal

- Dimension: public calculus behavior, support-matrix coverage, result simplification, domain conditions, trace quality, presentation, and verification residuals.
- Interpretation: matrix-oriented calculus lanes; classify failures by command, family, argument regime, domain regime, trace regime, presentation regime, or reusable pre-calculus dependency before adding isolated cases.
- Matrix axes: command, family, argument regime, domain regime, trace regime, presentation regime, and residual verification.
- `diff_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=263
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=344

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=774.09ms avg_case_ms=7.74 simplify=218.87ms avg_simplify_ms=2.19, sum total=200 failed=0 elapsed=706.50ms avg_case_ms=3.53 simplify=233.80ms avg_simplify_ms=1.17, product total=100 failed=0 elapsed=473.68ms avg_case_ms=4.74 simplify=135.68ms avg_simplify_ms=1.36, difference total=50 failed=0 elapsed=325.64ms avg_case_ms=6.51 simplify=103.52ms avg_simplify_ms=2.07
- Engine hotspots: sum simplify=233.80ms avg_simplify_ms=1.17 wall=706.50ms, shifted_quotient simplify=218.87ms avg_simplify_ms=2.19 wall=774.09ms, product simplify=135.68ms avg_simplify_ms=1.36 wall=473.68ms, difference simplify=103.52ms avg_simplify_ms=2.07 wall=325.64ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=774.09ms avg_case_ms=7.74 avg_simplify_ms=2.19, sum@0+100 failed=0 elapsed=516.63ms avg_case_ms=5.17 avg_simplify_ms=1.64, product@0+100 failed=0 elapsed=473.68ms avg_case_ms=4.74 avg_simplify_ms=1.36, difference@0+50 failed=0 elapsed=325.64ms avg_case_ms=6.51 avg_simplify_ms=2.07, sum@700+100 failed=0 elapsed=189.87ms avg_case_ms=1.90 avg_simplify_ms=0.70
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.65ms median_wire=12.72ms median_wall=48.38ms, sum@0+100 #173 sum runs=3 median_simplify=11.45ms median_wire=11.49ms median_wall=43.65ms, product@0+100 #175 product runs=3 median_simplify=11.41ms median_wire=11.46ms median_wall=43.63ms, difference@0+50 #174 difference runs=3 median_simplify=11.44ms median_wire=11.48ms median_wall=43.92ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.65ms median_wire=12.76ms median_wall=44.80ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.28s | passed=450 failed=0 total=450 avg_case=5.067ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.84s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 17.13s | passed=1 failed=0 |
