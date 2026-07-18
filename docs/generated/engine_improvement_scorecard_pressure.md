# Engine Improvement Scorecard

- Generated: 2026-07-18T12:20:15.125692+00:00
- Git branch: main
- Git commit: `df26fbd0852167c9eb8253ecb06e620387474f03`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=367

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=975.16ms avg_case_ms=9.75 simplify=274.33ms avg_simplify_ms=2.74, sum total=200 failed=0 elapsed=882.34ms avg_case_ms=4.41 simplify=288.63ms avg_simplify_ms=1.44, product total=100 failed=0 elapsed=602.54ms avg_case_ms=6.03 simplify=172.99ms avg_simplify_ms=1.73, difference total=50 failed=0 elapsed=397.97ms avg_case_ms=7.96 simplify=121.23ms avg_simplify_ms=2.42
- Engine hotspots: sum simplify=288.63ms avg_simplify_ms=1.44 wall=882.34ms, shifted_quotient simplify=274.33ms avg_simplify_ms=2.74 wall=975.16ms, product simplify=172.99ms avg_simplify_ms=1.73 wall=602.54ms, difference simplify=121.23ms avg_simplify_ms=2.42 wall=397.97ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=975.16ms avg_case_ms=9.75 avg_simplify_ms=2.74, sum@0+100 failed=0 elapsed=653.28ms avg_case_ms=6.53 avg_simplify_ms=2.07, product@0+100 failed=0 elapsed=602.54ms avg_case_ms=6.03 avg_simplify_ms=1.73, difference@0+50 failed=0 elapsed=397.97ms avg_case_ms=7.96 avg_simplify_ms=2.42, sum@700+100 failed=0 elapsed=229.06ms avg_case_ms=2.29 avg_simplify_ms=0.82
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.63ms median_wire=16.70ms median_wall=64.20ms, sum@0+100 #173 sum runs=3 median_simplify=15.57ms median_wire=15.62ms median_wall=59.03ms, product@0+100 #175 product runs=3 median_simplify=15.33ms median_wire=15.38ms median_wall=58.75ms, difference@0+50 #174 difference runs=3 median_simplify=15.61ms median_wire=15.66ms median_wall=58.69ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.82ms median_wire=12.90ms median_wall=48.30ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.86s | passed=450 failed=0 total=450 avg_case=6.356ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.47s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.92s | passed=1 failed=0 |
