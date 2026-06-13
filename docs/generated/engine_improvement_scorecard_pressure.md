# Engine Improvement Scorecard

- Generated: 2026-06-13T07:40:58.613008+00:00
- Git branch: main
- Git commit: `e3bf5db1e702bf5be1356b4136f2bdfa8ea94868`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=770.73ms avg_case_ms=7.71 simplify=218.20ms avg_simplify_ms=2.18, sum total=200 failed=0 elapsed=687.12ms avg_case_ms=3.44 simplify=229.08ms avg_simplify_ms=1.15, product total=100 failed=0 elapsed=471.17ms avg_case_ms=4.71 simplify=135.26ms avg_simplify_ms=1.35, difference total=50 failed=0 elapsed=319.21ms avg_case_ms=6.38 simplify=101.61ms avg_simplify_ms=2.03
- Engine hotspots: sum simplify=229.08ms avg_simplify_ms=1.15 wall=687.12ms, shifted_quotient simplify=218.20ms avg_simplify_ms=2.18 wall=770.73ms, product simplify=135.26ms avg_simplify_ms=1.35 wall=471.17ms, difference simplify=101.61ms avg_simplify_ms=2.03 wall=319.21ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=770.73ms avg_case_ms=7.71 avg_simplify_ms=2.18, sum@0+100 failed=0 elapsed=497.23ms avg_case_ms=4.97 avg_simplify_ms=1.59, product@0+100 failed=0 elapsed=471.17ms avg_case_ms=4.71 avg_simplify_ms=1.35, difference@0+50 failed=0 elapsed=319.21ms avg_case_ms=6.38 avg_simplify_ms=2.03, sum@700+100 failed=0 elapsed=189.89ms avg_case_ms=1.90 avg_simplify_ms=0.70
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.82ms median_wire=12.89ms median_wall=48.84ms, difference@0+50 #174 difference runs=3 median_simplify=11.42ms median_wire=11.46ms median_wall=43.62ms, product@0+100 #175 product runs=3 median_simplify=11.29ms median_wire=11.34ms median_wall=43.45ms, sum@0+100 #173 sum runs=3 median_simplify=11.30ms median_wire=11.35ms median_wall=43.30ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.62ms median_wire=10.69ms median_wall=39.94ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.25s | passed=450 failed=0 total=450 avg_case=5.000ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.80s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.91s | passed=1 failed=0 |
