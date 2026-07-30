# Engine Improvement Scorecard

- Generated: 2026-07-30T12:37:42.246050+00:00
- Git branch: main
- Git commit: `8dbb6e0e6fb3bf7f1a96dea66ce0fd89418d6da4`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=380

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=972.53ms avg_case_ms=9.73 simplify=274.34ms avg_simplify_ms=2.74, sum total=200 failed=0 elapsed=844.47ms avg_case_ms=4.22 simplify=270.09ms avg_simplify_ms=1.35, product total=100 failed=0 elapsed=598.70ms avg_case_ms=5.99 simplify=171.77ms avg_simplify_ms=1.72, difference total=50 failed=0 elapsed=389.67ms avg_case_ms=7.79 simplify=114.16ms avg_simplify_ms=2.28
- Engine hotspots: shifted_quotient simplify=274.34ms avg_simplify_ms=2.74 wall=972.53ms, sum simplify=270.09ms avg_simplify_ms=1.35 wall=844.47ms, product simplify=171.77ms avg_simplify_ms=1.72 wall=598.70ms, difference simplify=114.16ms avg_simplify_ms=2.28 wall=389.67ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=972.53ms avg_case_ms=9.73 avg_simplify_ms=2.74, sum@0+100 failed=0 elapsed=615.42ms avg_case_ms=6.15 avg_simplify_ms=1.88, product@0+100 failed=0 elapsed=598.70ms avg_case_ms=5.99 avg_simplify_ms=1.72, difference@0+50 failed=0 elapsed=389.67ms avg_case_ms=7.79 avg_simplify_ms=2.28, sum@700+100 failed=0 elapsed=229.05ms avg_case_ms=2.29 avg_simplify_ms=0.82
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.95ms median_wire=17.01ms median_wall=63.74ms, difference@0+50 #174 difference runs=3 median_simplify=14.97ms median_wire=15.01ms median_wall=57.38ms, sum@0+100 #173 sum runs=3 median_simplify=14.84ms median_wire=14.89ms median_wall=57.32ms, product@0+100 #175 product runs=3 median_simplify=14.94ms median_wire=14.99ms median_wall=56.87ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.82ms median_wire=12.89ms median_wall=49.06ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.81s | passed=450 failed=0 total=450 avg_case=6.244ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.11s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.73s | passed=1 failed=0 |
