# Engine Improvement Scorecard

- Generated: 2026-07-13T18:03:36.090037+00:00
- Git branch: main
- Git commit: `b37ddc2f034c350640d088034b9ac90c723a50f7`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=357

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=947.14ms avg_case_ms=9.47 simplify=262.80ms avg_simplify_ms=2.63, sum total=200 failed=0 elapsed=847.57ms avg_case_ms=4.24 simplify=272.80ms avg_simplify_ms=1.36, product total=100 failed=0 elapsed=583.92ms avg_case_ms=5.84 simplify=165.68ms avg_simplify_ms=1.66, difference total=50 failed=0 elapsed=393.49ms avg_case_ms=7.87 simplify=118.96ms avg_simplify_ms=2.38
- Engine hotspots: sum simplify=272.80ms avg_simplify_ms=1.36 wall=847.57ms, shifted_quotient simplify=262.80ms avg_simplify_ms=2.63 wall=947.14ms, product simplify=165.68ms avg_simplify_ms=1.66 wall=583.92ms, difference simplify=118.96ms avg_simplify_ms=2.38 wall=393.49ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=947.14ms avg_case_ms=9.47 avg_simplify_ms=2.63, sum@0+100 failed=0 elapsed=624.14ms avg_case_ms=6.24 avg_simplify_ms=1.93, product@0+100 failed=0 elapsed=583.92ms avg_case_ms=5.84 avg_simplify_ms=1.66, difference@0+50 failed=0 elapsed=393.49ms avg_case_ms=7.87 avg_simplify_ms=2.38, sum@700+100 failed=0 elapsed=223.43ms avg_case_ms=2.23 avg_simplify_ms=0.80
- Steady-state engine reruns: sum@0+100 #173 sum runs=3 median_simplify=16.46ms median_wire=16.51ms median_wall=59.55ms, shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.56ms median_wire=16.63ms median_wall=63.38ms, product@0+100 #175 product runs=3 median_simplify=15.03ms median_wire=15.08ms median_wall=56.96ms, difference@0+50 #174 difference runs=3 median_simplify=16.57ms median_wire=16.62ms median_wall=57.91ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.60ms median_wire=12.67ms median_wall=48.14ms
- Steady-state dominant expressions: sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.77s | passed=450 failed=0 total=450 avg_case=6.156ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.43s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.92s | passed=1 failed=0 |
