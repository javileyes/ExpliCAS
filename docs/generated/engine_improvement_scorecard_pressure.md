# Engine Improvement Scorecard

- Generated: 2026-06-15T22:03:47.698344+00:00
- Git branch: main
- Git commit: `e551dfc221f3c7abbdeda66036de2c39563afa49`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=353

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=807.43ms avg_case_ms=8.07 simplify=230.18ms avg_simplify_ms=2.30, sum total=200 failed=0 elapsed=707.91ms avg_case_ms=3.54 simplify=239.17ms avg_simplify_ms=1.20, product total=100 failed=0 elapsed=490.07ms avg_case_ms=4.90 simplify=142.48ms avg_simplify_ms=1.42, difference total=50 failed=0 elapsed=333.33ms avg_case_ms=6.67 simplify=106.12ms avg_simplify_ms=2.12
- Engine hotspots: sum simplify=239.17ms avg_simplify_ms=1.20 wall=707.91ms, shifted_quotient simplify=230.18ms avg_simplify_ms=2.30 wall=807.43ms, product simplify=142.48ms avg_simplify_ms=1.42 wall=490.07ms, difference simplify=106.12ms avg_simplify_ms=2.12 wall=333.33ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=807.43ms avg_case_ms=8.07 avg_simplify_ms=2.30, sum@0+100 failed=0 elapsed=511.58ms avg_case_ms=5.12 avg_simplify_ms=1.66, product@0+100 failed=0 elapsed=490.07ms avg_case_ms=4.90 avg_simplify_ms=1.42, difference@0+50 failed=0 elapsed=333.33ms avg_case_ms=6.67 avg_simplify_ms=2.12, sum@700+100 failed=0 elapsed=196.33ms avg_case_ms=1.96 avg_simplify_ms=0.74
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.36ms median_wire=13.44ms median_wall=50.42ms, product@0+100 #175 product runs=3 median_simplify=11.96ms median_wire=12.01ms median_wall=45.56ms, difference@0+50 #174 difference runs=3 median_simplify=11.96ms median_wire=12.02ms median_wall=45.32ms, sum@0+100 #173 sum runs=3 median_simplify=11.86ms median_wire=11.91ms median_wall=45.43ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.74ms median_wire=10.82ms median_wall=40.77ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.34s | passed=450 failed=0 total=450 avg_case=5.200ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.88s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.97s | passed=1 failed=0 |
