# Engine Improvement Scorecard

- Generated: 2026-07-31T06:39:56.129996+00:00
- Git branch: main
- Git commit: `8006028f5e75005c4f9601122636f886a5c897b5`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=971.95ms avg_case_ms=9.72 simplify=273.65ms avg_simplify_ms=2.74, sum total=200 failed=0 elapsed=840.23ms avg_case_ms=4.20 simplify=268.69ms avg_simplify_ms=1.34, product total=100 failed=0 elapsed=598.84ms avg_case_ms=5.99 simplify=171.72ms avg_simplify_ms=1.72, difference total=50 failed=0 elapsed=389.26ms avg_case_ms=7.79 simplify=114.19ms avg_simplify_ms=2.28
- Engine hotspots: shifted_quotient simplify=273.65ms avg_simplify_ms=2.74 wall=971.95ms, sum simplify=268.69ms avg_simplify_ms=1.34 wall=840.23ms, product simplify=171.72ms avg_simplify_ms=1.72 wall=598.84ms, difference simplify=114.19ms avg_simplify_ms=2.28 wall=389.26ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=971.95ms avg_case_ms=9.72 avg_simplify_ms=2.74, sum@0+100 failed=0 elapsed=611.35ms avg_case_ms=6.11 avg_simplify_ms=1.87, product@0+100 failed=0 elapsed=598.84ms avg_case_ms=5.99 avg_simplify_ms=1.72, difference@0+50 failed=0 elapsed=389.26ms avg_case_ms=7.79 avg_simplify_ms=2.28, sum@700+100 failed=0 elapsed=228.88ms avg_case_ms=2.29 avg_simplify_ms=0.82
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.74ms median_wire=16.81ms median_wall=63.50ms, difference@0+50 #174 difference runs=3 median_simplify=15.01ms median_wire=15.06ms median_wall=57.84ms, sum@0+100 #173 sum runs=3 median_simplify=14.90ms median_wire=14.94ms median_wall=56.49ms, product@0+100 #175 product runs=3 median_simplify=14.82ms median_wire=14.87ms median_wall=56.94ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.78ms median_wire=12.85ms median_wall=48.87ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.80s | passed=450 failed=0 total=450 avg_case=6.222ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.30s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.75s | passed=1 failed=0 |
