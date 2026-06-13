# Engine Improvement Scorecard

- Generated: 2026-06-13T19:01:39.640329+00:00
- Git branch: main
- Git commit: `4fc6f5cdd7053b613646522ccf7edcaa52086bf0`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=780.72ms avg_case_ms=7.81 simplify=220.09ms avg_simplify_ms=2.20, sum total=200 failed=0 elapsed=763.14ms avg_case_ms=3.82 simplify=233.27ms avg_simplify_ms=1.17, product total=100 failed=0 elapsed=474.17ms avg_case_ms=4.74 simplify=135.29ms avg_simplify_ms=1.35, difference total=50 failed=0 elapsed=323.31ms avg_case_ms=6.47 simplify=102.15ms avg_simplify_ms=2.04
- Engine hotspots: sum simplify=233.27ms avg_simplify_ms=1.17 wall=763.14ms, shifted_quotient simplify=220.09ms avg_simplify_ms=2.20 wall=780.72ms, product simplify=135.29ms avg_simplify_ms=1.35 wall=474.17ms, difference simplify=102.15ms avg_simplify_ms=2.04 wall=323.31ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=780.72ms avg_case_ms=7.81 avg_simplify_ms=2.20, sum@0+100 failed=0 elapsed=573.62ms avg_case_ms=5.74 avg_simplify_ms=1.63, product@0+100 failed=0 elapsed=474.17ms avg_case_ms=4.74 avg_simplify_ms=1.35, difference@0+50 failed=0 elapsed=323.31ms avg_case_ms=6.47 avg_simplify_ms=2.04, sum@700+100 failed=0 elapsed=189.53ms avg_case_ms=1.90 avg_simplify_ms=0.70
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.73ms median_wire=13.80ms median_wall=51.49ms, product@0+100 #175 product runs=3 median_simplify=11.81ms median_wire=11.87ms median_wall=44.78ms, sum@0+100 #173 sum runs=3 median_simplify=11.49ms median_wire=11.54ms median_wall=43.69ms, difference@0+50 #174 difference runs=3 median_simplify=11.82ms median_wire=11.87ms median_wall=44.37ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.60ms median_wire=10.68ms median_wall=39.80ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.34s | passed=450 failed=0 total=450 avg_case=5.200ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.83s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.92s | passed=1 failed=0 |
