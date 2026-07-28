# Engine Improvement Scorecard

- Generated: 2026-07-28T15:44:44.410106+00:00
- Git branch: main
- Git commit: `7f8150c44d383ac7d66f83b2d66ad031326dd36f`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=992.56ms avg_case_ms=9.93 simplify=281.00ms avg_simplify_ms=2.81, sum total=200 failed=0 elapsed=869.71ms avg_case_ms=4.35 simplify=286.23ms avg_simplify_ms=1.43, product total=100 failed=0 elapsed=617.13ms avg_case_ms=6.17 simplify=179.16ms avg_simplify_ms=1.79, difference total=50 failed=0 elapsed=409.53ms avg_case_ms=8.19 simplify=125.59ms avg_simplify_ms=2.51
- Engine hotspots: sum simplify=286.23ms avg_simplify_ms=1.43 wall=869.71ms, shifted_quotient simplify=281.00ms avg_simplify_ms=2.81 wall=992.56ms, product simplify=179.16ms avg_simplify_ms=1.79 wall=617.13ms, difference simplify=125.59ms avg_simplify_ms=2.51 wall=409.53ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=992.56ms avg_case_ms=9.93 avg_simplify_ms=2.81, sum@0+100 failed=0 elapsed=635.99ms avg_case_ms=6.36 avg_simplify_ms=2.01, product@0+100 failed=0 elapsed=617.13ms avg_case_ms=6.17 avg_simplify_ms=1.79, difference@0+50 failed=0 elapsed=409.53ms avg_case_ms=8.19 avg_simplify_ms=2.51, sum@700+100 failed=0 elapsed=233.72ms avg_case_ms=2.34 avg_simplify_ms=0.85
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.38ms median_wire=16.44ms median_wall=63.63ms, product@0+100 #175 product runs=3 median_simplify=15.38ms median_wire=15.44ms median_wall=58.67ms, sum@0+100 #173 sum runs=3 median_simplify=15.43ms median_wire=15.47ms median_wall=58.47ms, difference@0+50 #174 difference runs=3 median_simplify=15.46ms median_wire=15.52ms median_wall=58.58ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.10ms median_wire=13.17ms median_wall=49.39ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.89s | passed=450 failed=0 total=450 avg_case=6.422ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.37s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.87s | passed=1 failed=0 |
