# Engine Improvement Scorecard

- Generated: 2026-06-11T23:32:49.636679+00:00
- Git branch: main
- Git commit: `7d93d82cae679b944a79119cdc02cccaef45c273`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=775.36ms avg_case_ms=7.75 simplify=218.64ms avg_simplify_ms=2.19, sum total=200 failed=0 elapsed=681.99ms avg_case_ms=3.41 simplify=228.14ms avg_simplify_ms=1.14, product total=100 failed=0 elapsed=467.68ms avg_case_ms=4.68 simplify=134.23ms avg_simplify_ms=1.34, difference total=50 failed=0 elapsed=323.38ms avg_case_ms=6.47 simplify=102.35ms avg_simplify_ms=2.05
- Engine hotspots: sum simplify=228.14ms avg_simplify_ms=1.14 wall=681.99ms, shifted_quotient simplify=218.64ms avg_simplify_ms=2.19 wall=775.36ms, product simplify=134.23ms avg_simplify_ms=1.34 wall=467.68ms, difference simplify=102.35ms avg_simplify_ms=2.05 wall=323.38ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=775.36ms avg_case_ms=7.75 avg_simplify_ms=2.19, sum@0+100 failed=0 elapsed=491.43ms avg_case_ms=4.91 avg_simplify_ms=1.58, product@0+100 failed=0 elapsed=467.68ms avg_case_ms=4.68 avg_simplify_ms=1.34, difference@0+50 failed=0 elapsed=323.38ms avg_case_ms=6.47 avg_simplify_ms=2.05, sum@700+100 failed=0 elapsed=190.56ms avg_case_ms=1.91 avg_simplify_ms=0.70
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.88ms median_wire=12.95ms median_wall=49.55ms, difference@0+50 #174 difference runs=3 median_simplify=11.66ms median_wire=11.70ms median_wall=44.37ms, sum@0+100 #173 sum runs=3 median_simplify=11.55ms median_wire=11.60ms median_wall=44.08ms, product@0+100 #175 product runs=3 median_simplify=11.50ms median_wire=11.55ms median_wall=44.21ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.32ms median_wire=10.39ms median_wall=39.63ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.25s | passed=450 failed=0 total=450 avg_case=5.000ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.82s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 17.07s | passed=1 failed=0 |
