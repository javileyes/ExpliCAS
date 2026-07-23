# Engine Improvement Scorecard

- Generated: 2026-07-23T03:04:53.221615+00:00
- Git branch: main
- Git commit: `337536dbc3518f1fc5ac76c046b2c87586a0d8b1`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=974.56ms avg_case_ms=9.75 simplify=274.66ms avg_simplify_ms=2.75, sum total=200 failed=0 elapsed=888.31ms avg_case_ms=4.44 simplify=289.92ms avg_simplify_ms=1.45, product total=100 failed=0 elapsed=603.02ms avg_case_ms=6.03 simplify=173.92ms avg_simplify_ms=1.74, difference total=50 failed=0 elapsed=395.98ms avg_case_ms=7.92 simplify=120.81ms avg_simplify_ms=2.42
- Engine hotspots: sum simplify=289.92ms avg_simplify_ms=1.45 wall=888.31ms, shifted_quotient simplify=274.66ms avg_simplify_ms=2.75 wall=974.56ms, product simplify=173.92ms avg_simplify_ms=1.74 wall=603.02ms, difference simplify=120.81ms avg_simplify_ms=2.42 wall=395.98ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=974.56ms avg_case_ms=9.75 avg_simplify_ms=2.75, sum@0+100 failed=0 elapsed=656.80ms avg_case_ms=6.57 avg_simplify_ms=2.07, product@0+100 failed=0 elapsed=603.02ms avg_case_ms=6.03 avg_simplify_ms=1.74, difference@0+50 failed=0 elapsed=395.98ms avg_case_ms=7.92 avg_simplify_ms=2.42, sum@700+100 failed=0 elapsed=231.51ms avg_case_ms=2.32 avg_simplify_ms=0.83
- Steady-state engine reruns: sum@0+100 #173 sum runs=3 median_simplify=14.78ms median_wire=14.82ms median_wall=57.73ms, shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.45ms median_wire=16.52ms median_wall=69.04ms, difference@0+50 #174 difference runs=3 median_simplify=15.13ms median_wire=15.17ms median_wall=57.63ms, product@0+100 #175 product runs=3 median_simplify=15.74ms median_wire=15.80ms median_wall=59.85ms, sum@0+100 #157 sum runs=3 median_simplify=10.01ms median_wire=10.07ms median_wall=38.18ms
- Steady-state dominant expressions: sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.86s | passed=450 failed=0 total=450 avg_case=6.356ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.51s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.92s | passed=1 failed=0 |
