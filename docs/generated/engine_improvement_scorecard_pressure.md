# Engine Improvement Scorecard

- Generated: 2026-05-30T12:01:49.004698+00:00
- Git branch: main
- Git commit: `206261b78bdf191ef8d1ae7ba244824a75ccc3e5`
- Profile: `pressure`

## Generated Discovery Ledger

- Purpose: keep failed generated candidates visible without promoting them to live corpus.
- Observe-only discoveries: total=2
- By area: calculus / differentiation:1, calculus / domain-condition display:1
- Recent 1: `calculus / domain-condition display` - 2026-05-28 - Observe-only discovery: condition display must not scale-normalize periodic arguments
- Recent 2: `calculus / differentiation` - 2026-05-28 - Discovery observe-only: atanh exact-square denominator scale hits depth overflow

## Calculus Support Matrix Signal

- Dimension: public calculus behavior, support-matrix coverage, result simplification, domain conditions, trace quality, presentation, and verification residuals.
- Interpretation: matrix-oriented calculus lanes; classify failures by command, family, argument regime, domain regime, trace regime, presentation regime, or reusable pre-calculus dependency before adding isolated cases.
- Matrix axes: command, family, argument regime, domain regime, trace regime, presentation regime, and residual verification.
- `diff_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=263
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=340

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=610.73ms avg_case_ms=6.11 simplify=30.84ms avg_simplify_ms=0.31, sum total=200 failed=0 elapsed=523.10ms avg_case_ms=2.62 simplify=77.02ms avg_simplify_ms=0.39, product total=100 failed=0 elapsed=340.57ms avg_case_ms=3.41 simplify=20.61ms avg_simplify_ms=0.21, difference total=50 failed=0 elapsed=239.58ms avg_case_ms=4.79 simplify=26.91ms avg_simplify_ms=0.54
- Engine hotspots: sum simplify=77.02ms avg_simplify_ms=0.39 wall=523.10ms, shifted_quotient simplify=30.84ms avg_simplify_ms=0.31 wall=610.73ms, difference simplify=26.91ms avg_simplify_ms=0.54 wall=239.58ms, product simplify=20.61ms avg_simplify_ms=0.21 wall=340.57ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=610.73ms avg_case_ms=6.11 avg_simplify_ms=0.31, sum@0+100 failed=0 elapsed=374.70ms avg_case_ms=3.75 avg_simplify_ms=0.46, product@0+100 failed=0 elapsed=340.57ms avg_case_ms=3.41 avg_simplify_ms=0.21, difference@0+50 failed=0 elapsed=239.58ms avg_case_ms=4.79 avg_simplify_ms=0.54, sum@700+100 failed=0 elapsed=148.40ms avg_case_ms=1.48 avg_simplify_ms=0.31
- Steady-state engine reruns: sum@700+100 #2865 sum runs=3 median_simplify=5.37ms median_wire=5.40ms median_wall=5.71ms, sum@0+100 #105 sum runs=3 median_simplify=2.53ms median_wire=2.56ms median_wall=3.94ms, sum@0+100 #53 sum runs=3 median_simplify=3.86ms median_wire=3.90ms median_wall=5.34ms, difference@0+50 #54 difference runs=3 median_simplify=3.75ms median_wire=3.79ms median_wall=5.17ms, sum@700+100 #3021 sum runs=3 median_simplify=3.39ms median_wire=3.43ms median_wall=3.71ms
- Steady-state dominant expressions: sum@700+100 #2865 sum expr=(ln((x*y)^2) - ln(x^2) - ln(y^2)) + (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x))), sum@0+100 #105 sum expr=(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (exp(x) - exp(-x) - 2*sinh(x)), sum@0+100 #53 sum expr=(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x)))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 1.71s | passed=450 failed=0 total=450 avg_case=3.800ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.50s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 17.22s | passed=1 failed=0 |
