# Engine Improvement Scorecard

- Generated: 2026-06-03T11:51:52.458044+00:00
- Git branch: main
- Git commit: `fc8db9293b134ded9d91c6a6b0687af2333ebcfa`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=604.25ms avg_case_ms=6.04 simplify=30.78ms avg_simplify_ms=0.31, sum total=200 failed=0 elapsed=530.06ms avg_case_ms=2.65 simplify=77.57ms avg_simplify_ms=0.39, product total=100 failed=0 elapsed=341.80ms avg_case_ms=3.42 simplify=20.46ms avg_simplify_ms=0.20, difference total=50 failed=0 elapsed=239.20ms avg_case_ms=4.78 simplify=27.50ms avg_simplify_ms=0.55
- Engine hotspots: sum simplify=77.57ms avg_simplify_ms=0.39 wall=530.06ms, shifted_quotient simplify=30.78ms avg_simplify_ms=0.31 wall=604.25ms, difference simplify=27.50ms avg_simplify_ms=0.55 wall=239.20ms, product simplify=20.46ms avg_simplify_ms=0.20 wall=341.80ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=604.25ms avg_case_ms=6.04 avg_simplify_ms=0.31, sum@0+100 failed=0 elapsed=385.04ms avg_case_ms=3.85 avg_simplify_ms=0.47, product@0+100 failed=0 elapsed=341.80ms avg_case_ms=3.42 avg_simplify_ms=0.20, difference@0+50 failed=0 elapsed=239.20ms avg_case_ms=4.78 avg_simplify_ms=0.55, sum@700+100 failed=0 elapsed=145.01ms avg_case_ms=1.45 avg_simplify_ms=0.31
- Steady-state engine reruns: sum@700+100 #2865 sum runs=3 median_simplify=5.26ms median_wire=5.30ms median_wall=5.60ms, sum@0+100 #209 sum runs=3 median_simplify=3.23ms median_wire=3.28ms median_wall=5.34ms, sum@0+100 #53 sum runs=3 median_simplify=3.76ms median_wire=3.80ms median_wall=5.23ms, difference@0+50 #54 difference runs=3 median_simplify=3.38ms median_wire=3.42ms median_wall=4.81ms, sum@700+100 #3021 sum runs=3 median_simplify=3.37ms median_wire=3.40ms median_wall=3.69ms
- Steady-state dominant expressions: sum@700+100 #2865 sum expr=(ln((x*y)^2) - ln(x^2) - ln(y^2)) + (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x))), sum@0+100 #209 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x))), sum@0+100 #53 sum expr=(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x)))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 1.72s | passed=450 failed=0 total=450 avg_case=3.822ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.53s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 17.21s | passed=1 failed=0 |
