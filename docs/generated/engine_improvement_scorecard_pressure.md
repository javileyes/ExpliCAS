# Engine Improvement Scorecard

- Generated: 2026-06-03T06:20:13.758804+00:00
- Git branch: main
- Git commit: `62c22a1a523a613031dadd13950766694e80acd0`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=618.63ms avg_case_ms=6.19 simplify=32.34ms avg_simplify_ms=0.32, sum total=200 failed=0 elapsed=544.61ms avg_case_ms=2.72 simplify=81.17ms avg_simplify_ms=0.41, product total=100 failed=0 elapsed=368.82ms avg_case_ms=3.69 simplify=24.40ms avg_simplify_ms=0.24, difference total=50 failed=0 elapsed=247.69ms avg_case_ms=4.95 simplify=28.47ms avg_simplify_ms=0.57
- Engine hotspots: sum simplify=81.17ms avg_simplify_ms=0.41 wall=544.61ms, shifted_quotient simplify=32.34ms avg_simplify_ms=0.32 wall=618.63ms, difference simplify=28.47ms avg_simplify_ms=0.57 wall=247.69ms, product simplify=24.40ms avg_simplify_ms=0.24 wall=368.82ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=618.63ms avg_case_ms=6.19 avg_simplify_ms=0.32, sum@0+100 failed=0 elapsed=389.75ms avg_case_ms=3.90 avg_simplify_ms=0.48, product@0+100 failed=0 elapsed=368.82ms avg_case_ms=3.69 avg_simplify_ms=0.24, difference@0+50 failed=0 elapsed=247.69ms avg_case_ms=4.95 avg_simplify_ms=0.57, sum@700+100 failed=0 elapsed=154.86ms avg_case_ms=1.55 avg_simplify_ms=0.33
- Steady-state engine reruns: sum@700+100 #2865 sum runs=3 median_simplify=5.37ms median_wire=5.41ms median_wall=5.71ms, sum@0+100 #53 sum runs=3 median_simplify=3.90ms median_wire=3.94ms median_wall=5.37ms, difference@0+50 #54 difference runs=3 median_simplify=3.64ms median_wire=3.68ms median_wall=5.10ms, sum@700+100 #3021 sum runs=3 median_simplify=3.46ms median_wire=3.49ms median_wall=3.78ms, sum@0+100 #209 sum runs=3 median_simplify=3.31ms median_wire=3.36ms median_wall=5.47ms
- Steady-state dominant expressions: sum@700+100 #2865 sum expr=(ln((x*y)^2) - ln(x^2) - ln(y^2)) + (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x))), sum@0+100 #53 sum expr=(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x))), difference@0+50 #54 difference expr=(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) - (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x)))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 1.78s | passed=450 failed=0 total=450 avg_case=3.956ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.60s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 17.55s | passed=1 failed=0 |
