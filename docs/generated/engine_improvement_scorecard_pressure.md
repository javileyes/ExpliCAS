# Engine Improvement Scorecard

- Generated: 2026-05-29T12:54:52.046093+00:00
- Git branch: main
- Git commit: `f965b2ec0dae3215726ba7c8f354449102406433`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=652.28ms avg_case_ms=6.52 simplify=36.19ms avg_simplify_ms=0.36, sum total=200 failed=0 elapsed=554.21ms avg_case_ms=2.77 simplify=83.79ms avg_simplify_ms=0.42, product total=100 failed=0 elapsed=368.36ms avg_case_ms=3.68 simplify=25.38ms avg_simplify_ms=0.25, difference total=50 failed=0 elapsed=258.96ms avg_case_ms=5.18 simplify=31.56ms avg_simplify_ms=0.63
- Engine hotspots: sum simplify=83.79ms avg_simplify_ms=0.42 wall=554.21ms, shifted_quotient simplify=36.19ms avg_simplify_ms=0.36 wall=652.28ms, difference simplify=31.56ms avg_simplify_ms=0.63 wall=258.96ms, product simplify=25.38ms avg_simplify_ms=0.25 wall=368.36ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=652.28ms avg_case_ms=6.52 avg_simplify_ms=0.36, sum@0+100 failed=0 elapsed=391.79ms avg_case_ms=3.92 avg_simplify_ms=0.48, product@0+100 failed=0 elapsed=368.36ms avg_case_ms=3.68 avg_simplify_ms=0.25, difference@0+50 failed=0 elapsed=258.96ms avg_case_ms=5.18 avg_simplify_ms=0.63, sum@700+100 failed=0 elapsed=162.41ms avg_case_ms=1.62 avg_simplify_ms=0.36
- Steady-state engine reruns: sum@700+100 #2865 sum runs=3 median_simplify=5.63ms median_wire=5.67ms median_wall=6.01ms, difference@0+50 #54 difference runs=3 median_simplify=3.83ms median_wire=3.88ms median_wall=5.37ms, sum@0+100 #53 sum runs=3 median_simplify=4.10ms median_wire=4.15ms median_wall=5.62ms, sum@700+100 #3021 sum runs=3 median_simplify=3.73ms median_wire=3.78ms median_wall=4.08ms, sum@0+100 #209 sum runs=3 median_simplify=3.69ms median_wire=3.74ms median_wall=5.99ms
- Steady-state dominant expressions: sum@700+100 #2865 sum expr=(ln((x*y)^2) - ln(x^2) - ln(y^2)) + (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x))), difference@0+50 #54 difference expr=(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) - (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x))), sum@0+100 #53 sum expr=(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x)))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 1.83s | passed=450 failed=0 total=450 avg_case=4.067ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.72s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 18.50s | passed=1 failed=0 |
