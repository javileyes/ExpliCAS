# Engine Improvement Scorecard

- Generated: 2026-06-05T14:21:54.280889+00:00
- Git branch: main
- Git commit: `27b2122f3e3eace8301d06a4a45f3e23a6176028`
- Profile: `pressure`

## Generated Discovery Ledger

- Purpose: keep failed generated candidates visible without promoting them to live corpus.
- Observe-only discoveries: total=0
- Status: no open observe-only generated discoveries.

## Calculus Support Matrix Signal

- Dimension: public calculus behavior, support-matrix coverage, result simplification, domain conditions, trace quality, presentation, and verification residuals.
- Interpretation: matrix-oriented calculus lanes; classify failures by command, family, argument regime, domain regime, trace regime, presentation regime, or reusable pre-calculus dependency before adding isolated cases.
- Matrix axes: command, family, argument regime, domain regime, trace regime, presentation regime, and residual verification.
- `diff_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=263
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=343

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=600.08ms avg_case_ms=6.00 simplify=30.45ms avg_simplify_ms=0.30, sum total=200 failed=0 elapsed=524.33ms avg_case_ms=2.62 simplify=76.29ms avg_simplify_ms=0.38, product total=100 failed=0 elapsed=339.21ms avg_case_ms=3.39 simplify=20.49ms avg_simplify_ms=0.20, difference total=50 failed=0 elapsed=236.86ms avg_case_ms=4.74 simplify=26.86ms avg_simplify_ms=0.54
- Engine hotspots: sum simplify=76.29ms avg_simplify_ms=0.38 wall=524.33ms, shifted_quotient simplify=30.45ms avg_simplify_ms=0.30 wall=600.08ms, difference simplify=26.86ms avg_simplify_ms=0.54 wall=236.86ms, product simplify=20.49ms avg_simplify_ms=0.20 wall=339.21ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=600.08ms avg_case_ms=6.00 avg_simplify_ms=0.30, sum@0+100 failed=0 elapsed=373.46ms avg_case_ms=3.73 avg_simplify_ms=0.44, product@0+100 failed=0 elapsed=339.21ms avg_case_ms=3.39 avg_simplify_ms=0.20, difference@0+50 failed=0 elapsed=236.86ms avg_case_ms=4.74 avg_simplify_ms=0.54, sum@700+100 failed=0 elapsed=150.87ms avg_case_ms=1.51 avg_simplify_ms=0.32
- Steady-state engine reruns: sum@700+100 #2865 sum runs=3 median_simplify=5.28ms median_wire=5.32ms median_wall=5.63ms, sum@0+100 #53 sum runs=3 median_simplify=3.79ms median_wire=3.82ms median_wall=5.23ms, difference@0+50 #54 difference runs=3 median_simplify=3.62ms median_wire=3.66ms median_wall=5.05ms, sum@700+100 #3021 sum runs=3 median_simplify=3.43ms median_wire=3.47ms median_wall=3.77ms, sum@0+100 #209 sum runs=3 median_simplify=3.36ms median_wire=3.42ms median_wall=5.65ms
- Steady-state dominant expressions: sum@700+100 #2865 sum expr=(ln((x*y)^2) - ln(x^2) - ln(y^2)) + (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x))), sum@0+100 #53 sum expr=(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x))), difference@0+50 #54 difference expr=(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) - (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x)))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 1.70s | passed=450 failed=0 total=450 avg_case=3.778ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.48s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 17.09s | passed=1 failed=0 |
