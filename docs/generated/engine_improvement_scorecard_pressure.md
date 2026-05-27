# Engine Improvement Scorecard

- Generated: 2026-05-27T18:00:35.829959+00:00
- Git branch: main
- Git commit: `ddf150eb16dfa271adcf4cb383ccea18a42ac054`
- Profile: `pressure`

## Generated Discovery Ledger

- Purpose: keep failed generated candidates visible without promoting them to live corpus.
- Observe-only discoveries: total=0
- Status: no open observe-only generated discoveries.

## Calculus Support Matrix Signal

- Dimension: public calculus behavior, support-matrix coverage, result simplification, domain conditions, trace quality, presentation, and verification residuals.
- Interpretation: matrix-oriented calculus lanes; classify failures by command, family, argument regime, domain regime, trace regime, presentation regime, or reusable pre-calculus dependency before adding isolated cases.
- Matrix axes: command, family, argument regime, domain regime, trace regime, presentation regime, and residual verification.
- `diff_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=261
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=340

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=583.34ms avg_case_ms=5.83 simplify=31.22ms avg_simplify_ms=0.31, sum total=200 failed=0 elapsed=507.68ms avg_case_ms=2.54 simplify=75.77ms avg_simplify_ms=0.38, product total=100 failed=0 elapsed=329.07ms avg_case_ms=3.29 simplify=20.02ms avg_simplify_ms=0.20, difference total=50 failed=0 elapsed=237.41ms avg_case_ms=4.75 simplify=27.06ms avg_simplify_ms=0.54
- Engine hotspots: sum simplify=75.77ms avg_simplify_ms=0.38 wall=507.68ms, shifted_quotient simplify=31.22ms avg_simplify_ms=0.31 wall=583.34ms, difference simplify=27.06ms avg_simplify_ms=0.54 wall=237.41ms, product simplify=20.02ms avg_simplify_ms=0.20 wall=329.07ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=583.34ms avg_case_ms=5.83 avg_simplify_ms=0.31, sum@0+100 failed=0 elapsed=356.69ms avg_case_ms=3.57 avg_simplify_ms=0.44, product@0+100 failed=0 elapsed=329.07ms avg_case_ms=3.29 avg_simplify_ms=0.20, difference@0+50 failed=0 elapsed=237.41ms avg_case_ms=4.75 avg_simplify_ms=0.54, sum@700+100 failed=0 elapsed=150.99ms avg_case_ms=1.51 avg_simplify_ms=0.32
- Steady-state engine reruns: sum@700+100 #2865 sum runs=3 median_simplify=5.28ms median_wire=5.31ms median_wall=5.60ms, sum@0+100 #53 sum runs=3 median_simplify=4.09ms median_wire=4.13ms median_wall=5.59ms, sum@700+100 #3021 sum runs=3 median_simplify=3.41ms median_wire=3.45ms median_wall=3.73ms, difference@0+50 #54 difference runs=3 median_simplify=3.41ms median_wire=3.45ms median_wall=4.85ms, sum@0+100 #209 sum runs=3 median_simplify=3.25ms median_wire=3.31ms median_wall=5.18ms
- Steady-state dominant expressions: sum@700+100 #2865 sum expr=(ln((x*y)^2) - ln(x^2) - ln(y^2)) + (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x))), sum@0+100 #53 sum expr=(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x))), sum@700+100 #3021 sum expr=(2*ln(abs(x*y)) - 2*ln(abs(x)) - 2*ln(abs(y))) + (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x)))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 1.66s | passed=450 failed=0 total=450 avg_case=3.689ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.50s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 17.27s | passed=1 failed=0 |
