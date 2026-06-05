# Engine Improvement Scorecard

- Generated: 2026-06-05T09:14:06.162732+00:00
- Git branch: main
- Git commit: `b25f8d56d00e782611d844cd38aeb755c22c5b5c`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=604.03ms avg_case_ms=6.04 simplify=30.73ms avg_simplify_ms=0.31, sum total=200 failed=0 elapsed=529.51ms avg_case_ms=2.65 simplify=78.42ms avg_simplify_ms=0.39, product total=100 failed=0 elapsed=339.66ms avg_case_ms=3.40 simplify=20.33ms avg_simplify_ms=0.20, difference total=50 failed=0 elapsed=238.16ms avg_case_ms=4.76 simplify=27.28ms avg_simplify_ms=0.55
- Engine hotspots: sum simplify=78.42ms avg_simplify_ms=0.39 wall=529.51ms, shifted_quotient simplify=30.73ms avg_simplify_ms=0.31 wall=604.03ms, difference simplify=27.28ms avg_simplify_ms=0.55 wall=238.16ms, product simplify=20.33ms avg_simplify_ms=0.20 wall=339.66ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=604.03ms avg_case_ms=6.04 avg_simplify_ms=0.31, sum@0+100 failed=0 elapsed=383.96ms avg_case_ms=3.84 avg_simplify_ms=0.47, product@0+100 failed=0 elapsed=339.66ms avg_case_ms=3.40 avg_simplify_ms=0.20, difference@0+50 failed=0 elapsed=238.16ms avg_case_ms=4.76 avg_simplify_ms=0.55, sum@700+100 failed=0 elapsed=145.55ms avg_case_ms=1.46 avg_simplify_ms=0.31
- Steady-state engine reruns: sum@700+100 #2865 sum runs=3 median_simplify=5.11ms median_wire=5.15ms median_wall=5.44ms, sum@0+100 #53 sum runs=3 median_simplify=3.71ms median_wire=3.75ms median_wall=5.12ms, sum@0+100 #209 sum runs=3 median_simplify=3.18ms median_wire=3.22ms median_wall=5.27ms, difference@0+50 #54 difference runs=3 median_simplify=3.43ms median_wire=3.47ms median_wall=4.85ms, sum@700+100 #3021 sum runs=3 median_simplify=3.38ms median_wire=3.42ms median_wall=3.69ms
- Steady-state dominant expressions: sum@700+100 #2865 sum expr=(ln((x*y)^2) - ln(x^2) - ln(y^2)) + (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x))), sum@0+100 #53 sum expr=(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x))), sum@0+100 #209 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x)))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 1.71s | passed=450 failed=0 total=450 avg_case=3.800ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.50s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 17.05s | passed=1 failed=0 |
