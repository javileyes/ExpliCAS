# Engine Improvement Scorecard

- Generated: 2026-05-27T21:45:05.693187+00:00
- Git branch: main
- Git commit: `e7d776c323596b41c78e328ca3946ae617cd2b63`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=624.46ms avg_case_ms=6.24 simplify=35.94ms avg_simplify_ms=0.36, sum total=200 failed=0 elapsed=526.16ms avg_case_ms=2.63 simplify=82.87ms avg_simplify_ms=0.41, product total=100 failed=0 elapsed=357.07ms avg_case_ms=3.57 simplify=24.72ms avg_simplify_ms=0.25, difference total=50 failed=0 elapsed=252.46ms avg_case_ms=5.05 simplify=31.57ms avg_simplify_ms=0.63
- Engine hotspots: sum simplify=82.87ms avg_simplify_ms=0.41 wall=526.16ms, shifted_quotient simplify=35.94ms avg_simplify_ms=0.36 wall=624.46ms, difference simplify=31.57ms avg_simplify_ms=0.63 wall=252.46ms, product simplify=24.72ms avg_simplify_ms=0.25 wall=357.07ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=624.46ms avg_case_ms=6.24 avg_simplify_ms=0.36, sum@0+100 failed=0 elapsed=375.26ms avg_case_ms=3.75 avg_simplify_ms=0.49, product@0+100 failed=0 elapsed=357.07ms avg_case_ms=3.57 avg_simplify_ms=0.25, difference@0+50 failed=0 elapsed=252.46ms avg_case_ms=5.05 avg_simplify_ms=0.63, sum@700+100 failed=0 elapsed=150.90ms avg_case_ms=1.51 avg_simplify_ms=0.34
- Steady-state engine reruns: sum@700+100 #2865 sum runs=3 median_simplify=5.75ms median_wire=5.79ms median_wall=6.14ms, sum@0+100 #53 sum runs=3 median_simplify=4.15ms median_wire=4.20ms median_wall=5.71ms, difference@0+50 #54 difference runs=3 median_simplify=3.91ms median_wire=3.96ms median_wall=5.51ms, sum@0+100 #209 sum runs=3 median_simplify=3.54ms median_wire=3.59ms median_wall=5.53ms, difference@0+50 #26 difference runs=3 median_simplify=2.61ms median_wire=2.64ms median_wall=4.07ms
- Steady-state dominant expressions: sum@700+100 #2865 sum expr=(ln((x*y)^2) - ln(x^2) - ln(y^2)) + (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x))), sum@0+100 #53 sum expr=(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x))), difference@0+50 #54 difference expr=(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) - (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x)))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 1.76s | passed=450 failed=0 total=450 avg_case=3.911ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.68s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 18.36s | passed=1 failed=0 |
