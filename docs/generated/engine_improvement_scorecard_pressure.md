# Engine Improvement Scorecard

- Generated: 2026-06-04T10:45:52.106404+00:00
- Git branch: main
- Git commit: `e7b3983de1c59cb3b9c1fd5358a2504dd11e3212`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=616.69ms avg_case_ms=6.17 simplify=31.78ms avg_simplify_ms=0.32, sum total=200 failed=0 elapsed=545.90ms avg_case_ms=2.73 simplify=80.52ms avg_simplify_ms=0.40, product total=100 failed=0 elapsed=348.55ms avg_case_ms=3.49 simplify=21.25ms avg_simplify_ms=0.21, difference total=50 failed=0 elapsed=245.43ms avg_case_ms=4.91 simplify=28.48ms avg_simplify_ms=0.57
- Engine hotspots: sum simplify=80.52ms avg_simplify_ms=0.40 wall=545.90ms, shifted_quotient simplify=31.78ms avg_simplify_ms=0.32 wall=616.69ms, difference simplify=28.48ms avg_simplify_ms=0.57 wall=245.43ms, product simplify=21.25ms avg_simplify_ms=0.21 wall=348.55ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=616.69ms avg_case_ms=6.17 avg_simplify_ms=0.32, sum@0+100 failed=0 elapsed=394.20ms avg_case_ms=3.94 avg_simplify_ms=0.48, product@0+100 failed=0 elapsed=348.55ms avg_case_ms=3.49 avg_simplify_ms=0.21, difference@0+50 failed=0 elapsed=245.43ms avg_case_ms=4.91 avg_simplify_ms=0.57, sum@700+100 failed=0 elapsed=151.70ms avg_case_ms=1.52 avg_simplify_ms=0.32
- Steady-state engine reruns: sum@700+100 #2865 sum runs=3 median_simplify=5.36ms median_wire=5.39ms median_wall=5.70ms, sum@0+100 #53 sum runs=3 median_simplify=3.89ms median_wire=3.93ms median_wall=5.42ms, sum@0+100 #209 sum runs=3 median_simplify=3.33ms median_wire=3.40ms median_wall=5.51ms, sum@700+100 #3021 sum runs=3 median_simplify=3.44ms median_wire=3.47ms median_wall=3.76ms, difference@0+50 #54 difference runs=3 median_simplify=3.61ms median_wire=3.65ms median_wall=5.18ms
- Steady-state dominant expressions: sum@700+100 #2865 sum expr=(ln((x*y)^2) - ln(x^2) - ln(y^2)) + (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x))), sum@0+100 #53 sum expr=(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x))), sum@0+100 #209 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x)))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 1.76s | passed=450 failed=0 total=450 avg_case=3.911ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.60s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 17.76s | passed=1 failed=0 |
