# Engine Improvement Scorecard

- Generated: 2026-05-25T05:06:17.668196+00:00
- Git branch: main
- Git commit: `d8054a8f2291f09fb0ccb7cd5ac856107acc8069`
- Profile: `pressure`

## Generated Discovery Ledger

- Purpose: keep failed generated candidates visible without promoting them to live corpus.
- Observe-only discoveries: total=2
- By area: calculus / limit:1, orchestrator / exact-zero additive composition:1
- Recent 1: `calculus / limit` - 2026-05-24 - Observe-only discovery: limit command residuals cannot enter algebraic wrapper matrix
- Recent 2: `orchestrator / exact-zero additive composition` - 2026-05-24 - Discovery observe-only: hyperbolic angle-sum plus telescoping residual remains slow

## Calculus Support Matrix Signal

- Dimension: public calculus behavior, support-matrix coverage, result simplification, domain conditions, trace quality, presentation, and verification residuals.
- Interpretation: matrix-oriented calculus lanes; classify failures by command, family, argument regime, domain regime, trace regime, presentation regime, or reusable pre-calculus dependency before adding isolated cases.
- Matrix axes: command, family, argument regime, domain regime, trace regime, presentation regime, and residual verification.
- `diff_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=259
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=338

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=568.99ms avg_case_ms=5.69 simplify=34.18ms avg_simplify_ms=0.34, sum total=200 failed=0 elapsed=501.36ms avg_case_ms=2.51 simplify=83.50ms avg_simplify_ms=0.42, product total=100 failed=0 elapsed=338.60ms avg_case_ms=3.39 simplify=24.59ms avg_simplify_ms=0.25, difference total=50 failed=0 elapsed=235.24ms avg_case_ms=4.70 simplify=30.39ms avg_simplify_ms=0.61
- Engine hotspots: sum simplify=83.50ms avg_simplify_ms=0.42 wall=501.36ms, shifted_quotient simplify=34.18ms avg_simplify_ms=0.34 wall=568.99ms, difference simplify=30.39ms avg_simplify_ms=0.61 wall=235.24ms, product simplify=24.59ms avg_simplify_ms=0.25 wall=338.60ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=568.99ms avg_case_ms=5.69 avg_simplify_ms=0.34, sum@0+100 failed=0 elapsed=357.81ms avg_case_ms=3.58 avg_simplify_ms=0.49, product@0+100 failed=0 elapsed=338.60ms avg_case_ms=3.39 avg_simplify_ms=0.25, difference@0+50 failed=0 elapsed=235.24ms avg_case_ms=4.70 avg_simplify_ms=0.61, sum@700+100 failed=0 elapsed=143.55ms avg_case_ms=1.44 avg_simplify_ms=0.35
- Steady-state engine reruns: sum@700+100 #2865 sum runs=3 median_simplify=5.80ms median_wire=5.85ms median_wall=6.20ms, sum@0+100 #53 sum runs=3 median_simplify=4.15ms median_wire=4.20ms median_wall=5.47ms, difference@0+50 #54 difference runs=3 median_simplify=3.90ms median_wire=3.94ms median_wall=5.24ms, sum@700+100 #3021 sum runs=3 median_simplify=3.56ms median_wire=3.60ms median_wall=3.90ms, sum@0+100 #209 sum runs=3 median_simplify=3.35ms median_wire=3.41ms median_wall=5.69ms
- Steady-state dominant expressions: sum@700+100 #2865 sum expr=(ln((x*y)^2) - ln(x^2) - ln(y^2)) + (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x))), sum@0+100 #53 sum expr=(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x))), difference@0+50 #54 difference expr=(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) - (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x)))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 1.64s | passed=450 failed=0 total=450 avg_case=3.644ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.61s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 18.20s | passed=1 failed=0 |
