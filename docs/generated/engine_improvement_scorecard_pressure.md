# Engine Improvement Scorecard

- Generated: 2026-05-25T17:31:16.905188+00:00
- Git branch: main
- Git commit: `0e0f8b17d110820555956b4c650b7ef98a3260ed`
- Profile: `pressure`

## Generated Discovery Ledger

- Purpose: keep failed generated candidates visible without promoting them to live corpus.
- Observe-only discoveries: total=4
- By area: calculus / limit:2, calculus / differentiation:1, orchestrator / exact-zero additive composition:1
- Recent 1: `calculus / differentiation` - 2026-05-25 - Observe-only discovery: inverse-tangent symbolic denominator shortcut leaves condition and external-scale gaps
- Recent 2: `calculus / limit` - 2026-05-24 - Observe-only discovery: limit command residuals cannot enter algebraic wrapper matrix
- Recent 3: `orchestrator / exact-zero additive composition` - 2026-05-24 - Discovery observe-only: hyperbolic angle-sum plus telescoping residual remains slow

## Calculus Support Matrix Signal

- Dimension: public calculus behavior, support-matrix coverage, result simplification, domain conditions, trace quality, presentation, and verification residuals.
- Interpretation: matrix-oriented calculus lanes; classify failures by command, family, argument regime, domain regime, trace regime, presentation regime, or reusable pre-calculus dependency before adding isolated cases.
- Matrix axes: command, family, argument regime, domain regime, trace regime, presentation regime, and residual verification.
- `diff_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=259
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=340

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=633.14ms avg_case_ms=6.33 simplify=36.89ms avg_simplify_ms=0.37, sum total=200 failed=0 elapsed=544.29ms avg_case_ms=2.72 simplify=84.06ms avg_simplify_ms=0.42, product total=100 failed=0 elapsed=360.42ms avg_case_ms=3.60 simplify=24.23ms avg_simplify_ms=0.24, difference total=50 failed=0 elapsed=255.19ms avg_case_ms=5.10 simplify=30.93ms avg_simplify_ms=0.62
- Engine hotspots: sum simplify=84.06ms avg_simplify_ms=0.42 wall=544.29ms, shifted_quotient simplify=36.89ms avg_simplify_ms=0.37 wall=633.14ms, difference simplify=30.93ms avg_simplify_ms=0.62 wall=255.19ms, product simplify=24.23ms avg_simplify_ms=0.24 wall=360.42ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=633.14ms avg_case_ms=6.33 avg_simplify_ms=0.37, sum@0+100 failed=0 elapsed=387.66ms avg_case_ms=3.88 avg_simplify_ms=0.50, product@0+100 failed=0 elapsed=360.42ms avg_case_ms=3.60 avg_simplify_ms=0.24, difference@0+50 failed=0 elapsed=255.19ms avg_case_ms=5.10 avg_simplify_ms=0.62, sum@700+100 failed=0 elapsed=156.62ms avg_case_ms=1.57 avg_simplify_ms=0.35
- Steady-state engine reruns: sum@700+100 #2865 sum runs=3 median_simplify=6.34ms median_wire=6.39ms median_wall=6.75ms, sum@0+100 #53 sum runs=3 median_simplify=4.24ms median_wire=4.29ms median_wall=5.97ms, sum@700+100 #3021 sum runs=3 median_simplify=3.57ms median_wire=3.61ms median_wall=3.90ms, difference@0+50 #54 difference runs=3 median_simplify=3.84ms median_wire=3.89ms median_wall=5.33ms, sum@0+100 #25 sum runs=3 median_simplify=2.75ms median_wire=2.77ms median_wall=4.34ms
- Steady-state dominant expressions: sum@700+100 #2865 sum expr=(ln((x*y)^2) - ln(x^2) - ln(y^2)) + (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x))), sum@0+100 #53 sum expr=(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x))), sum@700+100 #3021 sum expr=(2*ln(abs(x*y)) - 2*ln(abs(x)) - 2*ln(abs(y))) + (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x)))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 1.79s | passed=450 failed=0 total=450 avg_case=3.978ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.71s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 18.57s | passed=1 failed=0 |
