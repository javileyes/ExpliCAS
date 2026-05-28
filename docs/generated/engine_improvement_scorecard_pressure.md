# Engine Improvement Scorecard

- Generated: 2026-05-28T11:16:00.170347+00:00
- Git branch: main
- Git commit: `eaf8a51b675e5f50c28332d85ad4b79e7b1fbf78`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=674.01ms avg_case_ms=6.74 simplify=39.65ms avg_simplify_ms=0.40, sum total=200 failed=0 elapsed=620.93ms avg_case_ms=3.10 simplify=102.23ms avg_simplify_ms=0.51, product total=100 failed=0 elapsed=381.07ms avg_case_ms=3.81 simplify=27.88ms avg_simplify_ms=0.28, difference total=50 failed=0 elapsed=288.42ms avg_case_ms=5.77 simplify=38.63ms avg_simplify_ms=0.77
- Engine hotspots: sum simplify=102.23ms avg_simplify_ms=0.51 wall=620.93ms, shifted_quotient simplify=39.65ms avg_simplify_ms=0.40 wall=674.01ms, difference simplify=38.63ms avg_simplify_ms=0.77 wall=288.42ms, product simplify=27.88ms avg_simplify_ms=0.28 wall=381.07ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=674.01ms avg_case_ms=6.74 avg_simplify_ms=0.40, sum@0+100 failed=0 elapsed=435.81ms avg_case_ms=4.36 avg_simplify_ms=0.58, product@0+100 failed=0 elapsed=381.07ms avg_case_ms=3.81 avg_simplify_ms=0.28, difference@0+50 failed=0 elapsed=288.42ms avg_case_ms=5.77 avg_simplify_ms=0.77, sum@700+100 failed=0 elapsed=185.13ms avg_case_ms=1.85 avg_simplify_ms=0.44
- Steady-state engine reruns: sum@700+100 #2865 sum runs=3 median_simplify=6.49ms median_wire=6.54ms median_wall=6.87ms, sum@0+100 #53 sum runs=3 median_simplify=4.47ms median_wire=4.52ms median_wall=6.10ms, sum@700+100 #3021 sum runs=3 median_simplify=4.36ms median_wire=4.40ms median_wall=4.76ms, difference@0+50 #26 difference runs=3 median_simplify=3.01ms median_wire=3.05ms median_wall=4.62ms, difference@0+50 #86 difference runs=3 median_simplify=3.23ms median_wire=3.27ms median_wall=4.82ms
- Steady-state dominant expressions: sum@700+100 #2865 sum expr=(ln((x*y)^2) - ln(x^2) - ln(y^2)) + (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x))), sum@0+100 #53 sum expr=(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x))), sum@700+100 #3021 sum expr=(2*ln(abs(x*y)) - 2*ln(abs(x)) - 2*ln(abs(y))) + (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x)))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 1.97s | passed=450 failed=0 total=450 avg_case=4.378ms |
| `calculus_diff_exhaustive_contract` | `pass` | 3.07s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 22.38s | passed=1 failed=0 |
