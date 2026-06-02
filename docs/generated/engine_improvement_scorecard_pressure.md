# Engine Improvement Scorecard

- Generated: 2026-06-02T20:56:20.831021+00:00
- Git branch: main
- Git commit: `af63536e6f8cf69fd724a1616bf9675d905eed38`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=628.19ms avg_case_ms=6.28 simplify=32.41ms avg_simplify_ms=0.32, sum total=200 failed=0 elapsed=534.23ms avg_case_ms=2.67 simplify=80.22ms avg_simplify_ms=0.40, product total=100 failed=0 elapsed=353.80ms avg_case_ms=3.54 simplify=22.15ms avg_simplify_ms=0.22, difference total=50 failed=0 elapsed=244.65ms avg_case_ms=4.89 simplify=28.56ms avg_simplify_ms=0.57
- Engine hotspots: sum simplify=80.22ms avg_simplify_ms=0.40 wall=534.23ms, shifted_quotient simplify=32.41ms avg_simplify_ms=0.32 wall=628.19ms, difference simplify=28.56ms avg_simplify_ms=0.57 wall=244.65ms, product simplify=22.15ms avg_simplify_ms=0.22 wall=353.80ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=628.19ms avg_case_ms=6.28 avg_simplify_ms=0.32, sum@0+100 failed=0 elapsed=382.83ms avg_case_ms=3.83 avg_simplify_ms=0.48, product@0+100 failed=0 elapsed=353.80ms avg_case_ms=3.54 avg_simplify_ms=0.22, difference@0+50 failed=0 elapsed=244.65ms avg_case_ms=4.89 avg_simplify_ms=0.57, sum@700+100 failed=0 elapsed=151.40ms avg_case_ms=1.51 avg_simplify_ms=0.32
- Steady-state engine reruns: sum@700+100 #2865 sum runs=3 median_simplify=5.67ms median_wire=5.72ms median_wall=6.02ms, sum@0+100 #53 sum runs=3 median_simplify=4.01ms median_wire=4.06ms median_wall=5.49ms, sum@0+100 #209 sum runs=3 median_simplify=3.50ms median_wire=3.55ms median_wall=5.70ms, difference@0+50 #54 difference runs=3 median_simplify=3.74ms median_wire=3.78ms median_wall=5.30ms, sum@0+100 #25 sum runs=3 median_simplify=2.61ms median_wire=2.63ms median_wall=4.12ms
- Steady-state dominant expressions: sum@700+100 #2865 sum expr=(ln((x*y)^2) - ln(x^2) - ln(y^2)) + (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x))), sum@0+100 #53 sum expr=(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x))), sum@0+100 #209 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x)))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 1.76s | passed=450 failed=0 total=450 avg_case=3.911ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.69s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 17.83s | passed=1 failed=0 |
