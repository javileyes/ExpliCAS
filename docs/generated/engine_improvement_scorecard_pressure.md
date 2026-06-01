# Engine Improvement Scorecard

- Generated: 2026-06-01T05:36:53.219939+00:00
- Git branch: main
- Git commit: `990006e12568e9b737b3ff13b8dbcb7e5eee83ad`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=599.88ms avg_case_ms=6.00 simplify=30.79ms avg_simplify_ms=0.31, sum total=200 failed=0 elapsed=514.03ms avg_case_ms=2.57 simplify=73.84ms avg_simplify_ms=0.37, product total=100 failed=0 elapsed=337.82ms avg_case_ms=3.38 simplify=20.08ms avg_simplify_ms=0.20, difference total=50 failed=0 elapsed=238.57ms avg_case_ms=4.77 simplify=27.11ms avg_simplify_ms=0.54
- Engine hotspots: sum simplify=73.84ms avg_simplify_ms=0.37 wall=514.03ms, shifted_quotient simplify=30.79ms avg_simplify_ms=0.31 wall=599.88ms, difference simplify=27.11ms avg_simplify_ms=0.54 wall=238.57ms, product simplify=20.08ms avg_simplify_ms=0.20 wall=337.82ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=599.88ms avg_case_ms=6.00 avg_simplify_ms=0.31, sum@0+100 failed=0 elapsed=369.80ms avg_case_ms=3.70 avg_simplify_ms=0.43, product@0+100 failed=0 elapsed=337.82ms avg_case_ms=3.38 avg_simplify_ms=0.20, difference@0+50 failed=0 elapsed=238.57ms avg_case_ms=4.77 avg_simplify_ms=0.54, sum@700+100 failed=0 elapsed=144.23ms avg_case_ms=1.44 avg_simplify_ms=0.31
- Steady-state engine reruns: sum@700+100 #2865 sum runs=3 median_simplify=5.15ms median_wire=5.18ms median_wall=5.48ms, sum@0+100 #53 sum runs=3 median_simplify=3.77ms median_wire=3.81ms median_wall=5.17ms, difference@0+50 #54 difference runs=3 median_simplify=3.37ms median_wire=3.41ms median_wall=4.79ms, sum@0+100 #209 sum runs=3 median_simplify=3.17ms median_wire=3.22ms median_wall=5.24ms, sum@700+100 #3021 sum runs=3 median_simplify=3.32ms median_wire=3.35ms median_wall=3.64ms
- Steady-state dominant expressions: sum@700+100 #2865 sum expr=(ln((x*y)^2) - ln(x^2) - ln(y^2)) + (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x))), sum@0+100 #53 sum expr=(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x))), difference@0+50 #54 difference expr=(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) - (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x)))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 1.69s | passed=450 failed=0 total=450 avg_case=3.756ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.48s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 16.77s | passed=1 failed=0 |
