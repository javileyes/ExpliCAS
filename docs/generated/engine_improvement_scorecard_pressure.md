# Engine Improvement Scorecard

- Generated: 2026-05-31T11:26:12.675660+00:00
- Git branch: main
- Git commit: `9e8659da74cc04894f131e276b62ce0fa1860e66`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=614.08ms avg_case_ms=6.14 simplify=31.06ms avg_simplify_ms=0.31, sum total=200 failed=0 elapsed=516.41ms avg_case_ms=2.58 simplify=75.85ms avg_simplify_ms=0.38, product total=100 failed=0 elapsed=343.33ms avg_case_ms=3.43 simplify=21.00ms avg_simplify_ms=0.21, difference total=50 failed=0 elapsed=241.89ms avg_case_ms=4.84 simplify=27.48ms avg_simplify_ms=0.55
- Engine hotspots: sum simplify=75.85ms avg_simplify_ms=0.38 wall=516.41ms, shifted_quotient simplify=31.06ms avg_simplify_ms=0.31 wall=614.08ms, difference simplify=27.48ms avg_simplify_ms=0.55 wall=241.89ms, product simplify=21.00ms avg_simplify_ms=0.21 wall=343.33ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=614.08ms avg_case_ms=6.14 avg_simplify_ms=0.31, sum@0+100 failed=0 elapsed=366.58ms avg_case_ms=3.67 avg_simplify_ms=0.44, product@0+100 failed=0 elapsed=343.33ms avg_case_ms=3.43 avg_simplify_ms=0.21, difference@0+50 failed=0 elapsed=241.89ms avg_case_ms=4.84 avg_simplify_ms=0.55, sum@700+100 failed=0 elapsed=149.83ms avg_case_ms=1.50 avg_simplify_ms=0.31
- Steady-state engine reruns: sum@700+100 #2865 sum runs=3 median_simplify=5.27ms median_wire=5.30ms median_wall=5.60ms, sum@0+100 #53 sum runs=3 median_simplify=3.86ms median_wire=3.89ms median_wall=5.30ms, sum@0+100 #25 sum runs=3 median_simplify=2.41ms median_wire=2.43ms median_wall=3.79ms, difference@0+50 #54 difference runs=3 median_simplify=3.68ms median_wire=3.72ms median_wall=5.12ms, sum@0+100 #209 sum runs=3 median_simplify=3.21ms median_wire=3.26ms median_wall=5.30ms
- Steady-state dominant expressions: sum@700+100 #2865 sum expr=(ln((x*y)^2) - ln(x^2) - ln(y^2)) + (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x))), sum@0+100 #53 sum expr=(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x))), sum@0+100 #25 sum expr=(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (cosh(x) + sinh(x) - e^x)

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 1.72s | passed=450 failed=0 total=450 avg_case=3.822ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.53s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 17.15s | passed=1 failed=0 |
