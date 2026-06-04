# Engine Improvement Scorecard

- Generated: 2026-06-04T13:59:46.751760+00:00
- Git branch: main
- Git commit: `01ff82245eb12cbc88e58e060c9414c22088d3ff`
- Profile: `pressure`

## Generated Discovery Ledger

- Purpose: keep failed generated candidates visible without promoting them to live corpus.
- Observe-only discoveries: total=3
- By area: calculus / differentiation:1, calculus / domain-condition display:1, calculus / integration:1
- Recent 1: `calculus / integration` - 2026-06-04 - Observe-only discovery: direct tanh symbolic scale lacks substitution substeps
- Recent 2: `calculus / domain-condition display` - 2026-05-28 - Observe-only discovery: condition display must not scale-normalize periodic arguments
- Recent 3: `calculus / differentiation` - 2026-05-28 - Discovery observe-only: atanh exact-square denominator scale hits depth overflow

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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=629.68ms avg_case_ms=6.30 simplify=34.23ms avg_simplify_ms=0.34, sum total=200 failed=0 elapsed=543.77ms avg_case_ms=2.72 simplify=82.50ms avg_simplify_ms=0.41, product total=100 failed=0 elapsed=357.54ms avg_case_ms=3.58 simplify=22.67ms avg_simplify_ms=0.23, difference total=50 failed=0 elapsed=245.31ms avg_case_ms=4.91 simplify=28.73ms avg_simplify_ms=0.57
- Engine hotspots: sum simplify=82.50ms avg_simplify_ms=0.41 wall=543.77ms, shifted_quotient simplify=34.23ms avg_simplify_ms=0.34 wall=629.68ms, difference simplify=28.73ms avg_simplify_ms=0.57 wall=245.31ms, product simplify=22.67ms avg_simplify_ms=0.23 wall=357.54ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=629.68ms avg_case_ms=6.30 avg_simplify_ms=0.34, sum@0+100 failed=0 elapsed=392.67ms avg_case_ms=3.93 avg_simplify_ms=0.50, product@0+100 failed=0 elapsed=357.54ms avg_case_ms=3.58 avg_simplify_ms=0.23, difference@0+50 failed=0 elapsed=245.31ms avg_case_ms=4.91 avg_simplify_ms=0.57, sum@700+100 failed=0 elapsed=151.09ms avg_case_ms=1.51 avg_simplify_ms=0.33
- Steady-state engine reruns: sum@700+100 #2865 sum runs=3 median_simplify=5.67ms median_wire=5.71ms median_wall=6.07ms, sum@0+100 #53 sum runs=3 median_simplify=3.97ms median_wire=4.02ms median_wall=5.53ms, sum@0+100 #25 sum runs=3 median_simplify=2.58ms median_wire=2.61ms median_wall=3.99ms, difference@0+50 #54 difference runs=3 median_simplify=3.73ms median_wire=3.77ms median_wall=5.18ms, sum@0+100 #209 sum runs=3 median_simplify=3.35ms median_wire=3.40ms median_wall=5.59ms
- Steady-state dominant expressions: sum@700+100 #2865 sum expr=(ln((x*y)^2) - ln(x^2) - ln(y^2)) + (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x))), sum@0+100 #53 sum expr=(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x))), sum@0+100 #25 sum expr=(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (cosh(x) + sinh(x) - e^x)

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 1.78s | passed=450 failed=0 total=450 avg_case=3.956ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.69s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 18.57s | passed=1 failed=0 |
