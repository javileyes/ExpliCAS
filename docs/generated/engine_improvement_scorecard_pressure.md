# Engine Improvement Scorecard

- Generated: 2026-06-11T14:12:39.957957+00:00
- Git branch: main
- Git commit: `2910e9be8bebcc38d4d7fdd4261654cb3f8a2a72`
- Profile: `pressure`

## Generated Discovery Ledger

- Purpose: keep failed generated candidates visible without promoting them to live corpus.
- Observe-only discoveries: total=10
- By area: calculus / integration:4, calculus / runtime:3, calculus / differentiation:2, calculus / robustness:1
- Recent 1: `calculus / integration` - 2026-06-08 - Discovery observe-only: polynomial cosecant/cotangent source-return still emits depth pressure
- Recent 2: `calculus / differentiation` - 2026-06-06 - Observe-only discovery: exact-square atanh scaled-root runtime is not caused by the global empty-domain check
- Recent 3: `calculus / differentiation` - 2026-06-06 - Observe-only discovery: exact-square inverse-root diff runtime is not fixed by raw target preservation

## Calculus Support Matrix Signal

- Dimension: public calculus behavior, support-matrix coverage, result simplification, domain conditions, trace quality, presentation, and verification residuals.
- Interpretation: matrix-oriented calculus lanes; classify failures by command, family, argument regime, domain regime, trace regime, presentation regime, or reusable pre-calculus dependency before adding isolated cases.
- Matrix axes: command, family, argument regime, domain regime, trace regime, presentation regime, and residual verification.
- `diff_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=263
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=344

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=765.97ms avg_case_ms=7.66 simplify=215.49ms avg_simplify_ms=2.15, sum total=200 failed=0 elapsed=705.07ms avg_case_ms=3.53 simplify=234.06ms avg_simplify_ms=1.17, product total=100 failed=0 elapsed=473.67ms avg_case_ms=4.74 simplify=135.80ms avg_simplify_ms=1.36, difference total=50 failed=0 elapsed=320.53ms avg_case_ms=6.41 simplify=101.46ms avg_simplify_ms=2.03
- Engine hotspots: sum simplify=234.06ms avg_simplify_ms=1.17 wall=705.07ms, shifted_quotient simplify=215.49ms avg_simplify_ms=2.15 wall=765.97ms, product simplify=135.80ms avg_simplify_ms=1.36 wall=473.67ms, difference simplify=101.46ms avg_simplify_ms=2.03 wall=320.53ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=765.97ms avg_case_ms=7.66 avg_simplify_ms=2.15, sum@0+100 failed=0 elapsed=514.47ms avg_case_ms=5.14 avg_simplify_ms=1.63, product@0+100 failed=0 elapsed=473.67ms avg_case_ms=4.74 avg_simplify_ms=1.36, difference@0+50 failed=0 elapsed=320.53ms avg_case_ms=6.41 avg_simplify_ms=2.03, sum@700+100 failed=0 elapsed=190.60ms avg_case_ms=1.91 avg_simplify_ms=0.71
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.93ms median_wire=12.99ms median_wall=55.12ms, sum@0+100 #173 sum runs=3 median_simplify=11.33ms median_wire=11.38ms median_wall=43.40ms, product@0+100 #175 product runs=3 median_simplify=11.30ms median_wire=11.35ms median_wall=43.52ms, difference@0+50 #174 difference runs=3 median_simplify=11.74ms median_wire=11.78ms median_wall=44.69ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=11.45ms median_wire=11.53ms median_wall=40.79ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.27s | passed=450 failed=0 total=450 avg_case=5.044ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.83s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 17.48s | passed=1 failed=0 |
