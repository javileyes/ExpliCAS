# Engine Improvement Scorecard

- Generated: 2026-06-21T00:15:15.515931+00:00
- Git branch: main
- Git commit: `e0044062703bfecd69d1a3365bc237f22fcc8607`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=353

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=785.45ms avg_case_ms=7.85 simplify=224.94ms avg_simplify_ms=2.25, sum total=200 failed=0 elapsed=700.88ms avg_case_ms=3.50 simplify=237.48ms avg_simplify_ms=1.19, product total=100 failed=0 elapsed=477.17ms avg_case_ms=4.77 simplify=137.04ms avg_simplify_ms=1.37, difference total=50 failed=0 elapsed=326.54ms avg_case_ms=6.53 simplify=104.76ms avg_simplify_ms=2.10
- Engine hotspots: sum simplify=237.48ms avg_simplify_ms=1.19 wall=700.88ms, shifted_quotient simplify=224.94ms avg_simplify_ms=2.25 wall=785.45ms, product simplify=137.04ms avg_simplify_ms=1.37 wall=477.17ms, difference simplify=104.76ms avg_simplify_ms=2.10 wall=326.54ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=785.45ms avg_case_ms=7.85 avg_simplify_ms=2.25, sum@0+100 failed=0 elapsed=504.96ms avg_case_ms=5.05 avg_simplify_ms=1.64, product@0+100 failed=0 elapsed=477.17ms avg_case_ms=4.77 avg_simplify_ms=1.37, difference@0+50 failed=0 elapsed=326.54ms avg_case_ms=6.53 avg_simplify_ms=2.10, sum@700+100 failed=0 elapsed=195.92ms avg_case_ms=1.96 avg_simplify_ms=0.73
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.01ms median_wire=13.09ms median_wall=49.13ms, difference@0+50 #174 difference runs=3 median_simplify=11.65ms median_wire=11.70ms median_wall=44.09ms, product@0+100 #175 product runs=3 median_simplify=11.63ms median_wire=11.68ms median_wall=44.38ms, sum@0+100 #173 sum runs=3 median_simplify=11.87ms median_wire=11.92ms median_wall=44.71ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.54ms median_wire=10.62ms median_wall=40.17ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.29s | passed=450 failed=0 total=450 avg_case=5.089ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.43s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.03s | passed=1 failed=0 |
