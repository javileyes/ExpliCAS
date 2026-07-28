# Engine Improvement Scorecard

- Generated: 2026-07-28T14:05:54.133332+00:00
- Git branch: main
- Git commit: `180a698a8ef636714e1328afdb073e2956b0d943`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=380

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=980.22ms avg_case_ms=9.80 simplify=274.86ms avg_simplify_ms=2.75, sum total=200 failed=0 elapsed=856.80ms avg_case_ms=4.28 simplify=278.73ms avg_simplify_ms=1.39, product total=100 failed=0 elapsed=610.18ms avg_case_ms=6.10 simplify=176.37ms avg_simplify_ms=1.76, difference total=50 failed=0 elapsed=399.11ms avg_case_ms=7.98 simplify=121.36ms avg_simplify_ms=2.43
- Engine hotspots: sum simplify=278.73ms avg_simplify_ms=1.39 wall=856.80ms, shifted_quotient simplify=274.86ms avg_simplify_ms=2.75 wall=980.22ms, product simplify=176.37ms avg_simplify_ms=1.76 wall=610.18ms, difference simplify=121.36ms avg_simplify_ms=2.43 wall=399.11ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=980.22ms avg_case_ms=9.80 avg_simplify_ms=2.75, sum@0+100 failed=0 elapsed=627.40ms avg_case_ms=6.27 avg_simplify_ms=1.97, product@0+100 failed=0 elapsed=610.18ms avg_case_ms=6.10 avg_simplify_ms=1.76, difference@0+50 failed=0 elapsed=399.11ms avg_case_ms=7.98 avg_simplify_ms=2.43, sum@700+100 failed=0 elapsed=229.41ms avg_case_ms=2.29 avg_simplify_ms=0.82
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.65ms median_wire=16.73ms median_wall=64.27ms, product@0+100 #175 product runs=3 median_simplify=15.36ms median_wire=15.41ms median_wall=59.31ms, difference@0+50 #174 difference runs=3 median_simplify=15.41ms median_wire=15.46ms median_wall=58.49ms, sum@0+100 #173 sum runs=3 median_simplify=15.17ms median_wire=15.22ms median_wall=58.38ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.99ms median_wire=13.06ms median_wall=49.62ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.85s | passed=450 failed=0 total=450 avg_case=6.333ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.38s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.85s | passed=1 failed=0 |
