# Engine Improvement Scorecard

- Generated: 2026-07-28T21:16:02.010741+00:00
- Git branch: main
- Git commit: `612255c39c8695ede712b785a8cb295a6d86d842`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=980.48ms avg_case_ms=9.80 simplify=278.41ms avg_simplify_ms=2.78, sum total=200 failed=0 elapsed=846.25ms avg_case_ms=4.23 simplify=275.32ms avg_simplify_ms=1.38, product total=100 failed=0 elapsed=602.59ms avg_case_ms=6.03 simplify=173.56ms avg_simplify_ms=1.74, difference total=50 failed=0 elapsed=395.82ms avg_case_ms=7.92 simplify=121.45ms avg_simplify_ms=2.43
- Engine hotspots: shifted_quotient simplify=278.41ms avg_simplify_ms=2.78 wall=980.48ms, sum simplify=275.32ms avg_simplify_ms=1.38 wall=846.25ms, product simplify=173.56ms avg_simplify_ms=1.74 wall=602.59ms, difference simplify=121.45ms avg_simplify_ms=2.43 wall=395.82ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=980.48ms avg_case_ms=9.80 avg_simplify_ms=2.78, sum@0+100 failed=0 elapsed=618.34ms avg_case_ms=6.18 avg_simplify_ms=1.93, product@0+100 failed=0 elapsed=602.59ms avg_case_ms=6.03 avg_simplify_ms=1.74, difference@0+50 failed=0 elapsed=395.82ms avg_case_ms=7.92 avg_simplify_ms=2.43, sum@700+100 failed=0 elapsed=227.91ms avg_case_ms=2.28 avg_simplify_ms=0.82
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.86ms median_wire=16.93ms median_wall=64.42ms, difference@0+50 #174 difference runs=3 median_simplify=14.94ms median_wire=15.00ms median_wall=57.77ms, product@0+100 #175 product runs=3 median_simplify=15.09ms median_wire=15.14ms median_wall=58.37ms, sum@0+100 #173 sum runs=3 median_simplify=14.90ms median_wire=14.96ms median_wall=57.12ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.61ms median_wire=12.68ms median_wall=49.06ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.83s | passed=450 failed=0 total=450 avg_case=6.289ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.28s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.86s | passed=1 failed=0 |
