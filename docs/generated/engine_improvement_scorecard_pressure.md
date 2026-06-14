# Engine Improvement Scorecard

- Generated: 2026-06-14T13:38:02.972292+00:00
- Git branch: main
- Git commit: `9c4e6b2a9ce2b71d9687f8c6780c94273ef88c66`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=349

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=782.65ms avg_case_ms=7.83 simplify=222.26ms avg_simplify_ms=2.22, sum total=200 failed=0 elapsed=687.19ms avg_case_ms=3.44 simplify=230.55ms avg_simplify_ms=1.15, product total=100 failed=0 elapsed=498.29ms avg_case_ms=4.98 simplify=144.23ms avg_simplify_ms=1.44, difference total=50 failed=0 elapsed=318.21ms avg_case_ms=6.36 simplify=100.86ms avg_simplify_ms=2.02
- Engine hotspots: sum simplify=230.55ms avg_simplify_ms=1.15 wall=687.19ms, shifted_quotient simplify=222.26ms avg_simplify_ms=2.22 wall=782.65ms, product simplify=144.23ms avg_simplify_ms=1.44 wall=498.29ms, difference simplify=100.86ms avg_simplify_ms=2.02 wall=318.21ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=782.65ms avg_case_ms=7.83 avg_simplify_ms=2.22, product@0+100 failed=0 elapsed=498.29ms avg_case_ms=4.98 avg_simplify_ms=1.44, sum@0+100 failed=0 elapsed=496.39ms avg_case_ms=4.96 avg_simplify_ms=1.60, difference@0+50 failed=0 elapsed=318.21ms avg_case_ms=6.36 avg_simplify_ms=2.02, sum@700+100 failed=0 elapsed=190.80ms avg_case_ms=1.91 avg_simplify_ms=0.71
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.78ms median_wire=12.85ms median_wall=48.88ms, product@0+100 #175 product runs=3 median_simplify=11.44ms median_wire=11.48ms median_wall=43.73ms, sum@0+100 #173 sum runs=3 median_simplify=11.23ms median_wire=11.28ms median_wall=43.24ms, difference@0+50 #174 difference runs=3 median_simplify=11.33ms median_wire=11.38ms median_wall=43.28ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.41ms median_wire=10.48ms median_wall=39.71ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.29s | passed=450 failed=0 total=450 avg_case=5.089ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.80s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.91s | passed=1 failed=0 |
