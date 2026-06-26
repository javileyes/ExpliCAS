# Engine Improvement Scorecard

- Generated: 2026-06-26T07:24:23.755165+00:00
- Git branch: main
- Git commit: `007a5eb840bc0582490816ab977e149b2d82fabb`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=808.95ms avg_case_ms=8.09 simplify=233.17ms avg_simplify_ms=2.33, sum total=200 failed=0 elapsed=716.12ms avg_case_ms=3.58 simplify=248.24ms avg_simplify_ms=1.24, product total=100 failed=0 elapsed=495.74ms avg_case_ms=4.96 simplify=147.47ms avg_simplify_ms=1.47, difference total=50 failed=0 elapsed=337.79ms avg_case_ms=6.76 simplify=109.52ms avg_simplify_ms=2.19
- Engine hotspots: sum simplify=248.24ms avg_simplify_ms=1.24 wall=716.12ms, shifted_quotient simplify=233.17ms avg_simplify_ms=2.33 wall=808.95ms, product simplify=147.47ms avg_simplify_ms=1.47 wall=495.74ms, difference simplify=109.52ms avg_simplify_ms=2.19 wall=337.79ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=808.95ms avg_case_ms=8.09 avg_simplify_ms=2.33, sum@0+100 failed=0 elapsed=513.72ms avg_case_ms=5.14 avg_simplify_ms=1.70, product@0+100 failed=0 elapsed=495.74ms avg_case_ms=4.96 avg_simplify_ms=1.47, difference@0+50 failed=0 elapsed=337.79ms avg_case_ms=6.76 avg_simplify_ms=2.19, sum@700+100 failed=0 elapsed=202.39ms avg_case_ms=2.02 avg_simplify_ms=0.79
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.26ms median_wire=13.34ms median_wall=50.44ms, sum@0+100 #173 sum runs=3 median_simplify=11.93ms median_wire=11.99ms median_wall=45.76ms, difference@0+50 #174 difference runs=3 median_simplify=11.81ms median_wire=11.86ms median_wall=45.15ms, product@0+100 #175 product runs=3 median_simplify=11.83ms median_wire=11.88ms median_wall=45.46ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=11.07ms median_wire=11.15ms median_wall=41.50ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.36s | passed=450 failed=0 total=450 avg_case=5.244ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.48s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.09s | passed=1 failed=0 |
