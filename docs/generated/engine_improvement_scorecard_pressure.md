# Engine Improvement Scorecard

- Generated: 2026-06-18T23:21:09.159008+00:00
- Git branch: main
- Git commit: `1c149c7e413310118c25465fdb21f6f46448c6f3`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=829.19ms avg_case_ms=8.29 simplify=238.43ms avg_simplify_ms=2.38, sum total=200 failed=0 elapsed=722.99ms avg_case_ms=3.61 simplify=246.79ms avg_simplify_ms=1.23, product total=100 failed=0 elapsed=505.41ms avg_case_ms=5.05 simplify=147.56ms avg_simplify_ms=1.48, difference total=50 failed=0 elapsed=336.67ms avg_case_ms=6.73 simplify=107.64ms avg_simplify_ms=2.15
- Engine hotspots: sum simplify=246.79ms avg_simplify_ms=1.23 wall=722.99ms, shifted_quotient simplify=238.43ms avg_simplify_ms=2.38 wall=829.19ms, product simplify=147.56ms avg_simplify_ms=1.48 wall=505.41ms, difference simplify=107.64ms avg_simplify_ms=2.15 wall=336.67ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=829.19ms avg_case_ms=8.29 avg_simplify_ms=2.38, sum@0+100 failed=0 elapsed=521.52ms avg_case_ms=5.22 avg_simplify_ms=1.71, product@0+100 failed=0 elapsed=505.41ms avg_case_ms=5.05 avg_simplify_ms=1.48, difference@0+50 failed=0 elapsed=336.67ms avg_case_ms=6.73 avg_simplify_ms=2.15, sum@700+100 failed=0 elapsed=201.47ms avg_case_ms=2.01 avg_simplify_ms=0.76
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.57ms median_wire=13.64ms median_wall=52.96ms, difference@0+50 #174 difference runs=3 median_simplify=12.44ms median_wire=12.49ms median_wall=46.37ms, sum@0+100 #173 sum runs=3 median_simplify=12.39ms median_wire=12.44ms median_wall=46.84ms, product@0+100 #175 product runs=3 median_simplify=12.40ms median_wire=12.45ms median_wall=47.90ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=11.44ms median_wire=11.52ms median_wall=43.36ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.40s | passed=450 failed=0 total=450 avg_case=5.333ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.56s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.06s | passed=1 failed=0 |
