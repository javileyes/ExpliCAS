# Engine Improvement Scorecard

- Generated: 2026-06-19T15:46:04.768932+00:00
- Git branch: main
- Git commit: `b27cf094938311e3fa3636c61b50f8e28f6edbb5`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=806.63ms avg_case_ms=8.07 simplify=231.41ms avg_simplify_ms=2.31, sum total=200 failed=0 elapsed=728.73ms avg_case_ms=3.64 simplify=248.35ms avg_simplify_ms=1.24, product total=100 failed=0 elapsed=491.64ms avg_case_ms=4.92 simplify=141.73ms avg_simplify_ms=1.42, difference total=50 failed=0 elapsed=336.62ms avg_case_ms=6.73 simplify=107.48ms avg_simplify_ms=2.15
- Engine hotspots: sum simplify=248.35ms avg_simplify_ms=1.24 wall=728.73ms, shifted_quotient simplify=231.41ms avg_simplify_ms=2.31 wall=806.63ms, product simplify=141.73ms avg_simplify_ms=1.42 wall=491.64ms, difference simplify=107.48ms avg_simplify_ms=2.15 wall=336.62ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=806.63ms avg_case_ms=8.07 avg_simplify_ms=2.31, sum@0+100 failed=0 elapsed=528.91ms avg_case_ms=5.29 avg_simplify_ms=1.74, product@0+100 failed=0 elapsed=491.64ms avg_case_ms=4.92 avg_simplify_ms=1.42, difference@0+50 failed=0 elapsed=336.62ms avg_case_ms=6.73 avg_simplify_ms=2.15, sum@700+100 failed=0 elapsed=199.82ms avg_case_ms=2.00 avg_simplify_ms=0.75
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.08ms median_wire=13.15ms median_wall=49.99ms, sum@0+100 #173 sum runs=3 median_simplify=11.74ms median_wire=11.80ms median_wall=44.50ms, difference@0+50 #174 difference runs=3 median_simplify=11.91ms median_wire=11.96ms median_wall=44.86ms, product@0+100 #175 product runs=3 median_simplify=12.15ms median_wire=12.21ms median_wall=45.53ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.91ms median_wire=10.99ms median_wall=40.93ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.37s | passed=450 failed=0 total=450 avg_case=5.267ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.48s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.04s | passed=1 failed=0 |
