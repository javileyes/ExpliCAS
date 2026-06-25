# Engine Improvement Scorecard

- Generated: 2026-06-25T07:57:02.186345+00:00
- Git branch: main
- Git commit: `c06e5358f3e812e89220f570a3e35df2109274f1`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=794.83ms avg_case_ms=7.95 simplify=227.44ms avg_simplify_ms=2.27, sum total=200 failed=0 elapsed=698.85ms avg_case_ms=3.49 simplify=237.36ms avg_simplify_ms=1.19, product total=100 failed=0 elapsed=484.92ms avg_case_ms=4.85 simplify=139.53ms avg_simplify_ms=1.40, difference total=50 failed=0 elapsed=326.21ms avg_case_ms=6.52 simplify=104.48ms avg_simplify_ms=2.09
- Engine hotspots: sum simplify=237.36ms avg_simplify_ms=1.19 wall=698.85ms, shifted_quotient simplify=227.44ms avg_simplify_ms=2.27 wall=794.83ms, product simplify=139.53ms avg_simplify_ms=1.40 wall=484.92ms, difference simplify=104.48ms avg_simplify_ms=2.09 wall=326.21ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=794.83ms avg_case_ms=7.95 avg_simplify_ms=2.27, sum@0+100 failed=0 elapsed=505.54ms avg_case_ms=5.06 avg_simplify_ms=1.65, product@0+100 failed=0 elapsed=484.92ms avg_case_ms=4.85 avg_simplify_ms=1.40, difference@0+50 failed=0 elapsed=326.21ms avg_case_ms=6.52 avg_simplify_ms=2.09, sum@700+100 failed=0 elapsed=193.31ms avg_case_ms=1.93 avg_simplify_ms=0.73
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.18ms median_wire=13.26ms median_wall=50.04ms, product@0+100 #175 product runs=3 median_simplify=11.67ms median_wire=11.72ms median_wall=44.50ms, sum@0+100 #173 sum runs=3 median_simplify=11.69ms median_wire=11.74ms median_wall=44.43ms, difference@0+50 #174 difference runs=3 median_simplify=11.70ms median_wire=11.74ms median_wall=44.31ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.63ms median_wire=10.70ms median_wall=40.30ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.31s | passed=450 failed=0 total=450 avg_case=5.133ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.44s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.05s | passed=1 failed=0 |
