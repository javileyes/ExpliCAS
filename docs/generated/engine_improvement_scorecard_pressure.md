# Engine Improvement Scorecard

- Generated: 2026-06-24T08:03:25.428729+00:00
- Git branch: main
- Git commit: `d1dd8a07d2dc15fbc72b6eb993ac3959dd77e1cb`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=793.02ms avg_case_ms=7.93 simplify=227.44ms avg_simplify_ms=2.27, sum total=200 failed=0 elapsed=699.80ms avg_case_ms=3.50 simplify=237.03ms avg_simplify_ms=1.19, product total=100 failed=0 elapsed=491.19ms avg_case_ms=4.91 simplify=143.40ms avg_simplify_ms=1.43, difference total=50 failed=0 elapsed=326.72ms avg_case_ms=6.53 simplify=104.59ms avg_simplify_ms=2.09
- Engine hotspots: sum simplify=237.03ms avg_simplify_ms=1.19 wall=699.80ms, shifted_quotient simplify=227.44ms avg_simplify_ms=2.27 wall=793.02ms, product simplify=143.40ms avg_simplify_ms=1.43 wall=491.19ms, difference simplify=104.59ms avg_simplify_ms=2.09 wall=326.72ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=793.02ms avg_case_ms=7.93 avg_simplify_ms=2.27, sum@0+100 failed=0 elapsed=505.78ms avg_case_ms=5.06 avg_simplify_ms=1.64, product@0+100 failed=0 elapsed=491.19ms avg_case_ms=4.91 avg_simplify_ms=1.43, difference@0+50 failed=0 elapsed=326.72ms avg_case_ms=6.53 avg_simplify_ms=2.09, sum@700+100 failed=0 elapsed=194.02ms avg_case_ms=1.94 avg_simplify_ms=0.73
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.96ms median_wire=13.03ms median_wall=49.86ms, difference@0+50 #174 difference runs=3 median_simplify=11.75ms median_wire=11.80ms median_wall=44.74ms, product@0+100 #175 product runs=3 median_simplify=11.79ms median_wire=11.84ms median_wall=44.64ms, sum@0+100 #173 sum runs=3 median_simplify=11.73ms median_wire=11.79ms median_wall=43.92ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.57ms median_wire=10.64ms median_wall=39.94ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.31s | passed=450 failed=0 total=450 avg_case=5.133ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.42s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.04s | passed=1 failed=0 |
