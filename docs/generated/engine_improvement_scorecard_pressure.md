# Engine Improvement Scorecard

- Generated: 2026-06-19T06:43:41.883170+00:00
- Git branch: main
- Git commit: `1291a5de740273806c52ed6dae4a63484542e3bb`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=808.72ms avg_case_ms=8.09 simplify=232.63ms avg_simplify_ms=2.33, sum total=200 failed=0 elapsed=723.79ms avg_case_ms=3.62 simplify=247.36ms avg_simplify_ms=1.24, product total=100 failed=0 elapsed=492.32ms avg_case_ms=4.92 simplify=142.87ms avg_simplify_ms=1.43, difference total=50 failed=0 elapsed=338.71ms avg_case_ms=6.77 simplify=109.23ms avg_simplify_ms=2.18
- Engine hotspots: sum simplify=247.36ms avg_simplify_ms=1.24 wall=723.79ms, shifted_quotient simplify=232.63ms avg_simplify_ms=2.33 wall=808.72ms, product simplify=142.87ms avg_simplify_ms=1.43 wall=492.32ms, difference simplify=109.23ms avg_simplify_ms=2.18 wall=338.71ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=808.72ms avg_case_ms=8.09 avg_simplify_ms=2.33, sum@0+100 failed=0 elapsed=519.17ms avg_case_ms=5.19 avg_simplify_ms=1.71, product@0+100 failed=0 elapsed=492.32ms avg_case_ms=4.92 avg_simplify_ms=1.43, difference@0+50 failed=0 elapsed=338.71ms avg_case_ms=6.77 avg_simplify_ms=2.18, sum@700+100 failed=0 elapsed=204.62ms avg_case_ms=2.05 avg_simplify_ms=0.77
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.09ms median_wire=13.16ms median_wall=49.74ms, product@0+100 #175 product runs=3 median_simplify=11.84ms median_wire=11.90ms median_wall=45.19ms, sum@0+100 #173 sum runs=3 median_simplify=11.81ms median_wire=11.86ms median_wall=44.77ms, difference@0+50 #174 difference runs=3 median_simplify=12.09ms median_wire=12.14ms median_wall=45.36ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.84ms median_wire=10.91ms median_wall=40.98ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.36s | passed=450 failed=0 total=450 avg_case=5.244ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.44s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.07s | passed=1 failed=0 |
