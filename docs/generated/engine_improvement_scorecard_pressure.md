# Engine Improvement Scorecard

- Generated: 2026-06-23T19:44:08.162075+00:00
- Git branch: main
- Git commit: `ebbfe767190a1d8b7857a1054be931e391408e4b`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=792.02ms avg_case_ms=7.92 simplify=227.14ms avg_simplify_ms=2.27, sum total=200 failed=0 elapsed=694.88ms avg_case_ms=3.47 simplify=235.33ms avg_simplify_ms=1.18, product total=100 failed=0 elapsed=500.83ms avg_case_ms=5.01 simplify=144.07ms avg_simplify_ms=1.44, difference total=50 failed=0 elapsed=324.96ms avg_case_ms=6.50 simplify=103.64ms avg_simplify_ms=2.07
- Engine hotspots: sum simplify=235.33ms avg_simplify_ms=1.18 wall=694.88ms, shifted_quotient simplify=227.14ms avg_simplify_ms=2.27 wall=792.02ms, product simplify=144.07ms avg_simplify_ms=1.44 wall=500.83ms, difference simplify=103.64ms avg_simplify_ms=2.07 wall=324.96ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=792.02ms avg_case_ms=7.92 avg_simplify_ms=2.27, sum@0+100 failed=0 elapsed=505.08ms avg_case_ms=5.05 avg_simplify_ms=1.64, product@0+100 failed=0 elapsed=500.83ms avg_case_ms=5.01 avg_simplify_ms=1.44, difference@0+50 failed=0 elapsed=324.96ms avg_case_ms=6.50 avg_simplify_ms=2.07, sum@700+100 failed=0 elapsed=189.80ms avg_case_ms=1.90 avg_simplify_ms=0.71
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.09ms median_wire=13.16ms median_wall=50.16ms, product@0+100 #175 product runs=3 median_simplify=11.58ms median_wire=11.63ms median_wall=44.21ms, sum@0+100 #173 sum runs=3 median_simplify=11.52ms median_wire=11.57ms median_wall=44.04ms, difference@0+50 #174 difference runs=3 median_simplify=11.81ms median_wire=11.86ms median_wall=44.47ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.68ms median_wire=10.75ms median_wall=40.23ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.31s | passed=450 failed=0 total=450 avg_case=5.133ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.42s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.03s | passed=1 failed=0 |
