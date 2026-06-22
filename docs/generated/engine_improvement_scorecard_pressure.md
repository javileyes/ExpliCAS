# Engine Improvement Scorecard

- Generated: 2026-06-22T20:25:00.376506+00:00
- Git branch: main
- Git commit: `15fef822ce67510e18c1a32b6bbbe431eb5c3390`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=781.71ms avg_case_ms=7.82 simplify=223.48ms avg_simplify_ms=2.23, sum total=200 failed=0 elapsed=700.22ms avg_case_ms=3.50 simplify=236.71ms avg_simplify_ms=1.18, product total=100 failed=0 elapsed=474.12ms avg_case_ms=4.74 simplify=136.46ms avg_simplify_ms=1.36, difference total=50 failed=0 elapsed=325.97ms avg_case_ms=6.52 simplify=104.31ms avg_simplify_ms=2.09
- Engine hotspots: sum simplify=236.71ms avg_simplify_ms=1.18 wall=700.22ms, shifted_quotient simplify=223.48ms avg_simplify_ms=2.23 wall=781.71ms, product simplify=136.46ms avg_simplify_ms=1.36 wall=474.12ms, difference simplify=104.31ms avg_simplify_ms=2.09 wall=325.97ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=781.71ms avg_case_ms=7.82 avg_simplify_ms=2.23, sum@0+100 failed=0 elapsed=505.88ms avg_case_ms=5.06 avg_simplify_ms=1.64, product@0+100 failed=0 elapsed=474.12ms avg_case_ms=4.74 avg_simplify_ms=1.36, difference@0+50 failed=0 elapsed=325.97ms avg_case_ms=6.52 avg_simplify_ms=2.09, sum@700+100 failed=0 elapsed=194.35ms avg_case_ms=1.94 avg_simplify_ms=0.73
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.11ms median_wire=13.18ms median_wall=50.04ms, sum@0+100 #173 sum runs=3 median_simplify=11.79ms median_wire=11.84ms median_wall=44.73ms, product@0+100 #175 product runs=3 median_simplify=11.83ms median_wire=11.89ms median_wall=44.96ms, difference@0+50 #174 difference runs=3 median_simplify=11.75ms median_wire=11.80ms median_wall=44.41ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.57ms median_wire=10.64ms median_wall=39.92ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.28s | passed=450 failed=0 total=450 avg_case=5.067ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.40s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.05s | passed=1 failed=0 |
