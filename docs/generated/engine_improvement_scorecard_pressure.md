# Engine Improvement Scorecard

- Generated: 2026-07-22T21:13:04.523837+00:00
- Git branch: main
- Git commit: `30c9fb2ffad1ac2dfdbb83719df6cf56aeda35c0`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=367

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=992.90ms avg_case_ms=9.93 simplify=278.59ms avg_simplify_ms=2.79, sum total=200 failed=0 elapsed=876.86ms avg_case_ms=4.38 simplify=287.08ms avg_simplify_ms=1.44, product total=100 failed=0 elapsed=608.48ms avg_case_ms=6.08 simplify=174.20ms avg_simplify_ms=1.74, difference total=50 failed=0 elapsed=398.83ms avg_case_ms=7.98 simplify=121.03ms avg_simplify_ms=2.42
- Engine hotspots: sum simplify=287.08ms avg_simplify_ms=1.44 wall=876.86ms, shifted_quotient simplify=278.59ms avg_simplify_ms=2.79 wall=992.90ms, product simplify=174.20ms avg_simplify_ms=1.74 wall=608.48ms, difference simplify=121.03ms avg_simplify_ms=2.42 wall=398.83ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=992.90ms avg_case_ms=9.93 avg_simplify_ms=2.79, sum@0+100 failed=0 elapsed=644.02ms avg_case_ms=6.44 avg_simplify_ms=2.04, product@0+100 failed=0 elapsed=608.48ms avg_case_ms=6.08 avg_simplify_ms=1.74, difference@0+50 failed=0 elapsed=398.83ms avg_case_ms=7.98 avg_simplify_ms=2.42, sum@700+100 failed=0 elapsed=232.83ms avg_case_ms=2.33 avg_simplify_ms=0.83
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.60ms median_wire=16.66ms median_wall=63.90ms, product@0+100 #175 product runs=3 median_simplify=15.27ms median_wire=15.31ms median_wall=58.75ms, difference@0+50 #174 difference runs=3 median_simplify=15.31ms median_wire=15.36ms median_wall=58.76ms, sum@0+100 #173 sum runs=3 median_simplify=16.11ms median_wire=16.16ms median_wall=59.06ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.92ms median_wire=12.99ms median_wall=49.52ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.88s | passed=450 failed=0 total=450 avg_case=6.400ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.59s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.93s | passed=1 failed=0 |
