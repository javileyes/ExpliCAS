# Engine Improvement Scorecard

- Generated: 2026-07-28T21:50:57.214144+00:00
- Git branch: main
- Git commit: `9a9e1761be8c5b30c363e21e9e5a954d84c93012`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=380

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=977.12ms avg_case_ms=9.77 simplify=275.60ms avg_simplify_ms=2.76, sum total=200 failed=0 elapsed=857.62ms avg_case_ms=4.29 simplify=278.80ms avg_simplify_ms=1.39, product total=100 failed=0 elapsed=606.31ms avg_case_ms=6.06 simplify=174.83ms avg_simplify_ms=1.75, difference total=50 failed=0 elapsed=398.32ms avg_case_ms=7.97 simplify=122.01ms avg_simplify_ms=2.44
- Engine hotspots: sum simplify=278.80ms avg_simplify_ms=1.39 wall=857.62ms, shifted_quotient simplify=275.60ms avg_simplify_ms=2.76 wall=977.12ms, product simplify=174.83ms avg_simplify_ms=1.75 wall=606.31ms, difference simplify=122.01ms avg_simplify_ms=2.44 wall=398.32ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=977.12ms avg_case_ms=9.77 avg_simplify_ms=2.76, sum@0+100 failed=0 elapsed=627.81ms avg_case_ms=6.28 avg_simplify_ms=1.97, product@0+100 failed=0 elapsed=606.31ms avg_case_ms=6.06 avg_simplify_ms=1.75, difference@0+50 failed=0 elapsed=398.32ms avg_case_ms=7.97 avg_simplify_ms=2.44, sum@700+100 failed=0 elapsed=229.82ms avg_case_ms=2.30 avg_simplify_ms=0.82
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.80ms median_wire=16.86ms median_wall=64.49ms, difference@0+50 #174 difference runs=3 median_simplify=15.10ms median_wire=15.14ms median_wall=57.67ms, product@0+100 #175 product runs=3 median_simplify=15.16ms median_wire=15.22ms median_wall=58.20ms, sum@0+100 #173 sum runs=3 median_simplify=15.11ms median_wire=15.15ms median_wall=57.99ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.90ms median_wire=12.98ms median_wall=49.49ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.84s | passed=450 failed=0 total=450 avg_case=6.311ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.38s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.85s | passed=1 failed=0 |
