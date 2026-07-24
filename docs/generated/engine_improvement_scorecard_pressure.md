# Engine Improvement Scorecard

- Generated: 2026-07-24T18:36:49.082570+00:00
- Git branch: main
- Git commit: `297f224388a72e11165f77aa454234a02442954f`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=996.73ms avg_case_ms=9.97 simplify=278.79ms avg_simplify_ms=2.79, sum total=200 failed=0 elapsed=856.86ms avg_case_ms=4.28 simplify=277.51ms avg_simplify_ms=1.39, product total=100 failed=0 elapsed=603.16ms avg_case_ms=6.03 simplify=173.22ms avg_simplify_ms=1.73, difference total=50 failed=0 elapsed=398.62ms avg_case_ms=7.97 simplify=121.54ms avg_simplify_ms=2.43
- Engine hotspots: shifted_quotient simplify=278.79ms avg_simplify_ms=2.79 wall=996.73ms, sum simplify=277.51ms avg_simplify_ms=1.39 wall=856.86ms, product simplify=173.22ms avg_simplify_ms=1.73 wall=603.16ms, difference simplify=121.54ms avg_simplify_ms=2.43 wall=398.62ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=996.73ms avg_case_ms=9.97 avg_simplify_ms=2.79, sum@0+100 failed=0 elapsed=625.98ms avg_case_ms=6.26 avg_simplify_ms=1.95, product@0+100 failed=0 elapsed=603.16ms avg_case_ms=6.03 avg_simplify_ms=1.73, difference@0+50 failed=0 elapsed=398.62ms avg_case_ms=7.97 avg_simplify_ms=2.43, sum@700+100 failed=0 elapsed=230.88ms avg_case_ms=2.31 avg_simplify_ms=0.82
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.88ms median_wire=16.95ms median_wall=63.60ms, product@0+100 #175 product runs=3 median_simplify=14.99ms median_wire=15.04ms median_wall=57.25ms, sum@0+100 #173 sum runs=3 median_simplify=16.18ms median_wire=16.23ms median_wall=62.21ms, difference@0+50 #174 difference runs=3 median_simplify=15.11ms median_wire=15.15ms median_wall=57.74ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.97ms median_wire=13.03ms median_wall=51.90ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.86s | passed=450 failed=0 total=450 avg_case=6.356ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.71s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.88s | passed=1 failed=0 |
